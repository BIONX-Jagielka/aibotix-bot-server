# aibotix/risk/protection.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Dict
import time

from alpaca.trading.requests import StopOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

log = logging.getLogger("aibotix.risk")

@dataclass
class ProtectionState:
    """Protection state for a single position"""
    symbol: str
    entry_price: float
    quantity: float
    lock_trigger_pct: float  # minimum 0.005 per spec
    current_lock_pct: float
    stop_order_id: Optional[str] = None
    
    # Hysteresis tracking
    consecutive_breaches: int = 0
    last_breach_time: float = 0
    last_unreal_pct: float = 0
    
    def __post_init__(self):
        # Enforce minimum lock trigger
        if self.lock_trigger_pct < 0.005:
            self.lock_trigger_pct = 0.005
            
    def is_lock_violation(self, unreal_pct: float) -> bool:
        """
        Check if current unrealized percentage violates profit lock
        Implements hysteresis: requires 2 consecutive breaches OR breach by > 0.0015
        """
        now = time.time()
        
        # Check if we're below the lock threshold
        if unreal_pct >= self.current_lock_pct:
            # Above threshold - reset breach tracking
            self.consecutive_breaches = 0
            self.last_unreal_pct = unreal_pct
            return False
            
        # Below threshold - check violation conditions
        breach_size = self.current_lock_pct - unreal_pct
        
        # Condition 1: Large breach (> 0.0015 absolute)
        if breach_size > 0.0015:
            log.warning(
                f"[RISK] Large lock breach detected: {self.symbol} "
                f"unreal_pct={unreal_pct:.4f} vs lock={self.current_lock_pct:.4f} "
                f"breach_size={breach_size:.4f}"
            )
            return True
            
        # Condition 2: Consecutive breach tracking
        if now - self.last_breach_time < 60:  # Within 1 minute of last breach
            self.consecutive_breaches += 1
        else:
            self.consecutive_breaches = 1  # Reset if too much time passed
            
        self.last_breach_time = now
        self.last_unreal_pct = unreal_pct
        
        if self.consecutive_breaches >= 2:
            log.warning(
                f"[RISK] Consecutive lock breach detected: {self.symbol} "
                f"unreal_pct={unreal_pct:.4f} vs lock={self.current_lock_pct:.4f} "
                f"breaches={self.consecutive_breaches}"
            )
            return True
            
        return False


class ProtectionManager:
    """
    Manages profit locks and protective stop orders for positions
    Section 4 implementation with broker-backed stops
    """
    
    def __init__(self, alpaca_trading_client):
        self.alpaca_client = alpaca_trading_client
        self.positions: Dict[str, ProtectionState] = {}
        
    def add_position(self, symbol: str, entry_price: float, quantity: float, 
                    initial_stop_price: float, lock_trigger_pct: float = 0.005) -> bool:
        """
        Add a new position with initial protective stop
        Returns True if stop order was successfully created
        """
        try:
            # Create initial protective stop order
            stop_request = StopOrderRequest(
                symbol=symbol,
                qty=abs(quantity),
                side=OrderSide.SELL if quantity > 0 else OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                stop_price=initial_stop_price
            )
            
            response = self.alpaca_client.submit_order(stop_request)
            stop_order_id = response.id
            
            # Create protection state
            state = ProtectionState(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                lock_trigger_pct=max(lock_trigger_pct, 0.005),  # Enforce minimum
                current_lock_pct=0.0,  # No lock initially
                stop_order_id=stop_order_id
            )
            
            self.positions[symbol] = state
            
            log.info(
                f"[RISK] Added protection for {symbol}: "
                f"qty={quantity}, entry=${entry_price:.4f}, "
                f"initial_stop=${initial_stop_price:.4f}, "
                f"stop_order_id={stop_order_id}"
            )
            
            return True
            
        except Exception as e:
            log.error(
                f"[RISK] Failed to create initial stop for {symbol}: {e}",
                extra={
                    "event_type": "stop_creation_failed",
                    "symbol": symbol,
                    "quantity": quantity,
                    "stop_price": initial_stop_price
                }
            )
            return False
            
    def update_position(self, symbol: str, current_price: float) -> bool:
        """
        Update position protection based on current price
        Handles profit lock activation and stop ratcheting
        Returns True if any action was taken
        """
        if symbol not in self.positions:
            return False
            
        state = self.positions[symbol]
        
        # Calculate unrealized percentage
        if state.quantity > 0:  # Long position
            unreal_pct = (current_price - state.entry_price) / state.entry_price
        else:  # Short position  
            unreal_pct = (state.entry_price - current_price) / state.entry_price
            
        # Check if profit lock should be activated/updated
        if unreal_pct > state.lock_trigger_pct and unreal_pct > state.current_lock_pct:
            # Update profit lock to new level (ratchet up)
            new_lock_pct = unreal_pct - 0.002  # 0.2% buffer below current profit
            
            if new_lock_pct > state.current_lock_pct:
                state.current_lock_pct = new_lock_pct
                
                # Calculate new stop price  
                if state.quantity > 0:  # Long position
                    new_stop_price = state.entry_price * (1 + new_lock_pct)
                else:  # Short position
                    new_stop_price = state.entry_price * (1 - new_lock_pct)
                    
                # Update stop order
                if self._update_stop_order(state, new_stop_price):
                    log.info(
                        f"[RISK] Profit lock ratcheted: {symbol} "
                        f"unreal_pct={unreal_pct:.4f} -> lock_pct={new_lock_pct:.4f} "
                        f"new_stop=${new_stop_price:.4f}"
                    )
                    return True
                    
        # Check for lock violation
        if state.current_lock_pct > 0 and state.is_lock_violation(unreal_pct):
            return self._exit_position_immediately(state, "profit_lock_violation")
            
        return False
        
    def _update_stop_order(self, state: ProtectionState, new_stop_price: float) -> bool:
        """
        Update the protective stop order price
        Attempts replace first, then cancel+recreate if that fails
        """
        if not state.stop_order_id:
            return False
            
        try:
            # Try to replace existing stop order (preferred method)
            try:
                from alpaca.trading.requests import ReplaceOrderRequest
                replace_request = ReplaceOrderRequest(
                    qty=abs(state.quantity),
                    stop_price=new_stop_price
                )
                response = self.alpaca_client.replace_order(state.stop_order_id, replace_request)
                state.stop_order_id = response.id
                return True
                
            except Exception as replace_error:
                # If replace fails, try cancel + recreate
                log.warning(f"[RISK] Stop order replace failed for {state.symbol}, trying cancel+recreate: {replace_error}")
                
                # Cancel existing order
                self.alpaca_client.cancel_order_by_id(state.stop_order_id)
                
                # Create new stop order
                stop_request = StopOrderRequest(
                    symbol=state.symbol,
                    qty=abs(state.quantity),
                    side=OrderSide.SELL if state.quantity > 0 else OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    stop_price=new_stop_price
                )
                
                response = self.alpaca_client.submit_order(stop_request)
                state.stop_order_id = response.id
                return True
                
        except Exception as e:
            # Complete stop update failure - force exit position
            log.error(
                f"[RISK] forced_exit_stop_update_failed: {state.symbol}, "
                f"qty={state.quantity}, last_stop_price={new_stop_price:.4f}, error={e}",
                extra={
                    "event_type": "forced_exit_stop_update_failed",
                    "symbol": state.symbol,
                    "quantity": state.quantity,
                    "last_stop_price": new_stop_price
                }
            )
            
            # Immediately market-exit the position
            return self._exit_position_immediately(state, "stop_update_failed")
            
    def _exit_position_immediately(self, state: ProtectionState, reason: str) -> bool:
        """
        Immediately exit position via market order and clean up protection state
        """
        try:
            # Submit market order to close position
            market_request = MarketOrderRequest(
                symbol=state.symbol,
                qty=abs(state.quantity),
                side=OrderSide.SELL if state.quantity > 0 else OrderSide.BUY,
                time_in_force=TimeInForce.IOC  # Immediate or cancel
            )
            
            response = self.alpaca_client.submit_order(market_request)
            
            # Cancel any existing stop order
            if state.stop_order_id:
                try:
                    self.alpaca_client.cancel_order_by_id(state.stop_order_id)
                except Exception as cancel_error:
                    log.warning(f"[RISK] Failed to cancel stop order {state.stop_order_id}: {cancel_error}")
            
            # Remove from protection
            if state.symbol in self.positions:
                del self.positions[state.symbol]
            
            log.warning(
                f"[RISK] Position immediately exited: {state.symbol}, "
                f"reason={reason}, qty={state.quantity}, order_id={response.id}",
                extra={
                    "event_type": "forced_exit",
                    "reason": reason,
                    "symbol": state.symbol,
                    "quantity": state.quantity,
                    "exit_order_id": response.id
                }
            )
            
            return True
            
        except Exception as e:
            log.error(
                f"[RISK] Failed to exit position {state.symbol}: {e}",
                extra={
                    "event_type": "forced_exit_failed",
                    "symbol": state.symbol,
                    "reason": reason,
                    "error": str(e)
                }
            )
            return False
            
    def remove_position(self, symbol: str) -> bool:
        """
        Remove position from protection (when position is closed normally)
        Cancels any existing stop orders
        """
        if symbol not in self.positions:
            return False
            
        state = self.positions[symbol]
        
        # Cancel stop order if it exists
        if state.stop_order_id:
            try:
                self.alpaca_client.cancel_order_by_id(state.stop_order_id)
                log.info(f"[RISK] Cancelled stop order {state.stop_order_id} for {symbol}")
            except Exception as e:
                log.warning(f"[RISK] Failed to cancel stop order for {symbol}: {e}")
                
        # Remove from tracking
        del self.positions[symbol]
        log.info(f"[RISK] Removed protection for {symbol}")
        return True
        
    def get_protected_symbols(self) -> list[str]:
        """Get list of symbols currently under protection"""
        return list(self.positions.keys())
        
    def get_protection_state(self, symbol: str) -> Optional[ProtectionState]:
        """Get protection state for a specific symbol"""
        return self.positions.get(symbol)