from __future__ import annotations
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..utils.http_client import ResilientSession, HttpConfig
from ..observability import log_stop_update_failure, log_structured, MetricEvent

log = logging.getLogger("aibotix.trading")

class StopOrderStatus(Enum):
    ACTIVE = "active"
    FILLED = "filled" 
    CANCELLED = "cancelled"
    FAILED = "failed"
    PENDING_UPDATE = "pending_update"

@dataclass
class StopOrder:
    """Reliable stop order tracking"""
    symbol: str
    order_id: Optional[str] = None
    stop_price: float = 0.0
    quantity: int = 0
    side: str = "sell"  # sell for long positions
    status: StopOrderStatus = StopOrderStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_broker_sync: Optional[datetime] = None
    update_attempts: int = 0
    max_update_attempts: int = 3
    fallback_enabled: bool = True

class ReliableStopManager:
    """
    Manages stop orders with broker sync and fallback mechanisms
    """
    
    def __init__(self, alpaca_trading_client=None, session: Optional[ResilientSession] = None):
        self.alpaca_client = alpaca_trading_client
        self.session = session or ResilientSession(HttpConfig())
        self.active_stops: Dict[str, StopOrder] = {}
        self.sync_interval = timedelta(minutes=5)  # Sync with broker every 5min
        self.stop_buffer = 0.02  # 2% buffer for stop placement
        
    def create_stop_order(self, symbol: str, stop_price: float, 
                         quantity: int, side: str = "sell") -> Optional[StopOrder]:
        """Create a new stop order with broker submission"""
        start_time = time.time()
        
        try:
            # Create stop order object
            stop_order = StopOrder(
                symbol=symbol,
                stop_price=stop_price,
                quantity=quantity,
                side=side,
                status=StopOrderStatus.PENDING_UPDATE
            )
            
            # Submit to broker
            if self.alpaca_client:
                order_id = self._submit_stop_to_broker(stop_order)
                if order_id:
                    stop_order.order_id = order_id
                    stop_order.status = StopOrderStatus.ACTIVE
                    stop_order.last_broker_sync = datetime.now()
                    
                    # Track the stop order
                    self.active_stops[symbol] = stop_order
                    
                    log_structured(MetricEvent(
                        event_type="stop_created",
                        symbol=symbol,
                        stop_price=stop_price,
                        order_id=order_id,
                        elapsed=time.time() - start_time
                    ))
                    
                    return stop_order
                else:
                    stop_order.status = StopOrderStatus.FAILED
                    
            # If broker submission failed, still track locally for fallback
            if stop_order.fallback_enabled:
                self.active_stops[symbol] = stop_order
                log.warning(f"Stop order for {symbol} using local fallback")
                return stop_order
                
        except Exception as e:
            log_stop_update_failure(
                symbol=symbol,
                operation="create",
                stop_price=stop_price,
                attempt=1,
                exception_type=type(e).__name__,
                elapsed=time.time() - start_time
            )
            
        return None
    
    def update_stop_price(self, symbol: str, new_stop_price: float) -> bool:
        """Update stop price with broker sync"""
        if symbol not in self.active_stops:
            return False
            
        start_time = time.time()
        stop_order = self.active_stops[symbol]
        old_price = stop_order.stop_price
        
        try:
            stop_order.update_attempts += 1
            stop_order.status = StopOrderStatus.PENDING_UPDATE
            
            # Try to update with broker
            if (stop_order.order_id and self.alpaca_client and 
                stop_order.update_attempts <= stop_order.max_update_attempts):
                
                success = self._update_stop_with_broker(stop_order, new_stop_price)
                if success:
                    stop_order.stop_price = new_stop_price
                    stop_order.status = StopOrderStatus.ACTIVE
                    stop_order.updated_at = datetime.now()
                    stop_order.last_broker_sync = datetime.now()
                    
                    log_structured(MetricEvent(
                        event_type="stop_updated",
                        symbol=symbol,
                        old_price=old_price,
                        new_price=new_stop_price,
                        elapsed=time.time() - start_time
                    ))
                    
                    return True
                    
            # Broker update failed, use local tracking
            if stop_order.fallback_enabled:
                stop_order.stop_price = new_stop_price
                stop_order.status = StopOrderStatus.ACTIVE
                stop_order.updated_at = datetime.now()
                
                log.warning(f"Stop update for {symbol} using local fallback")
                return True
                
        except Exception as e:
            log_stop_update_failure(
                symbol=symbol,
                operation="update",
                stop_price=new_stop_price,
                attempt=stop_order.update_attempts,
                exception_type=type(e).__name__,
                elapsed=time.time() - start_time
            )
            
        stop_order.status = StopOrderStatus.FAILED
        return False
    
    def check_stop_triggered(self, symbol: str, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if stop should be triggered (local fallback mechanism)
        Returns (triggered, reason)
        """
        if symbol not in self.active_stops:
            return False, None
            
        stop_order = self.active_stops[symbol]
        
        # If broker-backed and recently synced, trust broker status
        if (stop_order.order_id and stop_order.last_broker_sync and
            (datetime.now() - stop_order.last_broker_sync) < self.sync_interval):
            
            # Check broker status
            broker_status = self._check_broker_stop_status(stop_order)
            if broker_status in [StopOrderStatus.FILLED, StopOrderStatus.CANCELLED]:
                stop_order.status = broker_status
                return broker_status == StopOrderStatus.FILLED, f"broker_{broker_status.value}"
        
        # Local fallback check
        if stop_order.status == StopOrderStatus.ACTIVE:
            if stop_order.side == "sell" and current_price <= stop_order.stop_price:
                return True, "local_fallback"
            elif stop_order.side == "buy" and current_price >= stop_order.stop_price:
                return True, "local_fallback"
                
        return False, None

    def check_stop_triggers(self, symbol: str, current_price: float):
        """
        Backwards-compatible wrapper for legacy call-sites that expect
        check_stop_triggers(). Delegates to check_stop_triggered().
        """
        return self.check_stop_triggered(symbol, current_price)
    
    def remove_stop(self, symbol: str, reason: str = "manual") -> bool:
        """Remove stop order"""
        if symbol not in self.active_stops:
            return False
            
        stop_order = self.active_stops[symbol]
        
        try:
            # Try to cancel with broker
            if stop_order.order_id and self.alpaca_client:
                self._cancel_stop_with_broker(stop_order)
                
            # Remove from tracking
            del self.active_stops[symbol]
            
            log_structured(MetricEvent(
                event_type="stop_removed",
                symbol=symbol,
                reason=reason,
                order_id=stop_order.order_id
            ))
            
            return True
            
        except Exception as e:
            log.error(f"Error removing stop for {symbol}: {e}")
            # Still remove from local tracking
            if symbol in self.active_stops:
                del self.active_stops[symbol]
            return False
    
    def sync_with_broker(self) -> Dict[str, bool]:
        """Sync all stops with broker"""
        results = {}
        
        for symbol, stop_order in self.active_stops.items():
            try:
                if stop_order.order_id and self.alpaca_client:
                    status = self._check_broker_stop_status(stop_order)
                    stop_order.last_broker_sync = datetime.now()
                    
                    if status != stop_order.status:
                        log_structured(MetricEvent(
                            event_type="stop_status_change",
                            symbol=symbol,
                            old_status=stop_order.status.value,
                            new_status=status.value
                        ))
                        stop_order.status = status
                        
                    results[symbol] = True
                else:
                    results[symbol] = False
                    
            except Exception as e:
                log.error(f"Broker sync failed for {symbol}: {e}")
                results[symbol] = False
                
        return results
    
    def _submit_stop_to_broker(self, stop_order: StopOrder) -> Optional[str]:
        """Submit stop order to Alpaca"""
        try:
            from alpaca.trading.requests import StopOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            side = OrderSide.SELL if stop_order.side == "sell" else OrderSide.BUY
            
            request = StopOrderRequest(
                symbol=stop_order.symbol,
                qty=stop_order.quantity,
                side=side,
                time_in_force=TimeInForce.GTC,
                stop_price=stop_order.stop_price
            )
            
            response = self.alpaca_client.submit_order(request)
            return response.id if response else None
            
        except Exception as e:
            log.error(f"Failed to submit stop order to broker: {e}")
            return None
    
    def _update_stop_with_broker(self, stop_order: StopOrder, new_price: float) -> bool:
        """Update stop order with broker"""
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            
            request = ReplaceOrderRequest(
                qty=stop_order.quantity,
                time_in_force="gtc",
                stop_price=new_price
            )
            
            response = self.alpaca_client.replace_order_by_id(stop_order.order_id, request)
            return response is not None
            
        except Exception as e:
            log.error(f"Failed to update stop order with broker: {e}")
            return False
    
    def _check_broker_stop_status(self, stop_order: StopOrder) -> StopOrderStatus:
        """Check stop order status with broker"""
        try:
            order = self.alpaca_client.get_order_by_id(stop_order.order_id)
            
            status_map = {
                "new": StopOrderStatus.ACTIVE,
                "partially_filled": StopOrderStatus.ACTIVE,
                "filled": StopOrderStatus.FILLED,
                "done_for_day": StopOrderStatus.CANCELLED,
                "canceled": StopOrderStatus.CANCELLED,
                "expired": StopOrderStatus.CANCELLED,
                "replaced": StopOrderStatus.ACTIVE,
                "pending_cancel": StopOrderStatus.CANCELLED,
                "pending_replace": StopOrderStatus.PENDING_UPDATE,
                "rejected": StopOrderStatus.FAILED,
                "suspended": StopOrderStatus.FAILED,
                "pending_new": StopOrderStatus.PENDING_UPDATE,
                "calculated": StopOrderStatus.ACTIVE,
                "accepted": StopOrderStatus.ACTIVE,
                "accepted_for_bidding": StopOrderStatus.ACTIVE
            }
            
            return status_map.get(order.status, StopOrderStatus.FAILED)
            
        except Exception as e:
            log.error(f"Failed to check broker stop status: {e}")
            return StopOrderStatus.FAILED
    
    def _cancel_stop_with_broker(self, stop_order: StopOrder) -> bool:
        """Cancel stop order with broker"""
        try:
            self.alpaca_client.cancel_order_by_id(stop_order.order_id)
            return True
        except Exception as e:
            log.error(f"Failed to cancel stop order with broker: {e}")
            return False
    
    def get_active_stops(self) -> Dict[str, StopOrder]:
        """Get all active stop orders"""
        return {k: v for k, v in self.active_stops.items() 
                if v.status == StopOrderStatus.ACTIVE}
    
    def cleanup_old_stops(self, max_age: timedelta = timedelta(hours=24)):
        """Clean up old stop orders"""
        cutoff_time = datetime.now() - max_age
        
        to_remove = []
        for symbol, stop_order in self.active_stops.items():
            if (stop_order.status in [StopOrderStatus.FILLED, StopOrderStatus.CANCELLED, 
                                    StopOrderStatus.FAILED] and 
                stop_order.updated_at < cutoff_time):
                to_remove.append(symbol)
                
        for symbol in to_remove:
            del self.active_stops[symbol]
            
        if to_remove:
            log_structured(MetricEvent(
                event_type="stops_cleaned",
                count=len(to_remove)
            ))