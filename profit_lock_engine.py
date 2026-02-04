from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
from datetime import datetime, timedelta
import time


class LockPhase(str, Enum):
    NONE = "none"
    BREAKEVEN = "breakeven"
    LOCKING = "locking"
    TRAILING = "trailing"


@dataclass
class ProfitLockState:
    symbol: str
    side: str                 # "long" supported for now
    entry_price: float
    qty: float

    phase: LockPhase = LockPhase.NONE

    # High-water mark tracking
    peak_price: float = 0.0
    peak_unrealized_pct: float = 0.0   # e.g. 0.0125 for +1.25%

    # The locked floor we refuse to go below
    locked_min_profit_pct: float = -1.0  # -1 means "not set"
    stop_price: float = 0.0

    # Housekeeping
    last_stop_update_ts: float = 0.0
    last_peak_update_ts: float = field(default_factory=lambda: time.time())
    last_decision: str = ""
    order_id_stop: Optional[str] = None

    # Optional: track age of position if needed by caller
    opened_ts: float = field(default_factory=lambda: time.time())
    
    # TTL tracking for memory cleanup
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


# Ladder rules:
# - Use PEAK profit (not current) to advance locks.
# - Locked floor must NEVER decrease.
LOCK_LADDER = [
    # (trigger_peak_profit_pct, rule)
    (0.0035, "breakeven_plus_costs"),  # +0.35% peak -> break-even + costs
    (0.0075, 0.0025),                  # +0.75% peak -> lock +0.25%
    (0.0125, 0.0075),                  # +1.25% peak -> lock +0.75%
    (0.0200, "trail_0p0060"),           # +2.00% peak -> lock at peak - 0.60%
    (0.0300, "trail_0p0045"),           # +3.00% peak -> lock at peak - 0.45%
]

# TTL Configuration
PROFIT_LOCK_STATE_TTL_DAYS = 7


def _parse_trail_gap(rule: str) -> float:
    # "trail_0p0060" -> 0.0060
    return float(rule.replace("trail_", "").replace("p", "."))


def profit_lock_step(
    state: ProfitLockState,
    current_price: float,
    unrealized_pct: float,
    *,
    costs_pct: float = 0.0008,           # 0.08% default
    min_stop_step_pct: float = 0.0005,   # 0.05% min tighten increment
    update_cooldown_sec: int = 20,       # reduce order-update spam
) -> Tuple[bool, float, bool, str]:
    """
    Returns: (should_update_stop, new_stop_price, force_exit, reason)
    - should_update_stop: caller should cancel/replace stop order to new_stop_price
    - force_exit: lock violated (unrealized < locked floor) -> close position
    """
    if state.side.lower() != "long":
        # Safe no-op for now; do not break anything.
        return False, state.stop_price, False, "PROFIT_LOCK_NOOP_UNSUPPORTED_SIDE"

    now = time.time()

    # 1) Initialize peak tracking
    if state.peak_price <= 0:
        state.peak_price = current_price

    # 2) Update peak if new high
    if current_price > state.peak_price:
        state.peak_price = current_price
        if unrealized_pct > state.peak_unrealized_pct:
            state.peak_unrealized_pct = unrealized_pct
        state.last_peak_update_ts = now
        state.last_updated = datetime.utcnow()  # Update timestamp

    peak_pct = state.peak_unrealized_pct

    # 3) Determine desired locked floor based on ladder & peak (ratchet only)
    desired_locked_pct = state.locked_min_profit_pct

    for trigger, rule in LOCK_LADDER:
        if peak_pct >= trigger:
            if rule == "breakeven_plus_costs":
                desired_locked_pct = max(desired_locked_pct, costs_pct)
                state.phase = LockPhase.BREAKEVEN
            elif isinstance(rule, float):
                desired_locked_pct = max(desired_locked_pct, rule)
                state.phase = LockPhase.LOCKING
            elif isinstance(rule, str) and rule.startswith("trail_"):
                gap = _parse_trail_gap(rule)
                desired_locked_pct = max(desired_locked_pct, peak_pct - gap)
                state.phase = LockPhase.TRAILING

    # 4) Convert locked floor % -> stop price
    desired_stop = 0.0
    if desired_locked_pct >= 0:
        desired_stop = state.entry_price * (1.0 + desired_locked_pct)

    # 5) Ratchet stop only tighter, with cooldown + min step
    if desired_stop > state.stop_price and desired_stop > 0:
        can_update = (now - state.last_stop_update_ts) >= update_cooldown_sec

        if state.stop_price > 0:
            delta_pct = (desired_stop - state.stop_price) / state.stop_price
        else:
            delta_pct = 1.0

        big_enough = delta_pct >= min_stop_step_pct

        if can_update and big_enough:
            state.stop_price = desired_stop
            state.locked_min_profit_pct = desired_locked_pct
            state.last_stop_update_ts = now
            state.last_updated = datetime.utcnow()  # Update timestamp
            state.last_decision = (
                f"STOP_RATCHET stop={state.stop_price:.6f} lock={state.locked_min_profit_pct:.4%} phase={state.phase.value}"
            )
            return True, state.stop_price, False, state.last_decision

    # 6) Hard enforcement: if lock set and current unrealized falls below lock -> exit
    if state.locked_min_profit_pct >= 0 and unrealized_pct < state.locked_min_profit_pct:
        state.last_decision = (
            f"LOCK_VIOLATION unreal={unrealized_pct:.4%} < lock={state.locked_min_profit_pct:.4%} -> EXIT"
        )
        return False, state.stop_price, True, state.last_decision

    # 7) No action
    state.last_decision = "NO_CHANGE"
    return False, state.stop_price, False, state.last_decision


def cleanup_old_profit_lock_states(states: dict[str, ProfitLockState]) -> int:
    """
    Remove profit lock states older than TTL.
    Returns: number of states cleaned.
    """
    cutoff = datetime.utcnow() - timedelta(days=PROFIT_LOCK_STATE_TTL_DAYS)
    stale = [sym for sym, st in states.items() if st.last_updated < cutoff]
    for sym in stale:
        del states[sym]
    return len(stale)