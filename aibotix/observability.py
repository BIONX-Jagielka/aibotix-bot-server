# aibotix/observability.py
from __future__ import annotations
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Global metrics counters
_metrics: Dict[str, int] = defaultdict(int)

log = logging.getLogger("aibotix")

@dataclass
class MetricEvent:
    """Structured metric event for observability"""
    event_type: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    endpoint: Optional[str] = None
    error_type: Optional[str] = None
    attempt: Optional[int] = None
    elapsed: Optional[float] = None
    count: Optional[int] = None
    user_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

def inc_metric(name: str, value: int = 1) -> None:
    """Increment a global metric counter"""
    _metrics[name] += value

def get_metrics() -> Dict[str, int]:
    """Get current metric values"""
    return dict(_metrics)

def reset_metrics() -> None:
    """Reset all metrics (for testing)"""
    _metrics.clear()

def log_structured(event: MetricEvent) -> None:
    """Log a structured event with metrics"""
    extra = event.extra or {}
    extra.update({
        "event_type": event.event_type,
        "symbol": event.symbol,
        "timeframe": event.timeframe,
        "endpoint": event.endpoint,
        "error_type": event.error_type,
        "attempt": event.attempt,
        "elapsed": event.elapsed,
        "count": event.count,
    })
    # Remove None values
    extra = {k: v for k, v in extra.items() if v is not None}
    
    # Increment relevant metrics
    metric_key = f"{event.event_type}"
    if event.symbol:
        metric_key += f".{event.symbol}"
    inc_metric(metric_key)
    
    # Log at appropriate level
    if event.error_type or "failure" in event.event_type or "error" in event.event_type:
        log.warning(f"[{event.event_type.upper()}]", extra=extra)
        inc_metric("errors.total")
    else:
        log.info(f"[{event.event_type.upper()}]", extra=extra)

def log_fetch_failure(symbol: str, endpoint: str, timeframe: str, attempt: int, 
                     exception_type: str, elapsed: float) -> None:
    """Log a data fetch failure with structured metrics"""
    log_structured(MetricEvent(
        event_type="fetch_failure",
        symbol=symbol,
        endpoint=endpoint,
        timeframe=timeframe,
        attempt=attempt,
        error_type=exception_type,
        elapsed=elapsed
    ))

def log_empty_bars(symbol: str, timeframe: str, count: int) -> None:
    """Log empty bar detection"""
    log_structured(MetricEvent(
        event_type="empty_bars_detected",
        symbol=symbol,
        timeframe=timeframe,
        count=count
    ))
    inc_metric("data_quality.empty_bars")

def log_selector_instability(old_count: int, new_count: int, reason: str) -> None:
    """Log AI selector instability"""
    log_structured(MetricEvent(
        event_type="selector_instability",
        extra={"old_count": old_count, "new_count": new_count, "reason": reason}
    ))
    inc_metric("selector.instability")

def log_stop_update_failure(symbol: str, reason: str, stop_price: float) -> None:
    """Log stop order update failure"""
    log_structured(MetricEvent(
        event_type="stop_update_failure",
        symbol=symbol,
        error_type=reason,
        extra={"stop_price": stop_price}
    ))
    inc_metric("stops.update_failures")

def log_forced_exit(symbol: str, reason: str, price: float) -> None:
    """Log forced position exit"""
    log_structured(MetricEvent(
        event_type="forced_exit",
        symbol=symbol,
        extra={"reason": reason, "exit_price": price}
    ))
    inc_metric("exits.forced")

# Section 6: Additional metrics for Section 2-7 implementation
def log_alpaca_fetch_fail(symbol: str, error_type: str) -> None:
    """Log Alpaca fetch failure"""
    inc_metric("alpaca_fetch_fail_total")
    log_structured(MetricEvent(
        event_type="alpaca_fetch_fail",
        symbol=symbol,
        error_type=error_type
    ))

def log_alpaca_empty_bars(symbol: str) -> None:
    """Log Alpaca empty bars"""
    inc_metric("alpaca_empty_bars_total")
    log_structured(MetricEvent(
        event_type="alpaca_empty_bars",
        symbol=symbol
    ))

def log_bars_invalid(symbol: str, reason: str) -> None:
    """Log invalid bars detection"""
    inc_metric("bars_invalid_total")
    log_structured(MetricEvent(
        event_type="bars_invalid",
        symbol=symbol,
        extra={"reason": reason}
    ))

def log_selector_refresh(reason: str) -> None:
    """Log AI selector refresh by reason"""
    inc_metric("selector_refresh_total")
    inc_metric(f"selector_refresh_{reason}")
    log_structured(MetricEvent(
        event_type="selector_refresh",
        extra={"reason": reason}
    ))

def log_stop_update_failed() -> None:
    """Log stop order update failure"""
    inc_metric("stop_update_failed_total")

def log_forced_exit_total(reason: str) -> None:
    """Log forced exit by reason"""
    inc_metric("forced_exits_total")
    inc_metric(f"forced_exits_{reason}")

# Metrics flushing function
_last_flush_time = 0

def flush_metrics_to_logs():
    """Flush metrics to logs once per minute"""
    global _last_flush_time
    import time
    
    now = time.time()
    if now - _last_flush_time < 60:  # Don't flush more than once per minute
        return
        
    _last_flush_time = now
    current_metrics = get_metrics()
    
    if current_metrics:
        log.info(
            "Metrics flush",
            extra={
                "event_type": "metrics_flush",
                "metrics": current_metrics,
                "flush_interval_seconds": 60
            }
        )
        
        # Optional: Reset metrics after flush if desired
        # reset_metrics()