# aibotix/data/bars.py
from __future__ import annotations
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import TimeFrame

log = logging.getLogger("aibotix.data")

@dataclass
class BarFetchResult:
    """Structured result object for bar fetches"""
    status: str  # "OK" | "EMPTY" | "INVALID" | "FETCH_FAIL"
    df: Optional[pd.DataFrame]
    reason: str
    symbol: str

# Circuit breaker state - rolling 5-minute windows
_fetch_counters = {
    'total_attempts': deque(maxlen=300),  # 5 min * 60 sec
    'failures': deque(maxlen=300),
    'empty_bars': deque(maxlen=300), 
    'invalid_bars': deque(maxlen=300),
}

_data_degraded = False
_last_degraded_check = 0

def _record_fetch_event(event_type: str) -> None:
    """Record a fetch event with timestamp for circuit breaker"""
    now = time.time()
    _fetch_counters['total_attempts'].append(now)
    
    if event_type in ['failure', 'empty', 'invalid']:
        _fetch_counters[f'{event_type}_bars' if event_type != 'failure' else 'failures'].append(now)

def _cleanup_old_events() -> None:
    """Remove events older than 5 minutes"""
    now = time.time()
    cutoff = now - 300  # 5 minutes
    
    for counter in _fetch_counters.values():
        # Remove old events efficiently
        while counter and counter[0] < cutoff:
            counter.popleft()

def _update_degraded_status() -> None:
    """Update global degraded status based on recent failure rates"""
    global _data_degraded, _last_degraded_check
    
    now = time.time()
    if now - _last_degraded_check < 30:  # Check at most every 30 seconds
        return
    
    _cleanup_old_events()
    _last_degraded_check = now
    
    total = len(_fetch_counters['total_attempts'])
    if total < 5:  # Not enough data
        return
        
    failures = len(_fetch_counters['failures'])
    empty = len(_fetch_counters['empty_bars'])
    
    failure_rate = failures / total
    empty_rate = empty / total
    
    # Circuit breaker thresholds
    was_degraded = _data_degraded
    _data_degraded = failure_rate > 0.20 or empty_rate > 0.05
    
    if _data_degraded != was_degraded:
        log.warning(
            f"Data quality state changed: degraded={_data_degraded}, "
            f"failure_rate={failure_rate:.3f}, empty_rate={empty_rate:.3f}",
            extra={
                "event_type": "data_quality_change",
                "data_degraded": _data_degraded,
                "failure_rate": failure_rate,
                "empty_rate": empty_rate,
                "total_attempts": total
            }
        )

def is_data_degraded() -> Tuple[bool, Dict]:
    """Get current data degraded status and rates"""
    _update_degraded_status()
    
    total = len(_fetch_counters['total_attempts'])
    if total == 0:
        return False, {"failure_rate": 0.0, "empty_rate": 0.0, "total_attempts": 0}
        
    failure_rate = len(_fetch_counters['failures']) / total
    empty_rate = len(_fetch_counters['empty_bars']) / total
    
    return _data_degraded, {
        "failure_rate": failure_rate,
        "empty_rate": empty_rate,
        "total_attempts": total
    }

def _validate_bars(df: pd.DataFrame, symbol: str, timeframe: str, limit: int) -> Tuple[str, str]:
    """
    Validate bar data quality according to Section 2 rules
    Returns (status, reason)
    """
    if df is None or len(df) == 0:
        return "EMPTY", "NO_DATA"
    
    # Check for empty bars
    if len(df) == 0:
        return "EMPTY", "ZERO_BARS"
    
    # Continuity requirements
    if timeframe == "1Min":
        required_ratio = 0.70
        if limit >= 100:
            required_count = 70  # at least 70 from last 100
        else:
            required_count = max(1, int(limit * required_ratio))
    else:  # Daily
        required_ratio = 0.67
        if limit >= 120:
            required_count = 80  # at least 80 from last 120
        else:
            required_count = max(1, int(limit * required_ratio))
    
    if len(df) < required_count:
        return "INVALID", f"INSUFFICIENT_BARS_{len(df)}<{required_count}"
    
    # Check timestamp ordering and duplicates
    if not df.index.is_monotonic_increasing:
        return "INVALID", "INVALID_DATA_NOT_ORDERED"
    
    if df.index.duplicated().any():
        return "INVALID", "INVALID_DATA_DUPLICATES"
    
    # Check for valid OHLCV data
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return "INVALID", f"INVALID_DATA_MISSING_COLS_{missing_cols}"
    
    # Check for NaN in critical columns (allow some volume NaN)
    critical_cols = ['open', 'high', 'low', 'close']
    nan_counts = df[critical_cols].isna().sum()
    if nan_counts.sum() > 0:
        return "INVALID", f"INVALID_DATA_NAN_{nan_counts.to_dict()}"
    
    return "OK", "VALID"

def fetch_minute_bars(symbols: List[str], limit: int = 100, timeframe: str = "1Min") -> Dict[str, BarFetchResult]:
    """
    Fetch minute bars for multiple symbols with data quality gates
    """
    # Import here to avoid circular imports
    try:
        import bot.aibotix_trading_bot as bot_module
        data_client = getattr(bot_module, 'data_client', None)
    except ImportError:
        log.error("Cannot import bot module for data client access")
        return {symbol: BarFetchResult("FETCH_FAIL", None, "NO_CLIENT_ACCESS", symbol) for symbol in symbols}
    
    if data_client is None:
        log.error("Data client not initialized in bot module")
        return {symbol: BarFetchResult("FETCH_FAIL", None, "CLIENT_NOT_INITIALIZED", symbol) for symbol in symbols}
    
    results = {}
    
    for symbol in symbols:
        _record_fetch_event('attempt')
        start_time = time.time()
        
        try:
            # Use appropriate timeframe
            tf = TimeFrame.Minute if timeframe == "1Min" else TimeFrame.Day
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit, feed="iex")
            response = data_client.get_stock_bars(request)
            
            # Handle MultiIndex case
            df = response.df
            if isinstance(df.index, pd.MultiIndex):
                try:
                    df = df.xs(symbol)
                except KeyError:
                    _record_fetch_event('failure')
                    results[symbol] = BarFetchResult("FETCH_FAIL", None, "SYMBOL_NOT_IN_RESPONSE", symbol)
                    continue
            
            # Clean data
            df = df.dropna(subset=['close', 'high', 'low', 'open'])
            
            # Validate data quality
            status, reason = _validate_bars(df, symbol, timeframe, limit)
            
            if status == "EMPTY":
                _record_fetch_event('empty')
            elif status == "INVALID":
                _record_fetch_event('invalid')
            
            results[symbol] = BarFetchResult(status, df if status == "OK" else None, reason, symbol)
            
            elapsed = time.time() - start_time
            log.debug(f"Fetched {len(df) if df is not None else 0} bars for {symbol} in {elapsed:.3f}s, status={status}")
            
        except Exception as e:
            _record_fetch_event('failure')
            log.error(f"Failed to fetch bars for {symbol}: {e}")
            results[symbol] = BarFetchResult("FETCH_FAIL", None, f"EXCEPTION_{type(e).__name__}", symbol)
    
    _update_degraded_status()
    return results

def fetch_daily_bars(symbol: str, limit: int = 120, timeframe: str = "1Day") -> BarFetchResult:
    """
    Fetch daily bars for a single symbol with data quality gates
    """
    result = fetch_minute_bars([symbol], limit, timeframe)
    return result[symbol]