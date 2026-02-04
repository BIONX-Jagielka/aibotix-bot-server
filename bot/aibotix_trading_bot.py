from __future__ import annotations

from profit_lock_engine import ProfitLockState
# --- BotSession container for state (Patch B foundation) ---

# Global client registry for equity snapshots
_CLIENT_REGISTRY = {}  # key: f"{user_id}:{mode}" → Alpaca TradingClient instance

class BotSession:
    def __init__(self):
        self.USER_ID = None
        self.CURRENT_MODE = None
        self.api = None
        self.data_client = None

        # Core trading state
        self.consecutive_losses = 0
        self.last_trade_time = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.start_of_day_equity = None
        self.daily_realized_pnl = 0.0
        self.trading_halted_until = None
        # Market/session tracking for smarter entries
        self.market_open_ts = None          # datetime in NY tz for today's market open
        self.first_trade_done = False       # first entry after market open
        self.last_market_open_date = None   # date() in NY tz to reset first_trade_done daily

        # Per-ticker tracking
        self.per_ticker_loss_streak = {}
        self.per_ticker_cooloff_until = {}
        self.recently_traded = {}
        self.trade_history = {}
        self.ACTIVE_TICKERS = []
        
        # Smart Ticker Recovery Lock (per-ticker)
        self.ticker_recovery_lock = {}
        
        # Execution-level bad data suppression
        self.bad_ticker_until = {}   # ticker -> datetime until which ticker is skipped
        
        # Defensive logic state
        self.session_peak_equity = None
        self.buy_pause_until = None
        
        # Market session state + hard gates
        self.market_state = "UNKNOWN"  # OPEN | CLOSING | CLOSED | PREOPEN
        self.market_close_guard_until = None  # datetime; used to suppress trading after close sequence starts
        self.last_clock_check = None
        self.last_market_state_log = None
        
        # AI ticker management
        self.last_ai_refresh = None
        self.last_pnl_update = None
        
        # Ticker evaluation throttling
        self.last_ticker_eval = {}  # ticker -> datetime
        
        # Graduated risk management state
        self.risk_state = "NORMAL"  # NORMAL | DEGRADED | HALT
        self.last_risk_state_log = None
        self.per_ticker_loss_cooldown = {}  # ticker -> datetime until which ticker is cooled
        
        # Profit lock engine states
        self.profit_lock_states: dict[str, ProfitLockState] = {}
        
        # Supabase TTL cleanup state
        self.last_supabase_cleanup_at = None

# Multi-user session registry: one BotSession per (user_id, mode)
SESSIONS: dict[tuple[str | None, str | None], "BotSession"] = {}


def get_session(user_id: str | None, mode: str | None) -> "BotSession":
    """
    Return a BotSession for the given (user_id, mode) pair.

    This is the foundation for multi-user support:
    - Each (user_id, mode) has its own BotSession object
    - The global SESSION is set to the session currently being used
      by the worker/task so existing code keeps working.
    """
    global SESSION, SESSIONS
    key: tuple[str | None, str | None] = (user_id, mode)
    session = SESSIONS.get(key)
    if session is None:
        session = BotSession()
        session.USER_ID = user_id
        session.CURRENT_MODE = mode
        SESSIONS[key] = session
    # Make this session the "active" one for the current task/process
    SESSION = session
    return session

# Global session instance (future: per-user instances in worker)
SESSION = BotSession()
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
from alpaca.trading.client import TradingClient
# Alpaca v3 trading enums and order request
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.common.exceptions import APIError
from profit_lock_engine import profit_lock_step, cleanup_old_profit_lock_states
import pandas as pd
import time
import datetime
import pytz
import numpy as np
import os
from dotenv import load_dotenv
import logging
import asyncio
import random
from enum import Enum
from bot.ai_ticker_selector_aibotix import get_top_tickers

# Risk state enumeration for graduated risk management
class RiskState(Enum):
    NORMAL = "normal"      # Normal trading conditions
    DEGRADED = "degraded"  # Reduced position sizes, stricter criteria
    HALT = "halt"          # No new positions, only exits

# Alpaca data API imports for historical and latest trade data (v3)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame


# --- Supabase Logging Integration (Patch 1) ---
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        logging.info("Supabase logging initialised.")
    except Exception as e:
        logging.error(f"Failed to initialise Supabase client: {e}")

else:
    logging.warning("Supabase credentials missing — logging disabled.")

# Per-bot context (set by init_trading_client so logs are multi-user aware)
USER_ID: str | None = None
CURRENT_MODE: str | None = None  # 'paper' or 'live'

def supabase_log(message: str) -> None:
    global supabase, SESSION
    # Use the active BotSession so logs are tied to the correct user/mode
    user_id = getattr(SESSION, "USER_ID", None)
    mode = getattr(SESSION, "CURRENT_MODE", None)
    if supabase is None or user_id is None or mode is None:
        return
    try:
        supabase.table("bot_logs").insert({
            "user_id": user_id,
            "mode": mode,
            "message": message,
        }).execute()
    except Exception as e:
        logging.error(f"Supabase log error: {e}")

def cleanup_old_supabase_rows() -> None:
    """
    Clean up old rows from bot_logs and equity_history tables.
    Deletes rows older than CLEANUP_TTL_DAYS.
    All throttling logic handled internally.
    """
    global supabase, SESSION
    
    # Define now at the very first executable line
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    
    # Check cleanup interval (throttling logic moved inside function)
    if (SESSION.last_supabase_cleanup_at is not None and 
        (now - SESSION.last_supabase_cleanup_at).total_seconds() < SUPABASE_CLEANUP_INTERVAL_SECONDS):
        return  # Interval has not elapsed
    
    # Safety guards
    if supabase is None:
        return
    
    user_id = getattr(SESSION, "USER_ID", None)
    mode = getattr(SESSION, "CURRENT_MODE", None)
    if user_id is None or mode is None:
        return
    
    # Wrap full cleanup execution in try/except
    try:
        cutoff = now - timedelta(days=CLEANUP_TTL_DAYS)
        cutoff_str = cutoff.isoformat()
        
        n_logs = 0
        n_equity = 0
        
        # Clean bot_logs
        try:
            result = supabase.table("bot_logs").delete().lt("created_at", cutoff_str).eq("user_id", user_id).eq("mode", mode).execute()
            if hasattr(result, 'data') and result.data:
                n_logs = len(result.data)
        except Exception:
            pass  # Fail silently
        
        # Clean equity_history
        try:
            result = supabase.table("equity_history").delete().lt("timestamp", cutoff_str).eq("user_id", user_id).eq("mode", mode).execute()
            # Handle non-200/204 responses silently
            if hasattr(result, 'data') and result.data and hasattr(result, 'status_code') and result.status_code in [200, 204]:
                n_equity = len(result.data)
            elif hasattr(result, 'data') and result.data:
                n_equity = len(result.data)
        except Exception:
            pass  # Fail silently
        
        # Log only if rows were deleted
        if n_logs > 0 or n_equity > 0:
            logging.info(f"[CLEANUP] Deleted {n_logs} bot_logs rows and {n_equity} equity_history rows older than 7 days")
        
        # Update timestamp after cleanup attempt completes (even if 0 rows deleted)
        SESSION.last_supabase_cleanup_at = now
        
    except Exception as e:
        # Log exactly one warning on any exception
        logging.warning(f"Supabase cleanup failed: {e}")
        # Still update timestamp to prevent constant retries
        SESSION.last_supabase_cleanup_at = now

# Rate-limited transparency logging (prevents spam)
_last_reason_log: dict[str, datetime.datetime] = {}

def reason_log(key: str, message: str, min_seconds: int = 120) -> None:
    """Log a reason at most once per `min_seconds` per key."""
    try:
        now = datetime.datetime.now(ny_tz)
    except Exception:
        now = datetime.datetime.utcnow()
    last = _last_reason_log.get(key)
    if last and (now - last).total_seconds() < min_seconds:
        return
    _last_reason_log[key] = now
    supabase_log(message)

def ui_log(key: str, message: str, min_seconds: int = 60) -> None:
    """Rate-limited user-friendly log message."""
    try:
        reason_log(f"ui_{key}", f"ui | {message}", min_seconds=min_seconds)
    except Exception:
        pass  # Never block trading on UI log failures

def ui_event(message: str) -> None:
    """Immediate user-friendly event message."""
    try:
        supabase_log(f"ui | {message}")
    except Exception:
        pass  # Never block trading on UI log failures

# --- Clean user-facing heartbeat (1 per minute max) ---
_last_user_minute_log = {}

def user_minute_log(message: str):
    key = f"{SESSION.USER_ID}:{SESSION.CURRENT_MODE}"
    now = datetime.datetime.now(ny_tz)

    last = _last_user_minute_log.get(key)
    if last and (now - last).total_seconds() < 60:
        return

    _last_user_minute_log[key] = now
    supabase_log(f"ui | {message}")

# === Phase 2.2: Historical Hint Consumption (no calculations in bot) ===
HIST_HINT_CACHE_TTL_SEC = 300  # 5 min cache to reduce Supabase hits
_hist_hint_cache: dict[str, tuple[datetime.datetime, dict]] = {}  # key=user:mode:ticker -> (ts, hint)

def _hist_hint_cache_key(ticker: str) -> str:
    uid = getattr(SESSION, "USER_ID", None)
    mode = getattr(SESSION, "CURRENT_MODE", None)
    return f"{uid}:{mode}:{ticker}"

def get_historical_hint(ticker: str) -> dict | None:
    """
    Fetch precomputed historical context produced by the AI ticker selector.
    This bot MUST NOT compute history; it only consumes hints if available.
    Returns None if not available or any error occurs (safe default).
    Expected fields (optional): hist_volatility, avg_recovery_days, max_drawdown
    """
    if supabase is None:
        return None
    uid = getattr(SESSION, "USER_ID", None)
    mode = getattr(SESSION, "CURRENT_MODE", None)
    if uid is None or mode is None:
        return None

    # Cache
    try:
        now = datetime.datetime.now(ny_tz)
    except Exception:
        now = datetime.datetime.utcnow()

    ck = _hist_hint_cache_key(ticker)
    cached = _hist_hint_cache.get(ck)
    if cached:
        ts, hint = cached
        if (now - ts).total_seconds() < HIST_HINT_CACHE_TTL_SEC:
            return hint

    # IMPORTANT:
    # We query ai_tickers first because it already exists in your stack.
    # This will NOT break if columns do not exist — we handle exceptions and return None.
    try:
        # Try to pull additional hint columns if your AI selector stores them.
        # If your table doesn't have these columns yet, Supabase may error -> we catch and return None.
        res = (
            supabase.table("ai_tickers")
            .select("ticker,hist_volatility,avg_recovery_days,max_drawdown,updated_at")
            .eq("user_id", uid)
            .eq("mode", mode)
            .eq("ticker", ticker)
            .limit(1)
            .execute()
        )
        rows = getattr(res, "data", None) or []
        if not rows:
            return None

        row = rows[0] or {}
        hint = {
            "hist_volatility": row.get("hist_volatility") or 0.0,
            "avg_recovery_days": row.get("avg_recovery_days") or 20.0,
            "max_drawdown": row.get("max_drawdown") or 0.0,
        }

        # Remove any remaining None values and ensure numeric types
        safe_hint = {}
        for k, v in hint.items():
            if v is not None:
                try:
                    safe_hint[k] = float(v)
                except (ValueError, TypeError):
                    # Use safe defaults for invalid data
                    if k == "hist_volatility":
                        safe_hint[k] = 0.0
                    elif k == "avg_recovery_days":
                        safe_hint[k] = 20.0
                    else:
                        safe_hint[k] = 0.0
            else:
                # Provide safe defaults for NULL values
                if k == "hist_volatility":
                    safe_hint[k] = 0.0
                elif k == "avg_recovery_days":
                    safe_hint[k] = 20.0
                else:
                    safe_hint[k] = 0.0
        
        if not safe_hint:
            return None

        _hist_hint_cache[ck] = (now, safe_hint)
        return safe_hint

    except Exception:
        # Safe fallback: no hint
        return None

def apply_historical_hint_multiplier(ticker: str, qty: float) -> tuple[float, float, dict | None]:
    """
    Applies soft sizing modifiers based on historical hint.
    Returns: (new_qty, multiplier, hint)
    Never blocks entries. Never changes tier. Never changes logic beyond qty scaling.
    """
    if qty is None:
        return qty, 1.0, None
    try:
        qty_f = float(qty)
    except Exception:
        return qty, 1.0, None

    hint = get_historical_hint(ticker)
    if not hint:
        return qty_f, 1.0, None

    mult = 1.0

    # Soft volatility penalty if selector flagged high vol (expects numeric float volatility OR string tag)
    hv = hint.get("hist_volatility")
    try:
        if isinstance(hv, (int, float)) and float(hv) > 0.12:
            mult *= 0.85
    except Exception:
        pass
    if isinstance(hv, str) and hv.upper() in ("HIGH", "VERY_HIGH"):
        mult *= 0.85

    # Soft recovery bonus/penalty (expects numeric avg recovery days OR string tag)
    rec = hint.get("avg_recovery_days")
    try:
        if isinstance(rec, (int, float)) and float(rec) <= 20:
            mult *= 1.10
        elif isinstance(rec, (int, float)) and float(rec) >= 35:
            mult *= 0.85
    except Exception:
        pass
    if isinstance(rec, str) and rec.upper() in ("FAST",):
        mult *= 1.10
    if isinstance(rec, str) and rec.upper() in ("SLOW", "VERY_SLOW"):
        mult *= 0.85

    # Clamp safety: never negative, never explode
    mult = max(0.50, min(mult, 1.15))  # soft bounds
    new_qty = max(qty_f * mult, 0.0)
    return new_qty, mult, hint

def get_market_state() -> tuple[str, datetime.timedelta | None]:
    """Get current market state using SESSION.api.get_clock()"""
    if SESSION.api is None:
        return "CLOSED", None
    
    try:
        clock = SESSION.api.get_clock()
        if clock.is_open:
            time_to_close = clock.next_close - clock.timestamp if hasattr(clock, 'next_close') else None
            return "OPEN", time_to_close
        else:
            return "CLOSED", None
    except Exception:
        return "CLOSED", None

def in_closeout_window(time_to_close: datetime.timedelta | None) -> bool:
    """Returns True if time_to_close is not None and <= 15 minutes"""
    return time_to_close is not None and time_to_close <= datetime.timedelta(minutes=15)

# --- Clean user-facing heartbeat (1 per 5 minutes max) ---
_last_user_five_min_log = {}

def user_five_min_log(message: str):
    key = f"{SESSION.USER_ID}:{SESSION.CURRENT_MODE}"
    now = datetime.datetime.now(ny_tz)

    last = _last_user_five_min_log.get(key)
    if last and (now - last).total_seconds() < 300:
        return

    _last_user_five_min_log[key] = now
    supabase_log(f"ui | {message}")

load_dotenv()

API_KEY = None
API_SECRET = None

# CONFIGURATION
BASE_URL = 'https://paper-api.alpaca.markets'  # use live URL for real trading
# TICKERS will be dynamically set using AI ticker selector
# TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
RSI_PERIOD = 14

# Minimum bars required for safe indicator calculation
MIN_BARS_REQUIRED = max(RSI_PERIOD + 5, 50)
MAX_DRAWDOWN = 0.10  # 10%
MAX_TRADES_PER_HOUR = 5
MAX_TICKER_EXPOSURE = 0.30  # 30% of equity
MAX_CONCURRENT_POSITIONS = 30

# --- Position sizing safety caps (prevents oversized notional trades) ---
# Cap any single trade notional to a small slice of equity AND an absolute ceiling.
# This is separate from MAX_TICKER_EXPOSURE (portfolio concentration cap).
MAX_TRADE_NOTIONAL_PCT = 0.08      # allow up to 8% of equity per entry
MAX_TRADE_NOTIONAL_ABS = 7500.0    # allow larger trades on high equity accounts
LOSS_COOLDOWN = 300  # 5 minutes
STOP_LOSS_ATR_MULT = 1.5
TAKE_PROFIT_ATR_MULT = 2.0

RISK_PER_TRADE = 0.01  # 1% of equity
# Trade pacing & per-ticker caps
TRADE_SPACING_SECONDS = 15  # Minimum seconds between actions on the same ticker
MAX_TRADE_ATTEMPTS_PER_TICKER_PER_MIN = 2  # Avoid overtrading micro-conditions
# === Step 5: Server-side safety & reserve ===
BRACKET_ORDERS_ENABLED = True   # when True, BUY orders place server-side OCO TP/SL and local TP/trailing is disabled
RESERVE_FUND_PCT = 0.05         # keep 5% of equity unallocated as a safety buffer

# === Step 3: Enhanced Risk Controls ===
DAILY_MAX_LOSS_PCT = 0.0075       # 0.75% max loss triggers HALT state
DAILY_DEGRADED_LOSS_PCT = 0.0025  # 0.25% loss triggers DEGRADED state
DAILY_MAX_LOSS_DOLLARS = None      # optional hard dollar cap; set to a number (e.g., 200) to enforce alongside %
MAX_CONSECUTIVE_LOSSES_HALT = 5    # if this many losing exits occur in a day, halt trading until next session
PER_TICKER_MAX_LOSS_STREAK = 3     # per-ticker loss streak before cooling off that ticker
PER_TICKER_LOSS_COOLDOWN_MIN = 20  # minutes to cool off a ticker after individual loss (not streak)
PER_TICKER_COOLDOWN_MIN = 60     # minutes to cool off a ticker after it hits its loss streak

# Track recent per-ticker attempts (ticker, minute) -> count
recent_trade_attempts = {}

# --- Per-ticker trade attempt limiting ---
def _attempt_bucket_key():
    now = datetime.datetime.now(ny_tz)
    return now.strftime("%Y%m%d%H%M")

def can_attempt_trade(ticker: str) -> bool:
    bucket = _attempt_bucket_key()
    # Clean old buckets for this ticker
    keys_to_delete = [k for k in recent_trade_attempts.keys() if k.startswith(f"{ticker}|") and not k.endswith(bucket)]
    for k in keys_to_delete:
        del recent_trade_attempts[k]
    key = f"{ticker}|{bucket}"
    count = recent_trade_attempts.get(key, 0)
    if count >= MAX_TRADE_ATTEMPTS_PER_TICKER_PER_MIN:
        return False
    recent_trade_attempts[key] = count + 1
    return True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === Step 2: Dynamic per-user API initialization (Supabase-ready) ===
# WARNING:
# These globals are safe because AIBOTIX currently runs one active bot per worker.
# When multi-bot support is added, move these into per-session state.
def init_trading_client(api_key: str, api_secret: str, paper: bool = True, user_id: str | None = None, mode: str | None = None):
    """
    Initialize trading and data clients for the correct environment (paper or live)
    while attaching them to the correct BotSession.
    """

    global api, data_client, API_KEY, API_SECRET, USER_ID, CURRENT_MODE, SESSION

    API_KEY = api_key
    API_SECRET = api_secret

    # Determine effective mode
    # Explicit mode overrides paper=True/False
    if mode:
        effective_mode = mode
        paper_mode = mode == "paper"
    else:
        paper_mode = paper
        effective_mode = "paper" if paper else "live"

    # Select or create session
    session = get_session(user_id, effective_mode)

    USER_ID = user_id
    CURRENT_MODE = effective_mode

    # ---------- ENVIRONMENT SEPARATION PATCH ----------
    # Create Alpaca Trading client for correct environment
    api = TradingClient(api_key, api_secret, paper=paper_mode)

    # Create Alpaca Data client for correct environment
    data_client = StockHistoricalDataClient(api_key, api_secret)
    # ----------------------------------------------------

    # Store clients in registry for equity snapshots
    key = f"{user_id}:{effective_mode}"
    _CLIENT_REGISTRY[key] = {
        "trading_client": api,
        "data_client": data_client,
        "user_id": user_id,
        "mode": effective_mode
    }

    # Attach to session
    session.api = api
    session.data_client = data_client
    session.USER_ID = user_id
    session.CURRENT_MODE = effective_mode

    # Keep SESSION globally updated
    SESSION = session

    logging.info(f"[Client Init] mode={effective_mode} paper={paper_mode}")
    supabase_log(f"client_initialised | mode={effective_mode} | paper={paper_mode}")
    ui_event(f"Trading client connected in {effective_mode} mode.")

    return api

def get_trading_client(user_id: str, mode: str):
    """
    Retrieve the stored TradingClient + DataClient for equity snapshots.
    Returns None if not yet initialized.
    """
    key = f"{user_id}:{mode}"
    return _CLIENT_REGISTRY.get(key)

# Global placeholder. Start/stop routes will set this dynamically.
api: TradingClient | None = None

ny_tz = pytz.timezone('America/New_York')

REENTRY_COOLDOWN = 300  # 5 minutes

# Alpaca data client for historical and latest trade data
data_client: StockHistoricalDataClient | None = None

# --- Step 3 helpers: daily reset & halting ---

def _ny_now():
    return datetime.datetime.now(ny_tz)


def reset_daily_limits_if_new_day():
    """Reset daily counters if a new trading day has started."""
    now = _ny_now().date()
    # Initialize on first run
    if SESSION.start_of_day_equity is None:
        SESSION.start_of_day_equity = get_equity()
        SESSION.daily_realized_pnl = 0.0
        SESSION.consecutive_losses = 0
        SESSION.per_ticker_loss_streak = {}
        SESSION.per_ticker_cooloff_until = {}
        return
    # If date changed in NY timezone, reset
    if hasattr(reset_daily_limits_if_new_day, "_last_date"):
        last_date = reset_daily_limits_if_new_day._last_date
    else:
        last_date = now
    if now != last_date:
        SESSION.start_of_day_equity = get_equity()
        SESSION.daily_realized_pnl = 0.0
        SESSION.consecutive_losses = 0
        SESSION.per_ticker_loss_streak = {}
        SESSION.per_ticker_cooloff_until = {}
    reset_daily_limits_if_new_day._last_date = now


def _calc_daily_loss_limit():
    """Return the dollar loss limit for the current day based on start_of_day_equity and optional hard cap."""
    if SESSION.start_of_day_equity is None:
        return None
    pct_cap = SESSION.start_of_day_equity * DAILY_MAX_LOSS_PCT if DAILY_MAX_LOSS_PCT else None
    if DAILY_MAX_LOSS_DOLLARS is None:
        return pct_cap
    return max(DAILY_MAX_LOSS_DOLLARS, pct_cap or 0.0)


def get_risk_state():
    """Determine current risk state based on daily PnL and equity."""
    if SESSION.start_of_day_equity is None:
        return "NORMAL"
    
    current_equity = get_equity()
    daily_loss_pct = abs(SESSION.daily_realized_pnl) / SESSION.start_of_day_equity if SESSION.start_of_day_equity > 0 else 0
    
    # Check for HALT state (severe loss)
    halt_limit_pct = DAILY_MAX_LOSS_PCT if DAILY_MAX_LOSS_PCT else 0.0075
    if daily_loss_pct >= halt_limit_pct:
        return "HALT"
    
    # Check for DEGRADED state (moderate loss)
    degraded_limit_pct = DAILY_DEGRADED_LOSS_PCT if DAILY_DEGRADED_LOSS_PCT else 0.0025
    if daily_loss_pct >= degraded_limit_pct:
        return "DEGRADED"
    
    return "NORMAL"

def update_risk_state():
    """Update risk state and log transitions."""
    new_state = get_risk_state()
    if new_state not in ("NORMAL", "DEGRADED", "HALT"):
        logging.error(f"Invalid risk state returned: {new_state}. Forcing NORMAL.")
        new_state = "NORMAL"
    old_state = SESSION.risk_state
    
    if new_state != old_state:
        now = datetime.datetime.now(ny_tz)
        if SESSION.last_risk_state_log is None or (now - SESSION.last_risk_state_log).total_seconds() >= 300:
            logging.warning(f"Risk state transition: {old_state} → {new_state} (Daily PnL: ${SESSION.daily_realized_pnl:.2f})")
            supabase_log(f"risk_state_change | {old_state}_to_{new_state} | daily_pnl={SESSION.daily_realized_pnl:.2f}")
            SESSION.last_risk_state_log = now
        
        SESSION.risk_state = new_state
        
        # Recovery logging
        if old_state == "DEGRADED" and new_state == "NORMAL":
            logging.info("Risk state recovered to NORMAL - full trading resumed")
            ui_event("Risk managed - full trading capability restored")

def should_halt_trading():
    """True if trading should be completely halted (only in HALT state or manual halt)."""
    now = datetime.datetime.now(ny_tz)
    
    # Manual halt override
    if SESSION.trading_halted_until and now < SESSION.trading_halted_until:
        return True
    
    # Update risk state
    update_risk_state()
    
    # Only halt in HALT state, not DEGRADED
    if SESSION.risk_state == "HALT":
        if SESSION.api is None:
            logging.error("Trading client not initialised. Cannot check clock for halt logic.")
            SESSION.trading_halted_until = now + datetime.timedelta(hours=24)
            return True
        try:
            clock = SESSION.api.get_clock()
            SESSION.trading_halted_until = clock.next_open if hasattr(clock, "next_open") else now + datetime.timedelta(hours=24)
        except Exception:
            SESSION.trading_halted_until = now + datetime.timedelta(hours=24)
        logging.warning(f"HALT state triggered - trading halted until {SESSION.trading_halted_until}")
        return True
    
    # Global halt on many consecutive losses
    if MAX_CONSECUTIVE_LOSSES_HALT and SESSION.consecutive_losses >= MAX_CONSECUTIVE_LOSSES_HALT:
        if SESSION.api is None:
            logging.error("Trading client not initialised. Cannot check clock for halt logic.")
            SESSION.trading_halted_until = datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
            return True
        try:
            clock = SESSION.api.get_clock()
            SESSION.trading_halted_until = clock.next_open if hasattr(clock, "next_open") else datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
        except Exception:
            SESSION.trading_halted_until = datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
        logging.warning(f"Exceeded max consecutive losses ({SESSION.consecutive_losses}). Trading halted until {SESSION.trading_halted_until}.")
        return True
    
    return False

def is_ticker_in_cooldown(ticker: str) -> bool:
    """Check if ticker is in loss cooldown period."""
    cooldown_until = SESSION.per_ticker_loss_cooldown.get(ticker)
    if cooldown_until is None:
        return False
    
    now = datetime.datetime.now(ny_tz)
    if now < cooldown_until:
        return True
    
    # Cooldown expired, remove it
    del SESSION.per_ticker_loss_cooldown[ticker]
    logging.info(f"Ticker {ticker} cooldown expired - trading resumed")
    return False


def _register_trade_outcome(ticker: str, pnl: float):
    """Update loss streaks and potential halts based on trade P&L."""
    # Global loss streak
    if pnl < 0:
        SESSION.consecutive_losses += 1
        
        # Apply per-ticker loss cooldown
        cooldown_until = datetime.datetime.now(ny_tz) + datetime.timedelta(minutes=PER_TICKER_LOSS_COOLDOWN_MIN)
        SESSION.per_ticker_loss_cooldown[ticker] = cooldown_until
        logging.info(f"Ticker {ticker} entered {PER_TICKER_LOSS_COOLDOWN_MIN}min cooldown after loss (${pnl:.2f})")
        
    else:
        SESSION.consecutive_losses = 0
    
    # Per ticker loss streak & cooldown (for streak-based cooldowns)
    if ticker not in SESSION.per_ticker_loss_streak:
        SESSION.per_ticker_loss_streak[ticker] = 0
    if pnl < 0:
        SESSION.per_ticker_loss_streak[ticker] += 1
        if SESSION.per_ticker_loss_streak[ticker] >= PER_TICKER_MAX_LOSS_STREAK:
            SESSION.per_ticker_cooloff_until[ticker] = datetime.datetime.now(ny_tz) + datetime.timedelta(minutes=PER_TICKER_COOLDOWN_MIN)
            logging.warning(f"Cooling off {ticker} for {PER_TICKER_COOLDOWN_MIN} minutes due to loss streak.")
    else:
        SESSION.per_ticker_loss_streak[ticker] = 0
    
    # Global halt on many consecutive losses
    if MAX_CONSECUTIVE_LOSSES_HALT and SESSION.consecutive_losses >= MAX_CONSECUTIVE_LOSSES_HALT:
        if SESSION.api is None:
            logging.error("Trading client not initialised. Cannot check clock for halt logic.")
            SESSION.trading_halted_until = datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
            return
        try:
            clock = SESSION.api.get_clock()
            SESSION.trading_halted_until = clock.next_open if hasattr(clock, "next_open") else datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
        except Exception:
            SESSION.trading_halted_until = datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
        logging.warning(f"Exceeded max consecutive losses ({SESSION.consecutive_losses}). Trading halted until {SESSION.trading_halted_until}.")

def get_last_trade_price(symbol):
    global data_client
    if data_client is None:
        logging.error("Data client not initialised. Cannot fetch last trade price.")
        return None
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trade = data_client.get_stock_latest_trade(req)
        obj = trade.get(symbol) if isinstance(trade, dict) else trade
        price = getattr(obj, 'price', None)
        return float(price) if price is not None else None
    except Exception as e:
        logging.error(f"Failed to get last trade price for {symbol}: {e}")
        return None

class Strategy:
    def __init__(self, rsi_period, atr_period):
        self.rsi_period = rsi_period
        self.atr_period = atr_period

    def calculate_indicators(self, df):
        df['RSI'] = RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
        df['MACD'] = MACD(close=df['close']).macd_diff()
        bb = BollingerBands(close=df['close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()

        # Robust ATR using true range
        atr_ind = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.atr_period)
        df['ATR'] = atr_ind.average_true_range()
        # Volatility as a percent of price (e.g., 0.012 = 1.2%)
        df['ATR_PCT'] = df['ATR'] / df['close']
        return df

strategy = Strategy(RSI_PERIOD, 14)

# --- Utility Functions ---
def is_market_open(ticker=None):
    if SESSION.api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return False
    clock = SESSION.api.get_clock()
    if not clock.is_open:
        return False

    if ticker:
        try:
            asset = SESSION.api.get_asset(ticker)
            if not asset.tradable:
                logging.warning(f"{ticker} is not tradable.")
                return False
        except Exception as e:
            logging.error(f"Error checking tradability for {ticker}: {e}")
            return False

    return True

def get_account():
    api_instance = SESSION.api
    if api_instance is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return None
    return api_instance.get_account()

def get_open_position_count():
    if SESSION.api is None:
        return 0
    try:
        return len(SESSION.api.get_all_positions())
    except Exception:
        return 0

def get_equity():
    account = get_account()
    if account is None:
        return 0.0
    try:
        return float(account.equity)
    except Exception:
        return 0.0

def get_position(symbol):
    if SESSION.api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return 0, 0
    try:
        pos = SESSION.api.get_open_position(symbol)
        return float(pos.qty), float(pos.avg_entry_price)
    except Exception:
        return 0, 0

# --- Async helpers for blocking calls ---

async def is_market_open_async(ticker=None):
    """
    Async wrapper for is_market_open so we don't block the event loop with API calls.
    """
    return await asyncio.to_thread(is_market_open, ticker)


async def get_position_async(symbol):
    """
    Async wrapper for get_position so Alpaca calls run in a thread, not the event loop.
    """
    return await asyncio.to_thread(get_position, symbol)


def ny_now():
    """Shorthand for now in NY timezone."""
    return datetime.datetime.now(ny_tz)

# A) ENTRY QUALITY: Helper functions for confirmation-first approach
def check_entry_confirmation(df: pd.DataFrame, atr_pct: float) -> bool:
    """Entry confirmation gate using only existing df data. Returns True if conditions pass."""
    if df is None or len(df) < 20:
        return False
    
    try:
        # 1) Trend alignment: require close above short EMA
        close_series = df['close']
        short_ema = close_series.ewm(span=9).mean()
        current_close = close_series.iloc[-1]
        current_ema = short_ema.iloc[-1]
        
        if current_close <= current_ema:
            return False
        
        # 2) Breakout OR pullback confirmation
        highs_10 = df['high'].tail(10)
        prior_high = highs_10.iloc[:-1].max() if len(highs_10) > 1 else highs_10.iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_open = df['open'].iloc[-1] if 'open' in df.columns else current_close
        
        # Breakout: close > prior 10-bar high AND candle body >= 40% of range
        candle_range = current_high - current_low
        candle_body = abs(current_close - current_open)
        breakout_ok = (current_close > prior_high and 
                      candle_range > 0 and 
                      candle_body >= 0.4 * candle_range)
        
        # Pullback: price above EMA, current candle green after mild pullback
        prev_close = close_series.iloc[-2] if len(close_series) > 1 else current_close
        pullback_ok = (current_close >= current_ema and 
                      current_close >= prev_close and 
                      current_low <= current_ema * 1.005)  # small tolerance
        
        if not (breakout_ok or pullback_ok):
            return False
        
        # 3) Volatility sanity: ATR_PCT rising or stable
        if 'ATR_PCT' in df.columns and len(df) >= 5:
            recent_atr = df['ATR_PCT'].tail(5)
            if len(recent_atr) >= 2:
                atr_slope = recent_atr.iloc[-1] - recent_atr.iloc[0]
                if atr_slope < -0.001:  # contracting volatility
                    return False
        
        return True
        
    except Exception:
        return False  # Safe fallback

def check_anti_chop_filter(df: pd.DataFrame) -> bool:
    """Returns False if market is choppy (alternating green/red > 70%). True = OK to enter."""
    if df is None or len(df) < 10:
        return True  # Not enough data, allow entry
    
    try:
        last_10 = df.tail(10)
        if 'open' not in last_10.columns:
            return True  # No open data, allow entry
        
        # Count alternating candles
        candle_colors = []
        for _, row in last_10.iterrows():
            candle_colors.append(1 if row['close'] >= row['open'] else 0)  # 1=green, 0=red
        
        if len(candle_colors) < 8:
            return True
        
        # Count alternations
        alternations = 0
        for i in range(1, len(candle_colors)):
            if candle_colors[i] != candle_colors[i-1]:
                alternations += 1
        
        alternation_rate = alternations / (len(candle_colors) - 1)
        return alternation_rate <= 0.7  # Allow if <= 70% alternating
        
    except Exception:
        return True  # Safe fallback

# B) Loss clustering detection
def check_loss_cluster() -> bool:
    """Returns True if loss clustering detected and buy pause should be applied."""
    now = ny_now()
    recent_losses = []
    
    # Check all tickers for recent losses within 30 minutes
    for ticker_data in SESSION.trade_history.values():
        if 'last_loss' in ticker_data:
            loss_time = ticker_data['last_loss']
            if (now - loss_time).total_seconds() <= 1800:  # 30 minutes
                recent_losses.append(loss_time)
    
    # If 2+ losses within 30 minutes, trigger pause
    return len(recent_losses) >= 2

def mark_exec_bad_ticker(ticker: str, reason: str):
    # Defensive guard for future refactors / early calls
    if not hasattr(SESSION, "bad_ticker_until"):
        return
    until = ny_now() + datetime.timedelta(minutes=EXEC_BAD_TICKER_COOLDOWN_MIN)
    SESSION.bad_ticker_until[ticker] = until
    reason_log(
        f"exec_bad:{ticker}",
        f"ticker_exec_locked | {ticker} | reason={reason} | until={until.isoformat()}"
    )

# --- Trading Logic ---
class RateLimitError(Exception):
    pass

class MarketDataError(Exception):
    pass

def place_order(symbol, qty, side):
    if SESSION.api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return False
    try:
        current_price = get_last_trade_price(symbol)
        
        # Get quote for reference pricing
        bid, ask, mid, spread_pct = get_latest_quote(symbol)
        ref_price = ask if side.lower() == 'buy' else bid
        if ref_price is None:
            ref_price = current_price

        # Compute ATR if we intend to use bracket orders on BUY
        atr_for_bracket = None
        if BRACKET_ORDERS_ENABLED and side.lower() == 'buy':
            try:
                df_tmp = fetch_data(symbol)
                if df_tmp is not None and 'ATR' in df_tmp.columns and not pd.isna(df_tmp['ATR'].iloc[-1]):
                    atr_for_bracket = float(df_tmp['ATR'].iloc[-1])
            except Exception as _e:
                logging.warning(f"{symbol}: could not compute ATR for bracket; falling back to market-only. {_e}")

        # Detect fractional quantity to determine correct TimeInForce
        qty_float = float(qty)
        is_fractional = not qty_float.is_integer()
        time_in_force = TimeInForce.DAY if is_fractional else TimeInForce.GTC

        # Build order (use OCO TP/SL when enabled and ATR available)
        if BRACKET_ORDERS_ENABLED and side.lower() == 'buy' and ref_price is not None and atr_for_bracket:
            tp_price = round(ref_price + (TAKE_PROFIT_ATR_MULT * atr_for_bracket), 2)
            sl_price = round(ref_price - (STOP_LOSS_ATR_MULT * atr_for_bracket), 2)
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty_float,
                side=OrderSide.BUY,
                time_in_force=time_in_force,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )
        else:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty_float,  # Alpaca supports fractional qty for eligible assets
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=time_in_force,
            )

        SESSION.api.submit_order(order)
        logging.info(f"Order {side.upper()} {symbol} @ {datetime.datetime.now(ny_tz)} with quantity {qty}")
        SESSION.last_trade_time[symbol] = datetime.datetime.now(ny_tz)

        # F) Track opening time and first trade status for BUY orders
        if side.lower() == 'buy':
            SESSION.trade_history[symbol] = SESSION.trade_history.get(symbol, {})
            SESSION.trade_history[symbol]['opened_at'] = ny_now()
            SESSION.trade_history[symbol]['entry_ref_price'] = ref_price if ref_price else 'N/A'
            SESSION.trade_history[symbol]['entry_spread_pct'] = spread_pct if spread_pct is not None else 'N/A'
            # Mark first trade as done on successful BUY
            if not SESSION.first_trade_done:
                SESSION.first_trade_done = True

        # --- Supabase log order submitted (schema-safe) ---
        supabase_log(
            f"order_submitted | {side.upper()} {symbol} qty={qty} @ {datetime.datetime.now(ny_tz).isoformat()}"
        )
        
        # --- UI notification for BUY orders ---
        if side.lower() == 'buy':
            ui_event(f"Order placed: BUY {symbol}. Position opened.")

        if side.lower() == 'sell':
            SESSION.trade_history[symbol] = SESSION.trade_history.get(symbol, {})
            SESSION.trade_history[symbol]['last_sell'] = datetime.datetime.now(ny_tz)
            # realized PnL is handled on close via close_position()
        else:
            SESSION.trade_history[symbol] = SESSION.trade_history.get(symbol, {})
            SESSION.trade_history[symbol]['last_buy'] = datetime.datetime.now(ny_tz)
        return True

    except APIError as e:
        if 'rate limit' in str(e).lower():
            logging.error(f"Rate limit error: {e}")
            raise RateLimitError("Rate limit exceeded. Consider slowing down requests.")
        logging.error(f"API error for {symbol}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in place_order for {symbol}: {e}")
    return False

def close_position(symbol):
    if SESSION.api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return None
    try:
        # Snapshot before closing
        pos_qty, entry_price = get_position(symbol)
        current_price = get_last_trade_price(symbol)
        if pos_qty <= 0 or current_price is None or entry_price <= 0:
            SESSION.api.close_position(symbol)
            logging.info(f"Closed position on {symbol}")
            SESSION.last_trade_time[symbol] = datetime.datetime.now(ny_tz)
            # --- Supabase log position closed (PnL unknown) ---
            supabase_log(
                f"position_closed | {symbol} | pnl=unknown @ {datetime.datetime.now(ny_tz).isoformat()}"
            )
            return 0.0
        pnl = (current_price - entry_price) * pos_qty
        SESSION.api.close_position(symbol)
        logging.info(f"Closed position on {symbol} @ {datetime.datetime.now(ny_tz)} | qty={pos_qty:.4f} entry={entry_price:.4f} exit={current_price:.4f} pnl={pnl:.2f}")
        SESSION.last_trade_time[symbol] = datetime.datetime.now(ny_tz)
        SESSION.realized_pnl += pnl
        SESSION.daily_realized_pnl += pnl
        # Mark loss timing for re-entry logic
        SESSION.trade_history[symbol] = SESSION.trade_history.get(symbol, {})
        if pnl < 0:
            SESSION.trade_history[symbol]['last_loss'] = datetime.datetime.now(ny_tz)
        SESSION.trade_history[symbol]['last_sell'] = datetime.datetime.now(ny_tz)
        # Register outcome for streaks/halts
        _register_trade_outcome(symbol, pnl)
        # --- Supabase log position closed ---
        entry_ref_price = SESSION.trade_history.get(symbol, {}).get('entry_ref_price', 'N/A')
        entry_spread_pct = SESSION.trade_history.get(symbol, {}).get('entry_spread_pct', 'N/A')
        supabase_log(
            f"position_closed | {symbol} | pnl={pnl:.2f} entry_ref={entry_ref_price} exit_price={current_price:.4f} spread_pct={entry_spread_pct} @ {datetime.datetime.now(ny_tz).isoformat()}"
        )
        
        # --- UI notification ---
        profit_loss = "profit" if pnl >= 0 else "loss"
        ui_event(f"Position closed: {symbol} with ${pnl:.2f} {profit_loss}.")
        
        # === Smart Ticker Recovery Lock (loss-based) ===
        if pnl < 0:
            state = _get_recovery_state(symbol)
            state.locked = True
            state.locked_at = ny_now()
            state.reason = "loss_exit"
            state.exit_price = current_price
            state.lowest_since_lock = current_price
            state.bars_without_new_low = 0

            # Capture context at lock time
            try:
                df_lock = fetch_data(symbol)
                if df_lock is not None:
                    state.lock_rsi = float(df_lock['RSI'].iloc[-1])
                    state.lock_atr_pct = float(df_lock['ATR_PCT'].iloc[-1])
                    state.lock_score = SESSION.trade_history.get(symbol, {}).get('last_signal_score')
            except Exception:
                pass

            supabase_log(
                f"ticker_locked | {symbol} | reason=loss_exit "
                f"| rsi={state.lock_rsi} atr_pct={state.lock_atr_pct} score={state.lock_score}"
            )
        
        # STEP 6: Event-based cleanup on position close
        if symbol in SESSION.profit_lock_states:
            del SESSION.profit_lock_states[symbol]
        
        return pnl
    except Exception as e:
        logging.error(f"Failed to close {symbol}: {e}")
        return None

def cancel_replace_stop_for_symbol(alpaca_client, symbol: str, qty: float, stop_price: float, existing_order_id: str | None):
    """
    Cancel previous stop (if provided), then place a new stop-market sell for 'qty' at 'stop_price'.
    Returns: new_order_id (or None if placement failed).
    """
    # Cancel old stop if exists
    if existing_order_id:
        try:
            alpaca_client.cancel_order_by_id(existing_order_id)
        except Exception:
            # ignore cancel failures (order may be filled/cancelled)
            pass

    # Place new stop-market sell
    try:
        order = alpaca_client.submit_order(
            symbol=symbol,
            qty=str(qty),
            side="sell",
            type="stop",
            time_in_force="gtc",
            stop_price=str(stop_price),
        )
        return getattr(order, "id", None)
    except Exception:
        return None

def close_position_market(alpaca_client, symbol: str):
    try:
        alpaca_client.close_position(symbol)
        return True
    except Exception:
        return False

def get_latest_quote(symbol):
    """
    Returns (bid, ask, mid, spread_pct).
    Uses SESSION.data_client.
    Fail-safe: return (None, None, None, None) on any error.
    """
    try:
        if SESSION.data_client is None:
            return (None, None, None, None)
        
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote_response = SESSION.data_client.get_stock_latest_quote(req)
        
        if symbol not in quote_response:
            return (None, None, None, None)
            
        quote = quote_response[symbol]
        bid_price = getattr(quote, 'bid_price', None)
        ask_price = getattr(quote, 'ask_price', None)
        
        if bid_price is None or ask_price is None or bid_price <= 0 or ask_price <= 0:
            return (None, None, None, None)
            
        mid = (bid_price + ask_price) / 2
        spread_pct = (ask_price - bid_price) / mid if mid > 0 else None
        
        return (bid_price, ask_price, mid, spread_pct)
    except Exception as e:
        logging.warning(f"Quote fetch failed for {symbol}: {e}")
        return (None, None, None, None)

def spread_is_acceptable(spread_pct: float | None, atr_pct: float | None) -> bool:
    """Adaptive spread gate: requires spread <= absolute cap AND <= fraction of ATR%."""
    try:
        if spread_pct is None:
            return True  # if we can't compute spread, don't block; other gates still apply
        if spread_pct <= 0:
            return True
        if spread_pct > MAX_SPREAD_PCT:
            return False
        if atr_pct is None or atr_pct <= 0:
            return spread_pct <= MAX_SPREAD_PCT
        return spread_pct <= (MAX_SPREAD_ATR_FRACTION * atr_pct)
    except Exception:
        return True

# === Smart Ticker Recovery Lock structures and helpers ===
class TickerRecoveryState:
    def __init__(self):
        self.locked = False
        self.locked_at = None
        self.reason = None
        self.exit_price = None
        self.lowest_since_lock = None
        self.bars_without_new_low = 0
        self.lock_rsi = None
        self.lock_atr_pct = None
        self.lock_score = None


def _get_recovery_state(ticker: str) -> TickerRecoveryState:
    state = SESSION.ticker_recovery_lock.get(ticker)
    if state is None:
        state = TickerRecoveryState()
        SESSION.ticker_recovery_lock[ticker] = state
    return state

# Recovery-check logic helper
def _check_recovery_and_update_state(ticker: str, price: float, rsi: float | None,
                                     atr_pct: float | None, score: float | None) -> bool:
    """
    Returns True if ticker is eligible to trade.
    If locked, requires >=2 recovery conditions to unlock.
    """
    state = _get_recovery_state(ticker)
    if not state.locked:
        return True

    conditions_met = []

    # 1) RSI back to neutral
    if rsi is not None and 45 <= rsi <= 60:
        conditions_met.append("rsi_neutral")

    # 2) Price stabilisation (no new lows)
    if state.lowest_since_lock is None:
        state.lowest_since_lock = price

    if price < state.lowest_since_lock:
        state.lowest_since_lock = price
        state.bars_without_new_low = 0
    else:
        state.bars_without_new_low += 1

    if state.bars_without_new_low >= 3:
        conditions_met.append("price_stabilised")

    # 3) Signal score positive again
    if score is not None and score > 0:
        conditions_met.append("score_positive")

    # 4) ATR normalisation (avoid falling knife)
    if (
        atr_pct is not None
        and state.lock_atr_pct is not None
        and atr_pct <= state.lock_atr_pct * 1.10
    ):
        conditions_met.append("atr_normalised")

    if len(conditions_met) >= 2:
        state.locked = False
        supabase_log(
            f"ticker_released | {ticker} | conditions={','.join(conditions_met)}"
        )
        return True

    # Still locked
    reason_log(
        f"ticker_locked:{ticker}",
        f"ticker_locked_skip | {ticker} | waiting_for_recovery "
        f"| met={conditions_met} bars_no_low={state.bars_without_new_low}"
    )
    return False

def update_pnl():
    if SESSION.api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return
    SESSION.unrealized_pnl = 0.0
    try:
        positions = SESSION.api.get_all_positions()
    except Exception as e:
        logging.error(f"Could not fetch positions for P&L update: {e}")
        return
    for position in positions:
        try:
            symbol = position.symbol
            pos_qty = float(position.qty)
            entry_price = float(position.avg_entry_price)
            current_price = get_last_trade_price(symbol)
            if pos_qty > 0 and current_price is not None and entry_price > 0:
                SESSION.unrealized_pnl += (current_price - entry_price) * pos_qty
        except Exception as e:
            logging.error(f"P&L calc error for {position.symbol}: {e}")
    logging.info(f"Realized P&L: ${SESSION.realized_pnl:.2f}, Unrealized P&L: ${SESSION.unrealized_pnl:.2f}")
    
    # Track peak equity for defensive logic
    current_equity = get_equity()
    if SESSION.session_peak_equity is None:
        SESSION.session_peak_equity = current_equity
    else:
        SESSION.session_peak_equity = max(SESSION.session_peak_equity, current_equity)
    
    # Smart drawdown defense (NO HALT)
    drawdown_pct = 0
    if SESSION.session_peak_equity:
        drawdown_pct = (current_equity - SESSION.session_peak_equity) / SESSION.session_peak_equity
    
    if drawdown_pct <= -0.004:
        now = datetime.datetime.now(ny_tz)
        if not SESSION.buy_pause_until or now >= SESSION.buy_pause_until:
            SESSION.buy_pause_until = now + datetime.timedelta(minutes=10)
            supabase_log(f"defensive_mode | buy_paused_10min | drawdown={drawdown_pct:.2%}")
    
    # Optional: lightweight log line into bot_logs (does not assume bot_status schema)
    reason_log("pnl_update", f"pnl_update | realized={SESSION.realized_pnl:.2f} unrealized={SESSION.unrealized_pnl:.2f} equity={get_equity():.2f}", min_seconds=300)
    
    # Rate-limited equity update for users
    current_equity = get_equity()
    # ui_log("equity_update", f"Equity update: ${current_equity:.2f} (Realized ${SESSION.realized_pnl:.2f})")

# Patch 2.1: Add async wrapper for P&L update
async def update_pnl_async():
    """Async wrapper so P&L updates don’t block the event loop."""
    await asyncio.to_thread(update_pnl)

def calculate_position_size(symbol, risk_per_trade):
    equity = get_equity()
    # Use only equity after reserving safety buffer
    usable_equity = max(equity * (1 - RESERVE_FUND_PCT), 0)
    
    # Capital scaling for multiple concurrent positions
    MAX_CAPITAL_USAGE = 0.95  # keep safety reserve
    available_capital = equity * MAX_CAPITAL_USAGE
    # Allow stronger trades to use more capital without increasing risk
    per_trade_capital = available_capital / max(MAX_CONCURRENT_POSITIONS * 0.6, 1)
    
    max_risk = max(usable_equity * risk_per_trade, 0.01)  # ensure a tiny non-zero risk budget

    # Risk-state sizing (keeps bot trading in DEGRADED instead of freezing it)
    state = getattr(SESSION, "risk_state", "NORMAL")
    if state == "DEGRADED":
        max_risk *= 0.75   # reduced size, still trades
    elif state == "HALT":
        return 0.0

    df = fetch_data(symbol)
    if df is None or df.empty:
        return 0.0

    current_price = float(df['close'].iloc[-1])
    atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else 0.0
    atr_pct = float((atr / current_price)) if current_price > 0 else 0.0

    # Guard: avoid trading if volatility is too low/high
    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
        return 0.0

    # Risk per share ≈ ATR * STOP_LOSS_ATR_MULT
    risk_per_share = max(atr * STOP_LOSS_ATR_MULT, current_price * 0.002)  # at least 0.2% cushion
    if risk_per_share <= 0:
        return 0.0

    raw_qty = max_risk / risk_per_share

    # Cap by per-ticker equity exposure (portfolio concentration)
    max_dollars_per_ticker = usable_equity * MAX_EQUITY_FRACTION_PER_TICKER

    # Cap by per-trade notional (entry sizing) to prevent oversized single entries
    max_dollars_per_trade = usable_equity * MAX_TRADE_NOTIONAL_PCT if MAX_TRADE_NOTIONAL_PCT else max_dollars_per_ticker
    if MAX_TRADE_NOTIONAL_ABS is not None:
        max_dollars_per_trade = min(max_dollars_per_trade, float(MAX_TRADE_NOTIONAL_ABS))
    
    # Apply capital scaling for concurrent positions
    trade_capital = min(
        per_trade_capital,
        max_dollars_per_trade  # existing safety cap
    )

    # Apply both caps
    effective_max_dollars = min(max_dollars_per_ticker, trade_capital)
    if current_price > 0:
        raw_qty = min(raw_qty, effective_max_dollars / current_price)

    # Never allow position size to exceed available buying power
    account = get_account()
    if account is not None:
        try:
            buying_power = float(account.buying_power)
            max_qty_by_bp = buying_power / current_price if current_price > 0 else 0.0
            raw_qty = min(raw_qty, max_qty_by_bp)
        except Exception:
            pass

    return round(float(raw_qty), 4)

def scale_qty_by_score(qty, score):
    if score is None:
        return 0.0
    if score < 0.25:
        return qty * 0.55
    elif score < 0.40:
        return qty * 0.65
    elif score < 0.60:
        return qty * 0.85
    else:
        return qty

def fetch_data(symbol):
    global data_client
    if data_client is None:
        logging.error("Data client not initialised. Cannot fetch bar data.")
        return None
    try:
        # Try minute bars first
        req_min = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=1500)
        bars_min = data_client.get_stock_bars(req_min).df
        if isinstance(bars_min.index, pd.MultiIndex):
            try:
                bars_min = bars_min.xs(symbol)
            except Exception:
                pass
        bars_min = bars_min[bars_min['close'].notna() & bars_min['high'].notna() & bars_min['low'].notna()]

        df = bars_min
        if df.empty or len(df) < RSI_PERIOD + 1:
            logging.warning(f"{symbol}: Minute bars too short or missing. Trying daily fallback...")
            req_day = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=100)
            bars_day = data_client.get_stock_bars(req_day).df
            if isinstance(bars_day.index, pd.MultiIndex):
                try:
                    bars_day = bars_day.xs(symbol)
                except Exception:
                    pass
            df = bars_day[bars_day['close'].notna() & bars_day['high'].notna() & bars_day['low'].notna()]

        if df is None or df.empty:
            logging.warning(f"{symbol}: No bar data available at all. Skipping.")
            return None

        df = df.sort_index()
        
        # Guard: ensure enough bars before indicator calculation
        if df is None or df.empty or len(df) < MIN_BARS_REQUIRED:
            logging.warning(f"{symbol}: insufficient bars ({0 if df is None else len(df)}).")
            mark_exec_bad_ticker(symbol, "insufficient_bars")
            return None
        
        df = strategy.calculate_indicators(df.copy())
        
        # Check for NaN in critical indicators
        latest_row = df.iloc[-1]
        critical_indicators = ['RSI', 'ATR', 'ATR_PCT']
        for indicator in critical_indicators:
            if indicator in df.columns and pd.isna(latest_row[indicator]):
                mark_exec_bad_ticker(symbol, "indicator_nan")
                return None

        needed_cols = [c for c in ['RSI', 'ATR', 'ATR_PCT'] if c in df.columns]
        if df[needed_cols].dropna().shape[0] < 3:
            logging.warning(f"{symbol}: Not enough valid RSI/ATR data. Skipping.")
            return None

        return df
    except Exception as e:
        logging.error(f"{symbol}: Data/indicator failure - {e}")
        mark_exec_bad_ticker(symbol, "indicator_exception")
        return None

# Convert fetch_data to an asynchronous function
async def fetch_data_async(symbol):
    return await asyncio.to_thread(fetch_data, symbol)

# Convert place_order to an asynchronous function
async def place_order_async(symbol, qty, side):
    return await asyncio.to_thread(place_order, symbol, qty, side)

# Placeholder for sentiment analysis
async def get_sentiment_score(ticker):
    """
    Placeholder function for AI-based sentiment analysis.
    Returns a sentiment score between -1 (very negative) and 1 (very positive).
    """
    # Future integration: Use an AI model or API to fetch sentiment score
    return 0  # Neutral sentiment by default

# Weighted Signal Scoring System
INDICATOR_WEIGHTS = {
    'RSI': 0.5,
    'ATR': 0.3,
    'Sentiment': 0.2
}
# --- Signal thresholds & trade guards ---
SIGNAL_BUY_THRESHOLD = 0.12      # was 0.16 (too strict → no trades)
SIGNAL_SELL_THRESHOLD = -0.08    # was -0.10 (exit earlier on decay)
MICRO_PROFIT_TAKE = 0.0040       # keep friction-aware, slightly higher than before

# === Tiered Execution Thresholds ===
TIER1_THRESHOLD_MULT = 1.00   # full confidence
TIER2_THRESHOLD_MULT = 0.85   # good but not perfect
TIER2_SIZE_MULT = 0.50        # half-size entries

# Volatility guard (ignore ultra-quiet or too-wild regimes)
# NOTE: slightly widened to allow more tradable setups while still avoiding dead/chaotic regimes
MIN_ATR_PCT = 0.0006   # allow calmer names (0.06%)
MAX_ATR_PCT = 0.12     # slightly wider top-end; extreme names still filtered elsewhere

# Spread filtering (relaxed + adaptive)
# Many tickers (especially outside mega-caps) have spreads >0.15% even when tradable.
# We cap spread by BOTH an absolute % and a fraction of ATR% so we don't enter illiquid names.
MAX_SPREAD_PCT = 0.0040            # 0.40% absolute cap
MAX_SPREAD_ATR_FRACTION = 0.45     # allow spread up to 45% of ATR% (still adaptive)

# Entry quality guardrails
MAX_RSI_FOR_ENTRY = 65        # avoid chasing overbought moves on entry

# --- Execution-level data hygiene ---
EXEC_BAD_TICKER_COOLDOWN_MIN = 45   # minutes to ignore tickers with bad/insufficient data

# --- Active ticker rotation constants ---
AI_REFRESH_SECONDS = 600   # 10 minutes (more responsive during the session)
MIN_ACTIVE_TICKERS = 25    # keep a broader universe

# === Trade activity tuning (safe) ===
MVT_ENABLED = True
MVT_MIN_SCORE = 0.16                  # allow "good enough" setups when momentum/trend confirm
MVT_RSI_MIN = 42
MVT_RSI_MAX = 62
MVT_MIN_ATR_PCT = 0.0016              # 0.16% ATR% minimum for MVT
MVT_MINUTES_AFTER_OPEN = 15           # don't MVT in the very first minutes
FIRST_TRADE_BIAS_ENABLED = True
FIRST_TRADE_BIAS_WINDOW_MIN = 60      # only applies for first entry within first hour
FIRST_TRADE_SCORE_RELAX = 0.03        # reduce threshold by 0.03 for first trade only
FIRST_TRADE_RSI_MAX = 72              # allow a slightly higher RSI for first trade only

# Profit lock engine constants
PROFIT_LOCK_COSTS_PCT = 0.0008
PROFIT_LOCK_MIN_STEP_PCT = 0.0005
PROFIT_LOCK_COOLDOWN_SEC = 20

# Supabase TTL cleanup constants
CLEANUP_TTL_DAYS = 7
SUPABASE_CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours

# Position sizing caps
MIN_DOLLARS_PER_TRADE = 0.0  # We allow very small fractional trades
MAX_EQUITY_FRACTION_PER_TICKER = MAX_TICKER_EXPOSURE  # reuse your 30% cap

# Calculate weighted signal score
def calculate_signal_score(rsi, atr, sentiment, macd, close_price, bb_upper, bb_lower):
    if pd.isna(rsi) or pd.isna(atr) or pd.isna(sentiment) or pd.isna(macd) or pd.isna(bb_upper) or pd.isna(bb_lower):
        return None
    bb_score = 0
    if close_price < bb_lower:
        bb_score = 1
    elif close_price > bb_upper:
        bb_score = -1
    score = (
        INDICATOR_WEIGHTS['RSI'] * (1 - rsi / 100) +
        INDICATOR_WEIGHTS['ATR'] * atr +
        INDICATOR_WEIGHTS['Sentiment'] * sentiment +
        0.1 * macd +
        0.1 * bb_score
    )
    return score

# Add trailing stop-loss logic
def update_trailing_stop(entry_price, current_price, trail_amount, previous_stop):
    """
    Calculate the new trailing stop-loss price.
    """
    new_stop = max(previous_stop, current_price - trail_amount)
    return new_stop

def detect_market_regime(df):
    """
    Detect market regime based on current conditions.
    Returns: "TREND", "SQUEEZE", or "MEAN_REVERSION"
    """
    try:
        # Need at least 20 bars for regime detection
        if len(df) < 20:
            return "TREND"  # Default to trend for safety
            
        # Get recent price action (last 20 bars)
        recent_close = df['close'].tail(20)
        recent_atr_pct = df['ATR_PCT'].tail(5) if 'ATR_PCT' in df else None
        
        # Calculate price volatility and trend strength
        price_std = recent_close.std() / recent_close.mean() if recent_close.mean() > 0 else 0
        
        # Simple trend detection: compare current price to 10-period average
        sma_10 = recent_close.tail(10).mean()
        current_price = recent_close.iloc[-1]
        trend_strength = abs(current_price - sma_10) / sma_10 if sma_10 > 0 else 0
        
        # ATR-based volatility assessment
        avg_atr_pct = recent_atr_pct.mean() if recent_atr_pct is not None and len(recent_atr_pct) > 0 else 0.01
        
        # Regime classification logic
        if trend_strength > 0.02 and avg_atr_pct > 0.015:
            return "TREND"  # Strong directional movement with good volatility
        elif avg_atr_pct < 0.008:
            return "SQUEEZE"  # Low volatility, consolidating
        else:
            return "MEAN_REVERSION"  # Moderate volatility, likely range-bound
            
    except Exception as e:
        logging.warning(f"Market regime detection failed: {e}")
        return "TREND"  # Safe default

def is_last_30_minutes_of_market():
    """
    Check if we're in the last 30 minutes of market hours.
    Returns True if within 30 minutes of market close.
    """
    if SESSION.api is None:
        return False
    try:
        clock = SESSION.api.get_clock()
        if not clock.is_open:
            return True  # Market closed, treat as "last 30 minutes"
        
        time_to_close = clock.next_close - clock.timestamp
        return time_to_close <= datetime.timedelta(minutes=30)
    except Exception as e:
        logging.warning(f"Failed to check market close time: {e}")
        return False  # Safe default

def has_sufficient_buying_power(required_dollars):
    """
    Check if account has sufficient buying power for the trade.
    """
    account = get_account()
    if account is None:
        return False
    try:
        buying_power = float(account.buying_power)
        return buying_power >= required_dollars
    except Exception as e:
        logging.warning(f"Failed to check buying power: {e}")
        return False


# Global run-state is now controlled by the worker via task cancellation.
# The trading loop itself does not manage process lifetime.



async def safe_api_call(api_function, *args, **kwargs):
    retries = 5
    delay = 1
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(api_function, *args, **kwargs)
        except Exception as e:
            logging.error(f"API call failed: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2
    logging.critical("API call failed after multiple retries.")
    return None

async def close_all_positions():
    if SESSION.api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return
    try:
        positions = await asyncio.to_thread(SESSION.api.get_all_positions)
        for position in positions:
            symbol = position.symbol
            logging.info(f"Closing position for {symbol}.")
            await safe_api_call(SESSION.api.close_position, symbol)
    except Exception as e:
        logging.error(f"Failed to close all positions: {e}")


# Helper so the server can request a graceful stop; the worker will cancel the task.
def request_bot_stop():
    # --- Supabase log bot stop requested ---
    supabase_log(f"bot_stop_requested @ {datetime.datetime.now(ny_tz).isoformat()}")
    ui_event("Bot stop requested. Shutting down safely…")
    logging.info("Bot stop requested; worker will cancel trade_loop_async task.")

async def wait_until_market_open_or_preopen(preopen_minutes: int = 10):
    if SESSION.api is None:
        await asyncio.sleep(60)
        return "closed"

    clock = await asyncio.to_thread(SESSION.api.get_clock)

    if clock.is_open:
        return "open"

    now = clock.timestamp
    next_open = clock.next_open
    seconds_to_open = max((next_open - now).total_seconds(), 0)

    if seconds_to_open > preopen_minutes * 60:
        sleep_for = seconds_to_open - (preopen_minutes * 60)
        logging.info(f"Market closed. Sleeping until pre-open window ({int(sleep_for)}s).")
        await asyncio.sleep(sleep_for)
        return "preopen"

    logging.info("Pre-open window reached. Waiting for market open.")
    await asyncio.sleep(max(seconds_to_open, 1))
    return "open"

async def trade_loop_async(allowed_tickers=None):
    # Ensure correct session context
    get_session(SESSION.USER_ID, SESSION.CURRENT_MODE)
    if allowed_tickers is not None:
        # Override AI-selected tickers
        SESSION.ACTIVE_TICKERS = list(allowed_tickers)
    
    # Market-aware startup: instant wake-up
    while True:
        state = await wait_until_market_open_or_preopen()

        if state == "preopen":
            logging.info("Pre-open phase: waiting for market open.")
            continue

        if state == "open":
            logging.info("Market open detected — entering trading loop immediately.")
            ui_event("Market is open. Bot is now actively monitoring the market.")
            # Capture market open timestamp for "first trade bias" + MVT timing guards
            try:
                clock = await asyncio.to_thread(SESSION.api.get_clock)
                # Use NY timezone datetime for consistency with other logic
                open_dt = clock.timestamp.astimezone(ny_tz)
                SESSION.market_open_ts = open_dt
                SESSION.last_market_open_date = open_dt.date()
                SESSION.first_trade_done = False
            except Exception as e:
                logging.warning(f"Could not capture market open timestamp: {e}")
            
            # On market open, immediately get tickers if not overridden
            if allowed_tickers is None and not SESSION.ACTIVE_TICKERS:
                for retry in range(3):
                    try:
                        selected_tickers = await asyncio.to_thread(
                            get_top_tickers,
                            30,
                            SESSION.USER_ID,
                            SESSION.CURRENT_MODE
                        )
                        if selected_tickers:
                            SESSION.ACTIVE_TICKERS = list(selected_tickers)
                            logging.info(f"Market-open AI tickers loaded: {len(SESSION.ACTIVE_TICKERS)} tickers")
                            reason_log("ai_scan_complete", f"ai_scan_complete | tickers={len(SESSION.ACTIVE_TICKERS)}", min_seconds=300)
                            break
                        else:
                            logging.warning(f"AI ticker selector returned empty on retry {retry+1}/3")
                            if retry < 2:
                                await asyncio.sleep(10)
                    except Exception as e:
                        logging.error(f"AI ticker selector error on retry {retry+1}/3: {e}")
                        if retry < 2:
                            await asyncio.sleep(10)
            break
    
    logging.info(f"Trading loop active with {len(SESSION.ACTIVE_TICKERS)} tickers")
    # Daily safety reset for first-trade bias if worker spans multiple days
    try:
        today = ny_now().date()
        if SESSION.last_market_open_date != today:
            SESSION.first_trade_done = False
            SESSION.last_market_open_date = today
    except Exception:
        pass
    last_ticker_refresh = ny_now()
    last_heartbeat_log = ny_now()
    loop_counter = 0  # STEP 5: Loop counter for periodic cleanup
    while True:
        try:
            # STEP 5: Periodic profit lock state cleanup (every ~60 iterations ≈ 1 hour)
            loop_counter += 1
            if loop_counter % 60 == 0:
                cleaned_count = cleanup_old_profit_lock_states(SESSION.profit_lock_states)
                if cleaned_count > 0:
                    logging.info(f"[MEMORY] Cleaned {cleaned_count} stale profit lock states (TTL=7d)")
            
            # Supabase cleanup (throttling handled inside function)
            cleanup_old_supabase_rows()
            
            # Market state check with throttling (max once per 30 seconds)
            now = ny_now()
            if (SESSION.last_clock_check is None or 
                (now - SESSION.last_clock_check).total_seconds() >= 30):
                
                market_state, time_to_close = get_market_state()
                SESSION.market_state = market_state
                SESSION.last_clock_check = now
                
                # Handle market state transitions
                if market_state != "OPEN":
                    if market_state == "CLOSED" and (SESSION.last_market_state_log is None or 
                        (now - SESSION.last_market_state_log).total_seconds() > 300):
                        logging.info("Market closed. Bot sleeping.")
                        SESSION.last_market_state_log = now
                    await wait_until_market_open_or_preopen()
                    continue
                    
                # Check for closeout window
                if market_state == "OPEN" and in_closeout_window(time_to_close):
                    SESSION.market_state = "CLOSING"
                    try:
                        clock = await asyncio.to_thread(SESSION.api.get_clock)
                        SESSION.market_close_guard_until = clock.next_open if hasattr(clock, 'next_open') else now + datetime.timedelta(hours=24)
                    except Exception:
                        SESSION.market_close_guard_until = now + datetime.timedelta(hours=24)
                    
                    logging.info("Market closing – closing all positions and pausing entries.")
                    await close_all_positions()
                    await wait_until_market_open_or_preopen()
                    continue
            
            # Skip all processing if in closing state or guard active
            if (SESSION.market_state == "CLOSING" or 
                (SESSION.market_close_guard_until and now < SESSION.market_close_guard_until)):
                await asyncio.sleep(60)
                continue
            
            # Only proceed if market is open
            if SESSION.market_state != "OPEN":
                await asyncio.sleep(30)
                continue
                
            # Clean open-only scan cycle starts here
            user_minute_log("Bot running — scanning market for opportunities")
            user_five_min_log("Bot active — scanning market and evaluating AI signals")
            
            # Rate-limited evaluation logging
            reason_log("eval_loop", "bot_eval | evaluating trade opportunities", min_seconds=120)
            
            # 1. Daily resets and risk state management
            reset_daily_limits_if_new_day()
            risk_state = get_risk_state()
            
            # Only sleep in HALT state, not DEGRADED
            if risk_state == "HALT":
                logging.warning("Trading HALTED due to risk limits. Sleeping 5 minutes...")
                await asyncio.sleep(300)
                continue
            elif risk_state == "DEGRADED":
                logging.warning("Trading in DEGRADED state - reduced position sizes and stricter entry criteria")
                # Continue trading with degraded conditions

            # 2. Throttled P&L update (max once per 60 seconds)
            if (SESSION.last_pnl_update is None or 
                (now - SESSION.last_pnl_update).total_seconds() >= 60):
                await update_pnl_async()
                SESSION.last_pnl_update = now
            # 3. AI Ticker selection & active ticker maintenance (open market only)
            fresh_ai_tickers = []
            if SESSION.market_state == "OPEN":
                # Throttled AI ticker refresh (once per 60 seconds max)
                if (SESSION.last_ai_refresh is None or 
                    (now - SESSION.last_ai_refresh).total_seconds() >= 60):
                    try:
                        fresh_ai_tickers = await asyncio.to_thread(
                            get_top_tickers,
                            30,  # Get 30 tickers from AI
                            SESSION.USER_ID,
                            SESSION.CURRENT_MODE
                        )
                        SESSION.last_ai_refresh = now
                        if fresh_ai_tickers:
                            reason_log("ai_refresh", f"AI refresh: got {len(fresh_ai_tickers)} fresh tickers", min_seconds=300)
                    except Exception as e:
                        logging.warning(f"AI ticker refresh failed: {e}")
            
            # Maintain active ticker set, spreading attention across tickers
            try:
                current_positions = await asyncio.to_thread(SESSION.api.get_all_positions)
                position_symbols = [pos.symbol for pos in current_positions if hasattr(pos, 'symbol')]
            except Exception as e:
                logging.warning(f"Failed to fetch current positions: {e}")
                current_positions = []
                position_symbols = []
            
            # Combine fresh AI picks with existing positions to prevent early exits
            all_candidate_tickers = list(set(fresh_ai_tickers + position_symbols + SESSION.ACTIVE_TICKERS))
            
            # Prevent getting stuck on 1-2 tickers - rotate attention
            if len(all_candidate_tickers) > 4:
                # Spread attention across tickers every evaluation
                if not hasattr(SESSION, 'rotation_index'):
                    SESSION.rotation_index = 0
                
                # Select subset for this round (spread attention)
                batch_size = min(12, len(all_candidate_tickers))
                start_idx = SESSION.rotation_index % len(all_candidate_tickers)
                
                tickers_to_evaluate = []
                for i in range(batch_size):
                    ticker_idx = (start_idx + i) % len(all_candidate_tickers)
                    tickers_to_evaluate.append(all_candidate_tickers[ticker_idx])
                
                SESSION.rotation_index = (SESSION.rotation_index + batch_size) % len(all_candidate_tickers)
                
                reason_log("ticker_rotation", f"Rotating attention: evaluating {len(tickers_to_evaluate)} of {len(all_candidate_tickers)} candidate tickers", min_seconds=180)
            else:
                tickers_to_evaluate = all_candidate_tickers
            
            # 4. Concurrent ticker evaluation with semaphore (limit 10 simultaneous operations)
            if tickers_to_evaluate:
                semaphore = asyncio.Semaphore(10)
                
                async def process_ticker_with_limit(ticker):
                    async with semaphore:
                        try:
                            # Stale evaluation guard (spread attention, avoid spam)
                            if ticker in SESSION.last_ticker_eval:
                                last_eval = SESSION.last_ticker_eval[ticker]
                                if (now - last_eval).total_seconds() < 30:
                                    return None
                            
                            SESSION.last_ticker_eval[ticker] = now
                            return await process_ticker(ticker)
                        except Exception as e:
                            logging.warning(f"Error processing ticker {ticker}: {e}")
                            return None
                
                # Process all tickers concurrently within semaphore limit
                tasks = [process_ticker_with_limit(ticker) for ticker in tickers_to_evaluate]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log successful evaluations (throttled)
                successful_evals = len([r for r in results if r is not None and not isinstance(r, Exception)])
                if successful_evals > 0:
                    reason_log("eval_success", f"Successfully evaluated {successful_evals}/{len(tickers_to_evaluate)} tickers", min_seconds=120)
            
            # 5. Update active ticker set based on current positions
            SESSION.ACTIVE_TICKERS = list(set(
                position_symbols +
                list(SESSION.ACTIVE_TICKERS)[-20:]  # Keep last 20 active tickers for momentum
            ))
            
            # Clean evaluation cycle complete
            reason_log("eval_complete", f"Evaluation cycle complete | Active positions: {len(current_positions)} | Market: {SESSION.market_state}", min_seconds=120)
            
            # Clean cycle pause
            await asyncio.sleep(15)

        except RateLimitError as e:
            logging.error(e)
            await asyncio.sleep(60)  # Wait before retrying
        except asyncio.CancelledError:
            logging.info("trade_loop_async was cancelled, exiting gracefully.")
            return
        except Exception as e:
            logging.critical(f"Critical error in trade_loop: {e}")
            await asyncio.sleep(300)  # Wait before retrying
    logging.info("trade_loop_async has exited.")

# Process a single ticker asynchronously
async def process_ticker(ticker):
    try:
        await asyncio.sleep(random.uniform(0.2, 0.6))
        
        # Execution-level bad ticker cooldown
        bad_until = SESSION.bad_ticker_until.get(ticker)
        if bad_until and ny_now() < bad_until:
            return
        
        # Per-ticker loss cooldown check
        if is_ticker_in_cooldown(ticker):
            return
        
        # Respect global halts
        if should_halt_trading():
            return

        df = await fetch_data_async(ticker)
        if df is None or len(df) < MIN_BARS_REQUIRED:
            mark_exec_bad_ticker(ticker, "fetch_failed_or_short")
            return

        # Latest values
        rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df else np.nan
        atr = float(df['ATR'].iloc[-1]) if 'ATR' in df else np.nan
        close_price = float(df['close'].iloc[-1])
        macd = float(df['MACD'].iloc[-1]) if 'MACD' in df else 0.0
        bb_upper = float(df['BB_upper'].iloc[-1]) if 'BB_upper' in df else np.nan
        bb_lower = float(df['BB_lower'].iloc[-1]) if 'BB_lower' in df else np.nan
        atr_pct = float(df['ATR_PCT'].iloc[-1]) if 'ATR_PCT' in df else (atr / close_price if close_price else np.nan)

        # Position info
        pos_qty, entry_price = await get_position_async(ticker)
        have_position = pos_qty > 0

        # Clean up stale profit lock states for positions that no longer exist
        if ticker in SESSION.profit_lock_states and not have_position:
            del SESSION.profit_lock_states[ticker]

        # Sentiment (placeholder currently returns 0)
        sentiment = await get_sentiment_score(ticker)

        # Compute signal score (uses RSI/ATR/sentiment + small MACD/BB blend)
        score = calculate_signal_score(
            rsi=rsi, atr=atr_pct, sentiment=sentiment,
            macd=macd, close_price=close_price,
            bb_upper=bb_upper, bb_lower=bb_lower
        )
        if score is None:
            return

        # Quick volatility guard
        if pd.isna(atr_pct) or atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
            reason_log(
                f"atr_skip:{ticker}",
                f"entry_blocked | {ticker} | reason=atr_out_of_range | score={score} rsi={rsi} atr_pct={atr_pct}",
                min_seconds=300,
            )
            return

        # Spread gate (adaptive): apply size modifier instead of hard block
        bid, ask, mid, spread_pct = get_latest_quote(ticker)
        spread_size_mult = 1.0
        if spread_pct is not None and atr_pct is not None:
            if spread_pct > MAX_SPREAD_PCT:
                spread_size_mult = 0.40
            elif spread_pct > (MAX_SPREAD_ATR_FRACTION * atr_pct):
                spread_size_mult = 0.60
        
        # Keep existing logging for wide spreads (but don't block the trade)
        if not spread_is_acceptable(spread_pct, atr_pct) and not have_position:
            # Spread is handled via size reduction, not hard veto
            reason_log(
                f"spread_soft:{ticker}",
                f"spread_softened | {ticker} | score={score} rsi={rsi} atr_pct={atr_pct} spread_pct={spread_pct}",
                min_seconds=300,
            )

        # Cooldowns after recent actions
        now = ny_now()
        last_sell_time = SESSION.trade_history.get(ticker, {}).get("last_sell")
        if not have_position and last_sell_time:
            elapsed = (now - last_sell_time).total_seconds()
            # 1h after profit / 2h after loss handled elsewhere, but keep a basic 5-min safety here too
            if elapsed < REENTRY_COOLDOWN:
                logging.info(f"Skipping {ticker}: re-entry cooldown ({int(REENTRY_COOLDOWN - elapsed)}s left).")
                return

        # Partial entry constants
        PARTIAL_ENTRY_ENABLED = True
        PARTIAL_ENTRY_SCORE_FLOOR = 0.10     # was 0.14 (too strict)
        PARTIAL_ENTRY_MULTIPLIER = 0.35      # was 0.25 (too tiny to matter)

        # If signal is decent but not perfect, allow a small starter position.
        # This increases trade frequency without removing safety gates.
        partial_entry_allowed = (not have_position) and (score is not None) and (score >= PARTIAL_ENTRY_SCORE_FLOOR)

        # Detect market regime for adaptive strategy
        regime = detect_market_regime(df)
        logging.debug(f"[REGIME] {ticker} | regime={regime}")
        
        # Phase 1 Safety: SQUEEZE handling (soft sizing)
        # Instead of a hard ban (which can cause zero trades for long periods), we:
        # - Apply size reduction for SQUEEZE regimes
        # - Only hard block ultra-low volatility (extreme dead market)
        squeeze_size_mult = 1.0
        if regime == "SQUEEZE" and not have_position:
            if pd.isna(atr_pct) or atr_pct < 0.0009:
                logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=SQUEEZE_DEAD_MARKET atr%={atr_pct:.4f}")
                reason_log(f"squeeze_dead:{ticker}", f"entry_blocked | {ticker} | reason=squeeze_dead_market | score={score} rsi={rsi} atr_pct={atr_pct}")
                return
            else:
                squeeze_size_mult = 0.35  # Reduced size for SQUEEZE regime
                logging.debug(f"[SQUEEZE SIZE REDUCTION] {ticker} | regime={regime} atr%={atr_pct:.4f} mult={squeeze_size_mult}")

        # ================ SMART EXIT LOGIC (if holding) ================
        # C) FIX EXIT LOGIC CONFLICTS WITH BRACKET ORDERS
        if have_position:
            # STEP 7: Safety guard for invalid states - skip if entry_price <= 0 or qty <= 0
            if pos_qty > 0 and entry_price > 0 and close_price > 0:
                # Create/update profit lock state
                st = SESSION.profit_lock_states.get(ticker)
                if st is None:
                    st = ProfitLockState(
                        symbol=ticker,
                        side="long",
                        entry_price=float(entry_price),
                        qty=float(pos_qty),
                    )
                    SESSION.profit_lock_states[ticker] = st
                else:
                    # Keep peak/locks, but refresh qty/entry if broker reports change
                    st.qty = float(pos_qty)
                    st.entry_price = float(entry_price)
                
                # STEP 4: Compute current unrealized PCT
                current_price = float(close_price)
                unrealized_pct = (current_price - st.entry_price) / st.entry_price
                
                # STEP 5: Call profit lock engine
                should_update, new_stop_price, force_exit, reason = profit_lock_step(
                    st,
                    current_price=current_price,
                    unrealized_pct=unrealized_pct,
                    costs_pct=PROFIT_LOCK_COSTS_PCT,
                    min_stop_step_pct=PROFIT_LOCK_MIN_STEP_PCT,
                    update_cooldown_sec=PROFIT_LOCK_COOLDOWN_SEC,
                )
                
                # Log only when action happens
                if should_update or force_exit:
                    logging.info(
                        f"PROFIT_LOCK {ticker} | px={current_price:.4f} unreal={unrealized_pct:.4%} | {reason}"
                    )
                
                # STEP 7: Wire stop updates
                if should_update:
                    new_id = cancel_replace_stop_for_symbol(
                        alpaca_client=SESSION.api,
                        symbol=st.symbol,
                        qty=st.qty,
                        stop_price=new_stop_price,
                        existing_order_id=st.order_id_stop,
                    )
                    if new_id:
                        st.order_id_stop = new_id
                    else:
                        logging.info(f"STOP_UPDATE_FAILED {ticker} stop={new_stop_price:.6f}")
                
                # STEP 8: Force exit on lock violation
                if force_exit:
                    success = close_position_market(SESSION.api, ticker)
                    logging.info(f"LOCK_VIOLATION_EXIT {ticker} success={success} | {reason}")
                    # Don't delete state yet - let stale cleanup handle it
            
            # Existing exit logic continues below
            # When bracket orders are enabled, only run ATR circuit breaker and market close exits
            if BRACKET_ORDERS_ENABLED:
                # Calculate current PnL
                pnl_pct = 0.0
                if entry_price > 0:
                    pnl_pct = (close_price - entry_price) / entry_price
                
                # ATR-based circuit breaker ONLY
                atr_for_position = df['ATR'].iloc[-1]
                if not pd.isna(atr_for_position) and entry_price > 0:
                    atr_stop_pct = 2.2 * atr_for_position / entry_price
                    if pnl_pct <= -atr_stop_pct:
                        pnl = close_position(ticker)
                        SESSION.recently_traded[ticker] = ny_now()
                        SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                        supabase_log(f"atr_circuit_exit | {ticker} | atr_pct={atr_stop_pct:.4f}")
                        return
                
                # 2) Market closeout logic will handle closing positions at market close
                # (existing close_all_positions and closeout-window logic remains intact)
                
            else:
                # E) OVERTRADING CONTROL: Check minimum hold time for discretionary exits
                MIN_HOLD_SECONDS = 300  # 5 minutes
                opened_at = SESSION.trade_history.get(ticker, {}).get('opened_at')
                if opened_at:
                    hold_time_seconds = (ny_now() - opened_at).total_seconds()
                    if hold_time_seconds < MIN_HOLD_SECONDS:  # 5 minutes minimum hold
                        # Allow emergency exits but skip discretionary ones
                        pnl_pct = (close_price - entry_price) / entry_price if entry_price > 0 else 0.0
                        if pnl_pct > -0.006:  # Not an emergency
                            return
                
                # ---- Original Simplified Unified SMART EXIT LOGIC (when brackets disabled) ----
                # Calculate current PnL
                pnl_pct = 0.0
                if entry_price > 0:
                    pnl_pct = (close_price - entry_price) / entry_price

                # Track peak for reversal detection
                peak_key = f"{ticker}_peak_pnl"
                prev_peak = SESSION.recently_traded.get(peak_key, 0.0)
                new_peak = max(prev_peak, pnl_pct)
                SESSION.recently_traded[peak_key] = new_peak
                
                # Peak giveback exit (bracket orders disabled)
                giveback = new_peak - pnl_pct
                if (new_peak >= 0.0040 and pnl_pct >= 0.0010 and
                    ((0.0040 <= new_peak < 0.0080 and giveback >= 0.0015) or
                     (0.0080 <= new_peak < 0.0150 and giveback >= 0.0025) or
                     (0.0150 <= new_peak < 0.0300 and giveback >= 0.0040) or
                     (new_peak >= 0.0300 and giveback >= 0.0060))):
                    pnl = close_position(ticker)
                    SESSION.recently_traded[ticker] = ny_now()
                    SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                    supabase_log(f"peak_giveback_exit | {ticker} | peak={new_peak:.4%} pnl={pnl_pct:.4%} giveback={giveback:.4%}")
                    SESSION.recently_traded[peak_key] = 0.0
                    return

                # Thresholds
                MIN_LOCK_PROFIT = 0.001   # 0.1%
                REVERSAL_DROP   = 0.001   # 0.1%
                BASE_TP         = 0.03    # 3%
                EXT_TP          = 0.05    # 5%

                # Micro-profit snap exit (fast profit lock), skip for high-confidence trades
                if pnl_pct >= MICRO_PROFIT_TAKE and score < SIGNAL_BUY_THRESHOLD * 1.25:
                    pnl = close_position(ticker)
                    SESSION.recently_traded[ticker] = ny_now()
                    SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                    supabase_log(f"micro_tp_exit | {ticker} | pnl={pnl_pct:.2%}")
                    return

                # 1. Faster defensive loss exit at -0.3%
                if pnl_pct <= -0.003:
                    pnl = close_position(ticker)
                    SESSION.recently_traded[ticker] = ny_now()
                    SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                    supabase_log(f"defensive_exit | {ticker} | pnl={pnl_pct:.2%}")
                    return

                # 2. Exit if losing and score turns bad
                if pnl_pct <= 0 and score <= SIGNAL_SELL_THRESHOLD:
                    pnl = close_position(ticker)
                    SESSION.recently_traded[ticker] = ny_now()
                    SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                    logging.info(f"[EXIT] Defensive loss exit {ticker} | pnl={pnl_pct:.2%}")
                    return

                # 3. Reversal exit protecting >0.1% profits
                if pnl_pct >= MIN_LOCK_PROFIT and (new_peak - pnl_pct) >= REVERSAL_DROP:
                    pnl = close_position(ticker)
                    SESSION.recently_traded[ticker] = ny_now()
                    SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                    logging.info(f"[EXIT] Reversal exit {ticker} | pnl={pnl_pct:.2%}, peak={new_peak:.2%}")
                    return

                # 4. Standard 3% take-profit unless momentum strong
                if BASE_TP <= pnl_pct < EXT_TP:
                    momentum_weak = (score < SIGNAL_BUY_THRESHOLD) or (not pd.isna(rsi) and rsi >= 70)
                    if momentum_weak:
                        pnl = close_position(ticker)
                        SESSION.recently_traded[ticker] = ny_now()
                        SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                        logging.info(f"[EXIT] Base 3% TP exit {ticker} | pnl={pnl_pct:.2%}")
                        return

                # 5. High-profit exit above 5% if momentum weakens
                if pnl_pct >= EXT_TP:
                    momentum_weak = (score < SIGNAL_BUY_THRESHOLD) or (not pd.isna(rsi) and rsi >= 72)
                    if momentum_weak:
                        pnl = close_position(ticker)
                        SESSION.recently_traded[ticker] = ny_now()
                        SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                        logging.info(f"[EXIT] High-profit exit {ticker} | pnl={pnl_pct:.2%}")
                        return

                # 6. Fallback trailing stop
                trail_amount = atr * STOP_LOSS_ATR_MULT
                prev_stop = SESSION.recently_traded.get(f"{ticker}_stop", entry_price - trail_amount)
                new_stop = update_trailing_stop(entry_price, close_price, trail_amount, prev_stop)
                SESSION.recently_traded[f"{ticker}_stop"] = new_stop

                if close_price <= new_stop:
                    pnl = close_position(ticker)
                    SESSION.recently_traded[ticker] = ny_now()
                    SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                    logging.info(f"[EXIT] Trailing stop exit {ticker}")
                    return

                # 7. RSI safety exit
                if not pd.isna(rsi) and rsi >= 75 and pnl_pct > 0:
                    pnl = close_position(ticker)
                    SESSION.recently_traded[ticker] = ny_now()
                    SESSION.trade_history.setdefault(ticker, {})['last_sell'] = ny_now()
                    logging.info(f"[EXIT] RSI overbought safety exit {ticker} | rsi={rsi:.1f}")
                    return

        # If we're still holding after exit checks, do not evaluate entry logic for this ticker
        if have_position:
            return

        # === Smart Ticker Recovery Lock Guard ===
        if not _check_recovery_and_update_state(
            ticker=ticker,
            price=close_price,
            rsi=None if pd.isna(rsi) else rsi,
            atr_pct=None if pd.isna(atr_pct) else atr_pct,
            score=score
        ):
            return
            
        # === Continuation Gate - ALL conditions must be met for entry ===
        if len(df) >= 4:  # Need at least 4 candles for proper analysis
            # 1) Breakout confirmation: current close >= max of previous 3 highs
            prev_3_highs = df['high'].iloc[-4:-1]  # Previous 3 fully closed candles
            max_prev_high = prev_3_highs.max()
            if close_price < max_prev_high:
                logging.debug(f"[CONTINUATION GATE BLOCKED] {ticker} | reason=NO_BREAKOUT | close={close_price:.2f} < max_prev_high={max_prev_high:.2f}")
                return
            
            # 2) Momentum persistence: at least 2 of last 3 candles must be green
            prev_3_candles = df.iloc[-4:-1]  # Previous 3 fully closed candles
            green_count = sum(1 for _, candle in prev_3_candles.iterrows() if candle['close'] > candle['open'])
            if green_count < 2:
                logging.debug(f"[CONTINUATION GATE BLOCKED] {ticker} | reason=WEAK_MOMENTUM | green_count={green_count}/3")
                return
            
            # 3) Range expansion: current range >= 1.2 * avg range of last 5
            if len(df) >= 5:
                current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
                last_5_ranges = df.iloc[-6:-1].apply(lambda row: row['high'] - row['low'], axis=1)
                avg_range_last_5 = last_5_ranges.mean()
                if current_range < (1.2 * avg_range_last_5):
                    logging.debug(f"[CONTINUATION GATE BLOCKED] {ticker} | reason=NO_RANGE_EXPANSION | current={current_range:.3f} < required={1.2 * avg_range_last_5:.3f}")
                    return
            
            # 4) Not overextended: RSI <= 68
            if not pd.isna(rsi) and rsi > 68:
                logging.debug(f"[CONTINUATION GATE BLOCKED] {ticker} | reason=OVEREXTENDED | rsi={rsi:.1f} > 68")
                return
        
        # ================ ENTRY LOGIC (if flat) ================
        # Max concurrent positions safety guard
        if get_open_position_count() >= MAX_CONCURRENT_POSITIONS:
            return
            
        # B) STOP LOSS CLUSTERING: Check for loss cluster and apply pause
        if check_loss_cluster() and not SESSION.buy_pause_until:
            SESSION.buy_pause_until = ny_now() + datetime.timedelta(minutes=20)
            reason_log("loss_cluster_pause", "loss_cluster | pause_applied | 20min", min_seconds=300)
            
        # Defensive mode: Check if BUY entries are paused
        if SESSION.buy_pause_until:
            if datetime.datetime.now(ny_tz) < SESSION.buy_pause_until:
                return
            else:
                SESSION.buy_pause_until = None
                supabase_log("defensive_mode | buy_resumed")
        
        # B) Re-entry after loss rule
        last_loss_time = SESSION.trade_history.get(ticker, {}).get('last_loss')
        if last_loss_time:
            time_since_loss = (ny_now() - last_loss_time).total_seconds() / 60  # minutes
            if time_since_loss < 45:  # 45 minute minimum after loss
                return
        
        # A) ENTRY QUALITY: Entry confirmation gate
        if not check_entry_confirmation(df, atr_pct):
            return
        
        # E) OVERTRADING CONTROL: Anti-chop filter
        if not check_anti_chop_filter(df):
            return
        
        # Phase 1 Safety: Block new entries in last 30 minutes of market
        if is_last_30_minutes_of_market():
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=LAST_30_MINUTES")
            return
        
        # Conservative Entry Logic (Step 2)
        rsi_size_mult = 1.0  # Initialize RSI size multiplier

        # 1. RSI filter (base, with optional first-trade bias)
        rsi_max_allowed = 68
        rsi_size_mult = 1.0
        # First-trade-of-day bias: only relax RSI max for the very first entry after open, within window
        minutes_since_open = None
        if SESSION.market_open_ts:
            try:
                minutes_since_open = (ny_now() - SESSION.market_open_ts).total_seconds() / 60.0
            except Exception:
                minutes_since_open = None

        if FIRST_TRADE_BIAS_ENABLED and (not SESSION.first_trade_done) and minutes_since_open is not None and minutes_since_open <= FIRST_TRADE_BIAS_WINDOW_MIN:
            rsi_max_allowed = FIRST_TRADE_RSI_MAX

        if not pd.isna(rsi) and rsi > rsi_max_allowed:
            if rsi > MAX_RSI_FOR_ENTRY:
                rsi_size_mult = 0.60  # Reduced size for high RSI
                logging.debug(f"[RSI SIZE REDUCTION] {ticker} | rsi={rsi:.1f} max_allowed={rsi_max_allowed} mult={rsi_size_mult}")
            else:
                logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=RSI_TOO_HIGH rsi={rsi:.1f} max={rsi_max_allowed}")
                return

        # 2. ATR volatility filter (avoid weak or explosive regimes)
        if pd.isna(atr_pct) or atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=ATR_RANGE atr_pct={atr_pct:.4f}")
            return

        # 3. MACD confirmation (soft)
        # The strict "must be rising" check often blocks all entries.
        # We only block if MACD is clearly deteriorating AND below zero (bearish momentum).
        prev_macd = float(df['MACD'].iloc[-2]) if len(df) > 1 else macd
        macd_falling_hard = (macd < prev_macd * 0.85) and (macd < 0)
        if macd_falling_hard:
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=MACD_FALLING_HARD macd={macd:.4f} prev={prev_macd:.4f}")
            reason_log(f"macd_fall:{ticker}", f"entry_blocked | {ticker} | reason=macd_falling_hard | macd={macd:.4f} prev={prev_macd:.4f}")
            return

        momentum_confirmed = (macd >= prev_macd)
        prev_close = float(df['close'].iloc[-2]) if len(df) > 1 else close_price
        trend_confirmed = (close_price >= prev_close * 0.998)

        # 4. Bollinger Band filter – avoid chasing only when momentum is NOT confirming
        if (close_price > bb_upper and rsi > 70) and not momentum_confirmed:
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=BB_UPPER_OVERBOUGHT_NO_MOM price={close_price:.2f} bb_upper={bb_upper:.2f} rsi={rsi:.1f}")
            return

        # 5. Sentiment must be neutral or positive
        if sentiment < 0:
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=NEGATIVE_SENTIMENT sentiment={sentiment:.3f}")
            return

        # 6. Score threshold (conditional) + Minimum Viable Trade (MVT) + Market Regime Awareness
        base_threshold = SIGNAL_BUY_THRESHOLD * 1.05  # conservative baseline
        
        # Apply risk state threshold adjustment
        if SESSION.risk_state == "DEGRADED":
            base_threshold = base_threshold * 1.15  # Slightly higher threshold in degraded state
            logging.debug(f"[RISK THRESHOLD] {ticker} | DEGRADED state - increased threshold to {base_threshold:.3f}")
        
        threshold = base_threshold
        
        # Market regime-aware threshold adjustments (inspired by "Mastering the Trade")
        regime_multiplier = 1.0
        if regime == "TREND":
            regime_multiplier = 0.95  # Slightly easier entry in trending markets
        elif regime == "SQUEEZE":
            regime_multiplier = 1.10  # Harder entry during low volatility
        elif regime == "MEAN_REVERSION":
            regime_multiplier = 0.90  # Easier entry but will reduce size below
            
        threshold = base_threshold * regime_multiplier

        # Conditional relaxation when momentum + trend confirm (still safe because we keep ATR sizing + stops)
        if momentum_confirmed and trend_confirmed:
            threshold = max(MVT_MIN_SCORE, SIGNAL_BUY_THRESHOLD * 0.90 * regime_multiplier)

        # First-trade bias: only reduce threshold for the first entry after open (within the first hour)
        if FIRST_TRADE_BIAS_ENABLED and (not SESSION.first_trade_done) and minutes_since_open is not None and minutes_since_open <= FIRST_TRADE_BIAS_WINDOW_MIN:
            threshold = max(MVT_MIN_SCORE, threshold - FIRST_TRADE_SCORE_RELAX)

        # MVT path: allow "good enough" setups even if score is slightly under threshold, ONLY when conditions are true
        mvt_ok = False
        if MVT_ENABLED:
            if (
                minutes_since_open is not None and minutes_since_open >= MVT_MINUTES_AFTER_OPEN
                and (not pd.isna(rsi)) and (MVT_RSI_MIN <= rsi <= MVT_RSI_MAX)
                and (not pd.isna(atr_pct)) and (atr_pct >= MVT_MIN_ATR_PCT)
                and momentum_confirmed
                and trend_confirmed
                and sentiment >= 0
            ):
                mvt_ok = (score >= MVT_MIN_SCORE)

        # --- Entry decision ladder (full vs partial) ---
        entry_threshold = SIGNAL_BUY_THRESHOLD
        try:
            if FIRST_TRADE_BIAS_ENABLED and not SESSION.first_trade_done and SESSION.market_open_ts is not None:
                minutes_since_open = (ny_now() - SESSION.market_open_ts).total_seconds() / 60.0
                if minutes_since_open <= FIRST_TRADE_BIAS_WINDOW_MIN:
                    entry_threshold = max(SIGNAL_BUY_THRESHOLD - FIRST_TRADE_SCORE_RELAX, 0.0)
        except Exception:
            pass

        entry_mode = None  # "full" or "partial"
        if score is not None and score >= entry_threshold:
            entry_mode = "full"
        elif PARTIAL_ENTRY_ENABLED and score is not None and score >= PARTIAL_ENTRY_SCORE_FLOOR:
            entry_mode = "partial"

        # Only proceed if we have a valid entry mode and no position
        if entry_mode is None or have_position:
            if entry_mode is None:
                reason_log(
                    f"score_low:{ticker}",
                    f"entry_blocked | {ticker} | reason=score_too_low | score={score} threshold={entry_threshold} partial_floor={PARTIAL_ENTRY_SCORE_FLOOR}",
                    min_seconds=300,
                )
            return

        # 7. Trend acceleration – confirm short‑term momentum (allow flat)
        if not trend_confirmed:
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=WEAK_TREND close={close_price:.2f} prev={prev_close:.2f}")
            return

        # 8. Safety checks before entering
        if not can_attempt_trade(ticker):
            logging.info(f"Skipping {ticker}: attempt limit reached for current minute.")
            return
        if should_halt_trading():
            return

        # D) MAKE POSITION SIZING CONSISTENT WITH ENTRY MODE + EXISTING MULTIPLIERS
        base_qty = calculate_position_size(ticker, RISK_PER_TRADE)
        
        # Apply risk state scaling first
        if SESSION.risk_state == "DEGRADED":
            base_qty *= 0.5  # 50% reduction in DEGRADED state
        elif SESSION.risk_state == "HALT":
            return  # No new positions in HALT state

        # Apply entry mode sizing
        if entry_mode == "partial":
            base_qty *= PARTIAL_ENTRY_MULTIPLIER
        
        # Apply squeeze sizing (from existing logic)
        base_qty *= squeeze_size_mult

        # Step 1: Scale by score
        qty1 = scale_qty_by_score(base_qty, score)
        
        # Step 2: Apply historical hint multiplier
        qty2, hist_mult, hist_hint = apply_historical_hint_multiplier(ticker, qty1)
        
        # Step 3: Additional degraded state soft scaling
        if SESSION.risk_state == "DEGRADED":
            qty2 *= 0.65  # Additional soft reduction
        
        # Step 4: Apply remaining size modifiers from entry quality assessment
        final_mult = rsi_size_mult * spread_size_mult  # Include spread sizing
        final_mult = max(0.25, min(final_mult, 1.0))  # Clamp between 25% and 100%
        
        # Final quantity
        qty = qty2 * final_mult
        
        # Throttled logging for hint application
        if hist_hint is not None and hist_mult != 1.0:
            reason_log(
                f"hist_hint_{ticker}", 
                f"hist_hint_applied | {ticker} | mult={hist_mult:.2f}", 
                min_seconds=600  # 10 minutes
            )

        # 3) Add explicit qty_zero rejection logging
        if qty is None or qty <= 0:
            reason_log(
                f"qty_zero:{ticker}",
                f"entry_blocked | {ticker} | reason=qty_zero | mode={entry_mode} | score={score} rsi={rsi} atr_pct={atr_pct} spread_pct={spread_pct}",
                min_seconds=180,
            )
            return
        
        # Phase 1 Safety: Buying power guard
        required_dollars = qty * close_price
        if not has_sufficient_buying_power(required_dollars):
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=INSUFFICIENT_BUYING_POWER required=${required_dollars:.2f}")
            supabase_log("entry_blocked | insufficient_buying_power")
            # ui_log(
            #     f"bp_{ticker}",
            #     f"{ticker}: Entry skipped — insufficient buying power.",
            #     min_seconds=120
            # )
            return

        # Entry log with mode transparency
        logging.info(
            f"ENTRY {ticker} [{entry_mode.upper()}] | px={close_price:.2f} qty={qty:.4f} "
            f"score={score:.3f} thr={entry_threshold:.3f} rsi={rsi:.1f} atr%={atr_pct:.3%} "
            f"macd_rising={momentum_confirmed} regime={regime}"
        )
        # Trade intent transparency log
        supabase_log(
            f"trade_signal | BUY {ticker} | mode={entry_mode} score={score:.2f} regime={regime}"
        )
        ui_event(f"Signal found: BUY setup for {ticker}. Attempting entry…")

        if await place_order_async(ticker, qty, 'buy'):
            # G) OBSERVABILITY: Log consolidated cycle summary (throttled)
            reason_log(
                "cycle_summary",
                f"cycle | risk={SESSION.risk_state} | daily_pnl={SESSION.daily_realized_pnl:.2f} | "
                f"positions={get_open_position_count()} | market={SESSION.market_state}",
                min_seconds=300  # 5 minutes
            )
            # G) OBSERVABILITY: Log consolidated cycle summary (throttled)
            reason_log(
                "cycle_summary",
                f"cycle | risk={SESSION.risk_state} | daily_pnl={SESSION.daily_realized_pnl:.2f} | "
                f"positions={get_open_position_count()} | market={SESSION.market_state}",
                min_seconds=300  # 5 minutes
            )
            SESSION.first_trade_done = True
            SESSION.recently_traded[ticker] = ny_now()
            SESSION.recently_traded[f"{ticker}_stop"] = close_price - (atr * STOP_LOSS_ATR_MULT)
            SESSION.trade_history[ticker] = SESSION.trade_history.get(ticker, {})
            SESSION.trade_history[ticker]['last_buy'] = ny_now()
            SESSION.trade_history[ticker]['last_signal_score'] = float(score)
            SESSION.last_trade_time[ticker] = ny_now()

    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")


