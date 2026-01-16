# --- BotSession container for state (Patch B foundation) ---
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
from bot.ai_ticker_selector_aibotix import get_top_tickers

# Alpaca data API imports for historical and latest trade data (v3)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
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
TICKERS = []
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
MAX_TRADE_NOTIONAL_PCT = 0.05      # 5% of usable equity per entry
MAX_TRADE_NOTIONAL_ABS = 2500.0    # hard cap in account currency (USD). Set None to disable.
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
DAILY_MAX_LOSS_PCT = 0.03        # stop trading for the day if equity drawdown exceeds 3%
DAILY_MAX_LOSS_DOLLARS = None    # optional hard dollar cap; set to a number (e.g., 200) to enforce alongside %
MAX_CONSECUTIVE_LOSSES_HALT = 5  # if this many losing exits occur in a day, halt trading until next session
PER_TICKER_MAX_LOSS_STREAK = 3   # per-ticker loss streak before cooling off that ticker
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


def should_halt_trading():
    """True if trading should be halted due to daily loss cap, manual cooldown, or market close proximity."""
    now = datetime.datetime.now(ny_tz)
    if SESSION.trading_halted_until and now < SESSION.trading_halted_until:
        return True
    # Daily drawdown check
    loss_limit = _calc_daily_loss_limit()
    if loss_limit is not None and SESSION.daily_realized_pnl <= -abs(loss_limit):
        # Halt until next market open
        if SESSION.api is None:
            logging.error("Trading client not initialised. Cannot check clock for halt logic.")
            SESSION.trading_halted_until = now + datetime.timedelta(hours=24)
            return True
        try:
            clock = SESSION.api.get_clock()
            # Halt until next open time if available
            SESSION.trading_halted_until = clock.next_open if hasattr(clock, "next_open") else now + datetime.timedelta(hours=24)
        except Exception:
            SESSION.trading_halted_until = now + datetime.timedelta(hours=24)
        logging.warning(f"Daily loss limit reached (PnL: {SESSION.daily_realized_pnl:.2f}). Trading halted until {SESSION.trading_halted_until}.")
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


def _register_trade_outcome(ticker: str, pnl: float):
    """Update loss streaks and potential halts based on trade P&L."""
    # Global loss streak
    if pnl < 0:
        SESSION.consecutive_losses += 1
    else:
        SESSION.consecutive_losses = 0
    # Per ticker loss streak & cooldown
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
        if BRACKET_ORDERS_ENABLED and side.lower() == 'buy' and current_price is not None and atr_for_bracket:
            tp_price = round(current_price + (TAKE_PROFIT_ATR_MULT * atr_for_bracket), 2)
            sl_price = round(current_price - (STOP_LOSS_ATR_MULT * atr_for_bracket), 2)
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
        supabase_log(
            f"position_closed | {symbol} | pnl={pnl:.2f} @ {datetime.datetime.now(ny_tz).isoformat()}"
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
        return pnl
    except Exception as e:
        logging.error(f"Failed to close {symbol}: {e}")
        return None
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
    per_trade_capital = available_capital / MAX_CONCURRENT_POSITIONS
    
    max_risk = max(usable_equity * risk_per_trade, 0.01)  # ensure a tiny non-zero risk budget

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
    if score < 0.30:
        return qty * 0.25
    elif score < 0.45:
        return qty * 0.50
    elif score < 0.65:
        return qty * 0.75
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
        df['RSI'] = df['RSI'].interpolate().bfill()
        df['ATR'] = df['ATR'].interpolate().bfill()
        if 'ATR_PCT' in df.columns:
            df['ATR_PCT'] = df['ATR_PCT'].interpolate().bfill()

        needed_cols = [c for c in ['RSI', 'ATR', 'ATR_PCT'] if c in df.columns]
        if df[needed_cols].dropna().shape[0] < 3:
            logging.warning(f"{symbol}: Not enough valid RSI/ATR data even after interpolation. Skipping.")
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
SIGNAL_BUY_THRESHOLD = 0.20   # how strong the combined signal must be to enter
SIGNAL_SELL_THRESHOLD = -0.10 # if score flips negative enough while holding, exit
MICRO_PROFIT_TAKE = 0.001   # 0.10% fast profit snap exit

# === Tiered Execution Thresholds ===
TIER1_THRESHOLD_MULT = 1.00   # full confidence
TIER2_THRESHOLD_MULT = 0.85   # good but not perfect
TIER2_SIZE_MULT = 0.50        # half-size entries

# Volatility guard (ignore ultra-quiet or too-wild regimes)
# NOTE: slightly widened to allow more tradable setups while still avoiding dead/chaotic regimes
MIN_ATR_PCT = 0.002   # 0.2% of price
MAX_ATR_PCT = 0.09    # 9% of price

# Entry quality guardrails
MAX_RSI_FOR_ENTRY = 65        # avoid chasing overbought moves on entry

# --- Execution-level data hygiene ---
EXEC_BAD_TICKER_COOLDOWN_MIN = 45   # minutes to ignore tickers with bad/insufficient data

# --- Active ticker rotation constants ---
AI_REFRESH_SECONDS = 900   # 15 minutes (not 30)
MIN_ACTIVE_TICKERS = 20    # ensure we keep a broad universe

# === Trade activity tuning (safe) ===
MVT_ENABLED = True
MVT_MIN_SCORE = 0.18                  # allow "good enough" setups when momentum/trend confirm
MVT_RSI_MIN = 42
MVT_RSI_MAX = 62
MVT_MIN_ATR_PCT = 0.0020              # 0.20% ATR% minimum for MVT
MVT_MINUTES_AFTER_OPEN = 15           # don't MVT in the very first minutes
FIRST_TRADE_BIAS_ENABLED = True
FIRST_TRADE_BIAS_WINDOW_MIN = 60      # only applies for first entry within first hour
FIRST_TRADE_SCORE_RELAX = 0.02        # reduce threshold by 0.02 for first trade only
FIRST_TRADE_RSI_MAX = 70              # allow a slightly higher RSI for first trade only

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
    while True:
        try:
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
            
            # 1. Daily resets and halts
            reset_daily_limits_if_new_day()
            if should_halt_trading():
                logging.warning("Trading halted due to risk limits. Sleeping 5 minutes...")
                await asyncio.sleep(300)
                continue

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
                            15,  # Get 15 tickers from AI
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
                batch_size = min(8, len(all_candidate_tickers))
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
                                if (now - last_eval).total_seconds() < 60:
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

        # Quick volatility guard
        if pd.isna(atr_pct) or atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
            return

        # Position info
        pos_qty, entry_price = await get_position_async(ticker)
        have_position = pos_qty > 0

        # Cooldowns after recent actions
        now = ny_now()
        last_sell_time = SESSION.trade_history.get(ticker, {}).get("last_sell")
        if not have_position and last_sell_time:
            elapsed = (now - last_sell_time).total_seconds()
            # 1h after profit / 2h after loss handled elsewhere, but keep a basic 5-min safety here too
            if elapsed < REENTRY_COOLDOWN:
                logging.info(f"Skipping {ticker}: re-entry cooldown ({int(REENTRY_COOLDOWN - elapsed)}s left).")
                return

        # Sentiment (placeholder currently returns 0)
        sentiment = await get_sentiment_score(ticker)

        # Partial entry constants
        PARTIAL_ENTRY_ENABLED = True
        PARTIAL_ENTRY_SCORE_FLOOR = 0.14      # minimum score to allow partial entry
        PARTIAL_ENTRY_MULTIPLIER = 0.25       # 25% of normal size

        # Detect market regime for adaptive strategy
        regime = detect_market_regime(df)
        logging.debug(f"[REGIME] {ticker} | regime={regime}")
        
        # Phase 1 Safety: SQUEEZE handling (soft)
        # Instead of a hard ban (which can cause zero trades for long periods), we:
        # - Only block ultra-low volatility squeezes
        # - Otherwise allow entries but reduce position size later
        squeeze_soft_block = False
        if regime == "SQUEEZE" and not have_position:
            if pd.isna(atr_pct) or atr_pct < 0.0025:
                squeeze_soft_block = True
                logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=SQUEEZE_TOO_QUIET atr%={atr_pct:.4f}")
                reason_log(f"squeeze_quiet:{ticker}", f"entry_blocked | {ticker} | reason=squeeze_too_quiet | atr_pct={atr_pct:.4f}")
                # ui_log(f"entry_block_{ticker}", f"{ticker}: Entry blocked — volatility too quiet for safe trading.")
                return

        # Compute signal score (uses RSI/ATR/sentiment + small MACD/BB blend)
        score = calculate_signal_score(
            rsi=rsi, atr=atr_pct, sentiment=sentiment,
            macd=macd, close_price=close_price,
            bb_upper=bb_upper, bb_lower=bb_lower
        )
        if score is None:
            return

        # ================ SMART EXIT LOGIC (if holding) ================
        # ---- Simplified Unified SMART EXIT LOGIC ----
        if have_position:
            # Calculate current PnL
            pnl_pct = 0.0
            if entry_price > 0:
                pnl_pct = (close_price - entry_price) / entry_price

            # Track peak for reversal detection
            peak_key = f"{ticker}_peak_pnl"
            prev_peak = SESSION.recently_traded.get(peak_key, 0.0)
            new_peak = max(prev_peak, pnl_pct)
            SESSION.recently_traded[peak_key] = new_peak

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
        # ================ ENTRY LOGIC (if flat) ================
        # Max concurrent positions safety guard
        if get_open_position_count() >= MAX_CONCURRENT_POSITIONS:
            return
            
        # Defensive mode: Check if BUY entries are paused
        if SESSION.buy_pause_until:
            if datetime.datetime.now(ny_tz) < SESSION.buy_pause_until:
                logging.info(f"[DEFENSIVE MODE] Buy paused until {SESSION.buy_pause_until}")
                return
            else:
                SESSION.buy_pause_until = None
                supabase_log("defensive_mode | buy_resumed")
        
        # Phase 1 Safety: Block new entries in last 30 minutes of market
        if is_last_30_minutes_of_market():
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=LAST_30_MINUTES")
            return
        
        # Conservative Entry Logic (Step 2)

        # 1. RSI filter (base, with optional first-trade bias)
        rsi_max_allowed = 68
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
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=RSI_TOO_HIGH rsi={rsi:.1f} max={rsi_max_allowed}")
            return

        # 2. ATR volatility filter (avoid weak or explosive regimes)
        if pd.isna(atr_pct) or atr_pct < 0.0025 or atr_pct > 0.07:
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

        # --- Tiered Entry Decision ---
        tier = None

        if score >= threshold * TIER1_THRESHOLD_MULT:
            tier = "TIER1"
        elif score >= threshold * TIER2_THRESHOLD_MULT:
            tier = "TIER2"
        elif mvt_ok:
            tier = "MVT"
        elif (
            PARTIAL_ENTRY_ENABLED
            and score >= PARTIAL_ENTRY_SCORE_FLOOR
            and score < threshold * TIER2_THRESHOLD_MULT
        ):
            tier = "PARTIAL"
        else:
            if score >= threshold * 0.85:
                reason_log(
                    f"near_miss:{ticker}",
                    f"entry_blocked | {ticker} | reason=near_miss | score={score:.3f} threshold={threshold:.3f}"
                )
            logging.debug(f"[ENTRY BLOCKED] {ticker} | reason=LOW_SCORE score={score:.3f} threshold={threshold:.3f} base={base_threshold:.3f} mvt_ok={mvt_ok}")
            return

        if tier == "MVT" and score < threshold:
            logging.info(f"[ENTRY ALLOWED:MVT] {ticker} | score={score:.3f} (min={MVT_MIN_SCORE:.3f}) rsi={rsi:.1f} atr%={atr_pct:.3%} macd_rising={momentum_confirmed} mins_since_open={minutes_since_open:.0f}")

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

        qty = calculate_position_size(ticker, RISK_PER_TRADE)

        # Tier-based size adjustment
        if tier == "TIER2":
            qty = qty * TIER2_SIZE_MULT
        elif tier == "MVT":
            qty = qty * 0.40
        elif tier == "PARTIAL":
            qty = qty * PARTIAL_ENTRY_MULTIPLIER
        
        # Scale quantity based on signal score
        qty = scale_qty_by_score(qty, score)

        if qty <= 0:
            return

        # Market regime-aware position sizing adjustment
        if regime == "MEAN_REVERSION":
            # Reduce position size by 40% for mean reversion trades (higher risk of false signals)
            qty = qty * 0.6
            logging.debug(f"[REGIME SIZING] {ticker} | MEAN_REVERSION regime - reduced position size by 40%")

        if regime == "SQUEEZE":
            # SQUEEZE entries are allowed (unless too quiet), but size is reduced to limit risk.
            qty = qty * 0.5
            logging.debug(f"[REGIME SIZING] {ticker} | SQUEEZE regime - reduced position size by 50%")
        
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

        # Entry log with tier transparency
        if tier == "PARTIAL":
            logging.info(
                f"ENTRY {ticker} [PARTIAL] | score={score:.3f} threshold={threshold:.3f} qty={qty:.4f}"
            )
        
        logging.info(
            f"ENTRY {ticker} [{tier}] | px={close_price:.2f} qty={qty:.4f} "
            f"score={score:.3f} thr={threshold:.3f} rsi={rsi:.1f} atr%={atr_pct:.3%} "
            f"macd_rising={momentum_confirmed} regime={regime}"
        )
        # Trade intent transparency log
        supabase_log(
            f"trade_signal | BUY {ticker} | tier={tier} score={score:.2f} regime={regime}"
        )
        ui_event(f"Signal found: BUY setup for {ticker}. Attempting entry…")

        if await place_order_async(ticker, qty, 'buy'):
            SESSION.first_trade_done = True
            SESSION.recently_traded[ticker] = ny_now()
            SESSION.recently_traded[f"{ticker}_stop"] = close_price - (atr * STOP_LOSS_ATR_MULT)
            SESSION.trade_history[ticker] = SESSION.trade_history.get(ticker, {})
            SESSION.trade_history[ticker]['last_buy'] = ny_now()
            SESSION.trade_history[ticker]['last_signal_score'] = float(score)
            SESSION.last_trade_time[ticker] = ny_now()

    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")


