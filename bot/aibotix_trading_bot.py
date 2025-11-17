from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
import websockets
import json
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
import sys
from dotenv import load_dotenv
import logging
import asyncio
import signal
import threading
import matplotlib.pyplot as plt
import random
from bot.ai_ticker_selector_aibotix import get_top_tickers
try:
    import caffeine
except Exception:
    class _CaffeineShim:
        def on(self, display: bool = True):
            pass
    caffeine = _CaffeineShim()

# Alpaca data API imports for historical and latest trade data (v3)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame


import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = None
API_SECRET = None

# CONFIGURATION
BASE_URL = 'https://paper-api.alpaca.markets'  # use live URL for real trading
# TICKERS will be dynamically set using AI ticker selector
# TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
TICKERS = []
RSI_PERIOD = 14
MAX_DRAWDOWN = 0.10  # 10%
MAX_TRADES_PER_HOUR = 5
MAX_TICKER_EXPOSURE = 0.30  # 30% of equity
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
def init_trading_client(api_key: str, api_secret: str, paper: bool = True):
    """
    Initialise a TradingClient and data client instance for a specific user.
    Called by the bot start route with user-specific API keys from Supabase,
    or from __main__ when running this module directly.
    """
    global api, data_client, API_KEY, API_SECRET
    API_KEY = api_key
    API_SECRET = api_secret

    api = TradingClient(api_key, api_secret, paper=paper)
    data_client = StockHistoricalDataClient(api_key, api_secret)

    logging.info(f"Trading client initialised (paper={paper}).")
    return api

# Global placeholder. Start/stop routes will set this dynamically.
api: TradingClient | None = None

ny_tz = pytz.timezone('America/New_York')

trade_log = []
consecutive_losses = 0
last_trade_time = {}

# Add global variables to track P&L
realized_pnl = 0.0
unrealized_pnl = 0.0

# Daily risk tracking
start_of_day_equity = None
daily_realized_pnl = 0.0
trading_halted_until = None  # datetime when trading can resume

# Per-ticker loss management
per_ticker_loss_streak = {}       # {ticker: count}
per_ticker_cooloff_until = {}     # {ticker: datetime}

# Add a cooldown period for re-entry into the same stock
REENTRY_COOLDOWN = 300  # 5 minutes
recently_traded = {}

# Track last buy/sell timestamps for each ticker
trade_history = {}  # Tracks last buy/sell timestamps for each ticker

# Alpaca data client for historical and latest trade data
data_client: StockHistoricalDataClient | None = None

# --- Step 3 helpers: daily reset & halting ---

def _ny_now():
    return datetime.datetime.now(ny_tz)


def reset_daily_limits_if_new_day():
    """Reset daily counters if a new trading day has started."""
    global start_of_day_equity, daily_realized_pnl, consecutive_losses, per_ticker_loss_streak, per_ticker_cooloff_until
    now = _ny_now().date()
    # Initialize on first run
    if start_of_day_equity is None:
        start_of_day_equity = get_equity()
        daily_realized_pnl = 0.0
        consecutive_losses = 0
        per_ticker_loss_streak = {}
        per_ticker_cooloff_until = {}
        return
    # If date changed in NY timezone, reset
    if hasattr(reset_daily_limits_if_new_day, "_last_date"):
        last_date = reset_daily_limits_if_new_day._last_date
    else:
        last_date = now
    if now != last_date:
        start_of_day_equity = get_equity()
        daily_realized_pnl = 0.0
        consecutive_losses = 0
        per_ticker_loss_streak = {}
        per_ticker_cooloff_until = {}
    reset_daily_limits_if_new_day._last_date = now


def _calc_daily_loss_limit():
    """Return the dollar loss limit for the current day based on start_of_day_equity and optional hard cap."""
    if start_of_day_equity is None:
        return None
    pct_cap = start_of_day_equity * DAILY_MAX_LOSS_PCT if DAILY_MAX_LOSS_PCT else None
    if DAILY_MAX_LOSS_DOLLARS is None:
        return pct_cap
    return max(DAILY_MAX_LOSS_DOLLARS, pct_cap or 0.0)


def should_halt_trading():
    """True if trading should be halted due to daily loss cap, manual cooldown, or market close proximity."""
    global trading_halted_until
    # Manual/automatic halt window
    now = datetime.datetime.now(ny_tz)
    if trading_halted_until and now < trading_halted_until:
        return True
    # Daily drawdown check
    loss_limit = _calc_daily_loss_limit()
    if loss_limit is not None and daily_realized_pnl <= -abs(loss_limit):
        # Halt until next market open
        try:
            clock = api.get_clock()
            # Halt until next open time if available
            trading_halted_until = clock.next_open if hasattr(clock, "next_open") else now + datetime.timedelta(hours=24)
        except Exception:
            trading_halted_until = now + datetime.timedelta(hours=24)
        logging.warning(f"Daily loss limit reached (PnL: {daily_realized_pnl:.2f}). Trading halted until {trading_halted_until}.")
        return True
    return False


def _register_trade_outcome(ticker: str, pnl: float):
    """Update loss streaks and potential halts based on trade P&L."""
    global consecutive_losses, trading_halted_until
    # Global loss streak
    if pnl < 0:
        consecutive_losses += 1
    else:
        consecutive_losses = 0
    # Per ticker loss streak & cooldown
    if ticker not in per_ticker_loss_streak:
        per_ticker_loss_streak[ticker] = 0
    if pnl < 0:
        per_ticker_loss_streak[ticker] += 1
        if per_ticker_loss_streak[ticker] >= PER_TICKER_MAX_LOSS_STREAK:
            per_ticker_cooloff_until[ticker] = datetime.datetime.now(ny_tz) + datetime.timedelta(minutes=PER_TICKER_COOLDOWN_MIN)
            logging.warning(f"Cooling off {ticker} for {PER_TICKER_COOLDOWN_MIN} minutes due to loss streak.")
    else:
        per_ticker_loss_streak[ticker] = 0
    # Global halt on many consecutive losses
    if MAX_CONSECUTIVE_LOSSES_HALT and consecutive_losses >= MAX_CONSECUTIVE_LOSSES_HALT:
        try:
            clock = api.get_clock()
            trading_halted_until = clock.next_open if hasattr(clock, "next_open") else datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
        except Exception:
            trading_halted_until = datetime.datetime.now(ny_tz) + datetime.timedelta(hours=24)
        logging.warning(f"Exceeded max consecutive losses ({consecutive_losses}). Trading halted until {trading_halted_until}.")

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
    if api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return False
    clock = api.get_clock()
    if not clock.is_open:
        return False

    if ticker:
        try:
            asset = api.get_asset(ticker)
            if not asset.tradable:
                logging.warning(f"{ticker} is not tradable.")
                return False
        except Exception as e:
            logging.error(f"Error checking tradability for {ticker}: {e}")
            return False

    return True

def get_account():
    if api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return None
    return api.get_account()

def get_equity():
    account = get_account()
    if account is None:
        return 0.0
    return float(account.equity)

def get_position(symbol):
    if api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return 0, 0
    try:
        pos = api.get_open_position(symbol)
        return float(pos.qty), float(pos.avg_entry_price)
    except Exception:
        return 0, 0

# --- Trading Logic ---
class RateLimitError(Exception):
    pass

class MarketDataError(Exception):
    pass

def place_order(symbol, qty, side):
    global realized_pnl
    if api is None:
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

        # Build order (use OCO TP/SL when enabled and ATR available)
        if BRACKET_ORDERS_ENABLED and side.lower() == 'buy' and current_price is not None and atr_for_bracket:
            tp_price = round(current_price + (TAKE_PROFIT_ATR_MULT * atr_for_bracket), 2)
            sl_price = round(current_price - (STOP_LOSS_ATR_MULT * atr_for_bracket), 2)
            order = MarketOrderRequest(
                symbol=symbol,
                qty=float(qty),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )
        else:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=float(qty),  # Alpaca supports fractional qty for eligible assets
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )

        api.submit_order(order)
        logging.info(f"Order {side.upper()} {symbol} @ {datetime.datetime.now(ny_tz)} with quantity {qty}")
        last_trade_time[symbol] = datetime.datetime.now(ny_tz)

        if side.lower() == 'sell':
            trade_history[symbol] = trade_history.get(symbol, {})
            trade_history[symbol]['last_sell'] = datetime.datetime.now(ny_tz)
            # realized PnL is handled on close via close_position()
        else:
            trade_history[symbol] = trade_history.get(symbol, {})
            trade_history[symbol]['last_buy'] = datetime.datetime.now(ny_tz)
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
    global realized_pnl, daily_realized_pnl
    if api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return None
    try:
        # Snapshot before closing
        pos_qty, entry_price = get_position(symbol)
        current_price = get_last_trade_price(symbol)
        if pos_qty <= 0 or current_price is None or entry_price <= 0:
            api.close_position(symbol)
            logging.info(f"Closed position on {symbol}")
            last_trade_time[symbol] = datetime.datetime.now(ny_tz)
            return 0.0
        pnl = (current_price - entry_price) * pos_qty
        api.close_position(symbol)
        logging.info(f"Closed position on {symbol} @ {datetime.datetime.now(ny_tz)} | qty={pos_qty:.4f} entry={entry_price:.4f} exit={current_price:.4f} pnl={pnl:.2f}")
        last_trade_time[symbol] = datetime.datetime.now(ny_tz)
        realized_pnl += pnl
        daily_realized_pnl += pnl
        # Mark loss timing for re-entry logic
        trade_history[symbol] = trade_history.get(symbol, {})
        if pnl < 0:
            trade_history[symbol]['last_loss'] = datetime.datetime.now(ny_tz)
        trade_history[symbol]['last_sell'] = datetime.datetime.now(ny_tz)
        # Register outcome for streaks/halts
        _register_trade_outcome(symbol, pnl)
        return pnl
    except Exception as e:
        logging.error(f"Failed to close {symbol}: {e}")
        return None

def update_pnl():
    global realized_pnl, unrealized_pnl
    if api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return
    unrealized_pnl = 0.0
    try:
        positions = api.get_all_positions()
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
                unrealized_pnl += (current_price - entry_price) * pos_qty
        except Exception as e:
            logging.error(f"P&L calc error for {position.symbol}: {e}")
    logging.info(f"Realized P&L: ${realized_pnl:.2f}, Unrealized P&L: ${unrealized_pnl:.2f}")

def calculate_position_size(symbol, risk_per_trade):
    equity = get_equity()
    # Use only equity after reserving safety buffer
    usable_equity = max(equity * (1 - RESERVE_FUND_PCT), 0)
    max_risk = max(usable_equity * risk_per_trade, MIN_DOLLARS_PER_TRADE * 0.01)  # ensure non-zero

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

    # Cap by per-ticker equity exposure
    max_dollars = usable_equity * MAX_EQUITY_FRACTION_PER_TICKER
    if current_price > 0:
        raw_qty = min(raw_qty, max_dollars / current_price)

    # Also make sure we meet min dollars per trade if we do trade
    dollars = raw_qty * current_price
    if dollars < MIN_DOLLARS_PER_TRADE:
        return 0.0

    return round(float(raw_qty), 4)

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
        logging.error(f"{symbol}: Error fetching bars - {e}")
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
    logging.info(f"Fetching sentiment score for {ticker} (placeholder).")
    return 0  # Neutral sentiment by default

# Weighted Signal Scoring System
INDICATOR_WEIGHTS = {
    'RSI': 0.5,
    'ATR': 0.3,
    'Sentiment': 0.2
}
# --- Signal thresholds & trade guards ---
SIGNAL_BUY_THRESHOLD = 0.28   # how strong the combined signal must be to enter
SIGNAL_SELL_THRESHOLD = -0.10 # if score flips negative enough while holding, exit

# Volatility guard (ignore ultra-quiet or too-wild regimes)
MIN_ATR_PCT = 0.003   # 0.3% of price
MAX_ATR_PCT = 0.08    # 8% of price

# Entry quality guardrails
MAX_RSI_FOR_ENTRY = 65        # avoid chasing overbought moves on entry
MIN_BARS_REQUIRED = max(RSI_PERIOD + 5, 50)  # ensure enough data for indicators

# Position sizing caps
MIN_DOLLARS_PER_TRADE = 25.0
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

# Heartbeat monitoring
heartbeat_active = True

def heartbeat_monitor():
    while heartbeat_active:
        logging.info("Heartbeat: Bot is running.")
        time.sleep(300)  # Log every 5 minutes

# Modify signal handler to terminate the bot completely
def stop_bot(signal, frame):
    global heartbeat_active
    heartbeat_active = False
    logging.info("Stopping bot and exiting.")
    sys.exit(0)  # Gracefully exit the script

signal.signal(signal.SIGINT, stop_bot)
signal.signal(signal.SIGTERM, stop_bot)

# Fallback mechanism for API calls
def safe_api_call(api_function, *args, **kwargs):
    retries = 5
    delay = 1  # Start with 1 second delay

    for attempt in range(retries):
        try:
            return api_function(*args, **kwargs)
        except Exception as e:
            logging.error(f"API call failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff

    logging.critical("API call failed after multiple retries.")
    return None

# Market close handling
def close_all_positions():
    if api is None:
        logging.error("Trading client not initialised. Cannot proceed.")
        return
    try:
        positions = api.get_all_positions()
        for position in positions:
            symbol = position.symbol
            logging.info(f"Closing position for {symbol}.")
            safe_api_call(api.close_position, symbol)
    except Exception as e:
        logging.error(f"Failed to close all positions: {e}")

# Prevent system from sleeping
caffeine.on(display=True)

# Asynchronous trade loop
async def trade_loop_async():
    global consecutive_losses, TICKERS
    # Fetch initial tickers using the AI ticker selector with retry logic, only when market is open
    while True:
        if not is_market_open():
            logging.info("Market is currently closed. Waiting for market to open before fetching tickers...")
            await asyncio.sleep(300)
            continue
        # Wait for AI Ticker Selector to return tickers with retries
        max_attempts = 5
        retry_delay = 60  # seconds

        for attempt in range(1, max_attempts + 1):
            selected_tickers = get_top_tickers()
            if selected_tickers:
                logging.info(f"Top tickers received: {selected_tickers}")
                break
            else:
                logging.info(f"Waiting for AI ticker selector... (attempt {attempt}/{max_attempts})")
                await asyncio.sleep(retry_delay)
        else:
            logging.error("AI Ticker Selector failed to return tickers after multiple attempts. Exiting trading loop.")
            continue  # Or use `break` if you want to end the trading loop entirely
        TICKERS = selected_tickers
        if TICKERS:
            break
    logging.info(f"Selected tickers for trading: {TICKERS}")
    last_ticker_refresh = datetime.datetime.now(ny_tz)
    while True:
        try:
            # Step 3: daily resets & halts
            reset_daily_limits_if_new_day()
            if should_halt_trading():
                logging.warning("Trading halted due to risk limits. Sleeping 5 minutes...")
                await asyncio.sleep(300)
                continue

            logging.info("Bot is evaluating trade opportunities...")
            # Refresh tickers every hour
            now = datetime.datetime.now(ny_tz)
            if (now - last_ticker_refresh).total_seconds() > 3600:
                TICKERS = await asyncio.to_thread(get_top_tickers, 5)
                logging.info(f"Selected tickers for trading: {TICKERS}")
                last_ticker_refresh = now

            if not is_market_open():
                logging.info("Market closed. Sleeping 5 mins...")
                await asyncio.sleep(300)
                continue

            # Check if market is about to close
            clock = safe_api_call(api.get_clock)
            if clock and clock.next_close - clock.timestamp < datetime.timedelta(minutes=15):
                logging.info("Market is about to close. Closing all positions.")
                close_all_positions()
                await asyncio.sleep(900)  # Wait until market reopens
                continue

            update_pnl()  # Track P&L

            # Intelligent ticker rotation: close positions no longer in top tickers if profitable
            positions = api.get_all_positions()
            for position in positions:
                symbol = position.symbol
                if symbol not in TICKERS:
                    unrealized = float(position.unrealized_plpc)
                    if unrealized > 0.01:  # At least 1% profit
                        logging.info(f"Closing {symbol} as it is no longer in top tickers and has profit.")
                        close_position(symbol)

            trades_this_hour = 0
            hour_start = datetime.datetime.now(ny_tz)

            tasks = []
            for ticker in TICKERS:
                if not is_market_open(ticker):
                    continue

                now = datetime.datetime.now(ny_tz)
                if ticker in last_trade_time and (now - last_trade_time[ticker]).total_seconds() < TRADE_SPACING_SECONDS:
                    continue  # honor trade spacing

                if ticker in recently_traded and (now - recently_traded[ticker]).total_seconds() < REENTRY_COOLDOWN:
                    logging.info(f"Skipping {ticker} due to re-entry cooldown.")
                    continue

                # Per-ticker cooldown after loss streak
                cooloff_until = per_ticker_cooloff_until.get(ticker)
                if cooloff_until and datetime.datetime.now(ny_tz) < cooloff_until:
                    logging.info(f"{ticker} cooling off until {cooloff_until}.")
                    continue

                tasks.append(process_ticker(ticker))

            await asyncio.gather(*tasks)

            if consecutive_losses >= 3:
                logging.warning(f"Cooldown triggered due to 3 losses. Waiting {LOSS_COOLDOWN} seconds...")
                await asyncio.sleep(LOSS_COOLDOWN)
                consecutive_losses = 0

            if (datetime.datetime.now(ny_tz) - hour_start).total_seconds() < 60:
                await asyncio.sleep(30)

        except RateLimitError as e:
            logging.error(e)
            await asyncio.sleep(60)  # Wait before retrying
        except Exception as e:
            logging.critical(f"Critical error in trade_loop: {e}")
            await asyncio.sleep(300)  # Wait before retrying

# Process a single ticker asynchronously
async def process_ticker(ticker):
    try:
        await asyncio.sleep(random.uniform(0.2, 0.6))
        # Respect global halts
        if should_halt_trading():
            return

        df = await fetch_data_async(ticker)
        if df is None or len(df) < MIN_BARS_REQUIRED:
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
        pos_qty, entry_price = get_position(ticker)
        have_position = pos_qty > 0

        # Cooldowns after recent actions
        now = datetime.datetime.now(ny_tz)
        last_sell_time = trade_history.get(ticker, {}).get("last_sell")
        if not have_position and last_sell_time:
            elapsed = (now - last_sell_time).total_seconds()
            # 1h after profit / 2h after loss handled elsewhere, but keep a basic 5-min safety here too
            if elapsed < REENTRY_COOLDOWN:
                logging.info(f"Skipping {ticker}: re-entry cooldown ({int(REENTRY_COOLDOWN - elapsed)}s left).")
                return

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

        # ================ EXIT LOGIC (if holding) ================
        if have_position:
            if BRACKET_ORDERS_ENABLED:
                # Let server-side OCO (TP/SL) manage exits; skip local exit logic
                return
            # Dynamic trailing stop
            trail_amount = atr * STOP_LOSS_ATR_MULT
            prev_stop = recently_traded.get(f"{ticker}_stop", entry_price - trail_amount)
            new_stop = update_trailing_stop(entry_price, close_price, trail_amount, prev_stop)
            recently_traded[f"{ticker}_stop"] = new_stop

            # Take-profit level (ATR multiple)
            tp = entry_price + (TAKE_PROFIT_ATR_MULT * atr)

            # Hard stop if price pierces trailing stop
            if close_price <= new_stop:
                pnl = close_position(ticker)
                if should_halt_trading():
                    return
                recently_traded[ticker] = datetime.datetime.now(ny_tz)
                trade_history[ticker] = trade_history.get(ticker, {})
                trade_history[ticker]['last_sell'] = datetime.datetime.now(ny_tz)
                logging.info(f"Trailing stop triggered for {ticker} at {new_stop:.2f}")
                return

            # Profit-taking: price hits TP or RSI gets very overbought
            if close_price >= tp or (not pd.isna(rsi) and rsi >= 72):
                pnl = close_position(ticker)
                if should_halt_trading():
                    return
                recently_traded[ticker] = datetime.datetime.now(ny_tz)
                trade_history[ticker] = trade_history.get(ticker, {})
                trade_history[ticker]['last_sell'] = datetime.datetime.now(ny_tz)
                logging.info(f"Take-profit exit for {ticker} at {close_price:.2f}")
                return

            # Defensive exit if signal flips clearly negative
            if score <= SIGNAL_SELL_THRESHOLD:
                pnl = close_position(ticker)
                if should_halt_trading():
                    return
                recently_traded[ticker] = datetime.datetime.now(ny_tz)
                trade_history[ticker] = trade_history.get(ticker, {})
                trade_history[ticker]['last_sell'] = datetime.datetime.now(ny_tz)
                logging.info(f"Signal-based exit for {ticker} at {close_price:.2f} (score {score:.3f})")
            return

        # ================ ENTRY LOGIC (if flat) ================
        # Quality guard: avoid chasing high RSI on entry
        if not pd.isna(rsi) and rsi > MAX_RSI_FOR_ENTRY:
            return

        # Require sufficient positive score to enter
        if score >= SIGNAL_BUY_THRESHOLD:
            if not can_attempt_trade(ticker):
                logging.info(f"Skipping {ticker}: attempt limit reached for current minute.")
                return
            if should_halt_trading():
                return
            qty = calculate_position_size(ticker, RISK_PER_TRADE)
            if qty <= 0:
                return
            logging.info(f"ENTRY {ticker}: px={close_price:.2f}, qty={qty}, score={score:.3f}, rsi={rsi:.1f}, atr%={atr_pct:.3%}")
            if await place_order_async(ticker, qty, 'buy'):
                recently_traded[ticker] = datetime.datetime.now(ny_tz)
                # initialize trailing stop
                recently_traded[f"{ticker}_stop"] = close_price - (atr * STOP_LOSS_ATR_MULT)
                trade_history[ticker] = trade_history.get(ticker, {})
                trade_history[ticker]['last_buy'] = datetime.datetime.now(ny_tz)
                trade_history[ticker]['last_signal_score'] = float(score)
                last_trade_time[ticker] = datetime.datetime.now(ny_tz)

    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
# Placeholder for real-time market data stream
def start_websocket_stream():
    logging.info("WebSocket streaming would be initialized here for live data.")

# Backtesting function
def backtest(ticker, historical_data):
    logging.info(f"Starting backtest for {ticker}")

    # Initialize variables
    equity = 10000  # Starting equity in dollars
    position = 0  # Current position size
    entry_price = 0  # Entry price for the position
    trade_log = []

    # Apply strategy to historical data
    historical_data = strategy.calculate_indicators(historical_data)

    for i in range(len(historical_data)):
        row = historical_data.iloc[i]
        rsi = row['RSI']
        atr = row['ATR']
        close_price = row['close']

        # Check buy condition
        if position == 0 and rsi < 30:
            position = equity // close_price  # Buy as many shares as possible
            entry_price = close_price
            equity -= position * close_price
            trade_log.append((row.name, "BUY", close_price, position))

        # Check sell condition
        elif position > 0:
            tp = entry_price + (TAKE_PROFIT_ATR_MULT * atr)
            sl = entry_price - (STOP_LOSS_ATR_MULT * atr)

            if close_price >= tp or rsi > 70 or close_price <= sl:
                equity += position * close_price
                trade_log.append((row.name, "SELL", close_price, position))
                position = 0

    # Final equity calculation
    if position > 0:
        equity += position * historical_data.iloc[-1]['close']

    # Log results
    logging.info(f"Backtest complete for {ticker}. Final equity: ${equity:.2f}")

    # Plot results
    plot_backtest_results(historical_data, trade_log, ticker)

    return equity

# Plot backtest results
def plot_backtest_results(historical_data, trade_log, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['close'], label='Close Price', color='blue')

    for trade in trade_log:
        date, action, price, _ = trade
        color = 'green' if action == 'BUY' else 'red'
        plt.scatter(date, price, color=color, label=action, alpha=0.8)

    plt.title(f"Backtest Results for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Entry point for the asynchronous bot
if __name__ == "__main__":
    # Standalone execution: initialise trading and data clients from environment variables.
    env_key = os.getenv("ALPACA_API_KEY")
    env_secret = os.getenv("ALPACA_API_SECRET")

    if not env_key or not env_secret:
        logging.error("Environment variables ALPACA_API_KEY / ALPACA_API_SECRET are not set. "
                      "Cannot run standalone trading bot.")
        sys.exit(1)

    init_trading_client(env_key, env_secret, paper=True)

    # Start heartbeat monitor in a separate thread
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()

    start_websocket_stream()
    asyncio.run(trade_loop_async())

    # Load historical data (replace with actual data loading logic)
    try:
        if data_client is None:
            raise RuntimeError("Data client is not initialised for backtest.")
        req_bt = StockBarsRequest(
            symbol="AAPL",
            timeframe=TimeFrame.Day,
            start=None,
            end=None,
            limit=100
        )
        historical_data = data_client.get_stock_bars(req_bt).df
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        historical_data = None
        # Skip backtest if error
        sys.exit(0)

    if historical_data is None or historical_data.empty:
        logging.warning("Historical data is empty. Backtest aborted.")
        sys.exit(0)

    if hasattr(historical_data, 'index') and isinstance(historical_data.index, pd.MultiIndex):
        historical_data = historical_data.xs("AAPL")
    historical_data = historical_data.sort_index()

    # Run backtest
    final_equity = backtest("AAPL", historical_data)
    print(f"Final equity after backtest: ${final_equity:.2f}")


