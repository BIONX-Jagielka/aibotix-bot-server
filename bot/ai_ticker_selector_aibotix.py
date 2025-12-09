import asyncio
import logging
import os
import time
import random
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- Supabase Client for AI Ticker Persistence ---
from supabase import create_client, Client
from typing import List, Dict, Any, Optional, Tuple

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

if not supabase:
    raise RuntimeError(
        "[AI-SELECTOR ERROR] Supabase client failed to initialise. "
        "Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in environment variables."
    )

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

# --- Load Environment Variables ---
load_dotenv()

# --- Dynamic Client Initialization ---
def init_clients(mode: Optional[str] = None) -> Tuple[StockHistoricalDataClient, TradingClient]:
    """
    Initialise Alpaca clients using *AI selector–specific API keys*.
    Keys depend on the mode ("paper" or "live").
    This prevents AI selector from using user trading keys and keeps
    all market-scanning activity centralised and scalable.
    """

    effective_mode = (mode or os.getenv("BOT_MODE", "paper")).lower()
    paper = effective_mode == "paper"

    if paper:
        api_key = os.getenv("AI_SELECTOR_PAPER_API_KEY")
        api_secret = os.getenv("AI_SELECTOR_PAPER_API_SECRET")
    else:
        api_key = os.getenv("AI_SELECTOR_LIVE_API_KEY")
        api_secret = os.getenv("AI_SELECTOR_LIVE_API_SECRET")

    # Strict validation so Render errors are visible immediately
    if not api_key or not api_secret:
        raise RuntimeError(
            f"[AI-SELECTOR ERROR] Missing API keys for mode={effective_mode}. "
            "Ensure the following env variables exist:\n"
            "  - AI_SELECTOR_PAPER_API_KEY\n"
            "  - AI_SELECTOR_PAPER_API_SECRET\n"
            "  - AI_SELECTOR_LIVE_API_KEY\n"
            "  - AI_SELECTOR_LIVE_API_SECRET"
        )

    # Now safely initialise clients
    data_client = StockHistoricalDataClient(api_key, api_secret)
    trading_client = TradingClient(api_key, api_secret, paper=paper)

    return data_client, trading_client

# --- Logger Setup ---
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

RSI_PERIOD = 14

# Limits to control how many symbols we screen each run
MAX_TRADABLE_SCREEN = 400  # how many symbols to volume-screen
MAX_INDICATOR_TASKS = 60   # how many symbols to fetch intraday indicators for

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

async def fetch_indicators(symbol: str, mode: str):
    try:
        data_client, trading_client = init_clients(mode)
        bars_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=100)
        bars = data_client.get_stock_bars(bars_req).df
        if bars.empty:
            logging.warning(f"No data returned for {symbol}")
            return None

        # Require a minimum number of bars so ATR / EMA / volume windows have data
        if len(bars) < 30:
            logging.warning(f"Not enough intraday bars for {symbol}, skipping.")
            return None

        df = bars.copy()
        df['rsi'] = compute_rsi(df['close'], RSI_PERIOD)
        df['atr'] = compute_atr(df)
        df['ema_fast'] = df['close'].ewm(span=9).mean()
        df['ema_slow'] = df['close'].ewm(span=21).mean()
        df['ema_crossover'] = df['ema_fast'] > df['ema_slow']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['gap'] = df['close'] / df['close'].shift(1) - 1
        df['slope'] = df['close'].rolling(window=5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 5)

        return df
    except Exception as e:
        logging.warning(f"Fetch failed for {symbol}: {e}")
        # Do not raise here – return None so other symbols can still be processed
        return None

async def stage_a_screen_and_collect(mode: str, limit: int = 5):
    data_client, trading_client = init_clients(mode)
    failed_symbols = []  # Initialize the failed_symbols list
    try:
        assets = trading_client.get_all_assets()
        assets = [a for a in assets if a.status == 'active']
        logging.info(f"[DEBUG] Retrieved {len(assets)} active assets from Alpaca.")
    except Exception as e:
        logging.error(f"[ERROR] Failed to retrieve assets from Alpaca: {e}")
        assets = []

    tradable = [a.symbol for a in assets if a.tradable and a.symbol.isalpha() and a.exchange in ["NASDAQ", "NYSE"]]
    logging.info(f"[DEBUG] Total tradable symbols retrieved: {len(tradable)}")
    logging.info(f"[DEBUG] First 10 tradable symbols: {tradable[:10]}")
    volume_filtered = []
    for symbol in tradable[:MAX_TRADABLE_SCREEN]:
        try:
            logging.info(f"Checking symbol: {symbol}")
            d_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=1)
            bars = data_client.get_stock_bars(d_req).df

            # If no data, mark as failed and skip *before* trying to access bars.iloc[-1]
            if bars.empty:
                failed_symbols.append(symbol)
                if len(failed_symbols) <= 10:
                    logging.warning(f"[DEBUG] {symbol} failed - empty bars.")
                logging.warning(f"{symbol} returned empty bars.")
                continue

            # Now it's safe to inspect the last bar for debug logging
            if len(volume_filtered) < 5:
                logging.info(
                    f"[DEBUG] {symbol} - Volume: {bars.iloc[-1]['volume']}, "
                    f"Close: {bars.iloc[-1]['close']}, "
                    f"Dollar Volume: {bars.iloc[-1]['volume'] * bars.iloc[-1]['close']}"
                )

            vol = bars.iloc[-1]['volume']
            close = bars.iloc[-1]['close']
            dollar_volume = vol * close
            # Slightly relaxed filters so we don't end up with zero candidates in quiet sessions
            if vol > 50 and dollar_volume > 500 and close > 0.01:
                volume_filtered.append(symbol)
        except Exception as e:
            logging.warning(f"Error processing {symbol}: {e}")
            continue

    logging.info(f"[DEBUG] First 10 failed symbols: {failed_symbols[:10]}")
    logging.info(f"{len(volume_filtered)} tickers passed relaxed volume filter.")
    random.shuffle(volume_filtered)

    indicator_results = []

    # Limit how many symbols we fetch intraday indicators for
    symbols_for_indicators = volume_filtered[:MAX_INDICATOR_TASKS]

    if not symbols_for_indicators:
        logging.warning("No symbols passed the volume filter; returning empty indicator results.")
        return []

    tasks = [fetch_indicators(sym, mode) for sym in symbols_for_indicators]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for symbol, df in zip(symbols_for_indicators, results):
        # Skip any exceptions returned by asyncio.gather
        if isinstance(df, Exception):
            logging.warning(f"Indicator task for {symbol} raised an exception: {df}")
            continue

        if df is not None and len(df) > 10:
            latest = df.iloc[-1]
            logging.info(f"{symbol} - RSI: {latest['rsi']:.2f}, ATR: {latest['atr']:.4f}, EMA crossover: {latest['ema_crossover']}")
            indicator_results.append({
                "symbol": symbol,
                "rsi": latest['rsi'],
                "atr": latest['atr'],
                "atr_pct": latest['atr'] * 100,  # Convert to percentage
                "macd": latest.get('macd', 0),
                "sentiment": latest.get('sentiment', 0),
                "volume": latest.get('volume', 0),
                "ema_crossover": latest['ema_crossover'],
                "volume_ratio": latest['volume_ratio'],
                "slope": latest['slope'],
                "gap": latest['gap'],
                "score": unified_ai_score(
                    rsi=latest['rsi'],
                    atr=latest['atr'],
                    ema_crossover=latest['ema_crossover'],
                    volume_ratio=latest['volume_ratio'],
                    slope=latest['slope'],
                    gap=latest['gap'],
                ) or 0.0
            })

    logging.info(f"{len(indicator_results)} tickers gathered with indicators after screening.")
    return indicator_results


def get_open_position_symbols(mode: str):
    """
    Returns a set of symbols the account is already holding.
    Prevents the AI selector from opening duplicate positions.
    """
    try:
        data_client, trading_client = init_clients(mode)
        positions = trading_client.get_all_positions()
        return {p.symbol for p in positions}
    except Exception as e:
        logging.warning(f"Failed to retrieve open positions: {e}")
        return set()

def unified_ai_score(
    rsi: float,
    atr: float,
    ema_crossover: bool,
    volume_ratio: float,
    slope: float,
    gap: float,
) -> Optional[float]:
    """
    Unified multi-factor score used both for live AI screening and historical CSV-based ranking.
    This is conceptually aligned with the trading bot's signal logic: prefer healthy RSI,
    reasonable volatility, positive trend, good volume, and constructive gaps.
    """
    # Require RSI
    if pd.isna(rsi):
        return None

    # Safe defaults for other indicators
    if pd.isna(atr) or atr <= 0:
        atr = 1.0
    if pd.isna(volume_ratio) or volume_ratio <= 0:
        volume_ratio = 1.0
    if pd.isna(slope):
        slope = 0.0
    if pd.isna(gap):
        gap = 0.0

    score = 0.0

    # 1) RSI: prefer 40–60, softly accept 30–70, penalise extremes
    if 30 <= rsi <= 75:
        # bell-shaped preference around 50
        score += max(0.0, 1.5 - abs(rsi - 50) / 15.0)
    else:
        score -= 0.5

    # 2) ATR: prefer non-crazy volatility (smaller ATR is safer, but not zero)
    atr_clamped = min(max(atr, 0.01), 5.0)
    score += 0.5 / atr_clamped

    # 3) EMA crossover: bullish alignment
    if ema_crossover:
        score += 1.0

    # 4) Volume ratio: prefer above-average volume, cap contribution
    if volume_ratio > 1.0:
        score += min(1.5, (volume_ratio - 1.0) / 2.0)

    # 5) Slope: upward 5-bar trend is good, cap so a single spike doesn't dominate
    if slope > 0:
        score += min(1.5, slope * 100.0)

    # 6) Gap: very large gaps are risky; small gaps can be constructive
    if abs(gap) > 0.10:
        score -= 1.0
    elif abs(gap) > 0.02:
        score += 0.2

    return score

def score_tickers(indicator_results, mode: str, top_n: int = 5):
    """
    Score tickers using the unified multi-factor model so that
    live AI screening is aligned with the trading bot's signal logic.
    """
    scored = []
    for result in indicator_results:
        symbol = result['symbol']
        rsi = result['rsi']
        atr = result['atr']
        volume_ratio = result['volume_ratio']
        slope = result['slope']
        gap = result['gap']
        crossover = result['ema_crossover']

        score_val = unified_ai_score(
            rsi=rsi,
            atr=atr,
            ema_crossover=crossover,
            volume_ratio=volume_ratio,
            slope=slope,
            gap=gap,
        )
        if score_val is None:
            continue

        scored.append((symbol, score_val))

    # Apply soft feedback adjustments before sorting
    scored = adjust_scoring_with_feedback(scored)
    # Sort by unified AI score (highest first)
    sorted_scored = sorted(scored, key=lambda x: x[1], reverse=True)

    # Filter out duplicates and open positions
    open_symbols = get_open_position_symbols(mode)
    filtered = []
    seen = set()

    for symbol, score in sorted_scored:
        if symbol in seen:
            continue
        if symbol in open_symbols:
            logging.info(f"Skipping {symbol} (already an open position).")
            continue

        seen.add(symbol)
        filtered.append((symbol, score))

        if len(filtered) >= top_n:
            break

    top_symbols = [s for s, _ in filtered]
    logging.info(f"Top {len(top_symbols)} symbols by score (after filtering): {top_symbols}")
    return top_symbols

# NOTE:
# save_ai_tickers(user_id, mode, top_symbols, {}) 
# (Call this from worker or main process where user_id/mode context exists)

def log_selected_tickers_for_learning(indicator_results):
    try:
        log_file = "ai_ticker_learning_log.csv"
        rows = []
        for entry in indicator_results:
            # entry is a dict with indicator fields
            symbol = entry.get("symbol")
            if not symbol:
                continue

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "rsi": round(float(entry.get("rsi", 0.0)), 2),
                "atr": round(float(entry.get("atr", 0.0)), 4),
                "ema_crossover": bool(entry.get("ema_crossover", False)),
                "volume_ratio": round(float(entry.get("volume_ratio", 0.0)), 2),
                "slope": round(float(entry.get("slope", 0.0)), 4),
                "gap": round(float(entry.get("gap", 0.0)), 4),
            }
            rows.append(row)

        if not rows:
            logging.info("No indicator rows to log for learning.")
            return

        df = pd.DataFrame(rows)
        if os.path.exists(log_file):
            df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            df.to_csv(log_file, index=False)

        logging.info(f"{len(rows)} tickers logged to {log_file}")
    except Exception as e:
        logging.warning(f"Failed to log ticker data: {e}")

def adjust_scoring_with_feedback(scored_tickers, feedback_file="ai_ticker_feedback.csv"):
    if not os.path.exists(feedback_file):
        return scored_tickers  # No feedback data yet

    try:
        feedback_df = pd.read_csv(feedback_file)
        feedback_scores = feedback_df.groupby("symbol")["profit"].mean().to_dict()
        adjusted = []
        for symbol, score in scored_tickers:
            bonus = 0
            if symbol in feedback_scores:
                avg_profit = feedback_scores[symbol]
                if avg_profit > 0:
                    bonus += 1
                elif avg_profit < 0:
                    bonus -= 1
            adjusted.append((symbol, score + bonus))
        return adjusted
    except Exception as e:
        logging.warning(f"Failed to apply feedback: {e}")
        return scored_tickers

def record_trade_feedback(symbol, profit, filename="ai_ticker_feedback.csv"):
    try:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "profit": profit
        }

        df = pd.DataFrame([row])
        if os.path.exists(filename):
            df.to_csv(filename, mode="a", header=False, index=False)
        else:
            df.to_csv(filename, index=False)

        logging.info(f"Feedback for {symbol} recorded.")
    except Exception as e:
        logging.warning(f"Failed to record feedback: {e}")

def save_ai_tickers(user_id: str, mode: str, tickers):
    """
    Save AI-selected tickers to the ai_tickers table in a row-based format.

    tickers can be either:
      - a list of plain symbols: ["AAPL", "TSLA", ...]
      - or a list of dicts like:
        {
          "ticker": "AAPL",
          "rank": 1,
          "score": 0.89,
          "rsi": 41.2,
          "atrp": 0.011,
          "macd": 0.003,
          "sentiment": 0.10,
        }

    This function normalises both formats to the unified schema.
    """
    if not supabase:
        print("[AI-Tickers] Supabase client not configured — cannot save tickers.")
        return

    try:
        # Always clear old results for this user/mode
        supabase.table("ai_tickers").delete().match({
            "user_id": user_id,
            "mode": mode,
        }).execute()
    except Exception as e:
        print(f"[AI-Tickers] Failed to clear old tickers for user_id={user_id} mode={mode}: {e!r}")

    # Normalise to list of dicts
    payload = []
    if not tickers:
        print(f"[AI-Tickers] No tickers provided for user_id={user_id} mode={mode}")
        return

    if isinstance(tickers[0], str):
        # Simple list of symbols
        for idx, symbol in enumerate(tickers, start=1):
            payload.append({
                "user_id": user_id,
                "mode": mode,
                "ticker": symbol,
                "rank": idx,
            })
    else:
        # List of dicts
        for idx, t in enumerate(tickers, start=1):
            symbol = t.get("ticker") or t.get("symbol")
            if not symbol:
                continue

            payload.append({
                "user_id": user_id,
                "mode": mode,
                "ticker": symbol,
                "rank": t.get("rank", idx),
                "score": t.get("score"),
                "rsi": t.get("rsi"),
                "atrp": t.get("atrp") or t.get("atr_pct"),
                "macd": t.get("macd"),
                "sentiment": t.get("sentiment"),
            })

    if not payload:
        print(f"[AI-Tickers] Normalised payload is empty for user_id={user_id} mode={mode}")
        return

    try:
        supabase.table("ai_tickers").insert(payload).execute()
        print(f"[AI-Tickers] Saved {len(payload)} ticker rows for user_id={user_id} mode={mode}")
    except Exception as e:
        print(f"[AI-Tickers] Failed to save tickers for user_id={user_id} mode={mode}: {e!r}")


def fetch_ai_tickers(user_id: str, mode: str):
    """
    Retrieve previously saved AI tickers for this user+mode
    from the row-based ai_tickers table.

    Returns a list of ticker symbols ordered by rank:
      ["AAPL", "TSLA", ...]
    """
    if not supabase:
        print("[AI-Tickers] Supabase not configured — cannot fetch tickers.")
        return []

    try:
        resp = (
            supabase.table("ai_tickers")
            .select("ticker, rank")
            .eq("user_id", user_id)
            .eq("mode", mode)
            .order("rank", ascending=True)
            .execute()
        )

        rows = resp.data or []
        if not rows:
            return []

        tickers = [row["ticker"] for row in rows if row.get("ticker")]
        return tickers
    except Exception as e:
        print(f"[AI-Tickers] Failed to fetch tickers for user_id={user_id} mode={mode}: {e}")
        return []
        return []


if __name__ == "__main__":
    default_mode = os.getenv("BOT_MODE", "paper")
    results = asyncio.run(stage_a_screen_and_collect(default_mode))
    log_selected_tickers_for_learning(results)
    top = score_tickers(results, default_mode)

def get_top_tickers(limit: int, user_id: str, mode: str):
    """
    Central AI selection function.

    - Runs stage_a_screen_and_collect(mode) to compute indicators + scores.
    - Sorts results by 'score' (highest first).
    - Writes the top N tickers into the ai_tickers table (row-based).
    - Returns the ordered list of ticker symbols for the worker.

    Note: This function is synchronous and is expected to be called
    from a background thread (e.g. via asyncio.to_thread in worker.py),
    so it safely uses asyncio.run() internally.
    """
    if not supabase:
        logging.warning("[AI-Tickers] Supabase client not configured — returning empty ticker list.")
        return []

    # --- Step 1: compute scores via the async pipeline ---
    try:
        indicator_results = asyncio.run(stage_a_screen_and_collect(mode=mode))
    except Exception as e:
        logging.error("Error computing AI scores: %s", e)
        return []

    if not indicator_results:
        logging.warning("No indicator results from AI selector.")
        return []

    # Keep only entries that have a valid score
    scored = [r for r in indicator_results if r.get("score") is not None]
    if not scored:
        logging.warning("Indicator results contained no valid scores.")
        return []

    # Sort by score descending and take top 'limit'
    scored_sorted = sorted(scored, key=lambda r: r["score"], reverse=True)[:limit]

    rows = []
    tickers = []

    for idx, r in enumerate(scored_sorted, start=1):
        symbol = r.get("symbol")
        if not symbol:
            continue

        tickers.append(symbol)
        rows.append({
            "user_id": user_id,
            "mode": mode,
            "ticker": symbol,
            "rank": idx,
            "score": float(r.get("score") or 0.0),
            "rsi": float(r.get("rsi") or 0.0),
            "atrp": float(r.get("atr_pct") or 0.0),
            "macd": float(r.get("macd") or 0.0),
            "sentiment": float(r.get("sentiment") or 0.0),
        })

    if not rows:
        logging.warning("No valid rows produced by get_top_tickers.")
        return []

    # --- Step 2: overwrite ai_tickers for this user/mode ---
    try:
        supabase.table("ai_tickers").delete().match({
            "user_id": user_id,
            "mode": mode,
        }).execute()

        supabase.table("ai_tickers").insert(rows).execute()
        logging.info(
            "[AI-Tickers] Saved %d rows for user_id=%s mode=%s; symbols=%s",
            len(rows),
            user_id,
            mode,
            tickers,
        )
    except Exception as e:
        logging.error("[AI-Tickers] Failed to save AI tickers: %r", e)

    return tickers
