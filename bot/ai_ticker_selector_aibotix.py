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

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus

# --- Load Environment Variables ---
load_dotenv()

# --- Dynamic Client Initialization ---
def init_clients():
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    mode = os.getenv("BOT_MODE", "paper")
    paper = mode.lower() == "paper"

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

async def fetch_indicators(symbol):
    try:
        data_client, trading_client = init_clients()
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

async def stage_a_screen_and_collect(limit=5):
    data_client, trading_client = init_clients()
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

    tasks = [fetch_indicators(sym) for sym in symbols_for_indicators]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for symbol, df in zip(symbols_for_indicators, results):
        # Skip any exceptions returned by asyncio.gather
        if isinstance(df, Exception):
            logging.warning(f"Indicator task for {symbol} raised an exception: {df}")
            continue

        if df is not None and len(df) > 10:
            latest = df.iloc[-1]
            logging.info(f"{symbol} - RSI: {latest['rsi']:.2f}, ATR: {latest['atr']:.4f}, EMA crossover: {latest['ema_crossover']}")
            indicator_results.append((symbol, latest))

    logging.info(f"{len(indicator_results)} tickers gathered with indicators after screening.")
    return indicator_results


def score_tickers(indicator_results, top_n=5):
    scored = []
    for symbol, data in indicator_results:
        rsi = data['rsi']
        atr = data['atr']
        crossover = data['ema_crossover']
        volume_ratio = data['volume_ratio']
        slope = data['slope']
        gap = data['gap']

        # Require RSI, but make other indicators robust to NaNs by using safe defaults
        if pd.isna(rsi):
            continue

        # If ATR is missing or zero/negative, treat as neutral (avoid division-by-zero logic elsewhere)
        if pd.isna(atr) or atr <= 0:
            atr = 1.0

        # Treat missing volume_ratio as neutral (1.0 means "normal" volume)
        if pd.isna(volume_ratio) or volume_ratio <= 0:
            volume_ratio = 1.0

        # Treat missing slope/gap as flat / no gap
        if pd.isna(slope):
            slope = 0.0
        if pd.isna(gap):
            gap = 0.0

        score = 0
        if 40 <= rsi <= 70:
            score += 1
        if crossover:
            score += 1
        if volume_ratio > 1:
            score += 1
        if slope > 0:
            score += 1
        if abs(gap) > 0.01:
            score += 1

        scored.append((symbol, score))

    sorted_scored = sorted(scored, key=lambda x: x[1], reverse=True)
    sorted_scored = adjust_scoring_with_feedback(sorted_scored)
    top_symbols = [s for s, _ in sorted_scored[:top_n]]
    logging.info(f"Top {top_n} symbols by score: {top_symbols}")
    return top_symbols

def log_selected_tickers_for_learning(indicator_results):
    try:
        log_file = "ai_ticker_learning_log.csv"
        rows = []
        for symbol, data in indicator_results:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "rsi": round(data["rsi"], 2),
                "atr": round(data["atr"], 4),
                "ema_crossover": data["ema_crossover"],
                "volume_ratio": round(data["volume_ratio"], 2),
                "slope": round(data["slope"], 4),
                "gap": round(data["gap"], 4),
            }
            rows.append(row)

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

if __name__ == "__main__":
    results = asyncio.run(stage_a_screen_and_collect())
    log_selected_tickers_for_learning(results)
    top = score_tickers(results)

def get_top_tickers(indicator_log_csv='ai_ticker_learning_log.csv', top_n=5):
    try:
        df = pd.read_csv(indicator_log_csv)
        df.dropna(subset=['rsi', 'atr'], inplace=True)

        def score(row):
            rsi_score = max(0, 70 - row['rsi']) if row['rsi'] < 70 else 0
            atr_score = 1 / row['atr'] if row['atr'] > 0 else 0
            ema_score = 1 if row['ema_crossover'] else 0
            return rsi_score + atr_score + ema_score

        df['score'] = df.apply(score, axis=1)
        top_df = df.sort_values(by='score', ascending=False).head(top_n)

        logging.info(f"Top {top_n} tickers selected: {top_df['symbol'].tolist()}")
        return top_df['symbol'].tolist()

    except Exception as e:
        logging.error(f"Error in get_top_tickers: {e}")
        return []
