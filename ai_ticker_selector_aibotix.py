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


# --- Logger Setup ---
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def create_clients(api_key, api_secret, paper=True):
    data_client = StockHistoricalDataClient(api_key, api_secret)
    trading_client = TradingClient(api_key, api_secret, paper=paper)
    return data_client, trading_client

RSI_PERIOD = 14

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

async def fetch_indicators(symbol, data_client):
    try:
        bars_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=100)
        bars = data_client.get_stock_bars(bars_req).df
        if bars.empty:
            logging.warning(f"No data returned for {symbol}")
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
        raise  # Temporarily raise for debugging

async def stage_a_screen_and_collect(data_client, trading_client, limit=5):
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
    for symbol in tradable[:1000]:  # Increased from 300 to 1000
        try:
            logging.info(f"Checking symbol: {symbol}")
            d_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=1)
            bars = data_client.get_stock_bars(d_req).df
            if len(volume_filtered) < 5:
                logging.info(f"[DEBUG] {symbol} - Volume: {bars.iloc[-1]['volume']}, Close: {bars.iloc[-1]['close']}, Dollar Volume: {bars.iloc[-1]['volume'] * bars.iloc[-1]['close']}")
            if bars.empty:
                failed_symbols.append(symbol)
                if len(failed_symbols) <= 10:
                    logging.warning(f"[DEBUG] {symbol} failed - empty bars.")
                logging.warning(f"{symbol} returned empty bars.")
                continue
            vol = bars.iloc[-1]['volume']
            close = bars.iloc[-1]['close']
            dollar_volume = vol * close
            if vol > 100 and dollar_volume > 1_000 and close > 0.01:
                volume_filtered.append(symbol)
        except Exception as e:
            logging.warning(f"Error processing {symbol}: {e}")
            continue

    logging.info(f"[DEBUG] First 10 failed symbols: {failed_symbols[:10]}")
    logging.info(f"{len(volume_filtered)} tickers passed relaxed volume filter.")
    random.shuffle(volume_filtered)

    indicator_results = []
    tasks = [fetch_indicators(sym, data_client) for sym in volume_filtered[:100]]
    results = await asyncio.gather(*tasks)

    for symbol, df in zip(volume_filtered[:100], results):
        if df is not None and len(df) > 10:
            latest = df.iloc[-1]
            logging.info(f"{symbol} - RSI: {latest['rsi']:.2f}, ATR: {latest['atr']:.4f}, EMA crossover: {latest['ema_crossover']}")
            indicator_results.append((symbol, latest))

    logging.info(f"{len(indicator_results)} tickers gathered with indicators.")
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

        if pd.isna(rsi) or pd.isna(atr) or pd.isna(slope) or pd.isna(gap):
            continue

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


# New async function to be called externally (e.g., from main.py)
async def get_top_tickers_from_api_keys(api_key, api_secret, top_n=5, paper=True):
    data_client, trading_client = create_clients(api_key, api_secret, paper=paper)
    indicator_results = await stage_a_screen_and_collect(data_client, trading_client)
    log_selected_tickers_for_learning(indicator_results)
    top_tickers = score_tickers(indicator_results, top_n=top_n)
    return top_tickers

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
