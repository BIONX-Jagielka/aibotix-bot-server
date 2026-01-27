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
MAX_TRADABLE_SCREEN = 250  # how many symbols to volume-screen
MAX_INDICATOR_TASKS = 120   # how many symbols to fetch intraday indicators for

# Historical context constants
HISTORICAL_LOOKBACK_DAYS = 120
MIN_HISTORY_BARS = 60
HIST_VOL_PENALTY_THRESHOLD = 0.12
HIST_RECOVERY_LOOKBACK = 20

# --- Bad data memory (prevents re-scanning symbols with no bars) ---
BAD_DATA_COOLDOWN_SECONDS = 6 * 60 * 60  # 6 hours
_BAD_DATA_REGISTRY = {}  # symbol -> last_failed_timestamp

# 3) BAD_DATA ESCALATION: track daily failures for extended cooldowns
_DAILY_FAILURES = {}  # symbol -> {date: failure_count}

# 6) Stability Bias (soft memory)
_STABILITY_TRACKER = {}  # symbol -> {"appearances": int, "clean_resolutions": int, "last_seen": timestamp}

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

# --- Bad data registry helpers ---
def _is_in_bad_data_cooldown(symbol: str) -> bool:
    ts = _BAD_DATA_REGISTRY.get(symbol)
    if ts is None:
        return False
    
    # Check if symbol has escalated cooldown (24 hours for repeat daily failures)
    today = datetime.now().strftime('%Y-%m-%d')
    daily_failures = _DAILY_FAILURES.get(symbol, {})
    failure_count = daily_failures.get(today, 0)
    
    cooldown_seconds = 24 * 60 * 60 if failure_count >= 2 else BAD_DATA_COOLDOWN_SECONDS
    return (time.time() - ts) < cooldown_seconds

def _mark_bad_data(symbol: str):
    current_time = time.time()
    _BAD_DATA_REGISTRY[symbol] = current_time
    
    # Track daily failures for escalation
    today = datetime.now().strftime('%Y-%m-%d')
    if symbol not in _DAILY_FAILURES:
        _DAILY_FAILURES[symbol] = {}
    _DAILY_FAILURES[symbol][today] = _DAILY_FAILURES[symbol].get(today, 0) + 1

async def fetch_indicators(symbol: str, mode: str):
    try:
        data_client, trading_client = init_clients(mode)
        bars_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=100)
        bars = data_client.get_stock_bars(bars_req).df
        if bars.empty:
            _mark_bad_data(symbol)
            return None

        # 2) INTRADAY VIABILITY GATE: stricter requirements
        # Require at least 45 intraday minute bars
        if len(bars) < 45:
            _mark_bad_data(symbol)
            return None
        
        # Require total volume over last 20 bars > 15,000
        if len(bars) >= 20:
            recent_volume = bars['volume'].iloc[-20:].sum()
            if recent_volume <= 15_000:
                _mark_bad_data(symbol)
                return None
        else:
            # Not enough bars for volume check
            _mark_bad_data(symbol)
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
        _mark_bad_data(symbol)
        # Do not raise here – return None so other symbols can still be processed
        return None

def compute_historical_context(symbol: str, mode: str) -> dict | None:
    """
    Compute historical context for a symbol using daily bars.
    Returns dict with hist_volatility, max_drawdown, avg_recovery_days or None if insufficient data.
    """
    try:
        data_client, trading_client = init_clients(mode)
        bars_req = StockBarsRequest(
            symbol_or_symbols=symbol, 
            timeframe=TimeFrame.Day, 
            limit=HISTORICAL_LOOKBACK_DAYS
        )
        bars = data_client.get_stock_bars(bars_req).df
        
        if bars.empty or len(bars) < MIN_HISTORY_BARS:
            return None
            
        # Calculate daily returns
        daily_returns = bars['close'].pct_change().dropna()
        if len(daily_returns) < MIN_HISTORY_BARS:
            return None
            
        # Historical volatility (std of daily returns)
        hist_volatility = daily_returns.std()
        
        # Max drawdown calculation
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Average recovery days after >=3% pullbacks
        recovery_days = []
        prices = bars['close'].values
        
        for i in range(1, len(prices)):
            # Look for 3%+ pullback
            pullback_start = prices[i-1]
            current_price = prices[i]
            
            if (pullback_start - current_price) / pullback_start >= 0.03:
                # Found pullback, look for recovery
                recovery_target = pullback_start * 0.99  # 99% recovery
                
                for j in range(i+1, min(i+HIST_RECOVERY_LOOKBACK+1, len(prices))):
                    if prices[j] >= recovery_target:
                        recovery_days.append(j - i)
                        break
        
        avg_recovery_days = sum(recovery_days) / len(recovery_days) if recovery_days else HIST_RECOVERY_LOOKBACK + 1
        
        return {
            "hist_volatility": hist_volatility,
            "max_drawdown": max_drawdown,
            "avg_recovery_days": avg_recovery_days
        }
        
    except Exception as e:
        logging.warning(f"Historical context failed for {symbol}: {e}")
        return None

async def stage_a_screen_and_collect(mode: str, limit: int = 5):
    data_client, trading_client = init_clients(mode)
    failed_symbols = []  # Initialize the failed_symbols list
    try:
        assets = trading_client.get_all_assets()
        assets = [a for a in assets if a.status == 'active']
        logging.debug(f"Retrieved {len(assets)} active assets from Alpaca.")
    except Exception as e:
        logging.error(f"[ERROR] Failed to retrieve assets from Alpaca: {e}")
        assets = []

    tradable = [
        a.symbol for a in assets
        if a.tradable
        and a.asset_class == "us_equity"
        and a.symbol.isalpha()
        and len(a.symbol) <= 5
        and a.exchange in ["NASDAQ", "NYSE"]
    ]
    
    # 1) SYMBOL PRE-FILTER: exclude problematic symbols before API calls
    pre_filtered = []
    for symbol in tradable:
        # Exclude symbols ending with specific suffixes
        if symbol.endswith(('W', 'WS', 'U', 'R', 'P')):
            continue
        # Exclude symbols containing digits
        if any(char.isdigit() for char in symbol):
            continue
        # Exclude symbols longer than 5 characters (already handled above, but explicit)
        if len(symbol) > 5:
            continue
        pre_filtered.append(symbol)
    
    tradable = pre_filtered
    logging.debug(f"Total tradable symbols retrieved: {len(tradable)}")
    logging.debug(f"First 10 tradable symbols: {tradable[:10]}")
    volume_filtered = []
    for symbol in tradable[:MAX_TRADABLE_SCREEN]:
        if _is_in_bad_data_cooldown(symbol):
            continue
        try:
            logging.debug(f"Checking symbol: {symbol}")
            d_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=5)
            bars = data_client.get_stock_bars(d_req).df

            # If no data, mark as failed and skip *before* trying to access bars.iloc[-1]
            if bars.empty:
                failed_symbols.append(symbol)
                if len(failed_symbols) <= 10:
                    logging.warning(f"[DEBUG] {symbol} failed - empty bars.")
                logging.warning(f"{symbol} returned empty bars.")
                _mark_bad_data(symbol)
                continue

            # NOTE: During market hours Alpaca often returns only the current in-progress daily bar.
            # That is normal and should NOT exclude a symbol from screening.
            # We only treat DAILY bars as a light sanity check (e.g., price >= MIN_PRICE).

            # Now it's safe to inspect the last bar for debug logging
            if len(volume_filtered) < 5:
                logging.debug(
                    f"{symbol} - Volume: {bars.iloc[-1]['volume']}, "
                    f"Close: {bars.iloc[-1]['close']}, "
                    f"Dollar Volume: {bars.iloc[-1]['volume'] * bars.iloc[-1]['close']}"
                )

            # Daily bars are unreliable intraday for liquidity.
            # Only perform basic price sanity check here.
            close = bars.iloc[-1]['close']
            MIN_PRICE = 3.0

            if close >= MIN_PRICE:
                volume_filtered.append(symbol)
        except Exception as e:
            logging.warning(f"Error processing {symbol}: {e}")
            continue

    logging.info(f"[DEBUG] First 10 failed symbols: {failed_symbols[:10]}")
    logging.info(f"{len(volume_filtered)} tickers passed relaxed pre-screen filter.")
    random.shuffle(volume_filtered)

    indicator_results = []

    # Limit how many symbols we fetch intraday indicators for
    symbols_for_indicators = volume_filtered[:MAX_INDICATOR_TASKS]

    if not symbols_for_indicators:
        logging.warning("No symbols passed the pre-screen filter; returning empty indicator results.")
        return []

    tasks = [fetch_indicators(sym, mode) for sym in symbols_for_indicators]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 4) LOGGING COMPRESSION: track skipped symbols for summary
    skipped_invalid_count = 0
    
    for symbol, df in zip(symbols_for_indicators, results):
        # Skip any exceptions returned by asyncio.gather
        if isinstance(df, Exception):
            logging.warning(f"Indicator task for {symbol} raised an exception: {df}")
            continue

        if df is None:
            skipped_invalid_count += 1
            continue

        # Now process valid dataframe
        latest = df.iloc[-1]
        logging.info(f"{symbol} - RSI: {latest['rsi']:.2f}, ATR: {latest['atr']:.4f}, EMA crossover: {latest['ema_crossover']}")
        
        # 1) Market Regime Classification (internal only)
        regime = "transition"  # default
        if not pd.isna(latest['slope']) and not pd.isna(latest['ema_fast']) and not pd.isna(latest['ema_slow']):
            if latest['slope'] > 0.001 and latest['ema_fast'] > latest['ema_slow']:
                regime = "trend"
            elif abs(latest['slope']) < 0.0005:
                regime = "range"
        
        # 2) Volatility Expansion Bias (soft only)
        atr_expansion_bias = 0.0
        if len(df) >= 20:
            recent_atr = df['atr'].tail(10)
            if not recent_atr.empty and not pd.isna(recent_atr.iloc[-1]):
                current_atr = recent_atr.iloc[-1]
                mean_atr = recent_atr.mean()
                if current_atr <= mean_atr * 0.95:  # ATR contracting
                    atr_expansion_bias = -0.1  # small penalty
        
        # 4) Candle Structure Awareness (heuristic only)
            candle_structure_bias = 0.0
            if len(df) >= 15:
                recent_bars = df.tail(15)
                if not recent_bars.empty:
                    # Simple heuristics for structure detection
                    recent_ranges = recent_bars['high'] - recent_bars['low']
                    avg_range = recent_ranges.mean() if not recent_ranges.empty else 0
                    latest_range = recent_ranges.iloc[-1] if len(recent_ranges) > 0 else 0
                    
                    # Check for higher highs/lows pattern
                    highs = recent_bars['high'].tail(5)
                    lows = recent_bars['low'].tail(5)
                    
                    if len(highs) >= 3 and len(lows) >= 3:
                        # Impulsive: expanding ranges + higher highs
                        if latest_range > avg_range * 1.2 and highs.iloc[-1] > highs.iloc[-3]:
                            candle_structure_bias = 0.15  # positive bias for impulsive
                        # Compressing: shrinking ranges
                        elif latest_range < avg_range * 0.7:
                            candle_structure_bias = -0.1  # penalty for compression
            
            # Calculate initial score with enhanced unified_ai_score
            score = enhanced_unified_ai_score(
                rsi=latest['rsi'],
                atr=latest['atr'],
                ema_crossover=latest['ema_crossover'],
                volume_ratio=latest['volume_ratio'],
                slope=latest['slope'],
                gap=latest['gap'],
                regime=regime,
                atr_expansion_bias=atr_expansion_bias,
                candle_structure_bias=candle_structure_bias
            ) or 0.0
            
            # 6) Stability Bias (soft memory)
            stability_bias = 0.0
            stability_data = _STABILITY_TRACKER.get(symbol, {"appearances": 0, "clean_resolutions": 0})
            if stability_data["appearances"] > 0:
                resolution_rate = stability_data["clean_resolutions"] / stability_data["appearances"]
                if resolution_rate > 0.6:
                    stability_bias = 0.1  # reward consistent performers
                elif resolution_rate < 0.3 and stability_data["appearances"] >= 3:
                    stability_bias = -0.1  # penalize frequent non-performers
            
            # Update stability tracker
            _STABILITY_TRACKER[symbol] = {
                "appearances": stability_data["appearances"] + 1,
                "clean_resolutions": stability_data["clean_resolutions"],
                "last_seen": time.time()
            }
            
            # Apply stability bias to score
            score += stability_bias
            
            # Apply historical context modifier (Step 3)
            hist_context = compute_historical_context(symbol, mode)
            if hist_context is not None:
                original_score = score
                
                # Apply volatility penalty
                if hist_context["hist_volatility"] > HIST_VOL_PENALTY_THRESHOLD:
                    score *= 0.85
                
                # Apply recovery bonus
                if hist_context["avg_recovery_days"] <= HIST_RECOVERY_LOOKBACK:
                    score *= 1.10
                
                # Clamp to minimum
                score = max(0.01, score)
                
                # Rate-limited logging when context is applied (Step 4)
                if score != original_score:
                    logging.info(
                        f"HistContext | {symbol} | vol={hist_context['hist_volatility']:.3f} "
                        f"dd={hist_context['max_drawdown']:.2f} recov={hist_context['avg_recovery_days']:.0f}d"
                    )
            
            # STEP 2: Early session score dampening (not blocking)
            is_early_session = len(df) < 60
            if not is_early_session and df.index[-1] is not None:
                # Check if most recent bar is within first 45 minutes of session
                from datetime import time as dt_time, timedelta
                try:
                    last_bar_time = df.index[-1].time() if hasattr(df.index[-1], 'time') else None
                    if last_bar_time:
                        market_open = dt_time(9, 30)  # 9:30 AM ET
                        early_cutoff = (datetime.combine(datetime.today(), market_open) + timedelta(minutes=45)).time()
                        is_early_session = last_bar_time <= early_cutoff
                except:
                    pass
            
            if is_early_session:
                score *= 0.85
            
            # STEP 3: Score compression (anti-spike)
            score = np.tanh(score)
            score = max(0.01, score)
            
            # 5) Opportunity Scarcity (pre-score filtering)
            # Only add to results if score meets minimum quality threshold
            if score >= 0.15:  # Allow fewer but higher-quality candidates
                # STEP 1: Normalize ALL AI output fields (never None)
                rsi_norm = float(latest['rsi']) if latest['rsi'] is not None and not pd.isna(latest['rsi']) else 50.0
                atr_norm = float(latest['atr']) if latest['atr'] is not None and not pd.isna(latest['atr']) else 1.0
                atr_pct_norm = float(atr_norm * 100) if atr_norm is not None else 100.0
                macd_norm = float(latest.get('macd', 0)) if latest.get('macd') is not None else 0.0
                sentiment_norm = float(latest.get('sentiment', 0)) if latest.get('sentiment') is not None else 0.0
                hist_volatility_norm = float(hist_context.get("hist_volatility")) if hist_context and hist_context.get("hist_volatility") is not None else 0.0
                avg_recovery_days_norm = int(hist_context.get("avg_recovery_days")) if hist_context and hist_context.get("avg_recovery_days") is not None else int(HIST_RECOVERY_LOOKBACK + 1)
                max_drawdown_norm = float(hist_context.get("max_drawdown")) if hist_context and hist_context.get("max_drawdown") is not None else 0.0
                
                indicator_results.append({
                    "symbol": symbol,
                    "rsi": rsi_norm,
                    "atr": atr_norm,
                    "atr_pct": atr_pct_norm,
                    "macd": macd_norm,
                    "sentiment": sentiment_norm,
                    "volume": latest.get('volume', 0),
                    "ema_crossover": latest['ema_crossover'],
                    "volume_ratio": latest['volume_ratio'],
                    "slope": latest['slope'],
                    "gap": latest['gap'],
                    "score": score,
                    "hist_volatility": hist_volatility_norm,
                    "avg_recovery_days": avg_recovery_days_norm,
                    "max_drawdown": max_drawdown_norm,
                })

    # Enhanced logging with comprehensive summary and skipped count
    early_session_count = sum(
        1 for r in indicator_results
        if r.get("atr") is not None and r.get("atr_pct") is not None
    )
    is_early_session = early_session_count > (len(indicator_results) * 0.5)
    
    # Count quality tiers for opportunity scarcity insight
    high_quality_count = sum(1 for r in indicator_results if r.get('score', 0) > 0.5)
    symbols_analysed = len(symbols_for_indicators)
    symbols_returned = len(indicator_results)
    
    logging.info(
        f"[AI-Selector] Analyzed: {symbols_analysed} | Returned: {symbols_returned} | "
        f"High-quality: {high_quality_count} | Early-session: {is_early_session} | "
        f"Skipped {skipped_invalid_count} symbols due to intraday invalidity"
    )
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

def enhanced_unified_ai_score(
    rsi: float,
    atr: float,
    ema_crossover: bool,
    volume_ratio: float,
    slope: float,
    gap: float,
    regime: str = "transition",
    atr_expansion_bias: float = 0.0,
    candle_structure_bias: float = 0.0,
) -> Optional[float]:
    """
    Enhanced unified multi-factor score with market context awareness.
    Builds upon the original unified_ai_score with new regime and structure intelligence.
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

    # 3) RSI Context Reinterpretation (bias only, not entry signal)
    if 40 <= rsi <= 60:
        score += 0.1  # slight penalty for neutral RSI
    elif rsi > 60 and regime == "trend":
        score += 0.4  # positive bias for momentum in trend
    elif rsi < 40 and regime == "trend":
        score -= 0.2  # negative bias for weakness in trend
    elif 25 <= rsi <= 80:
        score += 0.2  # acceptable range bonus
    else:
        score -= 0.5  # penalty for extreme RSI

    # 2) ATR: prefer reasonable volatility with regime context
    atr_clamped = min(max(atr, 0.01), 5.0)
    base_atr_score = 0.5 / atr_clamped
    if regime == "trend":
        base_atr_score *= 1.1  # slight boost for volatility in trends
    score += base_atr_score

    # 3) EMA crossover: bullish alignment (bonus, not decisive)
    if ema_crossover:
        crossover_bonus = 0.5
        if regime == "trend":
            crossover_bonus *= 1.2  # stronger bonus in trend regime
        score += crossover_bonus

    # 4) Volume ratio: prefer above-average volume, cap contribution
    if volume_ratio > 1.0:
        score += min(1.5, (volume_ratio - 1.0) / 2.0)

    # 5) Slope: upward trend is good, context-aware
    if slope > 0:
        slope_score = min(1.5, slope * 100.0)
        if regime == "range":
            slope_score *= 0.7  # reduce slope importance in range-bound markets
        score += slope_score

    # 6) Gap: very large gaps are risky; small gaps can be constructive
    if abs(gap) > 0.10:
        score -= 1.0
    elif abs(gap) > 0.02:
        score += 0.2

    # Apply new bias factors
    score += atr_expansion_bias + candle_structure_bias

    return score

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

    # 1) RSI: prefer 40–60, softly accept 25–80, only penalise extremes
    if 25 <= rsi <= 80:
        # bell-shaped preference around 50, but no penalty for moderate ranges
        if 40 <= rsi <= 60:
            score += 1.5 - abs(rsi - 50) / 15.0
        else:
            score += 0.3  # small bonus for acceptable range
    else:
        score -= 0.5  # penalty only for extreme RSI

    # 2) ATR: prefer non-crazy volatility (smaller ATR is safer, but not zero)
    atr_clamped = min(max(atr, 0.01), 5.0)
    score += 0.5 / atr_clamped

    # 3) EMA crossover: bullish alignment (bonus, not decisive)
    if ema_crossover:
        score += 0.5

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

def score_tickers(indicator_results, mode: str, top_n: int = 30):
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
    Central AI selection function with breadth targets and stability.

    - Runs stage_a_screen_and_collect(mode) to compute indicators + scores.
    - Sorts results by 'score' (highest first).
    - Ensures at least min(limit, 20) symbols are returned.
    - Implements light continuity logic for 40-60% overlap with previous tickers.
    - Writes the top N tickers into the ai_tickers table (row-based).
    - Returns the ordered list of ticker symbols for the worker.

    Note: This function is synchronous and is expected to be called
    from a background thread (e.g. via asyncio.to_thread in worker.py),
    so it safely uses asyncio.run() internally.
    """
    if not supabase:
        logging.warning("[AI-Tickers] Supabase client not configured — returning empty ticker list.")
        return []

    # Fetch previously saved tickers for continuity
    previous_tickers = fetch_ai_tickers(user_id, mode)
    min_breadth_target = min(limit, 20)

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

    # Sort by score descending
    scored_sorted = sorted(scored, key=lambda r: r["score"], reverse=True)

    # --- Step 2: Implement stability logic (40-60% overlap preservation) ---
    final_selection = []
    used_symbols = set()
    
    # First, preserve high-quality previous tickers that still pass basic validity
    if previous_tickers:
        previous_valid = []
        for prev_symbol in previous_tickers:
            # Check if the previous symbol is still in our current results
            for result in scored_sorted:
                if result.get("symbol") == prev_symbol:
                    previous_valid.append(result)
                    break
        
        # Preserve up to 60% of target from previous selection (stability)
        preserve_count = min(len(previous_valid), int(limit * 0.6))
        for i in range(preserve_count):
            if previous_valid[i]["symbol"] not in used_symbols:
                final_selection.append(previous_valid[i])
                used_symbols.add(previous_valid[i]["symbol"])

    # Fill remaining slots with new high-scoring tickers
    for result in scored_sorted:
        symbol = result.get("symbol")
        if not symbol or symbol in used_symbols:
            continue
            
        final_selection.append(result)
        used_symbols.add(symbol)
        
        # Stop when we reach our limit
        if len(final_selection) >= limit:
            break

    # --- Step 3: Ensure minimum breadth target is met ---
    if len(final_selection) < min_breadth_target and len(scored_sorted) >= min_breadth_target:
        # Fill remaining slots with any available scored tickers
        for result in scored_sorted:
            symbol = result.get("symbol")
            if not symbol or symbol in used_symbols:
                continue
                
            final_selection.append(result)
            used_symbols.add(symbol)
            
            if len(final_selection) >= min_breadth_target:
                break

    rows = []
    tickers = []

    for idx, r in enumerate(final_selection, start=1):
        symbol = r.get("symbol")
        if not symbol:
            continue

        tickers.append(symbol)
        # STEP 4: Consistent row payload for Supabase (never NULL numerics)
        score_norm = float(r.get("score")) if r.get("score") is not None else 0.0
        rsi_norm = float(r.get("rsi")) if r.get("rsi") is not None else 50.0
        atrp_norm = float(r.get("atr_pct")) if r.get("atr_pct") is not None else 100.0
        macd_norm = float(r.get("macd")) if r.get("macd") is not None else 0.0
        sentiment_norm = float(r.get("sentiment")) if r.get("sentiment") is not None else 0.0
        hist_volatility_norm = float(r.get("hist_volatility")) if r.get("hist_volatility") is not None else 0.0
        avg_recovery_days_norm = int(r.get("avg_recovery_days")) if r.get("avg_recovery_days") is not None else int(HIST_RECOVERY_LOOKBACK + 1)
        max_drawdown_norm = float(r.get("max_drawdown")) if r.get("max_drawdown") is not None else 0.0
        
        rows.append({
            "user_id": user_id,
            "mode": mode,
            "ticker": symbol,
            "rank": idx,
            "score": score_norm,
            "rsi": rsi_norm,
            "atrp": atrp_norm,
            "macd": macd_norm,
            "sentiment": sentiment_norm,
            "hist_volatility": hist_volatility_norm,
            "avg_recovery_days": avg_recovery_days_norm,
            "max_drawdown": max_drawdown_norm,
        })

    if not rows:
        logging.warning("No valid rows produced by get_top_tickers.")
        return []

    # Log stability metrics
    overlap_count = len(set(tickers) & set(previous_tickers)) if previous_tickers else 0
    overlap_pct = (overlap_count / len(previous_tickers) * 100) if previous_tickers else 0
    logging.info(f"[AI-Stability] Selected {len(tickers)} tickers with {overlap_count}/{len(previous_tickers) if previous_tickers else 0} overlap ({overlap_pct:.1f}%)")

    # --- Step 4: overwrite ai_tickers for this user/mode ---
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
