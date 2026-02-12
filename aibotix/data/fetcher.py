# aibotix/data/fetcher.py
from __future__ import annotations
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

from ..utils.http_client import ResilientSession, HttpConfig
from ..observability import log_fetch_failure, log_empty_bars, log_structured, MetricEvent

log = logging.getLogger("aibotix.data")

class RobustDataFetcher:
    """Resilient market data fetcher with quality gates"""
    
    def __init__(self, alpaca_data_client=None, session: Optional[ResilientSession] = None):
        self.alpaca_client = alpaca_data_client
        self.session = session or ResilientSession(HttpConfig())
        self.min_bars_required = 25
        self.max_empty_bar_ratio = 0.1  # 10% max empty bars
        
    def fetch_bars_with_quality_check(self, symbol: str, timeframe: str = "1Min", 
                                     limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch bars with comprehensive quality checks
        Returns None if data quality is insufficient
        """
        start_time = time.time()
        
        try:
            # Attempt to fetch data using Alpaca client
            if self.alpaca_client:
                df = self._fetch_via_alpaca(symbol, timeframe, limit)
            else:
                log.warning("No Alpaca client available for data fetch", 
                           extra={"symbol": symbol, "timeframe": timeframe})
                return None
                
            if df is None or len(df) == 0:
                log_structured(MetricEvent(
                    event_type="fetch_empty_result",
                    symbol=symbol,
                    timeframe=timeframe,
                    elapsed=time.time() - start_time
                ))
                return None
                
            # Quality checks
            quality_issues = self._check_data_quality(df, symbol, timeframe)
            if quality_issues:
                log.warning(f"Data quality issues for {symbol}: {quality_issues}")
                return None
                
            # Log successful fetch
            log_structured(MetricEvent(
                event_type="fetch_success",
                symbol=symbol,
                timeframe=timeframe,
                count=len(df),
                elapsed=time.time() - start_time
            ))
            
            return df
            
        except Exception as e:
            elapsed = time.time() - start_time
            log_fetch_failure(
                symbol=symbol,
                endpoint="alpaca_bars",
                timeframe=timeframe,
                attempt=1,
                exception_type=type(e).__name__,
                elapsed=elapsed
            )
            return None
    
    def _fetch_via_alpaca(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data using Alpaca client with error handling"""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            # Map timeframe string to Alpaca TimeFrame
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, "Min"),
                "15Min": TimeFrame(15, "Min"),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            tf = tf_map.get(timeframe, TimeFrame.Minute)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=5)  # Get more data for quality
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_time,
                end=end_time,
                limit=limit,
                feed="iex"
            )
            
            response = self.alpaca_client.get_stock_bars(request)
            
            if not response or symbol not in response:
                return None
                
            bars = response[symbol]
            if not bars:
                return None
                
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            log.error(f"Alpaca fetch error for {symbol}: {e}")
            raise
    
    def _check_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[str]:
        """Check data quality and return list of issues"""
        issues = []
        
        if len(df) < self.min_bars_required:
            issues.append(f"insufficient_bars: {len(df)} < {self.min_bars_required}")
            
        # Check for empty bars (zero volume, identical OHLC)
        empty_bars = 0
        for idx, row in df.iterrows():
            if (row['volume'] == 0 or 
                (row['open'] == row['high'] == row['low'] == row['close'])):
                empty_bars += 1
                
        empty_ratio = empty_bars / len(df)
        if empty_ratio > self.max_empty_bar_ratio:
            issues.append(f"too_many_empty_bars: {empty_bars}/{len(df)} ({empty_ratio:.1%})")
            log_empty_bars(symbol, timeframe, empty_bars)
            
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            issues.append(f"nan_values: {nan_count}")
            
        # Check for zero/negative prices
        invalid_prices = ((df[['open','high','low','close']] <= 0).any(axis=1)).sum()
        if invalid_prices > 0:
            issues.append(f"invalid_prices: {invalid_prices}")
            
        # Check for inverted OHLC
        ohlc_issues = ((df['low'] > df['high']) | 
                      (df['open'] > df['high']) | 
                      (df['close'] > df['high']) |
                      (df['open'] < df['low']) |
                      (df['close'] < df['low'])).sum()
        if ohlc_issues > 0:
            issues.append(f"ohlc_violations: {ohlc_issues}")
            
        return issues