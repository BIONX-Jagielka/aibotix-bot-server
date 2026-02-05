# aibotix/selector/stable_selector.py
"""
DEPRECATED: Stability Filter for AI Ticker Selection

This module is being phased out in favor of integrating stability logic 
directly into the existing AI selector (bot/ai_ticker_selector_aibotix.py).

The existing AI selector already implements:
- 40-60% overlap preservation for stability
- Bad data cooldowns and registry tracking  
- Historical context and scoring
- Supabase persistence in ai_tickers table

This file now serves only as a compatibility wrapper to avoid breaking
existing integration code until full removal.
"""
from __future__ import annotations
import logging
from typing import List, Optional

log = logging.getLogger("aibotix.selector")

class StabilityFilter:
    """
    WRAPPER: Post-processes output from existing AI selector.
    Does NOT replace or compete with the authoritative AI selector.
    
    Purpose: Add optional filtering/rate-limiting on top of existing selection.
    The existing AI selector (get_top_tickers) remains the single source of truth.
    """
    
    def __init__(self):
        self.initialized = True
        log.info("StabilityFilter initialized as wrapper around existing AI selector")
    
    def filter_ai_selection(self, ai_selected_tickers: List[str], 
                           max_symbols: Optional[int] = None) -> List[str]:
        """
        Filter/post-process the output from the existing AI selector.
        
        Args:
            ai_selected_tickers: Output from get_top_tickers() function
            max_symbols: Optional limit (already applied by AI selector)
            
        Returns:
            Filtered ticker list (typically same as input since AI selector 
            already implements stability logic)
        """
        if not ai_selected_tickers:
            return []
        
        # The existing AI selector already implements stability logic:
        # - 40-60% overlap preservation  
        # - Bad data cooldowns
        # - Historical context scoring
        # - Comprehensive filtering
        
        # For now, pass through unchanged since stability is already built-in
        filtered_tickers = ai_selected_tickers[:max_symbols] if max_symbols else ai_selected_tickers
        
        log.debug(f"StabilityFilter: passed through {len(filtered_tickers)} tickers from AI selector")
        return filtered_tickers

# Backward compatibility aliases
SelectorState = dict  # Simplified for compatibility
StableAISelector = StabilityFilter  # Alias for existing integration code

def get_stability_filter() -> StabilityFilter:
    """Get a stability filter instance"""
    return StabilityFilter()