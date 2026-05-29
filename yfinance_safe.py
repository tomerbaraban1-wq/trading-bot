"""
Yahoo Finance Safe Wrapper
============================

Yahoo Finance occasionally returns 401 "Invalid Crumb" or rate-limits requests.
This module:
  1. Adds automatic retry with exponential backoff
  2. Suppresses noisy yfinance ERROR logs (we handle them gracefully)
  3. Caches negative results for 5 min so we don't hammer Yahoo
  4. Provides clean fallback values when Yahoo is down

Usage:
    from yfinance_safe import get_ticker_info_safe
    info = get_ticker_info_safe("AAPL")
    if info:
        # use info
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Negative result cache: ticker → expiry_ts
_negative_cache: dict[str, float] = {}
_NEGATIVE_TTL = 300   # 5 minutes — don't retry failed tickers too often

# Suppress yfinance's own noisy error logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def get_ticker_info_safe(ticker: str, max_retries: int = 2) -> Optional[dict]:
    """
    Safe wrapper around yf.Ticker(ticker).info that:
      - Retries on transient errors
      - Returns None on hard failure (instead of raising)
      - Caches negative results to avoid hammering
    """
    # Check negative cache
    expiry = _negative_cache.get(ticker)
    if expiry and time.time() < expiry:
        return None

    try:
        import yfinance as yf
    except ImportError:
        return None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            info = yf.Ticker(ticker).info
            if info and isinstance(info, dict) and len(info) > 1:
                # Clear negative cache on success
                _negative_cache.pop(ticker, None)
                return info
            # Empty response — treat as soft failure, retry
            last_error = "empty info dict"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            # Exponential backoff: 1s, 2s
            time.sleep(2 ** (attempt - 1))

    # All retries failed — cache negative result
    _negative_cache[ticker] = time.time() + _NEGATIVE_TTL
    logger.debug(f"[YF-SAFE] {ticker}: failed after {max_retries} retries — {last_error}")
    return None


def get_price_safe(ticker: str) -> Optional[float]:
    """Get current price from yfinance fast_info, with fallback to info."""
    expiry = _negative_cache.get(ticker)
    if expiry and time.time() < expiry:
        return None

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # Try fast_info first (faster, no crumb needed)
        try:
            p = float(getattr(t.fast_info, "last_price", 0) or 0)
            if p > 0:
                return p
        except Exception:
            pass
        # Fallback to info
        info = t.info
        if info and isinstance(info, dict):
            p = info.get("currentPrice") or info.get("regularMarketPrice")
            if p:
                return float(p)
    except Exception as e:
        logger.debug(f"[YF-SAFE] {ticker} price failed: {e}")
        _negative_cache[ticker] = time.time() + _NEGATIVE_TTL

    return None


def clear_negative_cache() -> int:
    """Clear all negative cache entries. Returns count cleared."""
    n = len(_negative_cache)
    _negative_cache.clear()
    return n


def get_negative_cache_stats() -> dict:
    """Get current negative cache state for diagnostics."""
    now = time.time()
    active = {t: exp - now for t, exp in _negative_cache.items() if exp > now}
    return {
        "total": len(_negative_cache),
        "active": len(active),
        "tickers": list(active.keys())[:10],
    }
