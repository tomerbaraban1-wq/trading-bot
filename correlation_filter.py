"""
Correlation Filter
==================

Prevents the bot from opening too many highly-correlated positions.
Example: holding both AMD and NVDA = double exposure to semiconductor risk.

Strategy:
  - Compute 20-day correlation between candidate and each open position
  - If correlation > 0.85 with an open position → skip the candidate
  - Cache results for 6 hours (correlations don't change fast)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Cache: (ticker_a, ticker_b) → (correlation, timestamp)
_corr_cache: dict[tuple[str, str], tuple[float, float]] = {}
_CACHE_TTL = 6 * 3600  # 6 hours

CORRELATION_THRESHOLD = 0.85   # Block if correlation above this
LOOKBACK_DAYS = 20


def _cache_key(a: str, b: str) -> tuple[str, str]:
    """Sorted tuple — (AAPL, NVDA) and (NVDA, AAPL) → same key."""
    return tuple(sorted([a, b]))  # type: ignore


def compute_correlation(ticker_a: str, ticker_b: str) -> Optional[float]:
    """
    Compute 20-day price correlation between two tickers.
    Returns None on error.
    """
    if ticker_a == ticker_b:
        return 1.0

    key = _cache_key(ticker_a, ticker_b)
    cached = _corr_cache.get(key)
    if cached and time.time() - cached[1] < _CACHE_TTL:
        return cached[0]

    try:
        import yfinance as yf
        data = yf.download(
            [ticker_a, ticker_b],
            period=f"{LOOKBACK_DAYS + 5}d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty or "Close" not in data.columns.get_level_values(0):
            return None

        closes = data["Close"].dropna()
        if len(closes) < LOOKBACK_DAYS // 2:
            return None

        # Daily returns
        returns = closes.pct_change().dropna()
        if ticker_a not in returns.columns or ticker_b not in returns.columns:
            return None

        corr = float(returns[ticker_a].corr(returns[ticker_b]))
        _corr_cache[key] = (corr, time.time())
        return corr

    except Exception as e:
        logger.debug(f"correlation({ticker_a}, {ticker_b}) failed: {e}")
        return None


def is_too_correlated(
    candidate: str,
    open_tickers: list[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[bool, str]:
    """
    Check if `candidate` is too correlated with any of `open_tickers`.

    Returns (blocked, reason). If blocked=True, candidate should be skipped.
    """
    if not open_tickers:
        return False, ""

    high_corr = []
    for existing in open_tickers:
        if existing == candidate:
            return True, f"כבר מחזיק {existing}"

        corr = compute_correlation(candidate, existing)
        if corr is None:
            continue
        if corr >= threshold:
            high_corr.append((existing, corr))

    if high_corr:
        # Sort by correlation, take highest
        high_corr.sort(key=lambda x: -x[1])
        existing, corr = high_corr[0]
        return True, f"מתאם {corr*100:.0f}% עם {existing}"

    return False, ""


def clean_cache() -> int:
    """Remove expired cache entries. Returns number removed."""
    now = time.time()
    expired = [k for k, (_, ts) in _corr_cache.items() if now - ts > _CACHE_TTL]
    for k in expired:
        _corr_cache.pop(k, None)
    return len(expired)
