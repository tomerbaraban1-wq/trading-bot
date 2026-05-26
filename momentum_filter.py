"""
Momentum Pre-Filter
====================

Before running expensive composite scoring on each stock,
quickly pre-rank all watchlist stocks by momentum.

Only the top N momentum stocks proceed to full scoring.
This makes the scan cycle faster and more focused on quality setups.

Momentum signals checked:
1. Price vs 52-week range (how close to highs)
2. Short-term return (5-day and 20-day)
3. Volume trend (is it increasing?)
4. Relative Strength vs SPY
5. Trend quality (above SMA20, SMA50)
"""

import asyncio
import logging
import time
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Cache: ticker -> (timestamp, momentum_score)
_momentum_cache: dict[str, tuple[float, float]] = {}
_CACHE_TTL = 20 * 60  # 20 minutes


def _fast_momentum_score(ticker: str) -> Optional[float]:
    """
    Calculate a quick momentum score (0-100) using yfinance.
    Returns None on error.
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta, timezone

        # Check cache first
        now = time.time()
        cached = _momentum_cache.get(ticker)
        if cached and now - cached[0] < _CACHE_TTL:
            return cached[1]

        # Get 60 days of data
        data = yf.download(ticker, period="60d", progress=False, auto_adjust=True)
        if data is None or len(data) < 20:
            return None

        prices = data["Close"].values
        volumes = data["Volume"].values
        current = float(prices[-1])

        score = 0.0

        # 1. Short-term return (5-day): 30 pts
        ret_5d = (current - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0
        if ret_5d > 3:
            score += 30
        elif ret_5d > 1:
            score += 20
        elif ret_5d > 0:
            score += 10
        elif ret_5d < -3:
            score -= 10

        # 2. 20-day return: 25 pts
        ret_20d = (current - prices[-20]) / prices[-20] * 100 if len(prices) >= 20 else 0
        if ret_20d > 8:
            score += 25
        elif ret_20d > 4:
            score += 18
        elif ret_20d > 0:
            score += 10
        elif ret_20d < -5:
            score -= 15

        # 3. Above SMA20: 20 pts
        sma20 = np.mean(prices[-20:])
        if current > sma20:
            score += 20
        else:
            score -= 10  # Below SMA20 = weak

        # 4. Volume trend (last 5 vs last 20): 15 pts
        avg_vol_20 = np.mean(volumes[-20:])
        avg_vol_5 = np.mean(volumes[-5:])
        vol_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1.0
        if vol_ratio > 1.3:
            score += 15  # Accelerating volume
        elif vol_ratio > 1.0:
            score += 8
        elif vol_ratio < 0.7:
            score -= 5  # Declining volume = weak

        # 5. Price position in 52-week range: 10 pts
        hi52 = max(prices)
        lo52 = min(prices)
        rng = hi52 - lo52
        if rng > 0:
            position = (current - lo52) / rng
            if position > 0.85:
                score += 10  # Near 52-week high = strength
            elif position > 0.60:
                score += 5
            elif position < 0.20:
                score -= 10  # Near 52-week low = weakness

        # Clamp to 0-100
        score = max(0.0, min(100.0, score + 30))  # +30 baseline

        _momentum_cache[ticker] = (now, score)
        return score

    except Exception as e:
        logger.debug(f"Momentum score failed for {ticker}: {e}")
        return None


def rank_by_momentum(tickers: list[str], top_n: int = 30) -> list[tuple[str, float]]:
    """
    Quickly rank all tickers by momentum score.
    Returns top_n tickers with their scores, sorted best-first.

    This is a synchronous function for use in threading.
    """
    scored = []

    for ticker in tickers:
        score = _fast_momentum_score(ticker)
        if score is not None:
            scored.append((ticker, score))
        # Skip tickers that fail (no data, delisted, etc.)

    # Sort by momentum score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        f"[MOMENTUM] Ranked {len(scored)}/{len(tickers)} tickers. "
        f"Top 5: {[(t, f'{s:.0f}') for t, s in scored[:5]]}"
    )

    return scored[:top_n]


async def get_top_momentum_tickers(
    tickers: list[str],
    top_n: int = 25,
    min_score: float = 50.0,
) -> list[str]:
    """
    Async wrapper for rank_by_momentum.
    Filters to only tickers above min_score.
    """
    try:
        ranked = await asyncio.to_thread(rank_by_momentum, tickers, top_n * 2)
        # Filter by minimum momentum score
        filtered = [t for t, s in ranked if s >= min_score][:top_n]

        if not filtered:
            logger.info("[MOMENTUM] No tickers above min_score — using top 10 by score")
            filtered = [t for t, s in ranked[:10]]

        return filtered

    except Exception as e:
        logger.error(f"Momentum filter failed: {e}")
        return tickers[:top_n]  # Fall back to first N tickers
