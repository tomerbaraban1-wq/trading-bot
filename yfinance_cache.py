"""
yfinance Data Cache
====================

Central cache for yfinance OHLCV data.  Every module that calls
yf.download() should go through get_ohlcv() instead.

Why this matters:
- During a 10-ticker scan, each ticker is downloaded 3-4 times
  (scoring, pattern recognition, ATR stop, relative-strength).
- Each download takes ~0.5-2s → 10 tickers × 4 downloads = 40-80s.
- With this cache, only the FIRST call per ticker fetches data.
  Subsequent calls return the in-memory copy instantly.
- Result: scan time drops from ~60s to ~10-15s.

TTL: 5 minutes (matches the scan interval).
"""

import logging
import threading
import time
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Cache structure: {ticker: (timestamp, DataFrame)}
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
# Last-known-GOOD data (long-lived). Used as a fallback when a fresh fetch fails
# (e.g. Yahoo 401 "Invalid Crumb"), so the bot decides on slightly-stale-but-real
# data instead of empty/corrupt data. Far safer for trade decisions.
_last_good: dict[str, tuple[float, pd.DataFrame]] = {}
_lock = threading.Lock()

# Cache lives for 5 minutes (matches scan cadence)
_TTL_SECONDS = 300

# Track cache stats for diagnostics
_stats = {"hits": 0, "misses": 0}


def get_ohlcv(
    ticker: str,
    period: str = "3mo",
    *,
    days: Optional[int] = None,
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return OHLCV data for *ticker*, fetching from yfinance only when needed.

    Parameters
    ----------
    ticker        : stock symbol, e.g. "AAPL"
    period        : yfinance period string (default "3mo")
    days          : if given, overrides period with start=today-days
    interval      : "1d" (default), "1h", "15m", etc.
    force_refresh : bypass cache and fetch fresh data

    Returns
    -------
    pandas DataFrame with columns: Open, High, Low, Close, Volume
    Empty DataFrame on error.
    """
    cache_key = f"{ticker}_{period}_{interval}"
    if days:
        cache_key = f"{ticker}_{days}d_{interval}"

    now = time.monotonic()

    if not force_refresh:
        with _lock:
            entry = _cache.get(cache_key)
            if entry:
                ts, df = entry
                if now - ts < _TTL_SECONDS:
                    _stats["hits"] += 1
                    return df.copy()

    # Cache miss — fetch fresh, with RETRY (transient Yahoo 401 "Invalid Crumb"
    # usually succeeds on a second attempt) and LAST-GOOD fallback.
    _stats["misses"] += 1

    kwargs: dict = {"interval": interval}
    if days:
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc)
        kwargs["start"] = end - timedelta(days=days)
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    last_err = "unknown"
    for attempt in range(3):
        try:
            df = yf.download(ticker, **kwargs, auto_adjust=True, progress=False)
            if df is None or df.empty:
                last_err = "empty result (possible rate-limit / 401)"
                time.sleep(0.4 * (attempt + 1))   # brief backoff, let yfinance refresh its crumb
                continue

            # Flatten MultiIndex columns (yfinance ≥ 0.2 quirk for single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            with _lock:
                _cache[cache_key] = (now, df)
                _last_good[cache_key] = (now, df)   # remember as last-known-good
            if attempt > 0:
                logger.info(f"[CACHE] {ticker}: fetched on retry #{attempt+1} ({len(df)} rows)")
            return df.copy()

        except Exception as e:
            last_err = str(e)
            time.sleep(0.4 * (attempt + 1))

    # All attempts failed — fall back to last-known-good data if we have any.
    with _lock:
        lg = _last_good.get(cache_key)
    if lg:
        age_min = (time.monotonic() - lg[0]) / 60.0
        logger.warning(
            f"[CACHE] {ticker}: fetch failed ({last_err}) — using last-good data "
            f"({age_min:.0f}m old) instead of empty/corrupt"
        )
        return lg[1].copy()

    logger.warning(f"[CACHE] {ticker}: fetch failed ({last_err}) and no cached fallback available")
    return pd.DataFrame()


def get_close_prices(ticker: str, period: str = "3mo") -> list[float]:
    """Convenience: return a flat list of closing prices."""
    df = get_ohlcv(ticker, period=period)
    if df.empty or "Close" not in df.columns:
        return []
    col = df["Close"]
    if hasattr(col, "squeeze"):
        col = col.squeeze()
    return [float(v) for v in col.dropna().values]


def prefetch_batch(tickers: list[str], period: str = "3mo") -> None:
    """
    Batch-download all tickers in ONE yfinance call, then populate the cache.

    yfinance batch download is ~5× faster than N individual downloads.
    Call this at the START of a scan cycle before individual lookups happen.
    """
    if not tickers:
        return

    try:
        import pandas as _pd
        df_all = yf.download(
            tickers,
            period=period,
            progress=False,
            auto_adjust=True,
        )

        now = time.monotonic()

        # yfinance returns (date, ticker) MultiIndex columns when fetching multiple
        for ticker in tickers:
            try:
                # Extract single ticker from batch result
                if isinstance(df_all.columns, pd.MultiIndex):
                    # MultiIndex: (OHLCV, ticker)
                    df_ticker = df_all.xs(ticker, axis=1, level=1, drop_level=True)
                else:
                    df_ticker = df_all  # single ticker returned flat

                if df_ticker.empty:
                    continue

                cache_key = f"{ticker}_{period}_1d"
                with _lock:
                    _df_copy = df_ticker.copy()
                    _cache[cache_key] = (now, _df_copy)
                    _last_good[cache_key] = (now, _df_copy)   # also keep as last-known-good

                logger.debug(f"[CACHE] prefetch: {ticker} loaded ({len(df_ticker)} rows)")

            except Exception as e:
                logger.debug(f"[CACHE] prefetch: {ticker} extract failed — {e}")

    except Exception as e:
        logger.warning(f"[CACHE] batch prefetch failed — {e}")


def get_cache_stats() -> dict:
    """Return hit/miss statistics."""
    with _lock:
        size = len(_cache)
    total = _stats["hits"] + _stats["misses"]
    hit_rate = _stats["hits"] / total * 100 if total else 0
    return {
        "cached_tickers": size,
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "hit_rate_pct": round(hit_rate, 1),
    }


def clear_cache() -> None:
    """Clear all cached data."""
    with _lock:
        _cache.clear()
    logger.info("[CACHE] Cache cleared")


def evict_expired() -> int:
    """Remove expired entries. Returns count of evicted entries."""
    now = time.monotonic()
    evicted = 0
    with _lock:
        expired_keys = [k for k, (ts, _) in _cache.items() if now - ts > _TTL_SECONDS]
        for k in expired_keys:
            del _cache[k]
            evicted += 1
    return evicted
