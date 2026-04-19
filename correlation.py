"""
Correlation Filter — Portfolio Concentration Guard

Why this matters:
  Buying NVDA when you already hold AMD creates hidden concentration risk —
  both stocks move together in GPU/AI news cycles. If the sector drops,
  you lose on both positions simultaneously, defeating the purpose of
  holding multiple names.

  A naive bot counts positions (e.g. "max 5 open trades") but ignores
  *economic* overlap. The correlation filter catches this.

What we compute:
  Pearson correlation of daily log-returns over the last LOOKBACK_DAYS.
  Threshold of 0.8 means "80% of daily price variance is shared" —
  these two assets are effectively the same bet.

  |corr| >= CORRELATION_THRESHOLD → block the new trade.

Fail-open policy:
  If yfinance data is unavailable (network error, rate limit, etc.)
  the check is skipped with a warning. We never block a trade purely
  due to a data-fetch failure.

Cache:
  Correlation between any two tickers is cached for CACHE_TTL seconds
  (default 30 min). Intraday re-computation is wasteful; correlations
  from daily returns don't change meaningfully in 30 minutes.

Public API
----------
  check(new_ticker)          → (blocked: bool, reason: str, details: dict)
  portfolio_matrix()         → dict  (N×N matrix of open-position correlations)
  get_status()               → dict  (for /status endpoint)

Environment variables
---------------------
  CORRELATION_THRESHOLD      float  default 0.8   (|r| >= this → block)
  CORRELATION_LOOKBACK_DAYS  int    default 60    (trading days of history)
  CORRELATION_MIN_PERIODS    int    default 30    (minimum observations required)
  CORRELATION_CACHE_TTL      int    default 1800  (seconds, 30 min)
"""

import logging
import math
import os
import threading
import time
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd

import database

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
THRESHOLD:    float = float(os.getenv("CORRELATION_THRESHOLD",     "0.8"))
LOOKBACK:     int   = int(os.getenv("CORRELATION_LOOKBACK_DAYS",  "60"))
MIN_PERIODS:  int   = int(os.getenv("CORRELATION_MIN_PERIODS",    "30"))
CACHE_TTL:    int   = int(os.getenv("CORRELATION_CACHE_TTL",      "1800"))

# ── Correlation cache ─────────────────────────────────────────────────────────
# key: (ticker_a, ticker_b) normalised alphabetically → (corr: float, ts: float)
_cache: dict[tuple[str, str], tuple[float, float]] = {}
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check(new_ticker: str) -> tuple[bool, str, dict]:
    """
    Check whether new_ticker is too correlated with any currently open position.

    Parameters
    ----------
    new_ticker : ticker symbol of the proposed new trade (e.g. "NVDA")

    Returns
    -------
    (blocked, reason, details)
        blocked = True  → skip this trade
        blocked = False → safe to proceed
        reason          → human-readable explanation
        details         → full correlation breakdown for logging / API response
    """
    new_ticker = new_ticker.upper()

    # ── Gather open position tickers ─────────────────────────────────────────
    open_trades  = database.get_open_trades()
    open_tickers = list({
        t["ticker"].upper()
        for t in open_trades
        if t.get("action") == "buy" and t["ticker"].upper() != new_ticker
    })

    if not open_tickers:
        return False, "no_open_positions", {
            "new_ticker": new_ticker,
            "open_tickers": [],
            "correlations": {},
            "max_correlation": None,
            "threshold": THRESHOLD,
        }

    # ── Compute correlations (cached) ─────────────────────────────────────────
    correlations: dict[str, float] = {}
    missing: list[str] = []

    for ot in open_tickers:
        cached = _get_cached(new_ticker, ot)
        if cached is not None:
            correlations[ot] = cached
            logger.debug(f"[CORR] {new_ticker}/{ot} = {cached:.4f} (cached)")
        else:
            missing.append(ot)

    if missing:
        # Fetch returns for new_ticker + all un-cached open tickers in one call
        to_fetch  = [new_ticker] + missing
        returns_df = _fetch_log_returns(to_fetch)

        if returns_df.empty or new_ticker not in returns_df.columns:
            logger.warning(
                f"[CORR] Could not fetch returns for {new_ticker} — skipping filter (fail-open)"
            )
            return False, "data_unavailable", {
                "new_ticker": new_ticker,
                "open_tickers": open_tickers,
                "correlations": correlations,
                "max_correlation": None,
                "threshold": THRESHOLD,
                "warning": "correlation data unavailable — check skipped",
            }

        new_series = returns_df[new_ticker]

        for ot in missing:
            if ot not in returns_df.columns:
                logger.warning(f"[CORR] No data for open position {ot} — skipping pair")
                continue

            pair = pd.concat([new_series, returns_df[ot]], axis=1).dropna()
            if len(pair) < MIN_PERIODS:
                logger.warning(
                    f"[CORR] Only {len(pair)} observations for {new_ticker}/{ot} "
                    f"(need {MIN_PERIODS}) — skipping pair"
                )
                continue

            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if math.isnan(corr):
                continue

            corr = round(corr, 4)
            correlations[ot] = corr
            _set_cached(new_ticker, ot, corr)
            logger.debug(f"[CORR] {new_ticker}/{ot} = {corr:.4f} (computed)")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if not correlations:
        # All pairs failed — fail-open
        return False, "all_pairs_failed", {
            "new_ticker": new_ticker,
            "open_tickers": open_tickers,
            "correlations": {},
            "max_correlation": None,
            "threshold": THRESHOLD,
            "warning": "all correlation pairs failed — check skipped",
        }

    # Rank by absolute correlation (symmetric: -0.9 is just as problematic as +0.9)
    ranked    = sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_name  = ranked[0][0]
    top_corr  = ranked[0][1]
    blocked   = abs(top_corr) >= THRESHOLD

    details = {
        "new_ticker":      new_ticker,
        "open_tickers":    open_tickers,
        "correlations":    {k: round(v, 4) for k, v in correlations.items()},
        "max_correlation": round(top_corr, 4),
        "correlated_with": top_name,
        "threshold":       THRESHOLD,
        "lookback_days":   LOOKBACK,
        "blocked":         blocked,
    }

    if blocked:
        reason = (
            f"{new_ticker} is {top_corr:+.2f} correlated with open position {top_name} "
            f"(threshold ±{THRESHOLD})"
        )
        logger.warning(f"[CORR] BLOCKED: {reason}")
    else:
        reason = f"max |r| = {top_corr:.2f} with {top_name} — below threshold {THRESHOLD}"
        logger.info(f"[CORR] OK: {new_ticker} — {reason}")

    return blocked, reason, details


def portfolio_matrix() -> dict:
    """
    Compute the full N×N correlation matrix for all currently open positions.
    Useful for /correlation endpoint and portfolio risk monitoring.
    """
    open_trades  = database.get_open_trades()
    open_tickers = list({
        t["ticker"].upper()
        for t in open_trades
        if t.get("action") == "buy"
    })

    if len(open_tickers) < 2:
        return {
            "tickers":   open_tickers,
            "matrix":    {},
            "computed_at": _utcnow(),
            "note": "need at least 2 open positions for a correlation matrix",
        }

    returns_df = _fetch_log_returns(open_tickers)
    if returns_df.empty:
        return {
            "tickers":     open_tickers,
            "matrix":      {},
            "computed_at": _utcnow(),
            "warning":     "data unavailable",
        }

    # Keep only tickers we got data for
    available = [t for t in open_tickers if t in returns_df.columns]
    corr_df   = returns_df[available].corr()

    matrix: dict[str, dict[str, float]] = {}
    for row in available:
        matrix[row] = {
            col: round(float(corr_df.loc[row, col]), 4)
            for col in available
        }

    # Find highest off-diagonal correlation
    max_corr = 0.0
    max_pair: tuple[str, str] = ("", "")
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            v = abs(matrix[a][b])
            if v > max_corr:
                max_corr = v
                max_pair = (a, b)

    return {
        "tickers":         available,
        "matrix":          matrix,
        "max_correlation": round(max_corr, 4),
        "max_pair":        list(max_pair),
        "threshold":       THRESHOLD,
        "lookback_days":   LOOKBACK,
        "computed_at":     _utcnow(),
    }


def get_status() -> dict:
    """Light-weight status dict for /status endpoint (no heavy computation)."""
    open_trades = database.get_open_trades()
    tickers = [t["ticker"].upper() for t in open_trades if t.get("action") == "buy"]
    with _cache_lock:
        cache_size = len(_cache)
    return {
        "threshold":     THRESHOLD,
        "lookback_days": LOOKBACK,
        "open_tickers":  tickers,
        "cache_entries": cache_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_log_returns(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch LOOKBACK days of daily log-returns for a list of tickers.
    Uses individual yf.Ticker calls (more robust than yf.download for small lists).

    Returns a DataFrame where each column = ticker, each row = daily log-return.
    Missing tickers are silently dropped.
    """
    closes: dict[str, pd.Series] = {}

    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period=f"{LOOKBACK}d", auto_adjust=True)
            if hist.empty or len(hist) < MIN_PERIODS:
                logger.warning(f"[CORR] Insufficient history for {t} ({len(hist)} bars)")
                continue
            closes[t] = hist["Close"].rename(t)
        except Exception as exc:
            logger.warning(f"[CORR] Failed to fetch {t}: {exc}")

    if not closes:
        return pd.DataFrame()

    df      = pd.concat(closes.values(), axis=1)
    # Log-returns: ln(P_t / P_{t-1}) — more stationary than simple returns
    returns = df.apply(lambda col: col.dropna().transform(lambda p: p / p.shift(1)).apply(
        lambda x: math.log(x) if x > 0 else float("nan")
    ))
    return returns.dropna(how="all")


def _cache_key(a: str, b: str) -> tuple[str, str]:
    """Normalise pair order so (A,B) and (B,A) share the same cache slot."""
    return (min(a, b), max(a, b))


def _get_cached(a: str, b: str) -> float | None:
    key = _cache_key(a, b)
    with _cache_lock:
        entry = _cache.get(key)
    if entry is None:
        return None
    corr, ts = entry
    if time.time() - ts > CACHE_TTL:
        with _cache_lock:
            _cache.pop(key, None)
        return None
    return corr


def _set_cached(a: str, b: str, corr: float) -> None:
    key = _cache_key(a, b)
    with _cache_lock:
        _cache[key] = (corr, time.time())


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
