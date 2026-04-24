"""
Market Regime Filter — ADX-Based Trend Detection

Why it matters:
  Trend-following strategies (momentum breakouts, high-composite-score buys) work
  best in trending markets. When the market is ranging/choppy, these signals produce
  more false positives than real breakouts. The ADX filter lets the bot recognise
  which environment it's in and act accordingly.

Rule:
  ADX >= ADX_THRESHOLD (default 25) → TRENDING  → allow trend-following trades
  ADX <  ADX_THRESHOLD              → RANGING   → skip (or flag for mean-reversion)

ADX formula (Wilder's method):
  +DM = max(high − prev_high, 0)  when > |low − prev_low|, else 0
  −DM = max(prev_low − low, 0)    when > (high − prev_high), else 0
  TR  = max(high−low, |high−prev_close|, |low−prev_close|)
  +DI = 100 × EWM(+DM) / EWM(TR)
  −DI = 100 × EWM(−DM) / EWM(TR)
  DX  = 100 × |+DI − −DI| / (+DI + −DI)
  ADX = EWM(DX, period)

Fail-open policy:
  If yfinance is unavailable, return "trending" so the trade is NOT blocked
  by a data failure. A warning is logged.

Cache:
  ADX is computed from daily bars and cached per ticker for ADX_CACHE_TTL seconds
  (default 30 min). ADX changes slowly — no need to recompute every scan cycle.

Public API:
  get_regime(ticker) → (regime: str, adx: float, details: dict)
  is_trending(ticker) → bool

Environment variables:
  ADX_PERIOD          int    default 14    (EWM span)
  ADX_THRESHOLD       float  default 25.0  (ADX below this = ranging)
  ADX_LOOKBACK_DAYS   int    default 60    (daily bars to fetch)
  ADX_CACHE_TTL       int    default 1800  (seconds, 30 min)
"""

import logging
import os
import threading
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ADX_PERIOD:        int   = int(os.getenv("ADX_PERIOD",        "14"))
ADX_THRESHOLD:     float = float(os.getenv("ADX_THRESHOLD",   "18.0"))
ADX_LOOKBACK_DAYS: int   = int(os.getenv("ADX_LOOKBACK_DAYS", "60"))
ADX_CACHE_TTL:     int   = int(os.getenv("ADX_CACHE_TTL",     "1800"))  # 30 min

# ── Cache — ticker → (adx, timestamp, regime_str) ─────────────────────────────
_cache: dict[str, tuple[float, float, str]] = {}
_lock   = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_regime(ticker: str) -> tuple[str, float, dict]:
    """
    Determine the market regime for a ticker using the ADX indicator.

    Parameters
    ----------
    ticker : e.g. "AAPL"

    Returns
    -------
    (regime, adx, details)
      regime  : "trending" | "ranging"
      adx     : float ADX value (0–100)
      details : breakdown dict for API / logging
    """
    ticker = ticker.upper()

    # ── Cache hit ──────────────────────────────────────────────────────────────
    now = time.time()
    with _lock:
        entry = _cache.get(ticker)
    if entry and now - entry[1] < ADX_CACHE_TTL:
        adx, _, regime = entry
        return regime, adx, {
            "ticker": ticker, "adx": round(adx, 2),
            "threshold": ADX_THRESHOLD, "regime": regime,
            "is_trending": regime == "trending", "cached": True,
        }

    # ── Compute ADX ───────────────────────────────────────────────────────────
    try:
        hist = yf.Ticker(ticker).history(
            period=f"{ADX_LOOKBACK_DAYS}d", auto_adjust=True
        )
        if hist.empty or len(hist) < ADX_PERIOD + 2:
            raise ValueError(
                f"only {len(hist)} bars — need {ADX_PERIOD + 2}"
            )
        adx = _compute_adx(hist)
        if adx < 0 or adx > 100:
            raise ValueError(f"ADX={adx:.2f} out of range")

    except Exception as exc:
        logger.warning(
            f"[ADX] {ticker}: computation failed ({exc}) — "
            f"assuming trending (fail-open)"
        )
        return "trending", 0.0, {
            "ticker": ticker, "adx": None,
            "error": str(exc), "regime": "trending (fail-open)",
            "is_trending": True,
        }

    regime = "trending" if adx >= ADX_THRESHOLD else "ranging"

    with _lock:
        _cache[ticker] = (adx, now, regime)

    emoji = "📈" if regime == "trending" else "↔️"
    logger.info(
        f"[ADX] {ticker}: ADX={adx:.1f} {emoji} {regime.upper()} "
        f"(threshold={ADX_THRESHOLD})"
    )

    details = {
        "ticker":      ticker,
        "adx":         round(adx, 2),
        "threshold":   ADX_THRESHOLD,
        "regime":      regime,
        "is_trending": regime == "trending",
        "cached":      False,
    }
    return regime, adx, details


def is_trending(ticker: str) -> bool:
    """
    Return True if market for this ticker is trending (ADX >= threshold).
    Fail-open: returns True when data is unavailable.
    """
    regime, _, _ = get_regime(ticker)
    return regime == "trending"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_adx(hist: pd.DataFrame) -> float:
    """
    Compute ADX_N using Wilder's EWM smoothing.
    Returns ADX as a float 0–100.
    """
    high  = hist["High"]
    low   = hist["Low"]
    close = hist["Close"]

    # ── True Range ────────────────────────────────────────────────────────────
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low  - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # ── Directional Movement ──────────────────────────────────────────────────
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )

    # ── Wilder's EWM (alpha = 1 / period) ────────────────────────────────────
    alpha = 1.0 / ADX_PERIOD
    tr_smooth    = tr.ewm(alpha=alpha,       adjust=False).mean()
    plus_smooth  = plus_dm.ewm(alpha=alpha,  adjust=False).mean()
    minus_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # ── +DI, −DI ──────────────────────────────────────────────────────────────
    safe_tr  = tr_smooth.replace(0, float("nan"))
    plus_di  = 100 * plus_smooth  / safe_tr
    minus_di = 100 * minus_smooth / safe_tr

    # ── DX → ADX ─────────────────────────────────────────────────────────────
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx     = 100 * (plus_di - minus_di).abs() / di_sum
    adx    = dx.ewm(alpha=alpha, adjust=False).mean()

    return float(adx.iloc[-1])
