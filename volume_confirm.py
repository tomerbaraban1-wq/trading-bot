"""
Volume Confirmation — Signal Validity Filter

Why volume matters:
  A price breakout on low volume is a false breakout.
  Institutional moves — the ones worth following — always come with
  elevated volume. If current volume is below the 20-bar average,
  the "breakout" is retail noise, not a real directional move.

Rule:
  current_volume >= VOLUME_MULTIPLIER × MA_20_volume
  → signal VALID
  → else REJECTED

We compare the current bar's volume to the rolling 20-period
volume moving average (SMA). The check uses intraday 5-minute bars
so it works during the trading day, not just at close.

Fail-open policy:
  If yfinance is unavailable, the check is skipped with a warning.
  We never block a trade purely due to a data-fetch failure.

Cache:
  Volume MA is cached per ticker for CACHE_TTL seconds (default 5 min)
  so rapid repeat checks within the same scan cycle don't re-fetch.

Public API:
  check(ticker)           → (passed: bool, reason: str, details: dict)
  get_current_ratio(ticker) → float | None  (current / MA ratio)

Environment variables:
  VOLUME_MULTIPLIER      float  default 0.8   (current must be >= 0.8× MA — lenient by design)
  VOLUME_MA_PERIOD       int    default 20    (bars in the MA)
  VOLUME_BAR_INTERVAL    str    default "5m"  (yfinance interval for intraday)
  VOLUME_CACHE_TTL       int    default 300   (seconds, 5 min)
"""

import logging
import os
import threading
import time

import yfinance as yf

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MULTIPLIER:   float = float(os.getenv("VOLUME_MULTIPLIER",   "0.8"))
MA_PERIOD:    int   = int(os.getenv("VOLUME_MA_PERIOD",      "20"))
BAR_INTERVAL: str   = os.getenv("VOLUME_BAR_INTERVAL",       "5m")
CACHE_TTL:    int   = int(os.getenv("VOLUME_CACHE_TTL",      "300"))

# ── Cache ─────────────────────────────────────────────────────────────────────
# ticker → (current_vol, ma_vol, ratio, timestamp)
_cache: dict[str, tuple[float, float, float, float]] = {}
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check(ticker: str) -> tuple[bool, str, dict]:
    """
    Validate that the current volume is sufficient to confirm the signal.

    Parameters
    ----------
    ticker : e.g. "AAPL"

    Returns
    -------
    (passed, reason, details)
      passed  = True  → volume confirms the signal, safe to proceed
      passed  = False → low volume, reject the signal
    """
    ticker = ticker.upper()

    current_vol, ma_vol, ratio = _get_volume_data(ticker)

    # Data unavailable → fail-open
    if current_vol is None or ma_vol is None or ma_vol == 0:
        logger.warning(
            f"[VOLUME] {ticker}: data unavailable — skipping check (fail-open)"
        )
        return True, "data_unavailable", {
            "ticker":      ticker,
            "passed":      True,
            "warning":     "volume data unavailable — check skipped",
            "multiplier":  MULTIPLIER,
            "ma_period":   MA_PERIOD,
        }

    passed = ratio >= MULTIPLIER

    details = {
        "ticker":        ticker,
        "current_volume": int(current_vol),
        "ma20_volume":   int(ma_vol),
        "ratio":         round(ratio, 3),
        "threshold":     MULTIPLIER,
        "ma_period":     MA_PERIOD,
        "bar_interval":  BAR_INTERVAL,
        "passed":        passed,
    }

    if passed:
        reason = (
            f"volume {ratio:.2f}× MA{MA_PERIOD} "
            f"({int(current_vol):,} vs {int(ma_vol):,}) — confirmed"
        )
        logger.info(f"[VOLUME] {ticker}: ✅ {reason}")
    else:
        reason = (
            f"volume only {ratio:.2f}× MA{MA_PERIOD} "
            f"({int(current_vol):,} vs {int(ma_vol):,}) — "
            f"need ≥ {MULTIPLIER}×"
        )
        logger.info(f"[VOLUME] {ticker}: ❌ {reason}")

    return passed, reason, details


def get_current_ratio(ticker: str) -> float | None:
    """Return current_volume / MA_volume ratio, or None if data unavailable."""
    _, _, ratio = _get_volume_data(ticker.upper())
    return ratio if ratio is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_volume_data(ticker: str) -> tuple[float | None, float | None, float | None]:
    """
    Fetch current bar volume and 20-bar MA volume.
    Returns (current_vol, ma_vol, ratio) — all None on failure.
    Uses a 5-minute cache to avoid hammering yfinance.
    """
    now = time.time()

    # Check cache
    with _cache_lock:
        entry = _cache.get(ticker)
    if entry and now - entry[3] < CACHE_TTL:
        return entry[0], entry[1], entry[2]

    try:
        hist = yf.Ticker(ticker).history(
            period="5d",
            interval=BAR_INTERVAL,
            auto_adjust=True,
        )

        if hist.empty or "Volume" not in hist.columns:
            raise ValueError("empty or missing Volume column")

        vol_series = hist["Volume"].dropna()

        if len(vol_series) < MA_PERIOD + 1:
            raise ValueError(
                f"only {len(vol_series)} bars — need {MA_PERIOD + 1}"
            )

        # Current bar = most recent completed bar
        current_vol = float(vol_series.iloc[-1])

        # MA over the preceding MA_PERIOD bars (exclude current bar)
        ma_vol = float(vol_series.iloc[-(MA_PERIOD + 1):-1].mean())

        if ma_vol <= 0:
            raise ValueError(f"MA volume = {ma_vol} (invalid)")

        ratio = round(current_vol / ma_vol, 4)

    except Exception as exc:
        logger.warning(f"[VOLUME] {ticker}: fetch failed — {exc}")
        return None, None, None

    with _cache_lock:
        _cache[ticker] = (current_vol, ma_vol, ratio, now)

    logger.debug(
        f"[VOLUME] {ticker}: current={int(current_vol):,} "
        f"MA{MA_PERIOD}={int(ma_vol):,} ratio={ratio:.3f}"
    )
    return current_vol, ma_vol, ratio
