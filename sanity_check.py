"""
Sanity Check — Pre-Trade Data Validation

Before placing any order the bot must verify that:

1. Price plausibility  — current price is close to the signal price.
   A large gap signals stale data, a flash-crash, or a fat-finger.

2. Price velocity      — price didn't move too far too fast.
   Compares two yfinance snapshots separated by ~1 second.
   Sudden moves trigger a "cooling-off" hold.

3. Data completeness   — required indicator fields are present and finite.
   Missing ATR, RSI, or volume = unreliable signal → reject.

4. Spread sanity       — bid/ask spread is within a reasonable range.
   An abnormally wide spread signals thin liquidity or broken quotes.

All checks are logged with full detail so every rejection is auditable.

Environment variables:
    SANITY_MAX_PRICE_DRIFT_PCT   float  default 1.0   (% gap signal vs live)
    SANITY_MAX_VELOCITY_PCT      float  default 2.0   (% move in 1 second)
    SANITY_MAX_SPREAD_PCT        float  default 0.5   (% bid/ask spread)
    SANITY_VELOCITY_SAMPLES      int    default 2     (snapshots for velocity)
    SANITY_CONFIRM_DELAY_SEC     float  default 1.5   (seconds between samples)
"""

import os
import time
import math
import logging
import threading
import yfinance as yf

logger = logging.getLogger(__name__)

MAX_PRICE_DRIFT_PCT:        float = float(os.getenv("SANITY_MAX_PRICE_DRIFT_PCT",    "1.0"))
MAX_VELOCITY_PCT:           float = float(os.getenv("SANITY_MAX_VELOCITY_PCT",      "2.0"))
MAX_SPREAD_PCT:             float = float(os.getenv("SANITY_MAX_SPREAD_PCT",        "0.5"))
CONFIRM_DELAY_SEC:          float = float(os.getenv("SANITY_CONFIRM_DELAY_SEC",     "1.5"))
MAX_CROSS_EXCHANGE_PCT:     float = float(os.getenv("SANITY_MAX_CROSS_EXCHANGE_PCT","0.5"))

# Required indicator keys for a BUY signal to be considered complete
REQUIRED_FIELDS = ("rsi", "macd", "volume_ratio", "atr")

# ── Price snapshot cache (used for velocity check) ────────────────────────────
# ticker → (price, timestamp)
_snap_lock = threading.Lock()
_snapshots: dict[str, tuple[float, float]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class SanityError(Exception):
    """Raised when a sanity check fails. Contains a human-readable reason."""


def run_all(
    ticker:        str,
    signal_price:  float,
    indicators:    dict | None = None,
) -> tuple[bool, str]:
    """
    Run all sanity checks for a given ticker before placing a BUY order.

    Parameters
    ----------
    ticker        : e.g. "AAPL"
    signal_price  : the price embedded in the TradingView/auto-invest signal
    indicators    : output of indicators.get_current_indicators() (optional but recommended)

    Returns
    -------
    (passed: bool, reason: str)
    passed=True  → safe to proceed
    passed=False → do NOT trade; reason explains why
    """
    ticker = ticker.upper()
    checks = [
        _check_data_completeness,
        _check_price_plausibility,
        _check_multi_source_price,
        _check_price_velocity,
        _check_spread,
    ]

    for check in checks:
        try:
            ok, reason = check(ticker, signal_price, indicators)
            if not ok:
                logger.warning(
                    f"[SANITY FAIL] {ticker} | check={check.__name__} | reason={reason}"
                )
                return False, reason
        except Exception as exc:
            # A crashed check is itself a sanity failure
            msg = f"{check.__name__} raised: {exc}"
            logger.error(f"[SANITY ERROR] {ticker} | {msg}")
            return False, msg

    logger.info(f"[SANITY OK] {ticker} | all checks passed")
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_data_completeness(
    ticker: str,
    signal_price: float,
    indicators: dict | None,
) -> tuple[bool, str]:
    """
    Reject if any required indicator field is missing or non-finite.
    """
    if signal_price <= 0:
        return False, f"signal_price={signal_price} is invalid"

    if indicators is None:
        # No indicator dict → warn but don't hard-reject (auto-invest path)
        logger.warning(f"[SANITY] {ticker}: no indicator dict provided — skipping completeness check")
        return True, "no_indicators"

    missing = []
    for field in REQUIRED_FIELDS:
        val = indicators.get(field)
        if val is None:
            missing.append(field)
        elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            missing.append(f"{field}=invalid({val})")

    if missing:
        return False, f"missing/invalid indicator fields: {missing}"

    return True, "complete"


def _check_price_plausibility(
    ticker: str,
    signal_price: float,
    indicators: dict | None,
) -> tuple[bool, str]:
    """
    Fetch live price and ensure it hasn't drifted more than MAX_PRICE_DRIFT_PCT
    from the signal price.

    A large drift means the signal was generated on stale data and the
    market has already moved past the entry opportunity (or worse —
    into a flash-crash).
    """
    live_price = _get_live_price(ticker)
    if live_price is None:
        return False, "could not fetch live price from exchange"

    drift_pct = abs(live_price - signal_price) / signal_price * 100
    if drift_pct > MAX_PRICE_DRIFT_PCT:
        return False, (
            f"price drift {drift_pct:.2f}% exceeds limit {MAX_PRICE_DRIFT_PCT}% "
            f"(signal=${signal_price:.4f} live=${live_price:.4f})"
        )

    logger.debug(f"[SANITY] {ticker}: price drift={drift_pct:.3f}% ✅")
    return True, f"drift={drift_pct:.3f}%"


def _check_multi_source_price(
    ticker: str,
    signal_price: float,
    indicators: dict | None,
) -> tuple[bool, str]:
    """
    Cross-validate the live price across two independent data sources:
      Source A — yfinance (1-minute bar close)
      Source B — Alpaca broker API (last_trade_price)

    If the two sources disagree by more than MAX_CROSS_EXCHANGE_PCT the
    price data is unreliable — there may be a feed error, stale quote,
    or genuine market dislocation.

    Fail-open: if EITHER source is unavailable, the check is skipped.
    Both sources must be available AND disagree to block the trade.
    """
    # Source A: yfinance
    yf_price = _get_live_price(ticker)
    if yf_price is None:
        logger.warning(f"[SANITY] {ticker}: multi-source check — yfinance unavailable, skipping")
        return True, "multi_source_yf_unavailable"

    # Source B: Alpaca broker API (lazy import to avoid circular dependency)
    try:
        import broker as _broker
        alpaca_price = _broker.get_price(ticker)
    except Exception as exc:
        logger.warning(
            f"[SANITY] {ticker}: multi-source check — broker API failed ({exc}), skipping"
        )
        return True, "multi_source_broker_unavailable"

    if not alpaca_price or alpaca_price <= 0:
        logger.debug(f"[SANITY] {ticker}: broker price unavailable — skipping multi-source check")
        return True, "multi_source_broker_unavailable"

    mid        = (yf_price + alpaca_price) / 2
    drift_pct  = abs(yf_price - alpaca_price) / mid * 100 if mid > 0 else 0

    if drift_pct > MAX_CROSS_EXCHANGE_PCT:
        return False, (
            f"multi-source price discrepancy {drift_pct:.2f}% exceeds "
            f"limit {MAX_CROSS_EXCHANGE_PCT}% "
            f"(yfinance=${yf_price:.4f}, broker=${alpaca_price:.4f})"
        )

    logger.debug(
        f"[SANITY] {ticker}: multi-source drift={drift_pct:.3f}% ✅ "
        f"(yf=${yf_price:.4f} broker=${alpaca_price:.4f})"
    )
    return True, f"multi_source_drift={drift_pct:.3f}%"


def _check_price_velocity(
    ticker: str,
    signal_price: float,
    indicators: dict | None,
) -> tuple[bool, str]:
    """
    Take two price snapshots separated by CONFIRM_DELAY_SEC and measure
    how fast the price is moving.

    If the price moves more than MAX_VELOCITY_PCT in that window the
    asset is in a momentum spike/crash and we must stand aside.
    """
    snap1 = _get_live_price(ticker)
    if snap1 is None:
        return False, "velocity check: could not get first price snapshot"

    time.sleep(CONFIRM_DELAY_SEC)   # blocking — called via asyncio.to_thread

    snap2 = _get_live_price(ticker)
    if snap2 is None:
        return False, "velocity check: could not get second price snapshot"

    velocity_pct = abs(snap2 - snap1) / snap1 * 100 if snap1 > 0 else 0

    # Store latest snapshot for future calls
    with _snap_lock:
        _snapshots[ticker] = (snap2, time.time())

    if velocity_pct > MAX_VELOCITY_PCT:
        return False, (
            f"price velocity {velocity_pct:.3f}%/{CONFIRM_DELAY_SEC:.1f}s "
            f"exceeds limit {MAX_VELOCITY_PCT}% "
            f"(${snap1:.4f} → ${snap2:.4f})"
        )

    logger.debug(
        f"[SANITY] {ticker}: velocity={velocity_pct:.4f}%/{CONFIRM_DELAY_SEC:.1f}s ✅"
    )
    return True, f"velocity={velocity_pct:.4f}%"


def _check_spread(
    ticker: str,
    signal_price: float,
    indicators: dict | None,
) -> tuple[bool, str]:
    """
    Verify bid/ask spread is within reasonable bounds.
    A wide spread signals illiquid or broken market conditions.

    Falls back gracefully if the broker doesn't expose bid/ask.
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.fast_info

        bid = float(getattr(info, "bid",       None) or 0)
        ask = float(getattr(info, "ask",       None) or 0)

        if bid <= 0 or ask <= 0:
            # fast_info doesn't always have bid/ask → skip silently
            logger.debug(f"[SANITY] {ticker}: bid/ask unavailable — skipping spread check")
            return True, "spread_unavailable"

        mid_price  = (bid + ask) / 2
        spread_pct = (ask - bid) / mid_price * 100 if mid_price > 0 else 0

        if spread_pct > MAX_SPREAD_PCT:
            return False, (
                f"spread {spread_pct:.3f}% exceeds limit {MAX_SPREAD_PCT}% "
                f"(bid=${bid:.4f} ask=${ask:.4f})"
            )

        logger.debug(f"[SANITY] {ticker}: spread={spread_pct:.4f}% ✅")
        return True, f"spread={spread_pct:.4f}%"

    except Exception as exc:
        # Non-fatal — spread check is best-effort
        logger.warning(f"[SANITY] {ticker}: spread check failed ({exc}) — skipping")
        return True, "spread_check_error"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_live_price(ticker: str) -> float | None:
    """Fetch latest trade price via yfinance 1-minute bar."""
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            if price > 0:
                return price
        info  = t.fast_info
        price = float(getattr(info, "last_price", None) or 0)
        return price if price > 0 else None
    except Exception as exc:
        logger.warning(f"[SANITY] _get_live_price({ticker}) failed: {exc}")
        return None
