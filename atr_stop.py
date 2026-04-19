"""
ATR-Based Trailing Stop — Dynamic Exit Engine

Problem with fixed stops:
  STOP_LOSS_PCT = 5% treats TSLA (daily ATR ~4%) the same as WMT (ATR ~0.5%).
  TSLA triggers the stop on any normal intraday move, locking in unnecessary
  losses. WMT gets 10× more room than it needs — you ride the loss down.

Hedge fund approach — ATR trailing stop:
  ┌─────────────────────────────────────────────────────────────────┐
  │  stop_distance = ATR_14  ×  ATR_MULTIPLIER                     │
  │                                                                 │
  │  clamped to:  [entry × MIN_STOP_PCT, entry × MAX_STOP_PCT]     │
  │                                                                 │
  │  Trailing rule:  stop = max(stop, high_watermark − stop_dist)  │
  │                  (stop can only move UP, never down)            │
  └─────────────────────────────────────────────────────────────────┘

  Example — TSLA @ $250, ATR=$10, multiplier=2.0:
    Initial stop:  $250 − $20 = $230  (8% room — appropriate for ATR=4%)
    Price → $290:  stop → $270         (locked in $20 above entry)
    Price → $310:  stop → $290         (locked in $60 profit zone)
    Price → $285:  $285 < $290 → EXIT  (trailing stop hit, booked $35/share)

  Example — WMT @ $80, ATR=$0.80, multiplier=2.0:
    Initial stop:  $80 − $1.60 = $78.40  (2% room — appropriate for ATR=1%)
    Tight stop cuts losses fast on a stable stock.

Persistence:
  Both atr_stop_price and high_watermark survive restarts via SQLite
  (trade_log columns atr_stop_price, high_watermark).

ATR cache:
  ATR is computed from daily bars and cached for 1 hour.
  Intraday ATR doesn't change meaningfully between monitor cycles.

Environment variables:
  ATR_STOP_MULTIPLIER   float  default 2.0   (ATR × multiplier = stop distance)
  ATR_STOP_MIN_PCT      float  default 1.5   (floor: never tighter than 1.5%)
  ATR_STOP_MAX_PCT      float  default 8.0   (ceiling: never wider than 8%)
  ATR_STOP_PERIOD       int    default 14    (ATR look-back period)
  ATR_STOP_CACHE_TTL    int    default 3600  (seconds, 1 hour)
"""

import logging
import math
import os
import threading
import time

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MULTIPLIER:  float = float(os.getenv("ATR_STOP_MULTIPLIER", "2.0"))
MIN_STOP_PCT: float = float(os.getenv("ATR_STOP_MIN_PCT",   "1.5"))  # %
MAX_STOP_PCT: float = float(os.getenv("ATR_STOP_MAX_PCT",   "8.0"))  # %
ATR_PERIOD:  int   = int(os.getenv("ATR_STOP_PERIOD",       "14"))
CACHE_TTL:   int   = int(os.getenv("ATR_STOP_CACHE_TTL",    "3600"))  # 1 hour

# ── ATR cache ─────────────────────────────────────────────────────────────────
# ticker → (atr_dollars: float, timestamp: float)
_atr_cache: dict[str, tuple[float, float]] = {}
_atr_lock   = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_initial_stop(ticker: str, entry_price: float) -> tuple[float, dict]:
    """
    Compute the initial ATR-based stop price when a trade opens.

    Parameters
    ----------
    ticker       : e.g. "TSLA"
    entry_price  : actual fill price of the BUY order

    Returns
    -------
    (stop_price, metadata)
      stop_price  : the dollar price below which we exit
      metadata    : dict with full breakdown for logging/audit
    """
    ticker = ticker.upper()
    atr    = _fetch_atr(ticker, entry_price)

    raw_stop_dist = atr * MULTIPLIER

    # Clamp distance to [MIN_STOP_PCT%, MAX_STOP_PCT%] of entry
    min_dist = entry_price * MIN_STOP_PCT / 100
    max_dist = entry_price * MAX_STOP_PCT / 100
    stop_dist = max(min_dist, min(max_dist, raw_stop_dist))

    stop_price = round(entry_price - stop_dist, 4)
    stop_pct   = round(stop_dist / entry_price * 100, 3)

    meta = {
        "atr":             round(atr, 4),
        "multiplier":      MULTIPLIER,
        "raw_stop_dist":   round(raw_stop_dist, 4),
        "stop_dist":       round(stop_dist, 4),
        "stop_pct":        stop_pct,
        "stop_price":      stop_price,
        "entry_price":     entry_price,
        "high_watermark":  entry_price,
        "binding":         (
            "min_pct" if stop_dist == min_dist else
            "max_pct" if stop_dist == max_dist else
            "atr"
        ),
    }

    logger.info(
        f"[ATR STOP] {ticker}: entry=${entry_price:.2f} | ATR=${atr:.4f} | "
        f"stop=${stop_price:.2f} ({stop_pct:.2f}% from entry) | "
        f"binding={meta['binding']}"
    )
    return stop_price, meta


def update_trailing_stop(
    ticker:          str,
    current_price:   float,
    current_stop:    float | None,
    high_watermark:  float,
    entry_price:     float,
) -> tuple[float, float, bool]:
    """
    Trail the stop upward as price rises.

    Called every monitoring cycle (default: every 60 seconds).

    Parameters
    ----------
    ticker          : e.g. "TSLA"
    current_price   : latest market price from broker
    current_stop    : last persisted stop_price (None → re-initialise)
    high_watermark  : highest price seen since entry
    entry_price     : original fill price

    Returns
    -------
    (new_stop, new_watermark, was_raised)
      new_stop       : updated stop price (always >= current_stop)
      new_watermark  : updated peak price seen
      was_raised     : True if stop moved up (for logging)
    """
    ticker = ticker.upper()
    atr    = _fetch_atr(ticker, entry_price)

    # Update high watermark
    new_wm = max(high_watermark, current_price)

    # Compute candidate stop from the new watermark
    raw_dist  = atr * MULTIPLIER
    min_dist  = entry_price * MIN_STOP_PCT / 100
    max_dist  = entry_price * MAX_STOP_PCT / 100
    stop_dist = max(min_dist, min(max_dist, raw_dist))

    candidate_stop = round(new_wm - stop_dist, 4)

    # Trailing rule: stop can only move UP
    if current_stop is None:
        new_stop = candidate_stop
        was_raised = True
    else:
        new_stop   = round(max(current_stop, candidate_stop), 4)
        was_raised = new_stop > current_stop

    if was_raised:
        logger.debug(
            f"[ATR STOP] {ticker}: stop raised ${current_stop} → ${new_stop} "
            f"| price=${current_price:.2f} | wm=${new_wm:.2f} | ATR=${atr:.4f}"
        )

    return new_stop, new_wm, was_raised


def should_exit(current_price: float, stop_price: float) -> bool:
    """Return True if current_price has breached the trailing stop."""
    return current_price <= stop_price


def should_exit_confirmed(
    ticker:        str,
    current_price: float,
    stop_price:    float,
    bar_interval:  str = "5m",
) -> tuple[bool, str]:
    """
    Flash-Crash Protection — 'Close on Confirmation' stop logic.

    Problem: a standard stop-loss fires on any tick below the threshold,
    including brief wicks that immediately recover. Flash crashes can
    shake out long positions that would have been profitable seconds later.

    Solution: only exit if BOTH conditions are true:
      1. Current price (tick) is below the stop level
      2. The last COMPLETED candle's close is also below the stop level

    If condition 1 is false → return (False, "above_stop")  [normal case]
    If condition 1 true but condition 2 false → hold, log a warning
    If condition 1 and 2 both true → confirmed exit

    Fail towards exiting: if candle data is unavailable, treat as confirmed
    (standard stop behaviour — don't suppress exits on data errors).

    Parameters
    ----------
    ticker        : e.g. "TSLA"
    current_price : latest tick from broker
    stop_price    : current trailing stop price
    bar_interval  : yfinance interval for confirmation candle (default "5m")

    Returns
    -------
    (should_exit, reason)
    """
    # Fast path — price above stop, nothing to do
    if current_price > stop_price:
        return False, f"price=${current_price:.2f} > stop=${stop_price:.2f}"

    # Price is at or below stop — check last completed candle for confirmation
    try:
        hist = yf.Ticker(ticker).history(
            period="1d", interval=bar_interval, auto_adjust=True
        )
        if hist.empty or len(hist) < 2:
            # Can't confirm — fail open (exit as normal)
            logger.warning(
                f"[FLASH] {ticker}: candle data unavailable — "
                f"applying standard stop exit"
            )
            return True, "candle_data_unavailable — standard exit applied"

        # Use [-2]: the LAST COMPLETED candle ([-1] may still be forming)
        last_close = float(hist["Close"].iloc[-2])

        if last_close <= stop_price:
            reason = (
                f"CONFIRMED: closed candle @ ${last_close:.2f} ≤ stop ${stop_price:.2f}"
            )
            logger.info(f"[FLASH] {ticker}: exit {reason}")
            return True, reason
        else:
            reason = (
                f"FLASH GUARD: tick=${current_price:.2f} < stop=${stop_price:.2f} "
                f"but last candle closed @ ${last_close:.2f} — waiting for close confirmation"
            )
            logger.warning(f"[FLASH] {ticker}: {reason}")
            return False, reason

    except Exception as exc:
        logger.warning(
            f"[FLASH] {ticker}: candle check failed ({exc}) — "
            f"applying standard stop exit"
        )
        return True, f"candle_check_error — standard exit applied"


def get_stop_summary(trade: dict) -> dict:
    """
    Return a human-readable stop summary for a trade dict (from database).
    Safe to call with legacy trades that have no atr_stop_price.
    """
    entry       = trade.get("entry_price", 0)
    stop        = trade.get("atr_stop_price")
    watermark   = trade.get("high_watermark", entry)

    if stop is None or entry <= 0:
        return {"type": "atr_trailing", "status": "not_initialized"}

    stop_pct    = round((entry - stop) / entry * 100, 2)
    locked_pct  = round((watermark - entry) / entry * 100, 2) if entry else 0
    return {
        "type":          "atr_trailing",
        "stop_price":    round(stop, 4),
        "high_wm":       round(watermark, 4),
        "entry_price":   round(entry, 4),
        "stop_pct_from_entry": stop_pct,
        "locked_profit_pct":   max(locked_pct, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ATR computation
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_atr(ticker: str, fallback_price: float) -> float:
    """
    Compute ATR_14 in dollar terms.
    Falls back to MAX_STOP_PCT × price / (multiplier) on error
    so the stop is always set even when yfinance is down.
    """
    # Check cache
    now = time.time()
    with _atr_lock:
        entry = _atr_cache.get(ticker)
    if entry and now - entry[1] < CACHE_TTL:
        return entry[0]

    try:
        hist = yf.Ticker(ticker).history(period="60d", auto_adjust=True)
        if hist.empty or len(hist) < ATR_PERIOD + 1:
            raise ValueError(f"Insufficient history: {len(hist)} bars")

        high  = hist["High"]
        low   = hist["Low"]
        close = hist["Close"]

        hl   = high - low
        hc   = (high - close.shift(1)).abs()
        lc   = (low  - close.shift(1)).abs()
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr  = float(tr.ewm(span=ATR_PERIOD, adjust=False).mean().iloc[-1])

        if atr <= 0 or math.isnan(atr):
            raise ValueError(f"ATR={atr} invalid")

    except Exception as exc:
        # Fallback: use midpoint of [min, max] stop distance
        mid_pct = (MIN_STOP_PCT + MAX_STOP_PCT) / 2 / 100
        atr = fallback_price * mid_pct / MULTIPLIER
        logger.warning(
            f"[ATR STOP] {ticker}: ATR fetch failed ({exc}) — "
            f"using fallback ATR=${atr:.4f}"
        )

    with _atr_lock:
        _atr_cache[ticker] = (atr, now)

    logger.debug(f"[ATR STOP] {ticker}: ATR_14 = ${atr:.4f}")
    return atr
