"""
Iceberg Orders — TWAP Execution Engine

Problem:
  A large market order moves the price against you.
  If you buy 500 shares of a thinly-traded stock at once:
    - Your own order exhausts the best ask levels
    - Market makers see the size and widen spreads
    - You get a worse average fill than the quoted price

Hedge fund solution — Iceberg / TWAP:
  Split the total order into N equal child orders ("slices"),
  executed at fixed intervals with random jitter so the pattern
  is not predictable by HFT algos.

  Each slice:
    1. Re-fetches the live price (price may have moved since last slice)
    2. Computes a fresh ATR-based limit price
    3. Submits a small limit order
    4. Waits ICEBERG_INTERVAL_SEC ± jitter before the next slice

  Result: a Time-Weighted Average Price (TWAP) that approximates the
  market price over the execution window, without spiking any single
  ask level.

Trigger condition (configurable via env):
  Iceberg activates when EITHER:
    qty  >= ICEBERG_THRESHOLD_SHARES  (default: 50 shares)
    OR
    qty * price >= ICEBERG_THRESHOLD_USD  (default: $2,000 notional)

Public API:
  should_use_iceberg(qty, price)   → bool
  iceberg_buy(ticker, qty, price)  → dict  (same shape as broker.submit_buy)
  get_status()                     → dict  (for /status endpoint)

Environment variables:
  ICEBERG_THRESHOLD_SHARES  int    default 50      trigger on share count
  ICEBERG_THRESHOLD_USD     float  default 2000    trigger on notional ($)
  ICEBERG_SLICE_SHARES      int    default 10      shares per child order
  ICEBERG_MIN_SLICES        int    default 2       minimum slices when triggered
  ICEBERG_MAX_SLICES        int    default 20      safety cap
  ICEBERG_INTERVAL_SEC      float  default 30      seconds between slices
  ICEBERG_JITTER_PCT        float  default 0.3     ±30% random interval jitter
  ICEBERG_TIMEOUT_SEC       float  default 20      per-slice broker timeout
"""

import asyncio
import logging
import math
import os
import random
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Any

import broker
from slippage import limit_buy_price

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
THRESHOLD_SHARES: int   = int(os.getenv("ICEBERG_THRESHOLD_SHARES", "50"))
THRESHOLD_USD:    float = float(os.getenv("ICEBERG_THRESHOLD_USD",   "2000"))
SLICE_SHARES:     int   = int(os.getenv("ICEBERG_SLICE_SHARES",      "10"))
MIN_SLICES:       int   = int(os.getenv("ICEBERG_MIN_SLICES",         "2"))
MAX_SLICES:       int   = int(os.getenv("ICEBERG_MAX_SLICES",        "20"))
INTERVAL_SEC:     float = float(os.getenv("ICEBERG_INTERVAL_SEC",    "30"))
JITTER_PCT:       float = float(os.getenv("ICEBERG_JITTER_PCT",      "0.3"))
TIMEOUT_SEC:      float = float(os.getenv("ICEBERG_TIMEOUT_SEC",     "20"))


# ── Live execution tracker (for /status) ──────────────────────────────────────
@dataclass
class _ActiveIceberg:
    ticker:        str
    total_qty:     float
    slices_total:  int
    slices_done:   int   = 0
    filled_qty:    float = 0.0
    avg_price:     float = 0.0
    started_at:    float = field(default_factory=time.time)
    last_slice_at: float = 0.0
    status:        str   = "running"   # running | done | partial

_active: dict[str, _ActiveIceberg] = {}
_active_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def should_use_iceberg(qty: float, price: float) -> bool:
    """
    Return True if this order is large enough to warrant iceberg splitting.

    Triggers when EITHER condition is met:
      - qty         >= ICEBERG_THRESHOLD_SHARES
      - qty × price >= ICEBERG_THRESHOLD_USD
    """
    if qty < 1:
        return False
    notional = qty * price
    return qty >= THRESHOLD_SHARES or notional >= THRESHOLD_USD


async def iceberg_buy(
    ticker:        str,
    total_qty:     float,
    initial_price: float,
) -> dict[str, Any]:
    """
    Execute a BUY order using iceberg slicing when warranted.

    If the order is below threshold, falls back to a single broker call
    (identical behaviour to the old code path).

    Parameters
    ----------
    ticker        : e.g. "AAPL"
    total_qty     : total shares to buy (from compute_position_size)
    initial_price : market price at signal time

    Returns
    -------
    dict with keys:
      price       → weighted average fill price across all slices
      order_id    → "iceberg_N" or broker order id for single orders
      filled_qty  → shares actually filled (may be < total_qty on partial)
      iceberg     → True | False
      slices      → list of per-slice results (empty for single orders)
    """
    ticker = ticker.upper()

    # ── Fast path: small order ────────────────────────────────────────────────
    if not should_use_iceberg(total_qty, initial_price):
        logger.info(
            f"[ICEBERG] {ticker}: {total_qty} shares @ ~${initial_price:.2f} "
            f"— below threshold, single order"
        )
        # Wrap sync function in thread to prevent blocking event loop
        lim = await asyncio.to_thread(limit_buy_price, initial_price, ticker)
        from utils import retry_sync
        order = await asyncio.wait_for(
            asyncio.to_thread(retry_sync, broker.submit_buy, ticker, total_qty, lim, max_retries=2),
            timeout=TIMEOUT_SEC,
        )
        fill = float(order.get("price") or lim)
        return {
            "price":      fill,
            "order_id":   order.get("order_id", ""),
            "filled_qty": total_qty,
            "iceberg":    False,
            "slices":     [],
        }

    # ── Compute slice plan ────────────────────────────────────────────────────
    slices = _plan_slices(total_qty)
    n      = len(slices)
    est_duration_min = (n - 1) * INTERVAL_SEC / 60

    logger.info(
        f"[ICEBERG] {ticker}: splitting {total_qty} shares into {n} slices "
        f"of ~{SLICE_SHARES} shares | interval={INTERVAL_SEC}s "
        f"| est. duration ~{est_duration_min:.1f}min"
    )

    # Notify Telegram that iceberg has started
    try:
        from telegram_bot import notify_iceberg_start
        asyncio.ensure_future(
            notify_iceberg_start(ticker, total_qty, n, INTERVAL_SEC)
        )
    except Exception:
        pass

    # Register in active tracker
    tracker = _ActiveIceberg(
        ticker=ticker, total_qty=total_qty, slices_total=n
    )
    with _active_lock:
        _active[ticker] = tracker

    # ── Execute slices ────────────────────────────────────────────────────────
    slice_results: list[dict] = []
    total_filled  = 0
    total_cost    = 0.0

    for i, slice_qty in enumerate(slices):
        # Wait between slices (not before first)
        if i > 0:
            jitter  = random.uniform(-JITTER_PCT, JITTER_PCT)
            delay   = INTERVAL_SEC * (1 + jitter)
            logger.debug(
                f"[ICEBERG] {ticker}: slice {i}/{n} — "
                f"waiting {delay:.1f}s before next order"
            )
            await asyncio.sleep(delay)

        # Fresh price for each slice
        try:
            fresh_price = await asyncio.wait_for(
                asyncio.to_thread(broker.get_price, ticker), timeout=10
            ) or initial_price
        except Exception:
            fresh_price = initial_price

        lim_price = limit_buy_price(fresh_price, ticker)

        # Submit child order
        try:
            from utils import retry_sync
            order = await asyncio.wait_for(
                asyncio.to_thread(
                    retry_sync, broker.submit_buy, ticker, slice_qty, lim_price,
                    max_retries=2
                ),
                timeout=TIMEOUT_SEC,
            )
            fill_price = float(order.get("price") or lim_price)
            order_id   = order.get("order_id", f"slice_{i+1}")

            slice_results.append({
                "slice":      i + 1,
                "qty":        slice_qty,
                "price":      fill_price,
                "limit":      lim_price,
                "order_id":   order_id,
                "market_ref": round(fresh_price, 4),
            })
            total_filled += slice_qty
            total_cost   += fill_price * slice_qty

            logger.info(
                f"[ICEBERG] {ticker}: slice {i+1}/{n} filled — "
                f"{slice_qty} shares @ ${fill_price:.4f} "
                f"(limit=${lim_price:.4f} | mkt=${fresh_price:.4f})"
            )

            # Update tracker
            with _active_lock:
                tracker.slices_done  = i + 1
                tracker.filled_qty   = total_filled
                tracker.avg_price    = total_cost / total_filled if total_filled else 0
                tracker.last_slice_at = time.time()

        except asyncio.TimeoutError:
            logger.error(
                f"[ICEBERG] {ticker}: slice {i+1}/{n} TIMED OUT — "
                f"continuing with remaining slices"
            )
        except Exception as exc:
            logger.error(
                f"[ICEBERG] {ticker}: slice {i+1}/{n} FAILED ({exc}) — "
                f"continuing with remaining slices"
            )

    # ── Aggregate result ──────────────────────────────────────────────────────
    if total_filled == 0:
        with _active_lock:
            _active.pop(ticker, None)
        raise RuntimeError(
            f"[ICEBERG] {ticker}: all {n} slices failed — no shares filled"
        )

    avg_price   = total_cost / total_filled
    is_partial  = total_filled < total_qty
    status      = "partial" if is_partial else "done"

    with _active_lock:
        tracker.status    = status
        tracker.avg_price = avg_price

    if is_partial:
        logger.warning(
            f"[ICEBERG] {ticker}: PARTIAL FILL — "
            f"{total_filled}/{total_qty} shares @ avg ${avg_price:.4f}"
        )
    else:
        logger.info(
            f"[ICEBERG] {ticker}: COMPLETE — "
            f"{total_filled} shares @ avg ${avg_price:.4f} "
            f"(initial mkt=${initial_price:.4f} | "
            f"slippage={((avg_price/initial_price)-1)*100:+.3f}%)"
        )

    # Notify Telegram completion
    try:
        from telegram_bot import notify_iceberg_done
        asyncio.ensure_future(
            notify_iceberg_done(ticker, total_filled, avg_price, n, is_partial)
        )
    except Exception:
        pass

    # Cleanup tracker after 5 minutes
    asyncio.ensure_future(_cleanup_tracker(ticker, delay=300))

    return {
        "price":      round(avg_price, 4),
        "order_id":   f"iceberg_{n}_slices",
        "filled_qty": total_filled,
        "iceberg":    True,
        "partial":    is_partial,
        "slices":     slice_results,
    }


def get_status() -> dict:
    """Return currently active iceberg executions (for /status endpoint)."""
    with _active_lock:
        return {
            ticker: asdict(tracker)
            for ticker, tracker in _active.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plan_slices(total_qty: float) -> list[float]:
    """
    Divide total_qty into equal child orders of SLICE_SHARES each.
    Uses true division (not //) to correctly handle fractional quantities.
    """
    raw_n  = math.ceil(total_qty / SLICE_SHARES) if SLICE_SHARES > 0 else 1
    n      = max(MIN_SLICES, min(MAX_SLICES, raw_n))

    # Use true division — safe for fractional shares
    base   = round(total_qty / n, 6)
    slices = [base] * (n - 1)
    # Last slice absorbs rounding remainder
    last   = round(total_qty - base * (n - 1), 6)
    if last > 0:
        slices.append(last)

    # Drop zero-qty slices (edge case: total_qty extremely small)
    slices = [s for s in slices if s > 0]
    return slices if slices else [total_qty]


async def _cleanup_tracker(ticker: str, delay: float = 300):
    """Remove ticker from active tracker after `delay` seconds."""
    await asyncio.sleep(delay)
    with _active_lock:
        _active.pop(ticker, None)
