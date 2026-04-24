"""
Slippage Estimation — ATR-Based Dynamic Model

Amateur approach (what we had before):
    limit_price = market_price × 1.001  (fixed 0.1% regardless of the asset)
    → TSLA (daily ATR ~4%) and WMT (daily ATR ~0.5%) get the same offset.
    → On high-volatility days the offset is too tight → orders don't fill.
    → On low-volatility days the offset is too wide → we overpay.

Hedge fund approach (what this module does):
    The limit offset scales with the asset's recent volatility (ATR).

    offset = ATR_14 / price × ATR_MULTIPLIER

    ATR_MULTIPLIER controls how many ATRs of room we give:
        0.1 → very tight  (fast markets, low fill risk)
        0.2 → standard    (default)
        0.5 → wide        (illiquid or high-volatility assets)

    The offset is also capped at MAX_SLIP_PCT to avoid absurd values on
    gapped opens or data errors.

Environment variables:
    SLIPPAGE_ATR_MULTIPLIER   float  default 0.2
    SLIPPAGE_MAX_PCT          float  default 0.5  (hard cap, %)
    SLIPPAGE_MIN_PCT          float  default 0.05 (floor, %)
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ATR_MULTIPLIER:  float = float(os.getenv("SLIPPAGE_ATR_MULTIPLIER", "0.2"))
MAX_SLIP_PCT:    float = float(os.getenv("SLIPPAGE_MAX_PCT",         "0.5"))   # 0.5%
MIN_SLIP_PCT:    float = float(os.getenv("SLIPPAGE_MIN_PCT",         "0.05"))  # 0.05%
ALERT_PCT:       float = float(os.getenv("SLIPPAGE_ALERT_PCT",       "0.1"))   # rolling avg threshold
ROLLING_N:       int   = int(os.getenv("SLIPPAGE_ROLLING_N",         "20"))    # rolling window

# ── ATR cache (5-minute TTL) ──────────────────────────────────────────────────
import time as _time
_atr_cache: dict[str, tuple[float, float]] = {}   # ticker → (atr_pct, ts)
_ATR_TTL = 300


def _fetch_atr_pct(ticker: str) -> float:
    """
    Fetch 14-day ATR as a percentage of current price.
    Returns the raw fraction (e.g. 0.025 for 2.5%).
    Falls back to MAX_SLIP_PCT / 100 on error.
    """
    now = _time.time()
    cached = _atr_cache.get(ticker)
    if cached and now - cached[1] < _ATR_TTL:
        return cached[0]

    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="30d")
        if hist.empty or len(hist) < 15:
            raise ValueError("insufficient history")

        high  = hist["High"]
        low   = hist["Low"]
        close = hist["Close"]

        # True Range
        hl   = high - low
        hc   = (high - close.shift()).abs()
        lc   = (low  - close.shift()).abs()
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr  = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        price = float(close.iloc[-1])
        atr_pct = atr / price if price > 0 else MAX_SLIP_PCT / 100

    except Exception as e:
        logger.warning(f"[SLIPPAGE] ATR fetch failed for {ticker}: {e} — using max_slip")
        atr_pct = MAX_SLIP_PCT / 100

    _atr_cache[ticker] = (atr_pct, now)
    return atr_pct


def _compute_offset_pct(ticker: str) -> float:
    """
    Compute the dynamic offset percentage for a given ticker.
    Clamped to [MIN_SLIP_PCT, MAX_SLIP_PCT].
    """
    atr_pct    = _fetch_atr_pct(ticker)
    offset_pct = atr_pct * ATR_MULTIPLIER * 100   # convert to %
    clamped    = max(MIN_SLIP_PCT, min(MAX_SLIP_PCT, offset_pct))
    logger.debug(
        f"[SLIPPAGE] {ticker} | ATR={atr_pct*100:.3f}% | "
        f"raw_offset={offset_pct:.3f}% | clamped={clamped:.3f}%"
    )
    return clamped


def limit_buy_price(market_price: float, ticker: str = "") -> float:
    """
    Aggressive limit buy: market_price + dynamic ATR-based offset.
    Slightly above market to guarantee fill without overpaying.
    """
    offset_pct = _compute_offset_pct(ticker) if ticker else MIN_SLIP_PCT
    return round(market_price * (1 + offset_pct / 100), 4)


def limit_sell_price(market_price: float, ticker: str = "") -> float:
    """
    Aggressive limit sell: market_price - dynamic ATR-based offset.
    Slightly below market to guarantee fill while protecting exit price.
    """
    offset_pct = _compute_offset_pct(ticker) if ticker else MIN_SLIP_PCT
    return round(market_price * (1 - offset_pct / 100), 4)


def estimate(market_price: float, qty: float, side: str, ticker: str = "") -> dict:
    """
    Full slippage estimate for a trade.
    Logs the breakdown and returns a metadata dict.
    """
    offset_pct = _compute_offset_pct(ticker) if ticker else MIN_SLIP_PCT

    if side == "buy":
        limit_price = round(market_price * (1 + offset_pct / 100), 4)
    else:
        limit_price = round(market_price * (1 - offset_pct / 100), 4)

    slip_per_share  = abs(limit_price - market_price)
    total_slip      = round(slip_per_share * qty, 4)
    slip_bps        = round((slip_per_share / market_price) * 10_000, 2) if market_price else 0

    result = {
        "ticker":           ticker,
        "side":             side,
        "qty":              qty,
        "market_price":     round(market_price, 4),
        "limit_price":      limit_price,
        "offset_pct":       round(offset_pct, 4),
        "slip_per_share":   round(slip_per_share, 4),
        "total_slip_usd":   total_slip,
        "slip_bps":         slip_bps,          # basis points — standard in HF
    }

    logger.info(
        f"[SLIPPAGE] {side.upper()} {qty}× {ticker} | "
        f"market=${market_price:.4f} → limit=${limit_price:.4f} | "
        f"offset={offset_pct:.3f}% ({slip_bps:.1f}bps) | "
        f"cost=${total_slip:.4f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Actual slippage tracking (signal price vs fill price)
# ─────────────────────────────────────────────────────────────────────────────

def record(
    signal_price: float,
    fill_price:   float,
    qty:          float,
    side:         str,    # "buy" | "sell"
    ticker:       str,
) -> dict:
    """
    Record the **actual** slippage for an executed order.

    Actual slippage is the difference between what the signal said the price
    would be and what the broker actually filled at.

    Direction convention (positive = cost to us):
      buy  → fill > signal is cost  (paid more than expected)
      sell → signal > fill is cost  (received less than expected)

    The observation is written to slippage_log in SQLite, then the rolling
    average is checked.  If it exceeds SLIPPAGE_ALERT_PCT a Telegram warning
    is sent.

    Parameters
    ----------
    signal_price : price embedded in the webhook / auto-invest signal
    fill_price   : actual execution price returned by the broker
    qty          : number of shares filled
    side         : "buy" or "sell"
    ticker       : e.g. "AAPL"

    Returns
    -------
    dict with full breakdown (same shape as estimate())
    """
    ticker = ticker.upper()

    if signal_price <= 0:
        logger.warning(f"[SLIPPAGE RECORD] {ticker}: invalid signal_price={signal_price} — skipping")
        return {}

    # Direction-aware signed slippage
    if side == "buy":
        raw_slip = fill_price - signal_price          # + = paid more (cost)
    else:
        raw_slip = signal_price - fill_price          # + = received less (cost)

    slip_pct      = raw_slip / signal_price * 100     # signed %
    abs_slip_pct  = abs(slip_pct)
    slip_per_share = abs(fill_price - signal_price)
    total_slip_usd = round(slip_per_share * qty, 4)
    slip_bps       = round(abs_slip_pct * 100, 2)

    row = {
        "ticker":         ticker,
        "side":           side,
        "qty":            qty,
        "signal_price":   round(signal_price, 4),
        "fill_price":     round(fill_price, 4),
        "slip_pct":       round(slip_pct, 4),
        "abs_slip_pct":   round(abs_slip_pct, 4),
        "slip_bps":       slip_bps,
        "slip_per_share": round(slip_per_share, 4),
        "total_slip_usd": total_slip_usd,
    }

    logger.info(
        f"[SLIPPAGE RECORD] {side.upper()} {qty}× {ticker} | "
        f"signal=${signal_price:.4f} → fill=${fill_price:.4f} | "
        f"slip={slip_pct:+.3f}% ({slip_bps:.1f}bps) | "
        f"cost=${total_slip_usd:.4f}"
    )

    try:
        import database as _db
        slip_id = _db.save_slippage(row)
        row["id"] = slip_id
    except Exception as exc:
        logger.warning(f"[SLIPPAGE RECORD] {ticker}: DB write failed: {exc}")

    # Rolling average alert (non-blocking, best-effort)
    _check_rolling_alert(ticker)

    return row


def _check_rolling_alert(ticker: str) -> None:
    """
    Compute the rolling {ROLLING_N}-trade average of abs_slip_pct.
    If it exceeds ALERT_PCT, send a Telegram warning.
    """
    try:
        import database as _db
        avg = _db.get_rolling_slippage(ROLLING_N)
        if avg > ALERT_PCT:
            msg = (
                f"⚠️ *High Slippage Alert*\n"
                f"Rolling {ROLLING_N}-trade average: *{avg:.3f}%* "
                f"(threshold: {ALERT_PCT}%)\n"
                f"Last trade: `{ticker}`\n"
                f"Review fill quality — consider adjusting limit price offsets."
            )
            logger.warning(f"[SLIPPAGE ALERT] rolling avg={avg:.3f}% > {ALERT_PCT}% — notifying")
            try:
                from telegram_bot import notify_slippage_alert
                import asyncio
                asyncio.ensure_future(notify_slippage_alert(avg, ticker, ROLLING_N, ALERT_PCT))
            except RuntimeError:
                # No running event loop (e.g. called from a thread during testing)
                logger.warning(f"[SLIPPAGE ALERT] No event loop — Telegram alert skipped: {msg}")
            except Exception as te:
                logger.warning(f"[SLIPPAGE ALERT] Telegram failed: {te}")
    except Exception as exc:
        logger.warning(f"[SLIPPAGE] Rolling alert check failed: {exc}")


def get_summary() -> dict:
    """Return aggregate slippage statistics from the database."""
    try:
        import database as _db
        return _db.get_slippage_summary()
    except Exception as exc:
        logger.warning(f"[SLIPPAGE] get_summary failed: {exc}")
        return {}
