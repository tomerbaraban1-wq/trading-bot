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

ATR_MULTIPLIER: float = float(os.getenv("SLIPPAGE_ATR_MULTIPLIER", "0.2"))
MAX_SLIP_PCT:   float = float(os.getenv("SLIPPAGE_MAX_PCT",         "0.5"))   # 0.5%
MIN_SLIP_PCT:   float = float(os.getenv("SLIPPAGE_MIN_PCT",         "0.05"))  # 0.05%

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


def estimate(market_price: float, qty: int, side: str, ticker: str = "") -> dict:
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
