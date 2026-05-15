"""
Sector Rotation Module
=======================
Identifies which market sectors are leading right now and prioritizes
stocks from those sectors in the scan cycle.

Why it matters:
  60-70% of a stock's move is explained by its sector's move.
  Buying NVDA when Tech is the #1 sector = sector tailwind.
  Buying NVDA when Tech is the worst sector = fighting the tide.

How it works:
  1. Downloads 20-day return for 11 sector ETFs (XLK, XLF, XLE, etc.)
  2. Ranks sectors from best to worst momentum
  3. Returns the top 3 sectors and their constituent stocks
  4. auto_invest_loop uses this to put sector leaders first in the scan queue

Cache: 2 hours (sector trends don't change minute-by-minute)

Public API
----------
  get_leading_sectors()         → list of (sector_name, etf, return_pct) sorted best→worst
  get_sector_for_ticker(ticker) → sector name or None
  get_sector_multiplier(ticker) → 1.0-1.3 (bonus for leading sector)
"""

import logging
import os
import threading
import time

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_TTL = int(os.getenv("SECTOR_CACHE_TTL", "7200"))   # 2 hours
LOOKBACK  = int(os.getenv("SECTOR_LOOKBACK_DAYS", "20"))  # 20-day momentum

# ── Sector ETF → Name mapping ─────────────────────────────────────────────────
SECTOR_ETFS = {
    "XLK":  "טכנולוגיה",
    "XLF":  "פיננסים",
    "XLV":  "בריאות",
    "XLE":  "אנרגיה",
    "XLY":  "צרכנות שיקולית",
    "XLP":  "צרכנות בסיסית",
    "XLI":  "תעשייה",
    "XLB":  "חומרים",
    "XLRE": "נדל\"ן",
    "XLU":  "תשתיות",
    "XLC":  "תקשורת",
}

# ── Ticker → Sector mapping (top holdings) ───────────────────────────────────
TICKER_SECTOR = {
    # Tech (XLK)
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AVGO":"XLK","AMD":"XLK","INTC":"XLK",
    "CSCO":"XLK","TXN":"XLK","QCOM":"XLK","IBM":"XLK","AMAT":"XLK","LRCX":"XLK",
    "KLAC":"XLK","MU":"XLK","ADI":"XLK","NOW":"XLK","CRM":"XLK","ADBE":"XLK",
    "INTU":"XLK","PANW":"XLK","CRWD":"XLK","FTNT":"XLK","SNPS":"XLK","CDNS":"XLK",
    "ORCL":"XLK","SAP":"XLK","ASML":"XLK","TSM":"XLK","ARM":"XLK","SMCI":"XLK",
    # Financials (XLF)
    "JPM":"XLF","BAC":"XLF","WFC":"XLF","GS":"XLF","MS":"XLF","AXP":"XLF",
    "V":"XLF","MA":"XLF","BLK":"XLF","SCHW":"XLF","CB":"XLF","PGR":"XLF",
    "SPGI":"XLF","MCO":"XLF","ICE":"XLF","CME":"XLF","COF":"XLF","USB":"XLF",
    "BK":"XLF","STT":"XLF","BRK-B":"XLF","KKR":"XLF","APO":"XLF","BX":"XLF",
    # Health (XLV)
    "UNH":"XLV","LLY":"XLV","JNJ":"XLV","ABBV":"XLV","MRK":"XLV","PFE":"XLV",
    "TMO":"XLV","ABT":"XLV","DHR":"XLV","AMGN":"XLV","ISRG":"XLV","VRTX":"XLV",
    "REGN":"XLV","BSX":"XLV","ELV":"XLV","CVS":"XLV","SYK":"XLV","ZTS":"XLV",
    "GILD":"XLV","MDT":"XLV","CI":"XLV","HUM":"XLV","NVO":"XLV","AZN":"XLV",
    # Energy (XLE)
    "XOM":"XLE","CVX":"XLE","COP":"XLE","EOG":"XLE","SLB":"XLE","OXY":"XLE",
    "PSX":"XLE","VLO":"XLE","MPC":"XLE","DVN":"XLE","TTE":"XLE","SHEL":"XLE",
    # Consumer Discretionary (XLY)
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","MCD":"XLY","SBUX":"XLY","NKE":"XLY",
    "TGT":"XLY","LOW":"XLY","TJX":"XLY","ROST":"XLY","CMG":"XLY","DPZ":"XLY",
    "BKNG":"XLY","ABNB":"XLY","UBER":"XLY",
    # Consumer Staples (XLP)
    "PG":"XLP","KO":"XLP","PEP":"XLP","PM":"XLP","MO":"XLP","MDLZ":"XLP",
    "CL":"XLP","KMB":"XLP","WMT":"XLP","COST":"XLP",
    # Industrials (XLI)
    "BA":"XLI","CAT":"XLI","HON":"XLI","GE":"XLI","MMM":"XLI","DE":"XLI",
    "UPS":"XLI","FDX":"XLI","ETN":"XLI","EMR":"XLI","RTX":"XLI","LMT":"XLI",
    "NOC":"XLI","GD":"XLI","LHX":"XLI","TDG":"XLI",
    # Communication (XLC)
    "GOOGL":"XLC","GOOG":"XLC","META":"XLC","NFLX":"XLC","DIS":"XLC","CMCSA":"XLC",
    "T":"XLC","VZ":"XLC","TMUS":"XLC",
    # Materials (XLB)
    "LIN":"XLB","APD":"XLB","ECL":"XLB","SHW":"XLB","FCX":"XLB","NEM":"XLB",
    # Real Estate (XLRE)
    "AMT":"XLRE","PLD":"XLRE","CCI":"XLRE","EQIX":"XLRE","PSA":"XLRE",
    "O":"XLRE","WELL":"XLRE","DLR":"XLRE",
}

# ── Cache ─────────────────────────────────────────────────────────────────────
_sector_cache: list[dict] = []
_cache_ts: float = 0.0
_lock = threading.Lock()


def get_leading_sectors() -> list[dict]:
    """
    Fetch 20-day returns for all sector ETFs and rank them.
    Returns list of dicts sorted best→worst:
      [{"etf": "XLK", "name": "טכנולוגיה", "return_pct": 5.2, "rank": 1}, ...]
    """
    global _sector_cache, _cache_ts
    now = time.time()

    with _lock:
        if _sector_cache and now - _cache_ts < CACHE_TTL:
            return list(_sector_cache)

    try:
        etf_list = list(SECTOR_ETFS.keys())
        _raw = yf.download(etf_list, period=f"{LOOKBACK + 5}d",
                           progress=False, auto_adjust=True)["Close"]
        if _raw.empty:
            return []
        # yf.download with 1 ETF returns Series; with multiple returns DataFrame
        import pandas as _pd
        prices = _pd.DataFrame(_raw) if hasattr(_raw, "to_frame") and not isinstance(_raw, _pd.DataFrame) else _raw

        results = []
        for etf in etf_list:
            if not hasattr(prices, "columns") or etf not in prices.columns:
                # Series case — single ETF
                if hasattr(prices, "name") and prices.name == etf:
                    series = prices.dropna()
                else:
                    continue
            else:
                series = prices[etf].dropna()
            if len(series) < 2:
                continue
            ret = (float(series.iloc[-1]) - float(series.iloc[0])) / float(series.iloc[0]) * 100
            results.append({
                "etf":        etf,
                "name":       SECTOR_ETFS[etf],
                "return_pct": round(ret, 2),
            })

        results.sort(key=lambda x: x["return_pct"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        with _lock:
            _sector_cache = results
            _cache_ts = now

        top3 = [f"{r['name']} {r['return_pct']:+.1f}%" for r in results[:3]]
        logger.info(f"[SECTOR] Top 3: {' | '.join(top3)}")
        return results

    except Exception as exc:
        logger.warning(f"[SECTOR] Failed to fetch sector data: {exc}")
        return []


def get_sector_for_ticker(ticker: str) -> str | None:
    """Return the sector ETF code for a given ticker, or None."""
    return TICKER_SECTOR.get(ticker.upper())


def get_sector_multiplier(ticker: str) -> float:
    """
    Return a position-size multiplier based on sector momentum rank.
      Rank 1-3  (leading)  → 1.20 (scan priority boost)
      Rank 4-7  (neutral)  → 1.00
      Rank 8-11 (lagging)  → 0.85 (de-prioritize)
    """
    etf = get_sector_for_ticker(ticker)
    if not etf:
        return 1.0

    sectors = get_leading_sectors()
    if not sectors:
        return 1.0

    for s in sectors:
        if s["etf"] == etf:
            rank = s["rank"]
            if rank <= 3:   return 1.20
            if rank <= 7:   return 1.00
            return 0.85

    return 1.0


def prioritize_by_sector(tickers: list[str]) -> list[str]:
    """
    Re-order a list of tickers: put tickers from leading sectors first.
    Used in auto_invest_loop to scan hot-sector stocks before cold ones.
    """
    sectors = get_leading_sectors()
    if not sectors:
        return tickers

    # Build rank map: etf → rank
    rank_map = {s["etf"]: s["rank"] for s in sectors}

    def _rank(ticker: str) -> int:
        etf = get_sector_for_ticker(ticker)
        return rank_map.get(etf, 6) if etf else 6  # unknown → neutral rank

    return sorted(tickers, key=_rank)
