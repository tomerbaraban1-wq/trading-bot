"""
Sector Diversification Guard
==============================

Prevents over-concentration in a single sector.

Rule: Never have more than 2 positions in the same sector.

Sector mapping based on common ETF sectors:
- Technology, Healthcare, Financials, Energy,
  Consumer Discretionary, Consumer Staples, Industrials,
  Materials, Real Estate, Utilities, Communications
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Comprehensive ticker→sector mapping
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "TSM": "Technology",
    "AVGO": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "MU": "Technology", "AMAT": "Technology", "KLAC": "Technology",
    "LRCX": "Technology", "MRVL": "Technology", "ON": "Technology",
    "ORCL": "Technology", "SAP": "Technology", "IBM": "Technology",
    "CSCO": "Technology", "HPQ": "Technology",

    # Software/Cloud
    "CRM": "Software", "GOOGL": "Software", "GOOG": "Software",
    "META": "Software", "ADBE": "Software", "NOW": "Software",
    "INTU": "Software", "SNOW": "Software", "PANW": "Software",

    # Consumer Tech
    "AMZN": "Consumer Tech", "TSLA": "Consumer Tech",
    "NFLX": "Consumer Tech", "UBER": "Consumer Tech",

    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
    "ABT": "Healthcare", "DHR": "Healthcare", "PFE": "Healthcare",
    "AMGN": "Healthcare", "GILD": "Healthcare", "BIIB": "Healthcare",
    "REGN": "Healthcare", "VRTX": "Healthcare", "BSX": "Healthcare",
    "MDT": "Healthcare", "SYK": "Healthcare", "ZTS": "Healthcare",
    "ELV": "Healthcare", "CVS": "Healthcare", "HUM": "Healthcare",
    "XLV": "Healthcare",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "V": "Financials", "MA": "Financials", "AXP": "Financials",
    "SCHW": "Financials", "C": "Financials", "USB": "Financials",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "SLB": "Energy", "PXD": "Energy",
    "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "OXY": "Energy", "GDX": "Energy", "XLE": "Energy",

    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "GIS": "Consumer Staples",

    # Consumer Discretionary
    "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "TGT": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "BKNG": "Consumer Discretionary",

    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "RTX": "Industrials", "UPS": "Industrials",
    "FDX": "Industrials", "DE": "Industrials", "LMT": "Industrials",

    # Materials & Commodities
    "GLD": "Materials", "SLV": "Materials", "NEM": "Materials",
    "FCX": "Materials", "BHP": "Materials", "RIO": "Materials",
    "LIN": "Materials", "APD": "Materials",

    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate",

    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "XLU": "Utilities",
}

MAX_POSITIONS_PER_SECTOR = int(2)  # max 2 stocks per sector


def get_sector(ticker: str) -> str:
    """Get sector for a ticker. Falls back to yfinance if not in map."""
    # Check static map first (fast)
    sector = SECTOR_MAP.get(ticker.upper())
    if sector:
        return sector

    # Try yfinance (slower, cached)
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
        SECTOR_MAP[ticker.upper()] = sector  # cache it
        return sector
    except Exception:
        return "Unknown"


def check_sector_concentration(
    new_ticker: str,
    open_tickers: list[str],
    max_per_sector: int = MAX_POSITIONS_PER_SECTOR,
) -> dict:
    """
    Check if adding new_ticker would over-concentrate in a sector.

    Returns:
    {
        "allowed": bool,
        "sector": str,
        "current_count": int,
        "max_allowed": int,
        "reason": str,
    }
    """
    new_sector = get_sector(new_ticker)

    if new_sector == "Unknown":
        return {"allowed": True, "sector": "Unknown", "current_count": 0,
                "max_allowed": max_per_sector, "reason": "Sector unknown — allowing"}

    # Count existing positions in same sector
    same_sector = []
    for t in open_tickers:
        if get_sector(t) == new_sector:
            same_sector.append(t)

    current_count = len(same_sector)
    allowed = current_count < max_per_sector

    if not allowed:
        reason = (
            f"Sector '{new_sector}' already has {current_count} positions "
            f"({', '.join(same_sector)}) — max {max_per_sector}"
        )
        logger.info(f"[SECTOR GUARD] {new_ticker} BLOCKED: {reason}")
    else:
        reason = f"Sector '{new_sector}': {current_count}/{max_per_sector} positions"

    return {
        "allowed": allowed,
        "sector": new_sector,
        "current_count": current_count,
        "max_allowed": max_per_sector,
        "same_sector_positions": same_sector,
        "reason": reason,
    }


def get_portfolio_sector_distribution(tickers: list[str]) -> dict:
    """Get sector distribution of current portfolio."""
    distribution: dict[str, list[str]] = {}
    for t in tickers:
        sector = get_sector(t)
        if sector not in distribution:
            distribution[sector] = []
        distribution[sector].append(t)
    return distribution
