"""
Market Intelligence Module
===========================

Advanced market analysis for:
1. Sector rotation detection
2. Market breadth analysis (advance/decline lines)
3. Key support/resistance identification
4. Volatility regime detection
5. Market microstructure analysis
6. Smart entry/exit signals
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET BREADTH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BreadthMetrics:
    """Market breadth and health indicators."""
    advances: int           # Number of stocks up
    declines: int          # Number of stocks down
    unchanged: int         # Number unchanged
    advance_decline_ratio: float  # Advances / Declines
    market_breadth_percent: float # Advances / (Advances + Declines) * 100
    strength_indicator: str # 🟢 Healthy, 🟡 Mixed, 🔴 Weak


async def get_market_breadth(market_index: str = "^GSPC") -> BreadthMetrics:
    """
    Analyze market breadth using advance/decline data.
    A healthy market has 60%+ of stocks advancing.
    """
    try:
        import yfinance as yf

        # Get S&P 500 stocks
        sp500_list = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMAZON",  # Top 5
            "META", "TSLA", "BERKB", "UNH", "JPM",       # Next 5
            "JNJ", "WMT", "V", "KO", "PG", "MA", "HD", "MCD", "INTC", "NFLX"
        ]

        # Quick sample of major stocks
        data = yf.download(sp500_list, period="1d", progress=False, auto_adjust=True)["Close"]

        if data.empty:
            return BreadthMetrics(0, 0, 0, 0, 0, "🔴 Error")

        # Calculate daily changes
        changes = (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100

        advances = (changes > 0).sum()
        declines = (changes < 0).sum()
        unchanged = (changes == 0).sum()

        ad_ratio = advances / max(declines, 1)
        breadth_pct = (advances / (advances + declines) * 100) if (advances + declines) > 0 else 0

        # Determine strength
        if breadth_pct > 65:
            strength = "🟢 Healthy - Strong broad-based advance"
        elif breadth_pct > 55:
            strength = "🟡 Mixed - Moderate participation"
        elif breadth_pct > 45:
            strength = "🟡 Weak - Narrow rally"
        else:
            strength = "🔴 Declining - Broad selloff"

        return BreadthMetrics(
            advances=advances,
            declines=declines,
            unchanged=unchanged,
            advance_decline_ratio=ad_ratio,
            market_breadth_percent=breadth_pct,
            strength_indicator=strength,
        )

    except Exception as e:
        logger.debug(f"Market breadth calculation failed: {e}")
        return BreadthMetrics(0, 0, 0, 0, 0, "🔴 Error")


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR ROTATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectorPerformance:
    """Sector performance metrics."""
    sector: str
    performance_pct: float
    rank: int  # 1 = best performer
    momentum: str  # 🔴 Declining, 🟡 Stable, 🟢 Accelerating
    recommendation: str  # Which sector to favor


async def analyze_sector_rotation() -> List[SectorPerformance]:
    """
    Analyze which sectors are leading/lagging.
    Identifies sector rotation opportunities.

    Sector ETFs used:
    - XLK: Technology
    - XLV: Healthcare
    - XLY: Consumer Discretionary
    - XLP: Consumer Staples
    - XLE: Energy
    - XLF: Financials
    - XLRE: Real Estate
    - XLI: Industrials
    - XLU: Utilities
    - XLRM: Materials (or XLRE)
    """
    try:
        import yfinance as yf

        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Financials": "XLF",
            "Real Estate": "XLRE",
            "Industrials": "XLI",
            "Utilities": "XLU",
            "Materials": "XLB",
        }

        # Get 30-day performance
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        performances = []
        for sector_name, ticker in sector_etfs.items():
            try:
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)["Close"]
                if len(data) < 2:
                    continue

                perf = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100)
                performances.append((sector_name, perf))
            except Exception:
                continue

        # Sort by performance
        performances.sort(key=lambda x: x[1], reverse=True)

        # Generate recommendations
        results = []
        for rank, (sector, perf) in enumerate(performances, 1):
            # Determine momentum
            if rank <= 3:
                momentum = "🟢 Accelerating - Strong outperformance"
            elif rank <= 5:
                momentum = "🟡 Stable - Moderate performance"
            elif rank <= 7:
                momentum = "🟡 Declining - Underperforming"
            else:
                momentum = "🔴 Weak - Significant underperformance"

            recommendation = "FAVOR 🟢" if rank <= 3 else "NEUTRAL 🟡" if rank <= 7 else "AVOID 🔴"

            results.append(SectorPerformance(
                sector=sector,
                performance_pct=perf,
                rank=rank,
                momentum=momentum,
                recommendation=recommendation,
            ))

        logger.info(f"[SECTOR] Top performer: {results[0].sector} (+{results[0].performance_pct:.2f}%)")
        return results

    except Exception as e:
        logger.debug(f"Sector rotation analysis failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORT/RESISTANCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SupportResistance:
    """Key support and resistance levels."""
    ticker: str
    support_levels: List[float]
    resistance_levels: List[float]
    current_price: float
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    distance_to_support_pct: float  # How far below current price
    distance_to_resistance_pct: float  # How far above current price


def find_support_resistance(ticker: str, period: int = 90) -> SupportResistance:
    """
    Identify key support and resistance levels using:
    - Previous swing highs/lows
    - Round numbers
    - Pivot points
    """
    try:
        import yfinance as yf

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=period)

        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return SupportResistance(ticker, [], [], 0, None, None, 0, 0)

        prices = data["Close"]
        current_price = prices.iloc[-1]

        # Find local highs and lows (swing points)
        window = 5
        supports = []
        resistances = []

        for i in range(window, len(prices) - window):
            # Local high
            if prices.iloc[i] == prices.iloc[i-window:i+window].max():
                resistances.append(prices.iloc[i])
            # Local low
            if prices.iloc[i] == prices.iloc[i-window:i+window].min():
                supports.append(prices.iloc[i])

        # Remove duplicates and sort
        supports = sorted(list(set(supports)), reverse=True)
        resistances = sorted(list(set(resistances)), reverse=True)

        # Keep only significant levels
        supports = supports[:3]  # Top 3 support levels
        resistances = resistances[:3]  # Top 3 resistance levels

        # Find nearest
        nearest_support = next((s for s in supports if s < current_price), None)
        nearest_resistance = next((r for r in resistances if r > current_price), None)

        dist_to_support = ((current_price - nearest_support) / current_price * 100) if nearest_support else 0
        dist_to_resistance = ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else 0

        return SupportResistance(
            ticker=ticker,
            support_levels=supports,
            resistance_levels=resistances,
            current_price=current_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            distance_to_support_pct=dist_to_support,
            distance_to_resistance_pct=dist_to_resistance,
        )

    except Exception as e:
        logger.debug(f"Support/resistance detection failed for {ticker}: {e}")
        return SupportResistance(ticker, [], [], 0, None, None, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VolatilityRegime:
    """Current volatility regime."""
    regime: str  # "EXPANSION", "CONTRACTION", "NORMAL"
    volatility_20d: float  # 20-day volatility
    volatility_5d: float   # 5-day volatility
    expansion_rate: float  # How fast expanding/contracting
    recommendation: str


async def detect_volatility_regime(ticker: str = "SPY") -> VolatilityRegime:
    """
    Detect volatility regime using 20-day vs 5-day volatility.

    - EXPANSION: 5d vol > 20d vol + 20% (market heating up)
    - CONTRACTION: 5d vol < 20d vol - 20% (market cooling)
    - NORMAL: Between the two
    """
    try:
        import yfinance as yf

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=60)

        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)["Close"]
        returns = data.pct_change().dropna()

        vol_20d = returns.tail(20).std() * np.sqrt(252) * 100
        vol_5d = returns.tail(5).std() * np.sqrt(252) * 100

        expansion_rate = ((vol_5d - vol_20d) / vol_20d * 100) if vol_20d > 0 else 0

        if expansion_rate > 20:
            regime = "🔴 EXPANSION - Volatility rising, caution warranted"
            rec = "⚠️ Reduce position size, tighten stops"
        elif expansion_rate < -20:
            regime = "🟢 CONTRACTION - Volatility falling, stability ahead"
            rec = "✅ Can increase position size, look for breakouts"
        else:
            regime = "🟡 NORMAL - Steady volatility"
            rec = "➡️ Maintain normal trading"

        return VolatilityRegime(
            regime=regime,
            volatility_20d=vol_20d,
            volatility_5d=vol_5d,
            expansion_rate=expansion_rate,
            recommendation=rec,
        )

    except Exception as e:
        logger.debug(f"Volatility regime detection failed: {e}")
        return VolatilityRegime("🔴 Error", 0, 0, 0, "Error calculating")


# ─────────────────────────────────────────────────────────────────────────────
# MARKET INTELLIGENCE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

async def get_market_intelligence_report() -> dict:
    """
    Generate comprehensive market intelligence report.
    """
    try:
        # Gather all market data in parallel
        breadth_task = get_market_breadth()
        sectors_task = analyze_sector_rotation()
        vol_regime_task = detect_volatility_regime()

        breadth = await breadth_task
        sectors = await sectors_task
        vol_regime = await vol_regime_task

        # Get SPY support/resistance
        spy_sr = await asyncio.to_thread(find_support_resistance, "SPY")

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_breadth": {
                "advances": breadth.advances,
                "declines": breadth.declines,
                "breadth_percent": breadth.market_breadth_percent,
                "strength": breadth.strength_indicator,
            },
            "sector_rotation": [
                {
                    "rank": s.rank,
                    "sector": s.sector,
                    "performance": s.performance_pct,
                    "recommendation": s.recommendation,
                }
                for s in sectors[:5]  # Top 5 sectors
            ],
            "volatility_regime": {
                "regime": vol_regime.regime,
                "volatility_5d": vol_regime.volatility_5d,
                "volatility_20d": vol_regime.volatility_20d,
                "recommendation": vol_regime.recommendation,
            },
            "spy_technicals": {
                "current_price": spy_sr.current_price,
                "nearest_support": spy_sr.nearest_support,
                "nearest_resistance": spy_sr.nearest_resistance,
                "distance_to_support": spy_sr.distance_to_support_pct,
                "distance_to_resistance": spy_sr.distance_to_resistance_pct,
            },
        }

        logger.info("[MARKET INTEL] Report generated successfully")
        return report

    except Exception as e:
        logger.error(f"Market intelligence report failed: {e}")
        return {"error": str(e)}
