"""
Pairs Trading & Hedging Module
================================

Advanced strategies for risk reduction:

1. Pairs Trading - Long one stock, short another (market-neutral)
2. Statistical Arbitrage - Mean reversion between correlated pairs
3. Hedging - Reduce directional exposure
4. Beta-neutral portfolios
5. Cointegration analysis
6. Ratio spreads
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradingPair:
    """A pair of stocks for pairs trading."""
    ticker1: str
    ticker2: str
    correlation: float
    cointegration_score: float  # 0-1, higher = better pair
    current_ratio: float
    historical_mean_ratio: float
    std_dev_ratio: float
    z_score: float  # How many std devs from mean
    signal: str  # "long_t1_short_t2", "long_t2_short_t1", "neutral", "exit"
    confidence: float


@dataclass
class HedgeRecommendation:
    """Hedge recommendation for a position."""
    ticker_held: str
    hedge_ticker: str
    hedge_ratio: float  # How much to short relative to long
    expected_correlation: float
    cost_basis: float
    reasoning: str


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION & COINTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

async def calculate_pair_correlation(ticker1: str, ticker2: str, days: int = 60) -> dict:
    """
    Calculate correlation between two stocks.
    Higher correlation = better pairs trading candidates.
    """
    try:
        import yfinance as yf

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Get data for both tickers
        data1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)["Close"]
        data2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)["Close"]

        if data1.empty or data2.empty:
            return {"error": "No data available"}

        # Returns
        returns1 = data1.pct_change().dropna()
        returns2 = data2.pct_change().dropna()

        # Align dates
        common_idx = returns1.index.intersection(returns2.index)
        if len(common_idx) < 20:
            return {"error": "Insufficient data"}

        returns1 = returns1[common_idx]
        returns2 = returns2[common_idx]

        # Pearson correlation
        correlation = float(np.corrcoef(returns1, returns2)[0, 1])

        # Spread analysis (price ratio)
        prices1 = data1[common_idx]
        prices2 = data2[common_idx]
        ratio = prices1 / prices2

        mean_ratio = float(ratio.mean())
        std_ratio = float(ratio.std())
        current_ratio = float(ratio.iloc[-1])

        # Z-score: how far from mean
        z_score = (current_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0

        # Simple cointegration score (1 - correlation of differences)
        try:
            diff = returns1 - returns2
            cointegration_score = 1 - abs(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.5
            cointegration_score = max(0, min(1, cointegration_score))
        except:
            cointegration_score = 0.5

        return {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "correlation": correlation,
            "cointegration_score": cointegration_score,
            "current_ratio": current_ratio,
            "mean_ratio": mean_ratio,
            "std_dev_ratio": std_ratio,
            "z_score": z_score,
            "data_points": len(common_idx),
        }

    except Exception as e:
        logger.error(f"Pair correlation failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# PAIRS TRADING SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

async def generate_pairs_signal(ticker1: str, ticker2: str) -> TradingPair:
    """
    Generate a pairs trading signal.

    Logic:
    - When z-score > +2: Pair is too high, short ticker1, long ticker2
    - When z-score < -2: Pair is too low, long ticker1, short ticker2
    - When |z-score| < 0.5: Exit positions
    """
    pair_data = await calculate_pair_correlation(ticker1, ticker2)

    if "error" in pair_data:
        return TradingPair(
            ticker1=ticker1, ticker2=ticker2,
            correlation=0, cointegration_score=0,
            current_ratio=0, historical_mean_ratio=0,
            std_dev_ratio=0, z_score=0,
            signal="error", confidence=0,
        )

    z_score = pair_data["z_score"]
    correlation = pair_data["correlation"]

    # Only trade highly correlated pairs
    if correlation < 0.7:
        signal = "low_correlation"
        confidence = 0
    elif abs(z_score) < 0.5:
        signal = "exit"  # Pair back to normal - close positions
        confidence = 0.5
    elif z_score > 2:
        signal = "short_t1_long_t2"  # Ratio too high
        confidence = min(1, (z_score - 1) / 3)
    elif z_score < -2:
        signal = "long_t1_short_t2"  # Ratio too low
        confidence = min(1, (abs(z_score) - 1) / 3)
    else:
        signal = "neutral"
        confidence = 0.2

    return TradingPair(
        ticker1=ticker1,
        ticker2=ticker2,
        correlation=correlation,
        cointegration_score=pair_data["cointegration_score"],
        current_ratio=pair_data["current_ratio"],
        historical_mean_ratio=pair_data["mean_ratio"],
        std_dev_ratio=pair_data["std_dev_ratio"],
        z_score=z_score,
        signal=signal,
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRE-DEFINED PAIRS
# ─────────────────────────────────────────────────────────────────────────────

# Well-known correlated pairs for stat-arb
DEFAULT_PAIRS = [
    # Tech
    ("AAPL", "MSFT"),
    ("GOOGL", "META"),
    ("NVDA", "AMD"),
    # Financials
    ("JPM", "BAC"),
    ("GS", "MS"),
    # Retail
    ("WMT", "TGT"),
    ("HD", "LOW"),
    # Energy
    ("XOM", "CVX"),
    # ETFs
    ("SPY", "IVV"),
    ("QQQ", "VOO"),
    # Pairs that should diverge sometimes
    ("KO", "PEP"),
    ("V", "MA"),
    # Telecom
    ("VZ", "T"),
]


async def scan_pairs_opportunities(custom_pairs: Optional[list] = None) -> list[dict]:
    """
    Scan pre-defined pairs for trading opportunities.
    """
    pairs_to_scan = custom_pairs or DEFAULT_PAIRS
    opportunities = []

    for ticker1, ticker2 in pairs_to_scan[:10]:  # Limit to 10 to avoid rate limits
        try:
            pair = await generate_pairs_signal(ticker1, ticker2)

            if pair.signal in ("long_t1_short_t2", "short_t1_long_t2") and pair.confidence > 0.5:
                opportunities.append({
                    "ticker1": pair.ticker1,
                    "ticker2": pair.ticker2,
                    "signal": pair.signal,
                    "z_score": pair.z_score,
                    "correlation": pair.correlation,
                    "confidence": pair.confidence,
                    "expected_action": (
                        f"Long {ticker1}, Short {ticker2}" if "long_t1" in pair.signal else
                        f"Short {ticker1}, Long {ticker2}"
                    ),
                })

        except Exception as e:
            logger.debug(f"Pairs scan error for {ticker1}/{ticker2}: {e}")

    # Sort by confidence
    opportunities.sort(key=lambda x: x["confidence"], reverse=True)
    return opportunities


# ─────────────────────────────────────────────────────────────────────────────
# HEDGING RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

# Common hedge candidates (negatively correlated)
HEDGE_CANDIDATES = {
    "SPY": ["SH", "VIXY"],          # S&P 500 inverse, VIX
    "QQQ": ["PSQ", "SQQQ"],         # Nasdaq inverse
    "AAPL": ["SH", "QID"],          # Tech hedges
    "TSLA": ["SH", "VIXY"],         # High-beta hedges
    "NVDA": ["SOXS", "SH"],         # Semiconductor inverse
    "AMD": ["SOXS", "SH"],          # Semiconductor inverse
    "default": ["SH", "VIXY"],      # Generic hedges
}


async def recommend_hedge(position_ticker: str, position_value: float, beta: float = 1.0) -> HedgeRecommendation:
    """
    Recommend a hedge for an existing long position.

    Hedge ratio depends on:
    - Beta of position
    - Volatility
    - Correlation with hedge
    """
    candidates = HEDGE_CANDIDATES.get(position_ticker, HEDGE_CANDIDATES["default"])
    hedge_ticker = candidates[0]

    # Hedge ratio: usually 30-50% of position value for partial hedge
    # Higher beta = need more hedge
    hedge_ratio = min(0.5, 0.3 * beta)
    hedge_value = position_value * hedge_ratio

    return HedgeRecommendation(
        ticker_held=position_ticker,
        hedge_ticker=hedge_ticker,
        hedge_ratio=hedge_ratio,
        expected_correlation=-0.7,  # Inverse ETFs target -1.0 correlation
        cost_basis=hedge_value,
        reasoning=(
            f"Partial hedge ({hedge_ratio:.0%}) of ${position_value:.2f} {position_ticker}. "
            f"Use {hedge_ticker} to offset downside risk. "
            "Hedge reduces but doesn't eliminate exposure."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO BETA CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

async def calculate_portfolio_beta() -> dict:
    """
    Calculate portfolio beta vs SPY.

    Beta > 1: More volatile than market
    Beta < 1: Less volatile than market
    Beta < 0: Inverse correlation
    """
    try:
        import broker
        import yfinance as yf

        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return {"error": "No positions"}

        # Get SPY data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=90)
        spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"]
        spy_returns = spy_data.pct_change().dropna()

        # Calculate weighted beta
        total_value = sum(float(p.market_value) for p in positions)
        if total_value == 0:
            return {"error": "Zero portfolio value"}

        weighted_beta = 0
        individual_betas = {}

        for p in positions:
            try:
                weight = float(p.market_value) / total_value
                ticker_data = yf.download(p.symbol, start=start_date, end=end_date, progress=False)["Close"]
                ticker_returns = ticker_data.pct_change().dropna()

                # Align indices
                common = ticker_returns.index.intersection(spy_returns.index)
                if len(common) < 20:
                    continue

                # Beta = covariance / variance
                covariance = np.cov(ticker_returns[common], spy_returns[common])[0, 1]
                variance = np.var(spy_returns[common])

                if variance > 0:
                    beta = covariance / variance
                    individual_betas[p.symbol] = float(beta)
                    weighted_beta += weight * beta

            except Exception as e:
                logger.debug(f"Beta calc for {p.symbol} failed: {e}")

        # Interpretation
        if weighted_beta > 1.3:
            interpretation = "🔴 High beta - very sensitive to market"
            hedge_needed = "Strong hedge recommended"
        elif weighted_beta > 1.0:
            interpretation = "🟠 Above-market beta"
            hedge_needed = "Consider partial hedge"
        elif weighted_beta > 0.7:
            interpretation = "🟡 Near-market beta"
            hedge_needed = "Optional hedge"
        elif weighted_beta > 0:
            interpretation = "🟢 Defensive portfolio"
            hedge_needed = "Minimal hedging needed"
        else:
            interpretation = "🟢 Inverse correlation - naturally hedged"
            hedge_needed = "No hedge needed"

        return {
            "portfolio_beta": float(weighted_beta),
            "interpretation": interpretation,
            "hedge_recommendation": hedge_needed,
            "individual_betas": individual_betas,
            "high_beta_positions": [
                {"ticker": t, "beta": b}
                for t, b in individual_betas.items()
                if b > 1.5
            ],
        }

    except Exception as e:
        logger.error(f"Portfolio beta calc failed: {e}")
        return {"error": str(e)}
