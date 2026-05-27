"""
Portfolio Optimization Module
==============================

Modern Portfolio Theory (MPT) implementation for optimal portfolio allocation.

Features:
1. Mean-Variance Optimization
2. Maximum Sharpe Ratio portfolio
3. Minimum Variance portfolio
4. Risk parity allocation
5. Correlation-aware rebalancing
6. Diversification scoring
7. Position sizing recommendations
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Recommended portfolio allocation."""
    tickers: list[str]
    weights: list[float]  # Percentage of capital per ticker
    expected_return: float  # Annualized %
    expected_volatility: float  # Annualized %
    sharpe_ratio: float
    diversification_score: float  # 0-100
    recommendation: str


# ─────────────────────────────────────────────────────────────────────────────
# COVARIANCE MATRIX CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

async def calculate_returns_matrix(tickers: list[str], days: int = 90) -> tuple:
    """
    Calculate returns matrix for portfolio optimization.

    Returns: (returns_dict, dates) where returns_dict maps ticker -> daily returns
    """
    try:
        import yfinance as yf

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        returns_dict = {}
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if data.empty:
                    continue

                prices = data["Close"].values
                daily_returns = np.diff(prices) / prices[:-1]
                returns_dict[ticker] = daily_returns
            except Exception as e:
                logger.debug(f"Failed to get returns for {ticker}: {e}")

        # Align lengths
        if returns_dict:
            min_length = min(len(r) for r in returns_dict.values())
            for t in returns_dict:
                returns_dict[t] = returns_dict[t][-min_length:]

        return returns_dict

    except Exception as e:
        logger.error(f"Returns matrix calculation failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAXIMUM SHARPE RATIO PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def calculate_max_sharpe_weights(returns_matrix: np.ndarray, risk_free_rate: float = 0.04) -> tuple:
    """
    Calculate weights that maximize Sharpe ratio.

    Uses random portfolio sampling (Monte Carlo) - simpler than scipy optimize.
    For production, scipy.optimize.minimize is more accurate.
    """
    n_assets = returns_matrix.shape[0]

    if n_assets == 0:
        return np.array([]), 0, 0, 0

    # Calculate mean returns and covariance
    mean_returns = np.mean(returns_matrix, axis=1) * 252  # Annualized
    cov_matrix = np.cov(returns_matrix) * 252  # Annualized

    # Run many random portfolios
    n_portfolios = 5000
    best_sharpe = -np.inf
    best_weights = None
    best_return = 0
    best_vol = 0

    for _ in range(n_portfolios):
        # Random weights summing to 1
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if portfolio_vol > 0:
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights
                best_return = portfolio_return
                best_vol = portfolio_vol

    return best_weights, best_return, best_vol, best_sharpe


# ─────────────────────────────────────────────────────────────────────────────
# MINIMUM VARIANCE PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def calculate_min_variance_weights(returns_matrix: np.ndarray) -> tuple:
    """Calculate minimum variance portfolio weights."""
    n_assets = returns_matrix.shape[0]

    if n_assets == 0:
        return np.array([]), 0, 0

    cov_matrix = np.cov(returns_matrix) * 252
    mean_returns = np.mean(returns_matrix, axis=1) * 252

    # Inverse covariance approach
    try:
        inv_cov = np.linalg.inv(cov_matrix + np.eye(n_assets) * 1e-6)  # Regularization
        ones = np.ones(n_assets)
        weights = inv_cov @ ones
        weights /= np.sum(weights)

        # Ensure no shorting (no negative weights)
        weights = np.maximum(weights, 0)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(n_assets) / n_assets

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        return weights, portfolio_return, portfolio_vol

    except Exception as e:
        logger.error(f"Min variance calc failed: {e}")
        return np.ones(n_assets) / n_assets, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# RISK PARITY ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_risk_parity_weights(returns_matrix: np.ndarray) -> np.ndarray:
    """
    Risk parity: each asset contributes equally to portfolio risk.

    Simple approach: weights inversely proportional to volatility.
    """
    if returns_matrix.shape[0] == 0:
        return np.array([])

    # Inverse volatility
    volatilities = np.std(returns_matrix, axis=1)
    inverse_vol = 1 / (volatilities + 1e-6)

    # Normalize to sum to 1
    weights = inverse_vol / np.sum(inverse_vol)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSIFICATION SCORING
# ─────────────────────────────────────────────────────────────────────────────

def calculate_diversification_score(weights: np.ndarray, returns_matrix: np.ndarray) -> float:
    """
    Calculate diversification score (0-100).

    Higher = more diversified.
    Based on:
    - Number of assets
    - Weight distribution (avoid concentration)
    - Correlations (lower = better)
    """
    if len(weights) == 0:
        return 0

    score = 0

    # Number of assets factor (0-25)
    score += min(25, len(weights) * 2.5)

    # Weight distribution (0-35)
    # HHI = sum of squared weights (1 = single asset, 1/n = equal)
    hhi = np.sum(weights ** 2)
    weight_score = max(0, 35 * (1 - hhi))
    score += weight_score

    # Correlation factor (0-40)
    if len(weights) > 1 and returns_matrix.shape[0] > 1:
        try:
            corr_matrix = np.corrcoef(returns_matrix)
            # Average pairwise correlation (excluding diagonal)
            n = len(corr_matrix)
            avg_corr = (np.sum(corr_matrix) - n) / (n * (n - 1))

            # Lower correlation = better diversification
            corr_score = max(0, 40 * (1 - abs(avg_corr)))
            score += corr_score
        except:
            score += 20  # Default

    return min(100, score)


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE PORTFOLIO OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

async def optimize_portfolio(
    candidate_tickers: list[str],
    strategy: str = "max_sharpe",  # "max_sharpe", "min_variance", "risk_parity"
    max_positions: int = 10,
) -> PortfolioAllocation:
    """
    Optimize portfolio allocation using selected strategy.
    """
    try:
        # Limit candidates
        tickers = candidate_tickers[:max_positions]

        # Get returns
        returns_dict = await calculate_returns_matrix(tickers, days=90)

        if not returns_dict:
            return PortfolioAllocation(
                tickers=[],
                weights=[],
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                diversification_score=0,
                recommendation="❌ Insufficient data",
            )

        # Build matrix
        valid_tickers = list(returns_dict.keys())
        returns_matrix = np.array([returns_dict[t] for t in valid_tickers])

        # Run optimization based on strategy
        if strategy == "max_sharpe":
            weights, ret, vol, sharpe = calculate_max_sharpe_weights(returns_matrix)
        elif strategy == "min_variance":
            weights, ret, vol = calculate_min_variance_weights(returns_matrix)
            sharpe = ret / vol if vol > 0 else 0
        elif strategy == "risk_parity":
            weights = calculate_risk_parity_weights(returns_matrix)
            mean_returns = np.mean(returns_matrix, axis=1) * 252
            ret = np.dot(weights, mean_returns)
            cov = np.cov(returns_matrix) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            sharpe = ret / vol if vol > 0 else 0
        else:
            return PortfolioAllocation(
                tickers=[], weights=[], expected_return=0, expected_volatility=0,
                sharpe_ratio=0, diversification_score=0,
                recommendation=f"❌ Unknown strategy: {strategy}",
            )

        # Calculate diversification
        div_score = calculate_diversification_score(weights, returns_matrix)

        # Generate recommendation
        recommendation = (
            f"✅ Optimized for {strategy}: " +
            ("Excellent diversification" if div_score > 70 else
             "Good diversification" if div_score > 50 else
             "Limited diversification - add more uncorrelated assets")
        )

        return PortfolioAllocation(
            tickers=valid_tickers,
            weights=weights.tolist() if hasattr(weights, 'tolist') else list(weights),
            expected_return=float(ret * 100),
            expected_volatility=float(vol * 100),
            sharpe_ratio=float(sharpe),
            diversification_score=float(div_score),
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        return PortfolioAllocation(
            tickers=[], weights=[], expected_return=0, expected_volatility=0,
            sharpe_ratio=0, diversification_score=0,
            recommendation=f"❌ Optimization error: {e}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# REBALANCING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_rebalancing_needs() -> dict:
    """
    Analyze if portfolio needs rebalancing.

    Compares current allocation to optimal allocation.
    """
    try:
        import broker

        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return {"error": "No positions"}

        # Current allocation
        total_value = sum(float(p.market_value) for p in positions)
        if total_value == 0:
            return {"error": "Zero portfolio value"}

        current_allocation = {
            p.symbol: float(p.market_value) / total_value
            for p in positions
        }

        # Get optimal allocation
        tickers = list(current_allocation.keys())
        optimal = await optimize_portfolio(tickers, strategy="max_sharpe")

        # Compare
        rebalancing_actions = []
        for i, ticker in enumerate(optimal.tickers):
            current_pct = current_allocation.get(ticker, 0)
            optimal_pct = optimal.weights[i] if i < len(optimal.weights) else 0
            difference = optimal_pct - current_pct

            if abs(difference) > 0.05:  # More than 5% difference
                action = "INCREASE" if difference > 0 else "DECREASE"
                rebalancing_actions.append({
                    "ticker": ticker,
                    "current_pct": current_pct * 100,
                    "optimal_pct": optimal_pct * 100,
                    "action": action,
                    "amount_pct": abs(difference) * 100,
                })

        # Sort by largest discrepancy
        rebalancing_actions.sort(key=lambda x: x["amount_pct"], reverse=True)

        return {
            "needs_rebalancing": len(rebalancing_actions) > 0,
            "actions_count": len(rebalancing_actions),
            "actions": rebalancing_actions[:5],  # Top 5
            "current_diversification": calculate_diversification_score(
                np.array(list(current_allocation.values())),
                await calculate_returns_matrix(list(current_allocation.keys())),
            ) if current_allocation else 0,
            "optimal_diversification": optimal.diversification_score,
            "recommendation": (
                "🔄 Significant rebalancing recommended" if len(rebalancing_actions) > 3 else
                "🟡 Minor rebalancing could improve allocation" if rebalancing_actions else
                "✅ Portfolio well-balanced"
            ),
        }

    except Exception as e:
        logger.error(f"Rebalancing analysis failed: {e}")
        return {"error": str(e)}
