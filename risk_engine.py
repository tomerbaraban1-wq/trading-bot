"""
Comprehensive Risk Management Engine
=====================================

Provides advanced risk metrics and scoring:
1. Value at Risk (VaR) - potential loss estimation
2. Sharpe Ratio - risk-adjusted returns
3. Sortino Ratio - downside risk-adjusted returns
4. Maximum Drawdown - worst loss from peak
5. Risk Score - 0-100 composite risk score
6. Position Risk Analysis - per-position risk metrics
7. Portfolio VaR - aggregate portfolio risk
8. Kelly Criterion - optimal position sizing
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position or portfolio."""
    value_at_risk_95: float  # 95% VaR (max loss in 95% of cases)
    value_at_risk_99: float  # 99% VaR (max loss in 99% of cases)
    expected_shortfall: float  # CVaR - average loss beyond VaR
    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return
    max_drawdown: float  # Worst drawdown
    win_rate: float  # % winning trades
    profit_factor: float  # Gross profit / Gross loss
    risk_score: float  # 0-100 composite (lower = safer)
    risk_level: str  # 🟢 LOW, 🟡 MODERATE, 🟠 HIGH, 🔴 EXTREME
    recommendations: List[str]


@dataclass
class PositionRiskAnalysis:
    """Risk analysis for a specific position."""
    ticker: str
    current_pnl: float
    unrealized_loss_potential: float  # If stop loss hit
    unrealized_gain_potential: float  # If TP hit
    risk_reward_ratio: float
    days_held: float
    risk_score: float  # 0-100
    recommendation: str


# ─────────────────────────────────────────────────────────────────────────────
# VALUE AT RISK (VaR)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_var(returns: list[float], confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk.

    VaR_95% = "We are 95% confident that losses won't exceed X dollars"

    Args:
        returns: historical returns (e.g., daily P&L)
        confidence: confidence level (0.95 or 0.99)

    Returns:
        VaR value (negative for losses)
    """
    if len(returns) < 5:
        return 0.0

    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    return float(sorted_returns[index]) if index < len(sorted_returns) else 0.0


def calculate_expected_shortfall(returns: list[float], confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR) - average loss when VaR is exceeded.
    More conservative than VaR.
    """
    if len(returns) < 5:
        return 0.0

    var = calculate_var(returns, confidence)
    tail_losses = [r for r in returns if r <= var]

    return float(np.mean(tail_losses)) if tail_losses else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SHARPE & SORTINO RATIOS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.04) -> float:
    """
    Sharpe Ratio: (return - risk_free_rate) / volatility

    Interpretation:
    > 1.0 = Good
    > 2.0 = Very good
    > 3.0 = Excellent
    """
    if len(returns) < 5:
        return 0.0

    daily_rf = risk_free_rate / 252  # Daily risk-free rate
    excess_returns = [r - daily_rf for r in returns]

    avg_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0.0

    # Annualize: multiply by sqrt(252) for daily returns
    return float(avg_return / std_return * np.sqrt(252))


def calculate_sortino_ratio(returns: list[float], risk_free_rate: float = 0.04) -> float:
    """
    Sortino Ratio: like Sharpe but only penalizes downside volatility.
    Better measure for asymmetric return distributions.
    """
    if len(returns) < 5:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = [r - daily_rf for r in returns]

    downside_returns = [r for r in excess_returns if r < 0]
    if not downside_returns:
        return 0.0

    avg_return = np.mean(excess_returns)
    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0.0

    return float(avg_return / downside_std * np.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_max_drawdown(equity_curve: list[float]) -> dict:
    """
    Calculate maximum drawdown from equity curve.

    Drawdown = peak-to-trough decline
    """
    if len(equity_curve) < 2:
        return {"max_drawdown": 0, "max_drawdown_pct": 0, "duration_days": 0}

    cumulative = np.array(equity_curve)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    drawdown_pct = drawdowns / running_max

    max_dd = float(np.min(drawdowns))
    max_dd_pct = float(np.min(drawdown_pct))

    # Find duration (consecutive days below peak)
    below_peak = drawdowns < 0
    max_duration = 0
    current_duration = 0
    for is_below in below_peak:
        if is_below:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct * 100,
        "duration_days": max_duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# KELLY CRITERION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion: optimal fraction of capital to risk per trade.

    f* = (bp - q) / b
    where:
      b = ratio of average win to average loss
      p = win probability
      q = loss probability (1 - p)

    Returns: optimal fraction (0-1), capped at 0.25 for safety
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    p = win_rate / 100
    q = 1 - p
    b = abs(avg_win / avg_loss)

    kelly = (b * p - q) / b

    # Use "Half Kelly" for safety (more conservative)
    safe_kelly = kelly * 0.5

    # Cap at 25% of capital per trade
    return max(0, min(0.25, safe_kelly))


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE RISK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_comprehensive_risk_metrics(returns: list[float], equity_curve: list[float]) -> RiskMetrics:
    """
    Calculate full risk metrics suite.
    """
    if not returns:
        return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, "🟢 LOW", ["No data"])

    # Calculate all metrics
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    es = calculate_expected_shortfall(returns, 0.95)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    dd_info = calculate_max_drawdown(equity_curve) if equity_curve else {"max_drawdown_pct": 0}

    # Win rate and profit factor
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = (len(wins) / len(returns) * 100) if returns else 0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    # Composite Risk Score (0-100)
    # Lower is safer
    risk_score = 0

    # Volatility component (0-30 points)
    volatility = np.std(returns)
    risk_score += min(30, volatility * 100)

    # Drawdown component (0-30 points)
    risk_score += min(30, abs(dd_info["max_drawdown_pct"]))

    # Sharpe ratio component (0-20 points)
    if sharpe < 0:
        risk_score += 20
    elif sharpe < 1:
        risk_score += 15
    elif sharpe < 2:
        risk_score += 10
    else:
        risk_score += 5

    # Win rate component (0-20 points)
    if win_rate < 40:
        risk_score += 20
    elif win_rate < 50:
        risk_score += 15
    elif win_rate < 60:
        risk_score += 10
    elif win_rate < 70:
        risk_score += 5

    # Classify risk level
    if risk_score < 25:
        risk_level = "🟢 LOW - Conservative risk profile"
    elif risk_score < 50:
        risk_level = "🟡 MODERATE - Balanced risk"
    elif risk_score < 75:
        risk_level = "🟠 HIGH - Elevated risk"
    else:
        risk_level = "🔴 EXTREME - Very high risk"

    # Generate recommendations
    recommendations = []

    if sharpe < 1:
        recommendations.append("📉 Sharpe Ratio בנמוך - שקול שיפור strategy או reduce risk")
    if abs(dd_info["max_drawdown_pct"]) > 15:
        recommendations.append(f"⚠️ Drawdown מקסימלי ({abs(dd_info['max_drawdown_pct']):.1f}%) גבוה מדי")
    if win_rate < 45:
        recommendations.append("🎯 Win rate נמוך - שקול הגבהת MIN_BUY_SCORE")
    if profit_factor < 1.5:
        recommendations.append("💰 Profit factor נמוך - הרוויחים לא מכסים מספיק על ההפסדים")
    if volatility > 0.05:
        recommendations.append("📊 Volatility גבוהה - שקול diversification או הקטנת position size")

    if not recommendations:
        recommendations.append("✅ Profile סיכון בריא - המשך כך!")

    return RiskMetrics(
        value_at_risk_95=var_95,
        value_at_risk_99=var_99,
        expected_shortfall=es,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=dd_info["max_drawdown_pct"],
        win_rate=win_rate,
        profit_factor=profit_factor,
        risk_score=risk_score,
        risk_level=risk_level,
        recommendations=recommendations,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_portfolio_risk() -> dict:
    """
    Analyze risk for entire portfolio.

    Includes:
    - Total portfolio exposure
    - Per-position risk
    - Concentration risk
    - Correlation risk
    """
    try:
        import database
        import broker

        conn = database.get_connection()

        # Get returns from last 30 days
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        rows = conn.execute("""
            SELECT pnl_gross, exit_time
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= ?
            ORDER BY exit_time ASC
        """, (thirty_days_ago,)).fetchall()

        if not rows:
            return {"error": "No trade history available"}

        returns = [r[0] for r in rows]
        equity_curve = list(np.cumsum(returns))

        # Comprehensive risk metrics
        risk_metrics = calculate_comprehensive_risk_metrics(returns, equity_curve)

        # Kelly Criterion
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = (len(wins) / len(returns) * 100) if returns else 0

        kelly = calculate_kelly_criterion(win_rate, avg_win, avg_loss)

        # Get current positions for concentration analysis
        positions = await asyncio.to_thread(broker.get_positions)
        total_value = sum(float(p.get('market_value', 0)) for p in positions) if positions else 0

        concentration = {}
        if positions and total_value > 0:
            for p in positions:
                pct = (float(p.get('market_value', 0)) / total_value * 100)
                if pct > 25:  # Concentration risk threshold
                    concentration[p.get('ticker')] = {
                        "pct_of_portfolio": pct,
                        "warning": "⚠️ Over 25% of portfolio - high concentration risk"
                    }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_metrics": {
                "var_95": risk_metrics.value_at_risk_95,
                "var_99": risk_metrics.value_at_risk_99,
                "expected_shortfall": risk_metrics.expected_shortfall,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "sortino_ratio": risk_metrics.sortino_ratio,
                "max_drawdown_pct": risk_metrics.max_drawdown,
                "win_rate": risk_metrics.win_rate,
                "profit_factor": risk_metrics.profit_factor,
                "risk_score": risk_metrics.risk_score,
                "risk_level": risk_metrics.risk_level,
            },
            "kelly_criterion": {
                "optimal_position_size": kelly,
                "interpretation": (
                    f"Risk {kelly*100:.1f}% per trade for optimal growth"
                    if kelly > 0 else
                    "Negative expectancy - reduce risk or improve strategy"
                ),
            },
            "concentration_risk": concentration,
            "recommendations": risk_metrics.recommendations,
        }

    except Exception as e:
        logger.error(f"Portfolio risk analysis failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# POSITION-LEVEL RISK
# ─────────────────────────────────────────────────────────────────────────────

def analyze_position_risk(
    ticker: str,
    entry_price: float,
    current_price: float,
    quantity: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    days_held: float,
) -> PositionRiskAnalysis:
    """
    Analyze risk for an individual position.
    """
    current_pnl = (current_price - entry_price) * quantity

    # Calculate potential loss/gain
    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
    take_profit_price = entry_price * (1 + take_profit_pct / 100)

    unrealized_loss = (stop_loss_price - current_price) * quantity
    unrealized_gain = (take_profit_price - current_price) * quantity

    # Risk:Reward ratio
    rr_ratio = abs(unrealized_gain / unrealized_loss) if unrealized_loss != 0 else 0

    # Calculate risk score (0-100)
    risk_score = 0

    # Days held component (positions getting stale)
    if days_held > 30:
        risk_score += 25
    elif days_held > 14:
        risk_score += 15
    elif days_held > 7:
        risk_score += 5

    # Current loss component
    pnl_pct = ((current_price - entry_price) / entry_price * 100)
    if pnl_pct < -10:
        risk_score += 35
    elif pnl_pct < -5:
        risk_score += 20
    elif pnl_pct < 0:
        risk_score += 10

    # R:R ratio component
    if rr_ratio < 1:
        risk_score += 25
    elif rr_ratio < 2:
        risk_score += 10

    # Generate recommendation
    if risk_score > 60:
        recommendation = "🔴 CONSIDER CLOSING - High risk position"
    elif risk_score > 40:
        recommendation = "🟠 TIGHTEN STOP LOSS - Elevated risk"
    elif risk_score > 20:
        recommendation = "🟡 MONITOR - Moderate risk"
    else:
        recommendation = "🟢 HOLD - Healthy position"

    return PositionRiskAnalysis(
        ticker=ticker,
        current_pnl=current_pnl,
        unrealized_loss_potential=unrealized_loss,
        unrealized_gain_potential=unrealized_gain,
        risk_reward_ratio=rr_ratio,
        days_held=days_held,
        risk_score=risk_score,
        recommendation=recommendation,
    )
