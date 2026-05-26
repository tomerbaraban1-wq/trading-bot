"""
Compound Growth Engine
=======================

Auto-reinvests profits to maximize compound growth.

Features:
1. Automatic profit reinvestment
2. Compound growth tracking
3. Future value projections
4. Reinvestment schedules
5. Tax-efficient compounding
6. Goal-based growth planning
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompoundProjection:
    """Future value projection."""
    initial_amount: float
    monthly_contribution: float
    annual_return_pct: float
    years: int
    final_amount: float
    total_contributions: float
    total_growth: float
    growth_pct: float
    monthly_breakdown: list[dict]


@dataclass
class CompoundingPlan:
    """Plan for systematic compounding."""
    current_capital: float
    target_capital: float
    monthly_savings: float
    expected_annual_return: float
    estimated_years_to_target: float
    monthly_growth_rate: float
    next_milestone: dict


# ─────────────────────────────────────────────────────────────────────────────
# FUTURE VALUE PROJECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_compound_growth(
    initial: float,
    monthly_contribution: float,
    annual_return_pct: float,
    years: int,
) -> CompoundProjection:
    """
    Calculate compound growth with monthly contributions.

    Uses standard compound interest formula:
    FV = PV * (1+r)^n + PMT * [((1+r)^n - 1) / r]
    """
    monthly_rate = (annual_return_pct / 100) / 12
    months = years * 12

    monthly_breakdown = []
    current = initial

    for month in range(1, months + 1):
        # Add monthly contribution at start of month
        current += monthly_contribution

        # Apply monthly growth
        current *= (1 + monthly_rate)

        # Record yearly snapshots
        if month % 12 == 0:
            year = month // 12
            monthly_breakdown.append({
                "year": year,
                "value": current,
                "contributions_to_date": initial + (monthly_contribution * month),
                "growth_to_date": current - (initial + monthly_contribution * month),
            })

    total_contributions = initial + (monthly_contribution * months)
    total_growth = current - total_contributions
    growth_pct = (total_growth / total_contributions * 100) if total_contributions else 0

    return CompoundProjection(
        initial_amount=initial,
        monthly_contribution=monthly_contribution,
        annual_return_pct=annual_return_pct,
        years=years,
        final_amount=current,
        total_contributions=total_contributions,
        total_growth=total_growth,
        growth_pct=growth_pct,
        monthly_breakdown=monthly_breakdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GOAL-BASED PLANNING
# ─────────────────────────────────────────────────────────────────────────────

def calculate_path_to_goal(
    current_capital: float,
    target_capital: float,
    monthly_savings: float,
    expected_annual_return: float,
) -> CompoundingPlan:
    """
    Calculate how long until reaching a target capital amount.

    Solves for time given:
    - Current capital
    - Target capital
    - Monthly savings
    - Expected return
    """
    monthly_rate = (expected_annual_return / 100) / 12

    current = current_capital
    months = 0

    # Simulate growth month by month
    while current < target_capital and months < 12 * 50:  # Cap at 50 years
        current += monthly_savings
        current *= (1 + monthly_rate)
        months += 1

    years = months / 12

    # Find next milestone
    milestones = [10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000, 10000000]
    next_milestone = next((m for m in milestones if m > current_capital), target_capital)

    months_to_milestone = 0
    current_sim = current_capital
    while current_sim < next_milestone and months_to_milestone < 12 * 50:
        current_sim += monthly_savings
        current_sim *= (1 + monthly_rate)
        months_to_milestone += 1

    monthly_growth_rate = ((target_capital / current_capital) ** (1/months) - 1) if months > 0 else 0

    return CompoundingPlan(
        current_capital=current_capital,
        target_capital=target_capital,
        monthly_savings=monthly_savings,
        expected_annual_return=expected_annual_return,
        estimated_years_to_target=years,
        monthly_growth_rate=monthly_growth_rate * 100,
        next_milestone={
            "amount": next_milestone,
            "months_to_reach": months_to_milestone,
            "years_to_reach": months_to_milestone / 12,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC REINVESTMENT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

async def get_reinvestment_summary() -> dict:
    """
    Track how profits have been reinvested.

    Shows:
    - Total profits reinvested
    - Capital growth from reinvestment
    - Compound effect over time
    """
    try:
        import database
        conn = database.get_connection()

        # Get all profitable trades
        rows = conn.execute("""
            SELECT pnl_gross, exit_time
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND pnl_gross > 0
            ORDER BY exit_time ASC
        """).fetchall()

        if not rows:
            return {"error": "No trade history"}

        total_profits = sum(r[0] for r in rows)
        num_trades = len(rows)

        # Calculate growth trajectory
        # Assume all profits reinvested
        initial = 1000  # Starting capital assumption
        current = initial
        cumulative_growth = []

        for pnl, exit_time in rows:
            current += pnl
            cumulative_growth.append({
                "date": exit_time,
                "value": current,
                "growth_pct": ((current - initial) / initial * 100),
            })

        # Statistics
        return {
            "total_winning_trades": num_trades,
            "total_profits_reinvested": total_profits,
            "growth_from_compounding": total_profits,
            "compounded_growth_pct": ((current - initial) / initial * 100),
            "current_value_estimate": current,
            "average_profit_per_trade": total_profits / num_trades if num_trades else 0,
            "trajectory": cumulative_growth[-20:],  # Last 20 trades
            "interpretation": (
                "🚀 Excellent compounding effect!" if cumulative_growth[-1]["growth_pct"] > 50 else
                "📈 Growing steadily through compounding" if cumulative_growth[-1]["growth_pct"] > 20 else
                "📊 Early compounding stage" if cumulative_growth[-1]["growth_pct"] > 0 else
                "⚠️ Compounding negative - review strategy"
            ),
        }

    except Exception as e:
        logger.error(f"Reinvestment summary failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compare_growth_scenarios(
    initial: float,
    monthly_contribution: float,
    years: int = 10,
) -> dict:
    """
    Compare growth under different return scenarios.

    Conservative: 5%/year (savings + bonds)
    Moderate: 10%/year (market average)
    Aggressive: 15%/year (active trading)
    Excellent: 20%/year (skilled trading)
    """
    scenarios = {
        "Conservative (5%)": calculate_compound_growth(initial, monthly_contribution, 5, years),
        "Moderate (10%)": calculate_compound_growth(initial, monthly_contribution, 10, years),
        "Aggressive (15%)": calculate_compound_growth(initial, monthly_contribution, 15, years),
        "Excellent (20%)": calculate_compound_growth(initial, monthly_contribution, 20, years),
        "Exceptional (30%)": calculate_compound_growth(initial, monthly_contribution, 30, years),
    }

    return {
        "initial_capital": initial,
        "monthly_contribution": monthly_contribution,
        "years": years,
        "total_contributions": initial + (monthly_contribution * 12 * years),
        "scenarios": [
            {
                "name": name,
                "final_amount": proj.final_amount,
                "total_growth": proj.total_growth,
                "growth_pct": proj.growth_pct,
                "multiplier": proj.final_amount / initial if initial else 0,
            }
            for name, proj in scenarios.items()
        ],
        "key_insight": (
            f"Going from 10% to 20% annual returns over {years} years "
            f"could nearly DOUBLE your final amount due to compound effect"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-COMPOUND RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

async def get_compounding_strategy() -> dict:
    """
    Generate personalized compounding strategy based on current portfolio.
    """
    try:
        import broker
        from continuous_learner import track_live_performance

        # Get current portfolio value
        positions = await asyncio.to_thread(broker.get_positions)
        portfolio_value = sum(float(p.market_value) for p in positions) if positions else 0

        # Get current performance
        perf = await asyncio.to_thread(track_live_performance)

        # Estimate annual return from recent performance
        # If win rate is 60% with 5% avg win and -2% avg loss:
        # Expected return = 0.6*5% - 0.4*2% = 2.2% per trade
        # If 50 trades/month: 110% / year (theoretical max)

        if perf.win_rate_today > 0 and perf.total_trades_today > 0:
            estimated_annual_return = min(50, perf.win_rate_today * 0.5)  # Conservative estimate
        else:
            estimated_annual_return = 15  # Default expectation

        # Generate plans for different targets
        targets = [
            ("Short-term", portfolio_value * 1.5, 500),
            ("Medium-term", portfolio_value * 3, 1000),
            ("Long-term", portfolio_value * 10, 2000),
        ]

        plans = []
        for name, target, monthly in targets:
            if portfolio_value > 0:
                plan = calculate_path_to_goal(
                    current_capital=portfolio_value,
                    target_capital=target,
                    monthly_savings=monthly,
                    expected_annual_return=estimated_annual_return,
                )
                plans.append({
                    "name": name,
                    "target": target,
                    "years_to_target": plan.estimated_years_to_target,
                    "monthly_savings_needed": monthly,
                })

        return {
            "current_portfolio_value": portfolio_value,
            "estimated_annual_return_pct": estimated_annual_return,
            "compounding_principle": (
                "Reinvesting profits causes exponential growth over time. "
                "Even small consistent gains can lead to massive wealth."
            ),
            "growth_plans": plans,
            "recommended_action": (
                "✅ Keep reinvesting all profits" if estimated_annual_return > 15 else
                "💡 Add monthly contributions to accelerate growth"
            ),
        }

    except Exception as e:
        logger.error(f"Compounding strategy failed: {e}")
        return {"error": str(e)}
