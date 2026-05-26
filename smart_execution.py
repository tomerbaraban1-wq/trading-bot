"""
Smart Order Execution Module
=============================

Advanced order execution strategies to minimize market impact and improve fill prices:

1. TWAP (Time-Weighted Average Price) - splits order over time
2. VWAP (Volume-Weighted Average Price) - splits proportional to volume
3. Iceberg Orders - hides large orders by showing only small portions
4. Implementation Shortfall - balances execution vs. market impact
5. Adaptive Limit Orders - adjusts based on market conditions
6. Smart Order Routing - chooses best execution method
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Plan for executing a large order."""
    strategy: str  # "TWAP", "VWAP", "ICEBERG", "MARKET", "LIMIT"
    total_quantity: float
    num_slices: int
    slice_size: float
    interval_seconds: float
    price_limit: Optional[float]
    estimated_completion_time: float  # minutes
    reasoning: str


@dataclass
class ExecutionStatus:
    """Status of an ongoing execution."""
    plan: ExecutionPlan
    slices_filled: int
    quantity_filled: float
    quantity_remaining: float
    avg_fill_price: float
    started_at: str
    progress_pct: float


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_execution_strategy(
    quantity: float,
    avg_daily_volume: float,
    current_volatility: float,
    urgency: str = "normal",  # "low", "normal", "high"
    current_spread_pct: float = 0.5,
) -> ExecutionPlan:
    """
    Select optimal execution strategy based on order characteristics.

    Rules:
    - Small order (<1% ADV) → MARKET or limit
    - Medium order (1-5% ADV) → TWAP
    - Large order (5-20% ADV) → VWAP
    - Huge order (>20% ADV) → ICEBERG with VWAP
    """
    volume_pct = (quantity / avg_daily_volume * 100) if avg_daily_volume > 0 else 100

    # Determine number of slices based on size and urgency
    if urgency == "high":
        # Need fast execution
        if volume_pct < 2:
            strategy = "MARKET"
            num_slices = 1
            interval = 0
        else:
            strategy = "TWAP"
            num_slices = min(5, int(volume_pct / 2))
            interval = 60  # 1 minute between slices

    elif urgency == "low":
        # Can take time, optimize for price
        if volume_pct > 5:
            strategy = "VWAP"
            num_slices = int(volume_pct * 1.5)
            interval = 300  # 5 minutes between slices
        else:
            strategy = "TWAP"
            num_slices = max(3, int(volume_pct * 2))
            interval = 180  # 3 minutes

    else:  # Normal urgency
        if volume_pct < 1:
            strategy = "LIMIT"
            num_slices = 1
            interval = 0
        elif volume_pct < 5:
            strategy = "TWAP"
            num_slices = max(3, int(volume_pct * 1.5))
            interval = 120  # 2 minutes
        elif volume_pct < 20:
            strategy = "VWAP"
            num_slices = max(5, int(volume_pct))
            interval = 180
        else:
            strategy = "ICEBERG"
            num_slices = max(10, int(volume_pct / 2))
            interval = 240

    # Adjust for high volatility (smaller slices, more frequent)
    if current_volatility > 40:
        num_slices = int(num_slices * 1.5)
        interval = int(interval * 0.7)

    slice_size = quantity / num_slices if num_slices > 0 else quantity
    estimated_time = (num_slices * interval) / 60  # in minutes

    # Build reasoning
    reasoning_parts = []
    reasoning_parts.append(f"Order is {volume_pct:.1f}% of ADV")
    reasoning_parts.append(f"Volatility: {current_volatility:.1f}")
    reasoning_parts.append(f"Urgency: {urgency}")
    reasoning_parts.append(f"Selected: {strategy}")

    return ExecutionPlan(
        strategy=strategy,
        total_quantity=quantity,
        num_slices=num_slices,
        slice_size=slice_size,
        interval_seconds=interval,
        price_limit=None,
        estimated_completion_time=estimated_time,
        reasoning=" | ".join(reasoning_parts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE LIMIT PRICING
# ─────────────────────────────────────────────────────────────────────────────

def calculate_adaptive_limit_price(
    current_price: float,
    side: str,  # "buy" or "sell"
    spread_pct: float,
    volatility: float,
    urgency: str = "normal",
) -> float:
    """
    Calculate intelligent limit price based on market conditions.

    For buys: limit price = current_price * (1 + offset)
    For sells: limit price = current_price * (1 - offset)

    Offset adjusts based on volatility and urgency.
    """
    # Base offset is 50% of spread
    base_offset = spread_pct / 2 / 100

    # Volatility adjustment: wider in volatile markets
    vol_multiplier = 1 + (volatility / 100)
    offset = base_offset * vol_multiplier

    # Urgency adjustment
    if urgency == "high":
        offset *= 2  # Pay more to ensure fill
    elif urgency == "low":
        offset *= 0.5  # Be patient for better price

    if side == "buy":
        # Buy slightly above market to ensure fill
        return current_price * (1 + offset)
    else:  # sell
        # Sell slightly below market
        return current_price * (1 - offset)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET IMPACT ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_market_impact(
    quantity: float,
    avg_daily_volume: float,
    volatility: float,
    spread_pct: float,
) -> dict:
    """
    Estimate market impact of an order.

    Uses square-root model:
    Impact = sigma * sqrt(participation_rate) * direction

    Where participation_rate = quantity / volume_during_execution
    """
    participation_rate = quantity / avg_daily_volume if avg_daily_volume > 0 else 1.0

    # Volatility-adjusted impact
    daily_vol = volatility / 100
    impact_pct = daily_vol * np.sqrt(participation_rate) * 100

    # Add half-spread cost
    half_spread = spread_pct / 2

    total_cost_pct = impact_pct + half_spread

    # Classify
    if total_cost_pct < 0.1:
        cost_level = "🟢 LOW"
    elif total_cost_pct < 0.5:
        cost_level = "🟡 MODERATE"
    elif total_cost_pct < 1.0:
        cost_level = "🟠 HIGH"
    else:
        cost_level = "🔴 EXTREME"

    return {
        "participation_rate_pct": participation_rate * 100,
        "estimated_impact_pct": impact_pct,
        "half_spread_cost_pct": half_spread,
        "total_cost_pct": total_cost_pct,
        "cost_level": cost_level,
        "recommendation": (
            "Split order to reduce impact" if total_cost_pct > 0.5 else
            "Execute normally" if total_cost_pct < 0.2 else
            "Use TWAP to minimize impact"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SLIPPAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_slippage(
    order_size_usd: float,
    avg_daily_volume_usd: float,
    volatility: float,
    spread_pct: float,
) -> dict:
    """
    Predict expected slippage for an order.

    Slippage = difference between expected and actual fill price.
    """
    # Size relative to typical volume
    size_factor = order_size_usd / avg_daily_volume_usd if avg_daily_volume_usd > 0 else 1.0

    # Base slippage from spread
    base_slippage_pct = spread_pct / 2

    # Size-adjusted slippage
    size_slippage = size_factor * volatility * 0.3

    # Total expected slippage
    total_slippage_pct = base_slippage_pct + size_slippage

    expected_slippage_usd = order_size_usd * (total_slippage_pct / 100)

    return {
        "expected_slippage_pct": total_slippage_pct,
        "expected_slippage_usd": expected_slippage_usd,
        "base_spread_cost": base_slippage_pct,
        "size_impact": size_slippage,
        "recommendation": (
            "✅ Acceptable slippage" if total_slippage_pct < 0.2 else
            "🟡 Consider TWAP/VWAP" if total_slippage_pct < 0.5 else
            "🔴 High slippage - split order"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE EXECUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_optimal_execution(
    ticker: str,
    quantity: float,
    side: str,
    urgency: str = "normal",
) -> dict:
    """
    Comprehensive analysis for optimal order execution.

    Returns recommended execution plan with all metrics.
    """
    try:
        import yfinance as yf

        # Get market data
        ticker_info = yf.Ticker(ticker)
        history = ticker_info.history(period="30d")

        if history.empty:
            return {"error": "No data available"}

        current_price = history["Close"].iloc[-1]
        avg_volume = history["Volume"].mean()
        avg_daily_volume_usd = avg_volume * current_price

        # Calculate volatility (annualized)
        returns = history["Close"].pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252) * 100)

        # Estimate spread (typically 0.05-0.5% for liquid stocks)
        spread_pct = 0.1  # Default 10 basis points

        # Calculate order metrics
        order_value_usd = quantity * current_price

        # Select strategy
        plan = select_execution_strategy(
            quantity=quantity,
            avg_daily_volume=avg_volume,
            current_volatility=volatility,
            urgency=urgency,
            current_spread_pct=spread_pct,
        )

        # Calculate adaptive limit price
        limit_price = calculate_adaptive_limit_price(
            current_price=current_price,
            side=side,
            spread_pct=spread_pct,
            volatility=volatility,
            urgency=urgency,
        )

        # Estimate impact and slippage
        impact = estimate_market_impact(
            quantity=quantity,
            avg_daily_volume=avg_volume,
            volatility=volatility,
            spread_pct=spread_pct,
        )

        slippage = predict_slippage(
            order_size_usd=order_value_usd,
            avg_daily_volume_usd=avg_daily_volume_usd,
            volatility=volatility,
            spread_pct=spread_pct,
        )

        return {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_details": {
                "quantity": quantity,
                "side": side,
                "current_price": current_price,
                "order_value_usd": order_value_usd,
                "urgency": urgency,
            },
            "market_metrics": {
                "avg_daily_volume": avg_volume,
                "volatility_pct": volatility,
                "estimated_spread_pct": spread_pct,
            },
            "execution_plan": {
                "strategy": plan.strategy,
                "num_slices": plan.num_slices,
                "slice_size": plan.slice_size,
                "interval_seconds": plan.interval_seconds,
                "estimated_completion_minutes": plan.estimated_completion_time,
                "recommended_limit_price": limit_price,
                "reasoning": plan.reasoning,
            },
            "cost_analysis": {
                "market_impact": impact,
                "expected_slippage": slippage,
                "total_cost_estimate_usd": order_value_usd * (impact["total_cost_pct"] / 100),
            },
        }

    except Exception as e:
        logger.error(f"Execution analysis failed for {ticker}: {e}")
        return {"error": str(e)}
