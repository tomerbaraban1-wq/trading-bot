"""
Dynamic Protection Layer
=========================

Advanced position protection that adapts to market conditions:

1. Trailing Stop Loss - moves up with price
2. Volatility-Adjusted Stops - wider in volatile markets
3. Time-Based Stops - tighten as time passes
4. Profit Protection Levels - lock in gains
5. Breakeven Move - move stop to breakeven after threshold
6. ATR-Based Stops - use Average True Range
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProtectionSettings:
    """Dynamic protection configuration."""
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 2.0  # Trail by 2%
    breakeven_threshold_pct: float = 3.0  # Move to BE after +3%
    profit_lock_levels: list = None  # Lock in profit at intervals
    volatility_multiplier: float = 1.0
    time_decay_enabled: bool = True
    max_holding_days: int = 14


@dataclass
class StopLossUpdate:
    """A stop loss update recommendation."""
    ticker: str
    current_price: float
    current_stop: float
    new_stop: float
    reason: str
    locked_profit_pct: float


# ─────────────────────────────────────────────────────────────────────────────
# ATR CALCULATION (Average True Range)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """
    Calculate Average True Range - measures volatility.

    ATR = average of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|
    """
    if len(highs) < period + 1:
        return 0

    true_ranges = []
    for i in range(1, len(highs)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i-1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return 0

    return float(np.mean(true_ranges[-period:]))


# ─────────────────────────────────────────────────────────────────────────────
# TRAILING STOP LOSS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    current_price: float,
    trailing_pct: float = 2.0,
) -> dict:
    """
    Calculate trailing stop loss.

    Stop trails up with price, but never moves down.
    Locks in profits as price rises.
    """
    # Trailing stop based on highest reached
    trailing_stop = highest_price * (1 - trailing_pct / 100)

    # Initial stop (if price hasn't moved up much)
    initial_stop = entry_price * (1 - trailing_pct / 100)

    # Use higher of the two
    stop_price = max(initial_stop, trailing_stop)

    # Calculate locked profit
    if current_price > entry_price:
        unrealized_gain_pct = (current_price - entry_price) / entry_price * 100
        locked_profit_pct = (stop_price - entry_price) / entry_price * 100
        protection_active = locked_profit_pct > 0
    else:
        unrealized_gain_pct = (current_price - entry_price) / entry_price * 100
        locked_profit_pct = 0
        protection_active = False

    return {
        "stop_price": stop_price,
        "unrealized_gain_pct": unrealized_gain_pct,
        "locked_profit_pct": locked_profit_pct,
        "protection_active": protection_active,
        "distance_from_stop_pct": ((current_price - stop_price) / current_price * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY-ADJUSTED STOPS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_volatility_stop(
    entry_price: float,
    atr: float,
    multiplier: float = 2.5,
    side: str = "long",
) -> float:
    """
    Calculate stop based on ATR (Average True Range).

    Wider stops in volatile markets, tighter in calm markets.
    Multiplier of 2.5x ATR is standard.
    """
    stop_distance = atr * multiplier

    if side == "long":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


# ─────────────────────────────────────────────────────────────────────────────
# TIME-DECAY STOPS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_time_decay_stop(
    entry_price: float,
    current_price: float,
    days_held: float,
    initial_stop_pct: float = 2.0,
    max_holding_days: int = 14,
) -> dict:
    """
    Tighten stop as time passes.

    Logic: The longer a position is held without hitting TP,
    the more likely it should be cut to free up capital.
    """
    # Linear decay
    decay_factor = min(1, days_held / max_holding_days)

    # As decay_factor approaches 1, stop tightens
    # Day 0: full stop loss (2%)
    # Day max: very tight stop (0.5% from current price)

    if days_held < 3:
        # First 3 days: full stop
        stop_pct = initial_stop_pct
        from_price = "entry"
        stop_price = entry_price * (1 - stop_pct / 100)
    elif days_held < 7:
        # Day 3-7: stop trails to breakeven
        stop_price = max(
            entry_price * (1 - initial_stop_pct / 100),
            entry_price * (1 - initial_stop_pct / 100 * 0.5),
        )
        from_price = "entry"
        stop_pct = initial_stop_pct * 0.5
    else:
        # Day 7+: tight trail from current price
        tighten_pct = 0.5 + (decay_factor * 1.5)
        stop_price = current_price * (1 - tighten_pct / 100)
        from_price = "current"
        stop_pct = tighten_pct

    return {
        "stop_price": stop_price,
        "stop_pct": stop_pct,
        "anchor": from_price,
        "decay_factor": decay_factor,
        "reasoning": f"Day {days_held:.1f}/{max_holding_days}: decay {decay_factor:.0%}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROFIT PROTECTION LEVELS
# ─────────────────────────────────────────────────────────────────────────────

def get_profit_protection_levels(
    entry_price: float,
    current_price: float,
) -> dict:
    """
    Set profit protection levels.

    As price rises, lock in gains at predefined thresholds:
    - +3%: Move stop to breakeven
    - +5%: Lock in +2%
    - +8%: Lock in +5%
    - +12%: Lock in +8%
    - +20%: Lock in +15%
    """
    gain_pct = (current_price - entry_price) / entry_price * 100

    protection_levels = [
        (20, 15, "🚀 Major gain - lock in +15%"),
        (12, 8, "🎯 Strong gain - lock in +8%"),
        (8, 5, "💰 Good gain - lock in +5%"),
        (5, 2, "✅ Decent gain - lock in +2%"),
        (3, 0, "🎯 Move to breakeven"),
    ]

    active_protection = None
    suggested_stop_pct = 0

    for threshold, lock_pct, message in protection_levels:
        if gain_pct >= threshold:
            active_protection = message
            suggested_stop_pct = lock_pct
            break

    if active_protection:
        new_stop = entry_price * (1 + suggested_stop_pct / 100)
    else:
        new_stop = None

    return {
        "current_gain_pct": gain_pct,
        "active_protection": active_protection,
        "suggested_stop_pct": suggested_stop_pct,
        "new_stop_price": new_stop,
        "next_threshold": next(
            (t for t, _, _ in protection_levels if t > gain_pct),
            None
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE PROTECTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_position_protection(
    ticker: str,
    entry_price: float,
    current_price: float,
    current_stop: float,
    days_held: float,
    highest_price: Optional[float] = None,
) -> dict:
    """
    Comprehensive protection analysis for a position.

    Recommends the optimal stop loss based on:
    - Current price action
    - Volatility (ATR)
    - Time held
    - Profit level
    """
    try:
        import yfinance as yf

        # Get recent price data for ATR
        data = yf.download(ticker, period="30d", progress=False, auto_adjust=True)
        if data.empty:
            atr = 0
        else:
            highs = [float(v) for v in data["High"].squeeze().dropna().values]
            lows = [float(v) for v in data["Low"].squeeze().dropna().values]
            closes = [float(v) for v in data["Close"].squeeze().dropna().values]
            atr = calculate_atr(highs, lows, closes)

        if highest_price is None:
            highest_price = current_price

        # Calculate all stop loss recommendations
        trailing = calculate_trailing_stop(entry_price, highest_price, current_price, 2.0)
        volatility_stop = calculate_volatility_stop(entry_price, atr, 2.5, "long")
        time_decay = calculate_time_decay_stop(entry_price, current_price, days_held)
        profit_protection = get_profit_protection_levels(entry_price, current_price)

        # Choose the most appropriate stop (highest, but reasonable)
        candidate_stops = [
            trailing["stop_price"],
            volatility_stop,
            time_decay["stop_price"],
        ]

        if profit_protection["new_stop_price"]:
            candidate_stops.append(profit_protection["new_stop_price"])

        # Filter to stops below current price (otherwise we'd close immediately)
        valid_stops = [s for s in candidate_stops if s < current_price * 0.999]

        if not valid_stops:
            optimal_stop = current_stop
        else:
            # Use highest valid stop (most protective)
            optimal_stop = max(valid_stops)

        # Determine reason for recommendation
        if optimal_stop == trailing["stop_price"]:
            reason = f"Trailing stop ({trailing['locked_profit_pct']:+.1f}% locked)"
        elif optimal_stop == volatility_stop:
            reason = f"ATR-based ({atr:.2f} volatility)"
        elif optimal_stop == time_decay["stop_price"]:
            reason = f"Time decay (day {days_held:.1f})"
        elif profit_protection["new_stop_price"] and optimal_stop == profit_protection["new_stop_price"]:
            reason = profit_protection["active_protection"]
        else:
            reason = "Current stop maintained"

        # Calculate improvement
        improvement_pct = ((optimal_stop - current_stop) / current_stop * 100) if current_stop > 0 else 0

        return {
            "ticker": ticker,
            "current_price": current_price,
            "entry_price": entry_price,
            "current_stop": current_stop,
            "recommended_stop": optimal_stop,
            "improvement_pct": improvement_pct,
            "should_update": improvement_pct > 1,  # Only update if >1% better
            "reason": reason,
            "details": {
                "trailing_stop": trailing,
                "atr": atr,
                "volatility_stop": volatility_stop,
                "time_decay": time_decay,
                "profit_protection": profit_protection,
            },
            "current_gain_pct": ((current_price - entry_price) / entry_price * 100),
        }

    except Exception as e:
        logger.error(f"Protection analysis failed for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROTECTION UPDATE
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_all_position_protections() -> list[dict]:
    """Analyze protection for all open positions."""
    try:
        import broker
        positions = await asyncio.to_thread(broker.get_positions)

        if not positions:
            return []

        recommendations = []
        for p in positions:
            try:
                entry_price = float(p.get('avg_entry_price', 0))
                current_price = float(p.get('current_price', 0))

                # Estimate days held (would need actual purchase date from DB)
                days_held = 5  # Placeholder

                # Get current stop from database (placeholder logic)
                current_stop = entry_price * 0.98  # Default 2% stop

                analysis = await analyze_position_protection(
                    ticker=p.get('ticker'),
                    entry_price=entry_price,
                    current_price=current_price,
                    current_stop=current_stop,
                    days_held=days_held,
                )

                if "error" not in analysis and analysis.get("should_update"):
                    recommendations.append(analysis)

            except Exception as e:
                logger.debug(f"Protection analysis for {p.get('ticker')} failed: {e}")

        return recommendations

    except Exception as e:
        logger.error(f"Batch protection analysis failed: {e}")
        return []
