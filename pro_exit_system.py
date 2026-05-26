"""
Professional Exit System
=========================

Pro traders never let winners turn into losers.
They also exit quickly on failed breakouts.

Rules:
1. Lock in 50% profit at 1.5× initial risk
2. Move stop to breakeven at 1× initial risk
3. Trail remaining at ATR×2 (not ATR×3)
4. Exit entire position if daily close breaks below key MA
5. Failed breakout rule: close immediately if stock drops back below entry level after breakout
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """A professional exit signal."""
    action: str             # "SELL_ALL", "SELL_HALF", "TIGHTEN_STOP", "HOLD"
    urgency: str            # "immediate", "end_of_day", "monitor"
    reason: str
    new_stop: Optional[float] = None
    qty_to_sell: Optional[float] = None


def calculate_initial_risk(entry_price: float, stop_price: float) -> float:
    """Calculate initial risk in dollar terms per share."""
    return max(0, entry_price - stop_price)


def check_profit_target_levels(
    entry_price: float,
    current_price: float,
    stop_price: float,
) -> dict:
    """
    Calculate profit target levels based on initial risk (R).

    Professional standard:
    1R = stop distance = breakeven signal
    2R = partial profit taking (sell 50%)
    3R = sell remaining, or trail tightly
    """
    initial_risk = entry_price - stop_price
    if initial_risk <= 0:
        return {"r_multiple": 0, "level_1r": entry_price, "level_2r": entry_price, "level_3r": entry_price}

    r_multiple = (current_price - entry_price) / initial_risk

    return {
        "r_multiple": r_multiple,
        "initial_risk": initial_risk,
        "level_1r": entry_price + initial_risk,       # 1R: move stop to BE
        "level_2r": entry_price + initial_risk * 2,   # 2R: sell 50%
        "level_3r": entry_price + initial_risk * 3,   # 3R: sell 75%
    }


async def check_failed_breakout(
    ticker: str,
    entry_price: float,
    current_price: float,
    entry_was_breakout: bool = True,
) -> Optional[ExitSignal]:
    """
    Detect failed breakouts — one of the most costly patterns.

    If price breaks above resistance then falls back below entry:
    → Exit immediately (failed breakout = strong sell signal)
    """
    if not entry_was_breakout:
        return None

    if current_price < entry_price * 0.998:  # 0.2% below entry = failed
        return ExitSignal(
            action="SELL_ALL",
            urgency="immediate",
            reason=f"Failed breakout: price ${current_price:.2f} < entry ${entry_price:.2f}",
        )

    return None


async def check_moving_average_break(
    ticker: str,
    current_price: float,
    entry_price: float,
    current_plpc: float,
) -> Optional[ExitSignal]:
    """
    Professional rule: exit if price closes below SMA20 while in position.
    Only applies when we already have some profit (>+2%).
    """
    if current_plpc < 2.0:
        return None  # Not enough profit to protect yet

    try:
        import yfinance as yf
        data = yf.download(ticker, period="30d", progress=False)["Close"]
        if data.empty:
            return None

        sma20 = float(data.tail(20).mean())

        if current_price < sma20 * 0.995:  # 0.5% below SMA20
            return ExitSignal(
                action="SELL_ALL",
                urgency="end_of_day",
                reason=f"Price broke below SMA20 (${sma20:.2f}) with profit ${current_plpc:+.1f}% — protect gains",
            )

        # Warning: approaching SMA20
        elif current_price < sma20 * 1.01:
            return ExitSignal(
                action="TIGHTEN_STOP",
                urgency="monitor",
                reason=f"Approaching SMA20 — tighten stop",
                new_stop=sma20 * 0.99,
            )

    except Exception as e:
        logger.debug(f"MA break check failed for {ticker}: {e}")

    return None


def generate_pro_exit_plan(
    entry_price: float,
    current_price: float,
    stop_price: float,
    qty: float,
) -> dict:
    """
    Generate a professional exit plan with multiple targets.

    Returns a structured exit plan that scales out at each R multiple.
    """
    targets = check_profit_target_levels(entry_price, current_price, stop_price)
    r = targets["r_multiple"]
    initial_risk = targets["initial_risk"]

    plan = {
        "current_r_multiple": r,
        "initial_risk_per_share": initial_risk,
        "status": "",
        "next_action": "",
        "exit_levels": [
            {
                "r_multiple": 1.0,
                "price": targets["level_1r"],
                "action": "Move stop to breakeven",
                "qty_pct": 0,
            },
            {
                "r_multiple": 2.0,
                "price": targets["level_2r"],
                "action": "Sell 50% of position",
                "qty_pct": 50,
            },
            {
                "r_multiple": 3.0,
                "price": targets["level_3r"],
                "action": "Sell 25% more (75% total)",
                "qty_pct": 25,
            },
        ],
    }

    if r >= 3.0:
        plan["status"] = f"🏆 +{r:.1f}R — Excellent! Consider 75% exit"
        plan["next_action"] = "Sell 25% more, trail remaining tightly"
    elif r >= 2.0:
        plan["status"] = f"💰 +{r:.1f}R — Good! Take partial profits"
        plan["next_action"] = "Sell 50% at market"
    elif r >= 1.0:
        plan["status"] = f"✅ +{r:.1f}R — Breakeven locked"
        plan["next_action"] = "Move stop to entry price"
    elif r >= 0:
        plan["status"] = f"🟡 +{r:.1f}R — Developing"
        plan["next_action"] = "Hold with current stop"
    else:
        plan["status"] = f"🔴 {r:.1f}R — Stop loss territory"
        plan["next_action"] = "Exit if stop hit"

    return plan
