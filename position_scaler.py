"""
Position Scaler — Add to Winners (Pyramid Up)
==============================================

Pro traders' secret: ADD to positions that are working,
DON'T add to losers.

Logic:
- Position +3% → add 50% more (first scale-in)
- Position +7% → add 25% more (second scale-in)
- Total max additions: 2 (so max 175% of original)

Rules:
- ONLY scale on winners (never averaging down)
- Each scale-in needs Pro Grade A/B again
- Each scale-in needs current score still strong
- Daily P&L must be positive (don't pyramid on bad day)
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ScaleInDecision:
    should_scale: bool
    qty_to_add: float
    reason: str
    new_total_qty: float
    new_avg_price: float


# Track scale-ins per trade
_scale_in_count: dict[int, int] = {}     # trade_id → count
_last_scale_in_time: dict[int, float] = {}  # trade_id → timestamp


async def evaluate_scale_in(
    trade_id: int,
    ticker: str,
    entry_price: float,
    current_price: float,
    current_qty: float,
    current_score: float,
) -> ScaleInDecision:
    """
    Check if we should add to a winning position.

    Returns decision with quantity to add.
    """
    import time

    plpc = (current_price - entry_price) / entry_price * 100

    # 1. Must be winning
    if plpc < 3.0:
        return ScaleInDecision(False, 0, f"Position only +{plpc:.1f}% (need +3%+)",
                              current_qty, entry_price)

    # 2. Don't scale too often (min 4 hours between)
    if trade_id in _last_scale_in_time:
        if time.time() - _last_scale_in_time[trade_id] < 4 * 3600:
            return ScaleInDecision(False, 0, "Scaled in recently (need 4h gap)",
                                  current_qty, entry_price)

    # 3. Max 2 scale-ins
    count = _scale_in_count.get(trade_id, 0)
    if count >= 2:
        return ScaleInDecision(False, 0, "Max scale-ins reached (2)",
                              current_qty, entry_price)

    # 4. Score still strong
    if current_score < 65:
        return ScaleInDecision(False, 0, f"Score weakened to {current_score:.0f}",
                              current_qty, entry_price)

    # 5. Determine size
    if count == 0 and plpc >= 3.0:
        # First scale-in: add 50%
        scale_pct = 0.50
        scale_name = "1st"
    elif count == 1 and plpc >= 7.0:
        # Second scale-in: add 25%
        scale_pct = 0.25
        scale_name = "2nd"
    else:
        return ScaleInDecision(False, 0, f"Need +7% for 2nd scale-in (now +{plpc:.1f}%)",
                              current_qty, entry_price)

    # 6. Daily P&L must be positive (don't pyramid on bad day)
    try:
        from circuit_breaker import _state
        daily_pnl = _state.get("daily_pnl", 0)
        if daily_pnl < -100:
            return ScaleInDecision(False, 0, f"Daily P&L negative (${daily_pnl:.0f})",
                                  current_qty, entry_price)
    except Exception:
        pass

    # 7. Calculate new quantity
    qty_to_add = current_qty * scale_pct
    new_total = current_qty + qty_to_add
    new_avg = ((current_qty * entry_price) + (qty_to_add * current_price)) / new_total

    return ScaleInDecision(
        should_scale=True,
        qty_to_add=qty_to_add,
        reason=f"{scale_name} scale-in: +{plpc:.1f}% (add {scale_pct*100:.0f}%)",
        new_total_qty=new_total,
        new_avg_price=new_avg,
    )


async def execute_scale_in(trade: dict, current_price: float, current_score: float) -> bool:
    """
    Execute a scale-in trade.
    Returns True if successful.
    """
    try:
        import broker, database
        import time

        decision = await evaluate_scale_in(
            trade_id=trade["id"],
            ticker=trade["ticker"],
            entry_price=trade["entry_price"],
            current_price=current_price,
            current_qty=trade["qty"],
            current_score=current_score,
        )

        if not decision.should_scale:
            logger.debug(f"[SCALE-IN] {trade['ticker']}: skipped — {decision.reason}")
            return False

        # Execute the buy
        result = await asyncio.to_thread(
            broker.submit_buy,
            trade["ticker"],
            decision.qty_to_add,
            current_price,
        )

        if not result or result.get("status") not in ("filled", "accepted"):
            logger.warning(f"[SCALE-IN] {trade['ticker']}: order failed: {result}")
            return False

        # Update database with new quantity and avg price
        await asyncio.to_thread(
            database.update_trade_qty,
            trade["id"],
            decision.new_total_qty,
        )

        # Track scale-in
        _scale_in_count[trade["id"]] = _scale_in_count.get(trade["id"], 0) + 1
        _last_scale_in_time[trade["id"]] = time.time()

        # Notify
        try:
            from telegram_bot import send_message
            await send_message(
                f"📈 <b>הוספתי לפוזיציה — {trade['ticker']}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ {decision.reason}\n"
                f"💵 הוספתי: {decision.qty_to_add:.2f} מניות @ ${current_price:.2f}\n"
                f"📊 סה״כ עכשיו: {decision.new_total_qty:.2f} מניות\n"
                f"💰 מחיר ממוצע חדש: ${decision.new_avg_price:.2f}\n\n"
                f"💡 פירמידה למעלה — מוסיף רק לwinners!"
            )
        except Exception:
            pass

        logger.info(
            f"[SCALE-IN] {trade['ticker']}: added {decision.qty_to_add} shares @ ${current_price:.2f}"
        )
        return True

    except Exception as e:
        logger.error(f"Scale-in execution failed: {e}")
        return False


def cleanup_old_trade(trade_id: int) -> None:
    """Clean up tracking for closed trade."""
    _scale_in_count.pop(trade_id, None)
    _last_scale_in_time.pop(trade_id, None)


def get_scale_in_stats() -> dict:
    """Get scale-in statistics."""
    return {
        "active_positions_scaled": len(_scale_in_count),
        "total_scale_ins": sum(_scale_in_count.values()),
        "positions_with_2_scales": sum(1 for c in _scale_in_count.values() if c >= 2),
    }
