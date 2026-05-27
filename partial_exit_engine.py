"""
Partial Exit Engine — Lock In Profits While Letting Winners Run
===============================================================

Strategy:
  Stage 1  (+5%)  → Sell 25% — cover part of risk
  Stage 2  (+10%) → Sell 25% more — free-ride with house money
  Stage 3  (+18%) → Sell 25% more — lock big gains
  Remaining 25%   → Trail with tight stop until exit

Rules:
  - Each stage fires ONCE per trade
  - Never sells if position < 2 shares (avoid fractional leftovers)
  - Sends Telegram alert on every partial exit
  - Updates trade qty in DB after each partial exit
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
STAGES = [
    {"id": 1, "trigger_pct": 5.0,  "sell_pct": 0.25, "label": "Stage 1 (+5%)"},
    {"id": 2, "trigger_pct": 10.0, "sell_pct": 0.25, "label": "Stage 2 (+10%)"},
    {"id": 3, "trigger_pct": 18.0, "sell_pct": 0.25, "label": "Stage 3 (+18%)"},
]

# Per-trade tracking: which stages have fired
_stages_fired: dict[int, set[int]] = {}   # trade_id → {stage_ids fired}
_last_partial_time: dict[int, float] = {} # trade_id → timestamp of last partial

MIN_QTY_REMAINING = 1.0   # never reduce below this many shares
MIN_STAGE_GAP_SECS = 600  # at least 10 min between partial exits


@dataclass
class PartialExitDecision:
    should_exit: bool
    stage_id: int
    qty_to_sell: float
    remaining_qty: float
    trigger_pct: float
    label: str
    reason: str


def get_next_stage(trade_id: int, plpc: float) -> Optional[dict]:
    """Return the next stage that should fire, or None."""
    fired = _stages_fired.get(trade_id, set())
    for stage in STAGES:
        if stage["id"] not in fired and plpc >= stage["trigger_pct"]:
            return stage
    return None


def evaluate_partial_exit(
    trade_id: int,
    current_qty: float,
    plpc: float,
) -> PartialExitDecision:
    """
    Check if a partial exit should be executed.
    Returns a decision with quantity to sell.
    """
    # Throttle — minimum gap between partial exits
    last_time = _last_partial_time.get(trade_id, 0)
    if time.time() - last_time < MIN_STAGE_GAP_SECS:
        return PartialExitDecision(
            False, 0, 0, current_qty, plpc,
            "waiting", f"Last partial was {(time.time()-last_time)/60:.0f}m ago"
        )

    stage = get_next_stage(trade_id, plpc)
    if not stage:
        return PartialExitDecision(
            False, 0, 0, current_qty, plpc,
            "no_stage", "No more stages to fire"
        )

    qty_to_sell = round(current_qty * stage["sell_pct"], 4)
    remaining = round(current_qty - qty_to_sell, 4)

    # Safety: never leave less than MIN_QTY_REMAINING
    if remaining < MIN_QTY_REMAINING:
        qty_to_sell = max(0, round(current_qty - MIN_QTY_REMAINING, 4))
        remaining = current_qty - qty_to_sell
        if qty_to_sell <= 0:
            return PartialExitDecision(
                False, stage["id"], 0, current_qty, plpc,
                stage["label"], "Too few shares to partial exit"
            )

    return PartialExitDecision(
        should_exit=True,
        stage_id=stage["id"],
        qty_to_sell=qty_to_sell,
        remaining_qty=remaining,
        trigger_pct=stage["trigger_pct"],
        label=stage["label"],
        reason=f"Profit +{plpc:.1f}% hit {stage['label']} target",
    )


async def execute_partial_exit(
    trade: dict,
    current_price: float,
    plpc: float,
) -> bool:
    """
    Execute a partial exit. Returns True if successful.

    Flow:
    1. Evaluate which stage to fire
    2. Submit sell order for partial qty
    3. Update DB with new qty
    4. Mark stage as fired
    5. Send Telegram notification
    """
    try:
        import broker, database

        trade_id = trade["id"]
        ticker = trade["ticker"]
        current_qty = float(trade.get("qty", 0))

        if current_qty <= MIN_QTY_REMAINING:
            return False

        decision = evaluate_partial_exit(trade_id, current_qty, plpc)

        if not decision.should_exit:
            return False

        logger.info(
            f"[PARTIAL EXIT] {ticker}: {decision.label} — "
            f"selling {decision.qty_to_sell:.2f} of {current_qty:.2f} shares @ ${current_price:.2f}"
        )

        # Execute sell
        result = await asyncio.to_thread(
            broker.submit_sell,
            ticker,
            decision.qty_to_sell,
            current_price,
        )

        if not result or result.get("status") not in ("filled", "accepted", "pending_new", "Submitted"):
            logger.warning(f"[PARTIAL EXIT] {ticker}: order failed: {result}")
            return False

        fill_price = float(result.get("price") or current_price)
        pnl_this_sell = (fill_price - trade["entry_price"]) * decision.qty_to_sell

        # Update qty in database
        await asyncio.to_thread(
            database.update_trade_qty,
            trade_id,
            decision.remaining_qty,
        )

        # Mark stage as fired
        if trade_id not in _stages_fired:
            _stages_fired[trade_id] = set()
        _stages_fired[trade_id].add(decision.stage_id)
        _last_partial_time[trade_id] = time.time()

        # Send Telegram notification
        try:
            from telegram_bot import send_message
            stage_emoji = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣"}.get(decision.stage_id, "📊")
            tv_link = f'<a href="https://www.tradingview.com/chart/?symbol={ticker}">{ticker}</a>'
            await send_message(
                f"💰 <b>רווח חלקי — {tv_link}</b>  {stage_emoji}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ {decision.label}: מכרתי {decision.qty_to_sell:.2f} מניות\n"
                f"📈 רווח על החלק: <b>${pnl_this_sell:+.2f}</b> (+{plpc:.1f}%)\n"
                f"💵 מחיר מכירה: ${fill_price:.2f}\n"
                f"📊 נשאר: {decision.remaining_qty:.2f} מניות (ממשיך לרוץ)\n\n"
                f"💡 <i>נועלים רווח ומשאירים runners!</i>"
            )
        except Exception as _te:
            logger.debug(f"[PARTIAL EXIT] Telegram notify failed: {_te}")

        logger.info(
            f"[PARTIAL EXIT] {ticker}: ✅ sold {decision.qty_to_sell:.2f} @ ${fill_price:.2f} "
            f"| P&L this sell: ${pnl_this_sell:+.2f} | remaining: {decision.remaining_qty:.2f}"
        )
        return True

    except Exception as e:
        logger.error(f"[PARTIAL EXIT] execute failed for {trade.get('ticker', '?')}: {e}")
        return False


def cleanup_trade(trade_id: int) -> None:
    """Clean up tracking when trade is fully closed."""
    _stages_fired.pop(trade_id, None)
    _last_partial_time.pop(trade_id, None)


def get_partial_exit_stats() -> dict:
    """Summary of partial exit activity."""
    total_stages = sum(len(s) for s in _stages_fired.values())
    return {
        "active_trades_with_partials": len(_stages_fired),
        "total_stages_fired": total_stages,
        "trades_at_stage_1": sum(1 for s in _stages_fired.values() if 1 in s),
        "trades_at_stage_2": sum(1 for s in _stages_fired.values() if 2 in s),
        "trades_at_stage_3": sum(1 for s in _stages_fired.values() if 3 in s),
    }
