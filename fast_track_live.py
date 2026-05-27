"""
Fast Track to Live Trading
============================

Aggressive but safe progression from paper → live in 1-2 weeks.

Stages:
  Stage 0 (Days 1-2): Validate on historical data
  Stage 1 (Days 3-5): Aggressive paper test
  Stage 2 (Days 6-10): Live Lite — $100 with max 1 position
  Stage 3 (Days 11-14): Live Mini — $500 with max 2 positions
  Stage 4 (Day 14+):   Live Full — $2,000+

Auto-promotion when criteria met.
Auto-demotion if any criteria failed.
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for each stage."""
    name: str
    description: str
    max_budget: float
    max_positions: int
    max_position_pct: float
    stop_loss_pct: float
    min_buy_score: int
    min_trades_to_promote: int      # min trades before promotion
    min_win_rate_to_promote: float
    min_days_in_stage: int
    max_daily_loss_pct: float
    max_consecutive_losses: int


# Aggressive but safe stages
STAGES = {
    0: StageConfig(
        name="VALIDATE",
        description="🔍 Validating historical performance",
        max_budget=0,  # No real trades
        max_positions=0,
        max_position_pct=0,
        stop_loss_pct=3.5,
        min_buy_score=70,
        min_trades_to_promote=30,   # From historical
        min_win_rate_to_promote=50.0,
        min_days_in_stage=1,
        max_daily_loss_pct=0,
        max_consecutive_losses=0,
    ),
    1: StageConfig(
        name="PAPER_AGGRESSIVE",
        description="📝 Paper trading with new filters",
        max_budget=10000,
        max_positions=4,
        max_position_pct=15,
        stop_loss_pct=3.5,
        min_buy_score=70,           # Strict
        min_trades_to_promote=10,
        min_win_rate_to_promote=45.0,
        min_days_in_stage=3,
        max_daily_loss_pct=3.0,
        max_consecutive_losses=4,
    ),
    2: StageConfig(
        name="LIVE_LITE",
        description="💵 Live with $100 — testing real execution",
        max_budget=100,              # Just $100!
        max_positions=1,             # ONE position only
        max_position_pct=80,         # The whole $100
        stop_loss_pct=2.5,           # Tight
        min_buy_score=72,           # Even stricter
        min_trades_to_promote=5,
        min_win_rate_to_promote=50.0,
        min_days_in_stage=4,
        max_daily_loss_pct=4.0,      # $4
        max_consecutive_losses=3,
    ),
    3: StageConfig(
        name="LIVE_MINI",
        description="💰 Live $500 — building confidence",
        max_budget=500,
        max_positions=2,
        max_position_pct=40,
        stop_loss_pct=3.0,
        min_buy_score=70,
        min_trades_to_promote=10,
        min_win_rate_to_promote=50.0,
        min_days_in_stage=3,
        max_daily_loss_pct=3.0,      # $15
        max_consecutive_losses=3,
    ),
    4: StageConfig(
        name="LIVE_FULL",
        description="🚀 Full live trading",
        max_budget=10000,
        max_positions=4,
        max_position_pct=15,
        stop_loss_pct=3.5,
        min_buy_score=65,
        min_trades_to_promote=999,
        min_win_rate_to_promote=999,
        min_days_in_stage=999,
        max_daily_loss_pct=2.0,
        max_consecutive_losses=3,
    ),
}


def get_current_stage() -> int:
    """Get current fast-track stage."""
    return int(os.getenv("FAST_TRACK_STAGE", "1"))


def get_stage_config(stage: int = None) -> StageConfig:
    """Get config for current or specified stage."""
    if stage is None:
        stage = get_current_stage()
    return STAGES.get(stage, STAGES[1])


def get_stage_start_date() -> Optional[str]:
    """When did current stage start."""
    return os.getenv("FAST_TRACK_STAGE_START")


def days_in_current_stage() -> float:
    """Days since current stage started."""
    start = get_stage_start_date()
    if not start:
        return 0
    try:
        start_dt = datetime.fromisoformat(start)
        return (datetime.now(timezone.utc) - start_dt.replace(tzinfo=timezone.utc)).total_seconds() / 86400
    except Exception:
        return 0


async def evaluate_stage_performance() -> dict:
    """
    Evaluate current stage performance.
    Returns whether to promote, demote, or stay.
    """
    stage = get_current_stage()
    config = get_stage_config(stage)

    try:
        import database
        conn = database.get_connection()

        # Get trades since stage started
        stage_start = get_stage_start_date() or (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        rows = conn.execute("""
            SELECT pnl_gross,
                   (julianday(COALESCE(exit_time, 'now')) - julianday(entry_time)) * 24 as hold_h
            FROM trade_log
            WHERE status NOT IN ('open')
            AND entry_time >= ?
        """, (stage_start,)).fetchall()

        days_in = days_in_current_stage()
        total_trades = len(rows)
        wins = sum(1 for r in rows if r[0] and r[0] > 0)
        losses = sum(1 for r in rows if r[0] and r[0] < 0)
        win_rate = (wins / max(total_trades, 1)) * 100
        total_pnl = sum(r[0] for r in rows if r[0])

        # Decision logic
        action = "STAY"
        reason = ""
        next_stage = stage

        # Check for demotion (losses)
        if total_pnl < 0 and total_trades >= 3:
            consecutive = 0
            max_consec = 0
            for r in rows[-5:]:
                if r[0] and r[0] < 0:
                    consecutive += 1
                    max_consec = max(max_consec, consecutive)
                else:
                    consecutive = 0
            if max_consec >= config.max_consecutive_losses:
                action = "DEMOTE"
                reason = f"{max_consec} consecutive losses — demoting"
                next_stage = max(0, stage - 1)

        # Check for promotion
        elif (
            total_trades >= config.min_trades_to_promote
            and win_rate >= config.min_win_rate_to_promote
            and days_in >= config.min_days_in_stage
            and total_pnl >= 0
        ):
            action = "PROMOTE"
            reason = f"Met all criteria — {win_rate:.0f}% WR on {total_trades} trades"
            next_stage = min(4, stage + 1)

        # Stay reasons
        else:
            reasons = []
            if total_trades < config.min_trades_to_promote:
                reasons.append(f"Need {config.min_trades_to_promote - total_trades} more trades")
            if win_rate < config.min_win_rate_to_promote:
                reasons.append(f"WR {win_rate:.0f}% < {config.min_win_rate_to_promote:.0f}%")
            if days_in < config.min_days_in_stage:
                reasons.append(f"Need {config.min_days_in_stage - days_in:.0f} more days")
            reason = " | ".join(reasons) if reasons else "all criteria met but PnL negative"

        return {
            "current_stage": stage,
            "stage_name": config.name,
            "days_in_stage": days_in,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "action": action,
            "next_stage": next_stage,
            "reason": reason,
            "criteria": {
                "min_trades": config.min_trades_to_promote,
                "min_win_rate": config.min_win_rate_to_promote,
                "min_days": config.min_days_in_stage,
            },
        }

    except Exception as e:
        logger.error(f"Stage evaluation failed: {e}")
        return {"error": str(e), "current_stage": stage}


async def advance_stage(force: bool = False) -> dict:
    """Move to next stage (with safety checks)."""
    eval_result = await evaluate_stage_performance()

    if not force and eval_result.get("action") != "PROMOTE":
        return {
            "success": False,
            "reason": f"Cannot promote — {eval_result.get('reason', 'criteria not met')}",
        }

    new_stage = eval_result.get("next_stage", get_current_stage())
    new_config = get_stage_config(new_stage)

    # Apply new stage settings
    os.environ["FAST_TRACK_STAGE"] = str(new_stage)
    os.environ["FAST_TRACK_STAGE_START"] = datetime.now(timezone.utc).isoformat()
    os.environ["MAX_BUDGET"] = str(new_config.max_budget)
    os.environ["MAX_OPEN_POSITIONS"] = str(new_config.max_positions)
    os.environ["MAX_POSITION_PCT"] = str(new_config.max_position_pct)
    os.environ["STOP_LOSS_PCT"] = str(new_config.stop_loss_pct)
    os.environ["MIN_BUY_SCORE"] = str(new_config.min_buy_score)
    os.environ["MAX_DAILY_LOSS_PCT"] = str(new_config.max_daily_loss_pct)
    os.environ["MAX_CONSECUTIVE_LOSSES"] = str(new_config.max_consecutive_losses)

    logger.info(f"[FAST TRACK] Stage {new_stage}: {new_config.name} — {new_config.description}")

    # Notify via Telegram
    try:
        from telegram_bot import send_message
        await send_message(
            f"🎯 <b>Fast Track: התקדמת לשלב {new_stage}!</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📊 {new_config.description}\n\n"
            f"<b>הגדרות חדשות:</b>\n"
            f"  💰 Budget: <b>${new_config.max_budget:,.0f}</b>\n"
            f"  📂 Max Positions: <b>{new_config.max_positions}</b>\n"
            f"  📊 Min Score: <b>{new_config.min_buy_score}</b>\n"
            f"  🛑 Stop Loss: <b>{new_config.stop_loss_pct}%</b>\n"
            f"  ⚠️ Max Daily Loss: <b>{new_config.max_daily_loss_pct}%</b>\n\n"
            f"📅 שלב הבא: {new_config.min_days_in_stage} ימים + "
            f"{new_config.min_trades_to_promote} עסקאות + "
            f"{new_config.min_win_rate_to_promote:.0f}% WR"
        )
    except Exception:
        pass

    return {
        "success": True,
        "old_stage": eval_result.get("current_stage"),
        "new_stage": new_stage,
        "config": asdict(new_config),
    }


async def auto_progress_check_loop():
    """
    Background loop: every 4 hours check if we should promote/demote.
    """
    await asyncio.sleep(60 * 60)   # Initial 1-hour wait
    while True:
        try:
            result = await evaluate_stage_performance()
            action = result.get("action")

            if action == "PROMOTE":
                await advance_stage(force=False)
            elif action == "DEMOTE":
                # Demote without confirmation
                new_stage = result.get("next_stage", 0)
                old_stage = result.get("current_stage", 1)
                old_config = get_stage_config(old_stage)
                new_config = get_stage_config(new_stage)

                os.environ["FAST_TRACK_STAGE"] = str(new_stage)
                os.environ["FAST_TRACK_STAGE_START"] = datetime.now(timezone.utc).isoformat()
                os.environ["MAX_BUDGET"] = str(new_config.max_budget)
                os.environ["MAX_OPEN_POSITIONS"] = str(new_config.max_positions)
                os.environ["MIN_BUY_SCORE"] = str(new_config.min_buy_score)

                try:
                    from telegram_bot import send_message
                    await send_message(
                        f"⚠️ <b>Fast Track: ירידה לשלב {new_stage}</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"📉 הסיבה: {result.get('reason')}\n"
                        f"🛡️ מגן עליך — חוזרים לשלב בטוח יותר\n"
                        f"💰 Budget: ${new_config.max_budget:,.0f}"
                    )
                except Exception:
                    pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Auto progress check failed: {e}")

        await asyncio.sleep(4 * 60 * 60)   # 4 hours


def get_status_report() -> str:
    """Format current fast-track status for Telegram."""
    stage = get_current_stage()
    config = get_stage_config(stage)
    days = days_in_current_stage()

    stage_emoji = {0: "🔍", 1: "📝", 2: "💵", 3: "💰", 4: "🚀"}
    e = stage_emoji.get(stage, "📊")

    return (
        f"{e} <b>Fast Track Status</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📍 Stage: <b>{stage}/4 — {config.name}</b>\n"
        f"📝 {config.description}\n"
        f"📅 ימים בשלב: {days:.1f}\n\n"
        f"<b>הגדרות פעילות:</b>\n"
        f"  💰 Budget: <b>${config.max_budget:,.0f}</b>\n"
        f"  📂 Max Positions: <b>{config.max_positions}</b>\n"
        f"  📊 Min Score: <b>{config.min_buy_score}</b>\n"
        f"  🛑 Stop Loss: <b>{config.stop_loss_pct}%</b>\n\n"
        f"<b>קריטריונים לקידום:</b>\n"
        f"  ✅ {config.min_trades_to_promote} עסקאות\n"
        f"  ✅ {config.min_win_rate_to_promote:.0f}%+ Win Rate\n"
        f"  ✅ {config.min_days_in_stage} ימים בשלב\n\n"
        f"💡 שלח /progress לבדיקה מפורטת"
    )
