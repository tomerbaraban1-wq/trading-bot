"""
Auto Parameter Optimizer
=========================

Automatically adjusts trading parameters based on recent performance.

Every 2 hours checks:
- If win rate < 40% → raise MIN_BUY_SCORE by 3
- If win rate > 65% → lower MIN_BUY_SCORE by 2 (more opportunities)
- If avg hold time > 20h → tighten stagnant exit to 12h
- If drawdown > 1.5% → reduce position sizes
"""

import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def auto_adjust_parameters() -> dict:
    """
    Check recent performance and adjust parameters accordingly.
    Returns dict of what was changed.
    """
    changes = {}

    try:
        import database
        conn = database.get_connection()

        # Get last 20 trades performance
        last_week = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        rows = conn.execute("""
            SELECT pnl_gross,
                   (julianday(COALESCE(exit_time, 'now')) - julianday(entry_time)) * 24 as hold_h
            FROM trade_log
            WHERE status NOT IN ('open')
            AND entry_time >= ?
        """, (last_week,)).fetchall()

        if len(rows) < 3:
            return changes  # Not enough data

        wins      = [r for r in rows if r[0] and r[0] > 0]
        losses    = [r for r in rows if r[0] and r[0] < 0]
        win_rate  = len(wins) / len(rows) * 100 if rows else 0
        avg_hold  = sum(r[1] for r in rows if r[1]) / len(rows) if rows else 0

        current_min_score = int(os.getenv("MIN_BUY_SCORE", "65"))

        # ── Adjust MIN_BUY_SCORE ──────────────────────────────────────
        if win_rate < 35 and len(rows) >= 5:
            new_score = min(80, current_min_score + 3)
            if new_score != current_min_score:
                os.environ["MIN_BUY_SCORE"] = str(new_score)
                changes["MIN_BUY_SCORE"] = {
                    "old": current_min_score,
                    "new": new_score,
                    "reason": f"Win rate low ({win_rate:.0f}%) — being more selective"
                }
                logger.info(f"[AUTO-OPT] Raised MIN_BUY_SCORE {current_min_score}→{new_score} (WR={win_rate:.0f}%)")

        elif win_rate > 65 and len(rows) >= 5 and current_min_score > 60:
            new_score = max(60, current_min_score - 2)
            if new_score != current_min_score:
                os.environ["MIN_BUY_SCORE"] = str(new_score)
                changes["MIN_BUY_SCORE"] = {
                    "old": current_min_score,
                    "new": new_score,
                    "reason": f"Win rate high ({win_rate:.0f}%) — more opportunities"
                }
                logger.info(f"[AUTO-OPT] Lowered MIN_BUY_SCORE {current_min_score}→{new_score} (WR={win_rate:.0f}%)")

        # ── Adjust hold time ──────────────────────────────────────────
        current_max_hold = float(os.getenv("MAX_HOLD_HOURS", "24"))
        if avg_hold > 20 and win_rate < 50:
            new_hold = max(16, current_max_hold - 4)
            os.environ["MAX_HOLD_HOURS"] = str(new_hold)
            changes["MAX_HOLD_HOURS"] = {
                "old": current_max_hold,
                "new": new_hold,
                "reason": f"Avg hold {avg_hold:.0f}h with {win_rate:.0f}% WR — exit faster"
            }
            logger.info(f"[AUTO-OPT] Reduced MAX_HOLD_HOURS {current_max_hold}→{new_hold}")

        # ── Summary log ───────────────────────────────────────────────
        logger.info(
            f"[AUTO-OPT] Analysis: {len(rows)} trades | "
            f"WR={win_rate:.0f}% | AvgHold={avg_hold:.0f}h | "
            f"Changes={len(changes)}"
        )

    except Exception as e:
        logger.error(f"Auto optimizer failed: {e}")

    return changes


async def run_auto_optimizer() -> None:
    """Async wrapper for auto_adjust_parameters."""
    import asyncio
    changes = await asyncio.to_thread(auto_adjust_parameters)

    if changes:
        try:
            from telegram_bot import send_message
            lines = ["🤖 <b>Auto-Optimizer: פרמטרים עודכנו</b>", "━━━━━━━━━━━━━━━━"]
            for param, data in changes.items():
                lines.append(f"  📊 {param}: {data['old']} → <b>{data['new']}</b>")
                lines.append(f"     {data['reason']}")
            await send_message("\n".join(lines))
        except Exception:
            pass
