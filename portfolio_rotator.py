"""
Portfolio Rotator — Strong replaces Weak
==========================================

When portfolio is full but a great new opportunity appears:
- Identify the WEAKEST current position
- If it's clearly underperforming → sell it
- Free up capital for the new opportunity

This prevents the bot from missing 80+ score setups
just because all slots are taken by 65 score positions.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RotationCandidate:
    """A position that could be rotated out."""
    ticker: str
    trade_id: int
    score: float
    plpc: float
    days_held: float
    reason_to_keep: str
    reason_to_rotate: str


@dataclass
class RotationDecision:
    should_rotate: bool
    ticker_to_sell: Optional[str]
    ticker_to_buy: Optional[str]
    score_improvement: float
    reason: str


async def find_weakest_position() -> Optional[RotationCandidate]:
    """Find the weakest current position that could be rotated."""
    try:
        import broker, database
        from scoring import get_composite_score
        from datetime import datetime, timezone

        open_trades = await asyncio.to_thread(database.get_open_trades)
        if not open_trades:
            return None

        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return None

        # Score each position
        candidates = []
        for pos in positions:
            try:
                trade = next((t for t in open_trades if t["ticker"] == pos.get('ticker')), None)
                if not trade:
                    continue

                # Get current score
                try:
                    score_result = await asyncio.wait_for(
                        asyncio.to_thread(get_composite_score, pos.get('ticker'), 5),
                        timeout=15,
                    )
                    score = score_result.get("composite_score", 50)
                except Exception:
                    score = 50

                plpc = float(pos.get('unrealized_plpc', 0)) * 100

                # Days held
                try:
                    entry_dt = datetime.fromisoformat(
                        str(trade.get("entry_time", "2000-01-01"))[:19].replace("Z", "")
                    ).replace(tzinfo=timezone.utc)
                    days_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
                except Exception:
                    days_held = 0

                # Reasons to keep
                keep_reason = ""
                if plpc >= 3:
                    keep_reason = f"In profit +{plpc:.1f}%"
                elif score >= 70:
                    keep_reason = f"Score still strong ({score:.0f})"
                elif days_held < 1:
                    keep_reason = "Just opened (< 1 day)"

                # Reasons to rotate
                rotate_reason = ""
                if score < 50 and plpc < 0:
                    rotate_reason = f"Score {score:.0f} + loss"
                elif score < 60 and days_held > 3:
                    rotate_reason = f"Score {score:.0f} + held {days_held:.0f} days"
                elif plpc < -2 and days_held > 1:
                    rotate_reason = f"Down {plpc:.1f}% after {days_held:.0f} days"

                candidates.append(RotationCandidate(
                    ticker=pos.get('ticker'),
                    trade_id=trade["id"],
                    score=score,
                    plpc=plpc,
                    days_held=days_held,
                    reason_to_keep=keep_reason,
                    reason_to_rotate=rotate_reason,
                ))

            except Exception as e:
                logger.debug(f"Rotation scoring failed for {pos.get('ticker')}: {e}")

        if not candidates:
            return None

        # Sort: weakest first (low score, negative PnL, longer held)
        candidates.sort(key=lambda c: (
            c.plpc < 0,             # negative PnL first
            -c.score,               # lowest score first
            -c.days_held,           # longest held first
        ))

        weakest = candidates[0]

        # Only suggest rotation if there's a clear reason
        if weakest.reason_to_rotate:
            return weakest

        return None

    except Exception as e:
        logger.error(f"find_weakest_position failed: {e}")
        return None


async def evaluate_rotation(
    new_ticker: str,
    new_score: float,
) -> RotationDecision:
    """
    Decide if we should rotate out the weakest position
    to make room for new_ticker.
    """
    # Don't rotate unless new opportunity is significantly better
    MIN_SCORE_IMPROVEMENT = 15

    weakest = await find_weakest_position()

    if not weakest:
        return RotationDecision(
            should_rotate=False,
            ticker_to_sell=None,
            ticker_to_buy=new_ticker,
            score_improvement=0,
            reason="No weak position to rotate",
        )

    score_improvement = new_score - weakest.score

    if score_improvement < MIN_SCORE_IMPROVEMENT:
        return RotationDecision(
            should_rotate=False,
            ticker_to_sell=weakest.ticker,
            ticker_to_buy=new_ticker,
            score_improvement=score_improvement,
            reason=f"Improvement {score_improvement:.0f} < {MIN_SCORE_IMPROVEMENT} needed",
        )

    # Don't sell a profitable position to buy something else
    if weakest.plpc >= 2.0:
        return RotationDecision(
            should_rotate=False,
            ticker_to_sell=weakest.ticker,
            ticker_to_buy=new_ticker,
            score_improvement=score_improvement,
            reason=f"Weakest is still profitable +{weakest.plpc:.1f}%",
        )

    return RotationDecision(
        should_rotate=True,
        ticker_to_sell=weakest.ticker,
        ticker_to_buy=new_ticker,
        score_improvement=score_improvement,
        reason=(
            f"Replacing {weakest.ticker} (score {weakest.score:.0f}, {weakest.plpc:+.1f}%) "
            f"with {new_ticker} (score {new_score:.0f}, +{score_improvement:.0f} better)"
        ),
    )


async def get_rotation_report() -> str:
    """Generate a report of rotation candidates."""
    try:
        import broker, database
        from scoring import get_composite_score

        open_trades = await asyncio.to_thread(database.get_open_trades)
        if not open_trades:
            return "📊 אין פוזיציות לסיבוב"

        positions = await asyncio.to_thread(broker.get_positions)

        lines = ["🔄 <b>Portfolio Rotation Analysis</b>", "━━━━━━━━━━━━━━━━"]

        scored = []
        for pos in positions:
            try:
                plpc = float(pos.get('unrealized_plpc', 0)) * 100
                # Quick score
                score_r = await asyncio.wait_for(
                    asyncio.to_thread(get_composite_score, pos.get('ticker'), 5),
                    timeout=10,
                )
                score = score_r.get("composite_score", 50)
                scored.append((pos.get('ticker'), score, plpc))
            except Exception:
                pass

        scored.sort(key=lambda x: x[1], reverse=True)

        for sym, score, plpc in scored:
            emoji = "💪" if score >= 70 else "🟡" if score >= 60 else "⚠️"
            pnl_e = "🟢" if plpc >= 0 else "🔴"
            lines.append(f"{emoji} {sym}: Score <b>{score:.0f}</b> | {pnl_e} {plpc:+.1f}%")

        if scored:
            weakest = scored[-1]
            lines.extend([
                "",
                f"📉 <b>הפוזיציה החלשה: {weakest[0]}</b>",
                f"  Score: {weakest[1]:.0f} | P&L: {weakest[2]:+.1f}%",
                "",
                "💡 הבוט יסובב אם יראה הזדמנות עם ציון 85+",
            ])

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Rotation analysis error: {e}"
