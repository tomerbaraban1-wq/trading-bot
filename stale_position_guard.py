"""
Stale Position Guard
====================

Monitors positions that have been held too long without progress.

Rules:
- Position held > 5 days with < +2% gain → alert + consider selling
- Position held > 7 days with any loss → immediate sell
- Position held > MAX_HOLD_HOURS → sell unconditionally

Also handles wide stops:
- If ATR stop > 7% from entry → tighten to 5%
- If position held > 3 days, progressively tighten stop
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


async def scan_stale_positions() -> list[dict]:
    """
    Find positions that are stale (held too long without progress).
    Returns list of recommendations.
    """
    recommendations = []

    try:
        import broker, database
        from datetime import datetime, timezone

        positions = await asyncio.to_thread(broker.get_positions)
        open_trades = await asyncio.to_thread(database.get_open_trades)
        if not positions or not open_trades:
            return []

        trade_map = {t["ticker"]: t for t in open_trades}
        now = datetime.now(timezone.utc)

        for p in positions:
            trade = trade_map.get(p.symbol)
            if not trade:
                continue

            try:
                cur_price = float(p.current_price)
                entry_price = float(p.avg_entry_price)
                unrealized_plpc = float(p.unrealized_plpc) * 100

                # Calculate days held
                entry_time = trade.get("entry_time")
                if not entry_time:
                    continue

                entry_dt = datetime.fromisoformat(
                    str(entry_time).replace("Z", "+00:00")
                ).replace(tzinfo=timezone.utc)
                days_held = (now - entry_dt).total_seconds() / 86400
                hours_held = days_held * 24

                # Check conditions
                action = None
                reason = None
                urgency = "medium"

                # Rule 1: >7 days with any loss → sell immediately
                if days_held > 7 and unrealized_plpc < 0:
                    action = "SELL"
                    urgency = "high"
                    reason = f"הוחזקה {days_held:.0f} ימים בהפסד ({unrealized_plpc:+.1f}%)"

                # Rule 2: >5 days with minimal gain (<+2%) → consider selling
                elif days_held > 5 and unrealized_plpc < 2.0:
                    action = "REVIEW"
                    urgency = "medium"
                    reason = f"הוחזקה {days_held:.0f} ימים עם רווח נמוך ({unrealized_plpc:+.1f}%)"

                # Rule 3: >MAX_HOLD_HOURS → sell
                max_hold = float(os.getenv("MAX_HOLD_HOURS", "24"))
                if hours_held > max_hold * 1.5:  # 150% of limit
                    action = "SELL"
                    urgency = "high"
                    reason = f"מעל {max_hold*1.5:.0f}ש' ({hours_held:.0f}ש' נוכחי)"

                # Rule 4: Wide stop (>7%) on position held >3 days → tighten
                atr_stop = trade.get("atr_stop_price")
                if atr_stop and entry_price:
                    stop_pct = (entry_price - atr_stop) / entry_price * 100
                    if stop_pct > 7 and days_held > 3:
                        action = action or "TIGHTEN_STOP"
                        reason = reason or f"Stop רחב {stop_pct:.1f}% אחרי {days_held:.0f} ימים"

                if action:
                    recommendations.append({
                        "ticker": p.symbol,
                        "action": action,
                        "urgency": urgency,
                        "days_held": days_held,
                        "unrealized_plpc": unrealized_plpc,
                        "current_price": cur_price,
                        "entry_price": entry_price,
                        "atr_stop": atr_stop,
                        "reason": reason,
                    })

            except Exception as e:
                logger.debug(f"Stale check failed for {p.symbol}: {e}")

    except Exception as e:
        logger.error(f"Stale position scan failed: {e}")

    # Sort by urgency
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: urgency_order.get(r["urgency"], 99))
    return recommendations


async def tighten_wide_stop(ticker: str, trade: dict, cur_price: float) -> bool:
    """Tighten a wide stop to 5% from current price."""
    try:
        import database
        new_stop = cur_price * 0.95  # 5% from current price
        current_stop = trade.get("atr_stop_price", 0)

        if new_stop > current_stop:
            await asyncio.to_thread(
                database.update_trade_stop,
                trade["id"],
                new_stop,
                trade.get("high_watermark", cur_price),
            )
            logger.info(f"[STALE GUARD] {ticker}: tightened stop ${current_stop:.2f} → ${new_stop:.2f}")
            return True
    except Exception as e:
        logger.debug(f"Stop tighten failed for {ticker}: {e}")
    return False


async def notify_stale_positions(recommendations: list[dict]) -> None:
    """Send Telegram notification about stale positions."""
    if not recommendations:
        return

    try:
        from telegram_bot import send_message

        lines = ["⏰ <b>פוזיציות ישנות — בדיקה נדרשת</b>", "━━━━━━━━━━━━━━━━"]

        for r in recommendations[:5]:
            icon = "🔴" if r["urgency"] == "high" else "🟡"
            lines.append(
                f"{icon} <b>{r['ticker']}</b>  {r['days_held']:.0f} ימים  "
                f"{r['unrealized_plpc']:+.1f}%"
            )
            lines.append(f"  📌 {r['reason']}")
            if r["action"] == "SELL":
                lines.append(f"  → ⚡ מומלץ למכור!")
            elif r["action"] == "TIGHTEN_STOP":
                lines.append(f"  → 🛑 מהדק stop")

        await send_message("\n".join(lines))

    except Exception as e:
        logger.debug(f"Stale notification failed: {e}")
