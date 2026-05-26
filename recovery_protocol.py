"""
Recovery Protocol
==================

Automatic protocol that activates after consecutive losses.

When consecutive losses >= 2:
  - Tighten stop losses on open positions
  - Skip new buys until conditions improve
  - Reduce position sizes
  - Send status to Telegram

When win streak resumes:
  - Gradually return to normal
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# States
NORMAL = "normal"
CAUTION = "caution"    # 2 consecutive losses
RECOVERY = "recovery"  # 3+ consecutive losses
PAUSE = "pause"        # Circuit breaker tripped


def get_current_mode() -> str:
    """Get current trading mode based on recent performance."""
    try:
        from circuit_breaker import _state
        if _state.get("tripped"):
            return PAUSE
        consecutive = _state.get("consecutive_losses", 0)
        if consecutive >= 3:
            return RECOVERY
        elif consecutive >= 2:
            return CAUTION
        return NORMAL
    except Exception:
        return NORMAL


def get_mode_multipliers(mode: str) -> dict:
    """Get position size and threshold multipliers for each mode."""
    modes = {
        NORMAL:   {"position_mult": 1.0, "score_add": 0,  "label": "🟢 נורמלי"},
        CAUTION:  {"position_mult": 0.7, "score_add": 5,  "label": "🟡 זהירות"},
        RECOVERY: {"position_mult": 0.4, "score_add": 10, "label": "🟠 התאוששות"},
        PAUSE:    {"position_mult": 0.0, "score_add": 99, "label": "🔴 עצירה"},
    }
    return modes.get(mode, modes[NORMAL])


async def tighten_stops_after_loss() -> list[str]:
    """
    After a loss, tighten stops on all open positions by 30%.
    Returns list of tickers that had stops tightened.
    """
    tightened = []
    try:
        import broker, database
        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return []

        open_trades = await asyncio.to_thread(database.get_open_trades)
        trade_map = {t["ticker"]: t for t in (open_trades or [])}

        for p in positions:
            trade = trade_map.get(p.symbol)
            if not trade:
                continue

            cur_price = float(p.current_price)
            cur_stop = trade.get("atr_stop_price")
            if not cur_stop:
                continue

            # Tighten by 30%: move stop 30% closer to current price
            stop_distance = cur_price - cur_stop
            new_stop = cur_price - (stop_distance * 0.7)

            # Only tighten if new stop is higher
            if new_stop > cur_stop:
                await asyncio.to_thread(
                    database.update_trade_stop,
                    trade["id"],
                    new_stop,
                    trade.get("high_watermark", cur_price),
                )
                tightened.append(p.symbol)
                logger.info(
                    f"[RECOVERY] {p.symbol}: tightened stop "
                    f"${cur_stop:.2f} → ${new_stop:.2f}"
                )

        if tightened:
            try:
                from telegram_bot import send_message
                await send_message(
                    f"🛡️ <b>הידוק Stops אחרי הפסד</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📉 הידקתי stops על: {', '.join(tightened)}\n"
                    f"🔒 Stop הוקרב ב-30% למחיר נוכחי\n"
                    f"💡 מגן על רווחים קיימים"
                )
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Tighten stops failed: {e}")

    return tightened


async def send_recovery_status() -> None:
    """Send Telegram message about current recovery mode."""
    mode = get_current_mode()
    if mode == NORMAL:
        return  # Don't spam on normal

    multipliers = get_mode_multipliers(mode)

    try:
        from circuit_breaker import _state
        consecutive = _state.get("consecutive_losses", 0)
        daily_pnl = _state.get("daily_pnl", 0)
    except Exception:
        consecutive = 0
        daily_pnl = 0

    try:
        from telegram_bot import send_message
        await send_message(
            f"{multipliers['label']}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"❌ הפסדים רצופים: {consecutive}\n"
            f"💰 P&L יומי: ${daily_pnl:+.2f}\n"
            f"📉 גודל פוזיציה: {multipliers['position_mult']*100:.0f}% מנורמלי\n"
            f"📊 ציון מינימלי: +{multipliers['score_add']} נקודות\n"
            f"{'⛔ לא קונה עד לעצירה' if mode == PAUSE else '✅ ממשיך לקנות בזהירות'}"
        )
    except Exception as e:
        logger.debug(f"Recovery status notification failed: {e}")


def should_skip_buy_in_recovery(mode: str) -> bool:
    """Should we skip buying in current mode?"""
    return mode == PAUSE


def apply_recovery_to_score(score: float, mode: str) -> float:
    """Raise the effective minimum score requirement in recovery mode."""
    multipliers = get_mode_multipliers(mode)
    # Don't change the score, but return the extra threshold needed
    return multipliers["score_add"]
