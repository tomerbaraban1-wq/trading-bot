"""
Activity Broadcaster — 24/7 Live Updates
==========================================

Sends Telegram updates for EVERY meaningful bot activity:
  🔍 Scanning starts
  📊 Score computed for a stock
  ✅ Stock passes filters → buying
  ❌ Stock blocked → reason
  💰 Position opened
  📈 Position update (significant move)
  🛑 Stop-loss triggered
  💵 Take-profit hit
  🧠 Training started/finished
  📚 Learning insight discovered
  🌙 Market closed (1/hour reminder)
  💓 Heartbeat alive (20-45 min)

Smart features:
  - Throttling per category (no spam)
  - Batching scan reports (one msg per scan, not per ticker)
  - Quiet hours (1 AM – 7 AM Israel — no notifications unless critical)
  - Priority tiers (CRITICAL bypasses all throttling)
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class Priority:
    """Priority levels — higher = more important, bypasses throttling."""
    QUIET = 0     # only sent during work hours (skip 1 AM – 7 AM)
    INFO = 1      # standard activity
    IMPORTANT = 2  # always sent
    CRITICAL = 3   # always sent + force flag (skip dedup)


# Last-send tracker per category: category → last_sent_ts
_last_sent: dict[str, float] = defaultdict(float)

# Throttle intervals (seconds) per category
_THROTTLE = {
    "scan_start":      300,   # 5 min between scan-start messages
    "training_start":  900,   # 15 min — don't spam training start
    "training_done":   900,
    "heartbeat":         0,   # always send heartbeats (controlled by sender)
    "buy":               0,   # ALWAYS send buy
    "sell":              0,   # ALWAYS send sell
    "stop_loss":         0,   # ALWAYS send stop
    "take_profit":       0,   # ALWAYS send TP
    "learning":        600,   # 10 min between learning insights
    "market_state":   3600,   # 1h between market state msgs
    "scan_summary":     60,   # 1 min between scan summary msgs
    "position_alert": 1800,   # 30 min per ticker alert
}


def _is_quiet_hours() -> bool:
    """Quiet hours = non-critical Telegram messages are silenced.

    DISABLED by default — the user asked for a full 24/7 step-by-step feed, and the
    nightly 1 AM–7 AM silence looked like the bot was "pausing" (it isn't: it runs
    ~360-540 actions/hour all night, it just wasn't narrating). To restore the night
    silence, set QUIET_HOURS_ENABLED=true (optionally QUIET_HOURS_START / _END, 0-23).
    """
    import os
    if os.getenv("QUIET_HOURS_ENABLED", "false").strip().lower() not in ("true", "1", "yes"):
        return False
    now_utc = datetime.now(timezone.utc)
    il_off = 3 if 3 <= now_utc.month <= 10 else 2
    h = (now_utc + timedelta(hours=il_off)).hour
    start = int(os.getenv("QUIET_HOURS_START", "1"))
    end = int(os.getenv("QUIET_HOURS_END", "7"))
    return start <= h < end


def _il_time_str() -> str:
    """Return current Israel time as HH:MM."""
    now_utc = datetime.now(timezone.utc)
    il_off = 3 if 3 <= now_utc.month <= 10 else 2
    return (now_utc + timedelta(hours=il_off)).strftime("%H:%M")


async def broadcast(
    category: str,
    message: str,
    priority: int = Priority.INFO,
    key: str = "",   # optional per-key throttling (e.g. ticker name)
) -> bool:
    """
    Send a categorized activity update to Telegram.

    Returns True if sent, False if throttled / silenced.

    Args:
      category: one of the keys in _THROTTLE (or any string, default throttle=60s)
      message: HTML-formatted body (Hebrew preferred)
      priority: Priority.* level
      key: optional sub-key (e.g. ticker) to throttle per-key
    """
    # ── Quiet hours filter ──────────────────────────────────────────────
    if _is_quiet_hours() and priority < Priority.IMPORTANT:
        logger.debug(f"[BROADCAST] {category} silenced (quiet hours)")
        return False

    # ── Throttling ──────────────────────────────────────────────────────
    throttle_key = f"{category}:{key}" if key else category
    throttle_sec = _THROTTLE.get(category, 60)
    if priority < Priority.IMPORTANT and throttle_sec > 0:
        now = time.time()
        last = _last_sent.get(throttle_key, 0)
        if now - last < throttle_sec:
            logger.debug(f"[BROADCAST] {category} throttled (last {now-last:.0f}s ago)")
            return False
        _last_sent[throttle_key] = now

    # ── Send ────────────────────────────────────────────────────────────
    try:
        from telegram_bot import send_message
        force = priority >= Priority.CRITICAL
        await send_message(message, force=force)
        logger.debug(f"[BROADCAST] sent: {category}")
        return True
    except Exception as e:
        logger.warning(f"[BROADCAST] send failed for {category}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPERS — easy to call from anywhere
# ─────────────────────────────────────────────────────────────────────────────

async def announce_scan_start(n_tickers: int, cash: float) -> None:
    """🔍 Bot starts a scan cycle."""
    await broadcast(
        "scan_start",
        f"🔍 <b>סורק {n_tickers} מניות</b>  <i>({_il_time_str()})</i>\n"
        f"💰 מזומן זמין: ${cash:,.0f}",
        priority=Priority.QUIET,
    )


async def announce_scan_done(n_passed: int, n_failed: int, top_picks: list[str] | None = None) -> None:
    """📊 Scan completed."""
    lines = [
        f"📊 <b>סריקה הסתיימה</b>  <i>({_il_time_str()})</i>",
        f"━━━━━━━━━━━━━━━━",
        f"✅ עברו: {n_passed}  |  ❌ נחסמו: {n_failed}",
    ]
    if top_picks:
        lines.append(f"🎯 מועמדים: {', '.join(top_picks[:5])}")
    await broadcast("scan_summary", "\n".join(lines), priority=Priority.QUIET)


async def announce_training_start(mode: str, tickers: list[str]) -> None:
    """🧠 Bot starts learning from history — each ticker is a clickable TradingView chart link."""
    def _tv(t):
        return f'<a href="https://www.tradingview.com/chart/?symbol={t}">{t}</a>'
    sample = " · ".join(_tv(t) for t in tickers[:6])
    if len(tickers) > 6:
        sample += f" + {len(tickers) - 6} עוד"
    await broadcast(
        "training_start",
        f"🧠 <b>מתחיל אימון — {mode}</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📈 מניות (לחץ לגרף TradingView):\n{sample}\n"
        f"⏳ מנתח היסטוריה...",
        priority=Priority.INFO,
    )


async def announce_training_done(
    tickers_analyzed: int, win_rate: float, avg_return: float, optimal_score: int
) -> None:
    """🎓 Training completed."""
    icon = "✅" if win_rate >= 50 else "⚠️" if win_rate >= 35 else "❌"
    await broadcast(
        "training_done",
        f"🎓 <b>אימון הושלם</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 ניתחתי: {tickers_analyzed} מניות\n"
        f"{icon} הצלחה: <b>{win_rate:.1f}%</b>\n"
        f"📈 תשואה ממוצעת: {avg_return:+.2f}%\n"
        f"🎯 ציון אופטימלי: <b>{optimal_score}</b>",
        priority=Priority.INFO,
    )


async def announce_buy(ticker: str, price: float, qty: float, score: float, reason: str = "") -> None:
    """💰 Position opened — with inline action buttons."""
    notional = price * qty
    msg = (
        f"💰 <b>קניתי {ticker}</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💲 מחיר: ${price:.2f}  |  📦 כמות: {qty:.4f}\n"
        f"💼 הושקע: ${notional:,.2f}\n"
        f"🎯 ציון: {score:.0f}/100\n"
        f"{('📝 ' + reason) if reason else ''}"
    )

    # Send with inline buttons for quick actions
    try:
        from telegram_bot import send_message_with_inline
        buttons = [
            [
                {"text": "📊 פרטים", "callback_data": f"info:{ticker}"},
                {"text": "📰 חדשות", "callback_data": f"news:{ticker}"},
            ],
            [
                {"text": "💸 מכור עכשיו", "callback_data": f"sell:{ticker}"},
                {"text": "🎯 ציון מלא", "callback_data": f"score:{ticker}"},
            ],
        ]
        await send_message_with_inline(msg, buttons=buttons)
    except Exception as e:
        logger.debug(f"Inline buttons failed, falling back: {e}")
        await broadcast("buy", msg, priority=Priority.IMPORTANT)


async def announce_sell(ticker: str, price: float, pnl: float, reason: str = "") -> None:
    """💸 Position closed."""
    pnl_icon = "📈" if pnl >= 0 else "📉"
    pnl_word = "רווח" if pnl >= 0 else "הפסד"
    cat = "take_profit" if pnl >= 0 else "stop_loss"
    await broadcast(
        cat,
        f"💸 <b>מכרתי {ticker}</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💲 מחיר יציאה: ${price:.2f}\n"
        f"{pnl_icon} {pnl_word}: <b>${abs(pnl):+,.2f}</b>\n"
        f"{('📝 ' + reason) if reason else ''}",
        priority=Priority.IMPORTANT,
    )


async def announce_market_closed() -> None:
    """🌙 Market is closed — bot is training/learning."""
    await broadcast(
        "market_state",
        f"🌙 <b>שוק סגור</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💤 הבוט לא ישן — הוא עובד:\n"
        f"  🧠 לומד מעסקאות עבר\n"
        f"  📊 מעדכן ציון אופטימלי\n"
        f"  📰 קורא חדשות\n"
        f"  🔍 מנתח גרפים",
        priority=Priority.QUIET,
    )


async def announce_market_open() -> None:
    """🟢 Market just opened — bot starts active trading."""
    await broadcast(
        "market_state",
        f"🟢 <b>השוק נפתח!</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🚀 הבוט עובר למצב מסחר אקטיבי:\n"
        f"  🔍 סורק כל 4 דק'\n"
        f"  🎯 מחפש מניות עם ציון 58+\n"
        f"  ⚡ מבצע קנייה אוטומטית",
        priority=Priority.IMPORTANT,
    )


async def announce_learning_insight(insight: str) -> None:
    """📚 Bot discovered a new learning."""
    await broadcast(
        "learning",
        f"📚 <b>תובנה חדשה</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{insight}",
        priority=Priority.QUIET,
    )


async def announce_position_alert(ticker: str, pct: float, reason: str) -> None:
    """⚠️ Significant move on an open position."""
    icon = "📈" if pct >= 0 else "📉"
    await broadcast(
        "position_alert",
        f"{icon} <b>{ticker}</b> — {pct:+.1f}%\n"
        f"<i>{reason}</i>",
        priority=Priority.INFO,
        key=ticker,
    )


async def announce_heartbeat(activity: str, equity: float, positions: int, pnl: float) -> None:
    """💓 Periodic heartbeat — bot is alive."""
    pnl_icon = "📈" if pnl >= 0 else "📉"
    await broadcast(
        "heartbeat",
        f"💓 <b>הבוט פעיל ועובד</b>  <i>({_il_time_str()})</i>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{activity}\n"
        f"💼 שווי תיק: ${equity:,.0f}\n"
        f"📂 פוזיציות: {positions}\n"
        f"{pnl_icon} P&L פתוח: ${pnl:+,.2f}",
        priority=Priority.QUIET,
    )
