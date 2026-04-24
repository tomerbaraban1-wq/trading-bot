"""
Telegram Notification System

Design principles:
  1. Every trade event fires a structured notification — no silent executions.
  2. Critical errors are rate-limited (max 1 per type per 5 min) to prevent spam.
  3. send_message() retries up to 3× with exponential back-off — one Telegram
     hiccup must not lose an important alert.
  4. All functions are fire-and-forget (async) — callers use asyncio.ensure_future()
     so notifications never block the trading pipeline.

Notification inventory:
  Trade events    → notify_trade_open, notify_trade_close
  Emergency       → notify_emergency
  Iceberg         → notify_iceberg_start, notify_iceberg_done
  Errors          → notify_error  (rate-limited, 5-min cooldown per error type)
  Circuit breaker → notify_circuit_breaker_tripped
  Budget          → notify_budget_warning
  Daily summary   → notify_daily_summary
  Weekly report   → notify_weekly_report

Backward-compat aliases (keep old call-sites working):
  notify_buy  → notify_trade_open (thin wrapper)
  notify_sell → notify_trade_close (thin wrapper)
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone

import aiohttp

from config import settings

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = getattr(settings, "TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID:   str = getattr(settings, "TELEGRAM_CHAT_ID",   "")

_SEND_TIMEOUT_SEC    = 10
_MAX_RETRIES         = 3
_RETRY_BASE_DELAY    = 2.0   # seconds (doubles each retry)
ERROR_COOLDOWN_SEC   = 300   # 5 minutes between identical error types

# ── Error rate limiter ────────────────────────────────────────────────────────
_error_cooldown: dict[str, float] = {}   # error_key → last_sent_ts
_cooldown_lock = threading.Lock()


def _enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _is_rate_limited(error_key: str) -> bool:
    with _cooldown_lock:
        last = _error_cooldown.get(error_key, 0)
        return (time.time() - last) < ERROR_COOLDOWN_SEC


def _mark_sent(error_key: str) -> None:
    with _cooldown_lock:
        _error_cooldown[error_key] = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Core sender — with retry + back-off
# ─────────────────────────────────────────────────────────────────────────────

async def send_message(text: str) -> bool:
    """
    Send a Telegram message with up to 3 retries on failure.
    Returns True on success, False if all retries failed.
    Silently no-ops when bot token / chat ID are not configured.
    """
    if not _enabled():
        return False

    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text[:4096],   # Telegram max message length
        "parse_mode": "HTML",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=_SEND_TIMEOUT_SEC),
                ) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.warning(
                        f"Telegram HTTP {resp.status} (attempt {attempt}/{_MAX_RETRIES}): {body[:200]}"
                    )
        except asyncio.TimeoutError:
            logger.warning(f"Telegram timeout (attempt {attempt}/{_MAX_RETRIES})")
        except Exception as exc:
            logger.warning(f"Telegram error (attempt {attempt}/{_MAX_RETRIES}): {exc}")

        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

    logger.error("Telegram: all retries failed — message dropped")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Trade notifications
# ─────────────────────────────────────────────────────────────────────────────

async def notify_trade_open(
    ticker:          str,
    qty:             int,
    price:           float,
    notional:        float,
    score:           float       = 0,
    sentiment_score: int         = 0,
    trade_id:        int | None  = None,
    is_iceberg:      bool        = False,
    n_slices:        int         = 0,
) -> None:
    """Rich BUY notification with position sizing and scoring context."""
    iceberg_line = (
        f"\n🧊 פיצול הזמנה: {n_slices} חלקים"
        if is_iceberg else ""
    )
    id_line = f"\n🔖 עסקה #{trade_id}" if trade_id else ""
    await send_message(
        f"🟢 <b>קנייה — {ticker}</b>\n"
        f"📦 {qty} מניות במחיר ${price:.2f}  (סה״כ ${notional:,.2f})\n"
        f"🎯 ציון: {score:.0f}/100  |  סנטימנט: {sentiment_score}/10"
        f"{iceberg_line}"
        f"{id_line}"
    )


async def notify_trade_close(
    ticker:         str,
    qty:            int,
    entry_price:    float,
    exit_price:     float,
    pnl_gross:      float,
    pnl_net:        float,
    tax_reserved:   float,
    duration_hours: float       = 0.0,
    reason:         str         = "",
    trade_id:       int | None  = None,
) -> None:
    """Rich SELL notification with full P&L breakdown and trade duration."""
    win      = pnl_gross >= 0
    emoji    = "💰" if win else "🔴"
    pct      = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
    dur_str  = _fmt_duration(duration_hours)
    reason_line = f"\n📌 סיבה: {reason}" if reason else ""
    id_line     = f"\n🔖 עסקה #{trade_id}" if trade_id else ""

    await send_message(
        f"{emoji} <b>מכירה — {ticker}</b>\n"
        f"📦 {qty} מניות  |  ⏱ {dur_str}\n"
        f"💵 כניסה ${entry_price:.2f} → יציאה ${exit_price:.2f}  ({pct:+.2f}%)\n"
        f"{'📈' if win else '📉'} רווח/הפסד: <b>${pnl_gross:+.2f}</b>  |  "
        f"נטו: ${pnl_net:+.2f}  |  מס: ${tax_reserved:.2f}"
        f"{reason_line}"
        f"{id_line}"
    )


async def notify_emergency(ticker: str, reason: str) -> None:
    """Emergency exit alert."""
    await send_message(
        f"🚨 <b>יציאת חירום — {ticker}</b>\n"
        f"⚠️ {reason}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error notifications (rate-limited)
# ─────────────────────────────────────────────────────────────────────────────

async def notify_error(
    error_type: str,
    ticker:     str  = "",
    detail:     str  = "",
) -> None:
    """
    Send a critical error alert — rate-limited to one message per error_type
    per ERROR_COOLDOWN_SEC (default 5 min) to prevent Telegram spam.

    error_type examples:
      "api_timeout"        broker or sentiment API timed out
      "order_failed"       broker rejected or errored on the order
      "insufficient_funds" budget check blocked a trade
      "sentiment_fail"     Groq/LLM call failed
      "loop_error"         background task threw an unexpected exception
      "stop_loss_fail"     stop-loss monitor couldn't exit a position
    """
    key = f"{error_type}:{ticker}" if ticker else error_type
    if _is_rate_limited(key):
        logger.debug(f"Telegram: rate-limiting error notification '{key}'")
        return

    _mark_sent(key)

    ticker_line = f"  •  מניה: <b>{ticker}</b>" if ticker else ""
    detail_line = f"\n💬 {detail[:300]}"          if detail  else ""

    await send_message(
        f"⚠️ <b>שגיאה — {error_type.replace('_', ' ').upper()}</b>"
        f"{ticker_line}"
        f"{detail_line}\n"
        f"<i>⏰ {_utcnow()}</i>"
    )


async def notify_circuit_breaker_tripped(
    daily_pnl:    float,
    loss_limit:   float,
    trip_reason:  str = "",
) -> None:
    """Alert when the daily loss circuit breaker fires — highest-priority message."""
    _mark_sent("circuit_breaker")   # suppress repeat for rest of day
    await send_message(
        f"🔴🔴 <b>עצור! הפסד יומי מקסימלי הושג</b> 🔴🔴\n"
        f"🛑 אין קניות נוספות להיום\n"
        f"📉 רווח/הפסד יומי: <b>${daily_pnl:+.2f}</b>  "
        f"(מגבלה ${loss_limit:.2f})\n"
        f"💬 {trip_reason}\n"
        f"<i>⏰ {_utcnow()}</i>"
    )


async def notify_budget_warning(reason: str, cash_available: float) -> None:
    """Warn when budget check blocks a trade — rate-limited."""
    if _is_rate_limited("budget_warning"):
        return
    _mark_sent("budget_warning")
    await send_message(
        f"💸 <b>אזהרת תקציב</b>\n"
        f"💵 מזומן זמין: ${cash_available:.2f}\n"
        f"📌 {reason}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Iceberg notifications
# ─────────────────────────────────────────────────────────────────────────────

async def notify_iceberg_start(
    ticker:       str,
    total_qty:    int,
    n_slices:     int,
    interval_sec: float,
) -> None:
    duration_min = (n_slices - 1) * interval_sec / 60
    await send_message(
        f"🧊 <b>פיצול הזמנה — {ticker}</b>\n"
        f"📦 {total_qty} מניות → {n_slices} חלקים × ~{max(1, total_qty // n_slices)} מניות\n"
        f"⏱ מרווח: {interval_sec:.0f} שניות  |  משך משוער: ~{duration_min:.0f} דקות"
    )


async def notify_iceberg_done(
    ticker:     str,
    filled_qty: int,
    avg_price:  float,
    n_slices:   int,
    is_partial: bool,
) -> None:
    status = "⚠️ בוצע חלקית" if is_partial else "✅ הושלם"
    await send_message(
        f"🧊 <b>פיצול הזמנה {status} — {ticker}</b>\n"
        f"📦 {filled_qty} מניות בוצעו ב-{n_slices} חלקים\n"
        f"💵 מחיר ממוצע: ${avg_price:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Periodic summaries
# ─────────────────────────────────────────────────────────────────────────────

async def notify_daily_summary(
    total_trades:       int,
    wins:               int,
    losses:             int,
    total_pnl:          float,
    open_positions:     int,
    equity:             float,
    tax_reserved:       float = 0.0,
    realized_pnl_net:   float = 0.0,
) -> None:
    """Enhanced daily summary including equity, tax, and net P&L."""
    win_rate   = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl_emoji  = "📈" if total_pnl >= 0 else "📉"
    tax_line   = f"\n🧾 מס שהופרש היום: ${tax_reserved:.2f}" if tax_reserved > 0 else ""
    net_line   = f"\n💳 רווח נטו (אחרי מס): ${realized_pnl_net:+.2f}" if realized_pnl_net else ""

    await send_message(
        f"📊 <b>סיכום יומי</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🔄 עסקאות: {total_trades}  (רווח: {wins}  /  הפסד: {losses})\n"
        f"🎯 אחוז הצלחה: {win_rate:.1f}%\n"
        f"{pnl_emoji} רווח/הפסד: <b>${total_pnl:+.2f}</b>"
        f"{net_line}"
        f"{tax_line}\n"
        f"📂 פוזיציות פתוחות: {open_positions}\n"
        f"💼 שווי תיק: ${equity:,.2f}"
    )


async def notify_weekly_report(report_html: str) -> None:
    """Send pre-formatted weekly performance report HTML."""
    await send_message(report_html)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases  (keep old call-sites in heartbeat.py working)
# ─────────────────────────────────────────────────────────────────────────────

async def notify_buy(
    ticker:          str,
    qty:             int,
    price:           float,
    score:           float,
    sentiment:       int,
) -> None:
    """Legacy alias → notify_trade_open."""
    await notify_trade_open(
        ticker=ticker, qty=qty, price=price,
        notional=round(price * qty, 2),
        score=score, sentiment_score=sentiment,
    )


async def notify_sell(
    ticker:    str,
    price:     float,
    pnl_gross: float,
    reason:    str,
) -> None:
    """
    Legacy alias used by heartbeat stop_loss / take_profit / smart_sell.
    Sends a concise sell message (no entry price / duration available here).
    """
    win   = pnl_gross >= 0
    emoji = "💰" if win else "🔴"
    await send_message(
        f"{emoji} <b>מכירה — {ticker}</b>\n"
        f"💵 יציאה @ ${price:.2f}  |  רווח/הפסד: <b>${pnl_gross:+.2f}</b>\n"
        f"📌 {reason}"
    )


async def notify_slippage_alert(
    avg_slip_pct: float,
    ticker:       str,
    rolling_n:    int,
    threshold:    float,
) -> None:
    """
    Fired when the rolling-average actual slippage exceeds the configured
    threshold.  Warns the operator to review execution quality.
    """
    if not _enabled():
        return
    if _is_rate_limited("slippage_alert"):
        return
    _mark_sent("slippage_alert")

    await send_message(
        f"⚠️ <b>התראת סחירות גבוהה</b>\n"
        f"ממוצע {rolling_n} עסקאות אחרונות: <b>{avg_slip_pct:.3f}%</b> "
        f"(מגבלה: {threshold}%)\n"
        f"עסקה אחרונה: <code>{ticker}</code>\n"
        f"בדוק איכות ביצוע — ייתכן שנדרש כיוונון מחירי הגבלה.\n"
        f"🕒 {_utcnow()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_duration(hours: float) -> str:
    if hours < 1 / 60:
        return "< דקה"
    if hours < 1:
        return f"{int(hours * 60)} דקות"
    if hours < 24:
        return f"{hours:.1f} שעות"
    return f"{hours / 24:.1f} ימים"
