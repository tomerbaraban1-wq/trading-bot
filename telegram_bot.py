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

# Warn at startup if Telegram is misconfigured
if TELEGRAM_BOT_TOKEN and not TELEGRAM_CHAT_ID:
    logger.warning("⚠️  TELEGRAM_BOT_TOKEN set but TELEGRAM_CHAT_ID missing — notifications disabled")
elif TELEGRAM_CHAT_ID and not TELEGRAM_BOT_TOKEN:
    logger.warning("⚠️  TELEGRAM_CHAT_ID set but TELEGRAM_BOT_TOKEN missing — notifications disabled")
elif not TELEGRAM_BOT_TOKEN:
    logger.info("ℹ️  Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable)")

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
    Send a message to Telegram AND Discord (if configured).
    Returns True if at least one channel succeeded.
    """
    # Send to Discord in parallel (tracked task — won't be silently dropped)
    try:
        from discord_bot import send_discord as _send_discord
        import asyncio as _asyncio
        task = _asyncio.create_task(_send_discord(text))
        task.add_done_callback(
            lambda t: logger.debug(f"Discord send failed: {t.exception()}")
            if not t.cancelled() and t.exception() else None
        )
    except Exception:
        pass

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
    qty:             float,
    price:           float,
    notional:        float,
    score:           float       = 0,
    sentiment_score: int         = 0,
    trade_id:        int | None  = None,
    is_iceberg:      bool        = False,
    n_slices:        int         = 0,
) -> None:
    """Rich BUY notification with position sizing and scoring context."""
    qty_str = f"{qty:.4f}" if qty != int(qty) else str(int(qty))
    iceberg_line = f"\n🧊 פיצול הזמנה: {n_slices} חלקים" if is_iceberg else ""
    id_line = f"\n🔖 עסקה #{trade_id}" if trade_id else ""
    try:
        from telegram_chat import _fmt_price as _fp
        _price_str = _fp(price)
        _notional_str = _fp(notional)
    except Exception:
        _price_str = f"${price:.2f}"
        _notional_str = f"${notional:,.2f}"
    # Score quality label
    if score >= 75:   q = "🔥 מצוין"
    elif score >= 65: q = "✅ טוב"
    elif score >= 58: q = "⚠️ גבולי"
    else:             q = "📊 רגיל"

    sent_label = "😨 פחד — הזדמנות" if sentiment_score <= 4 else ("🟢 חיובי" if sentiment_score >= 7 else "😐 ניטרלי")

    # Fetch stop & TP from DB for immediate context
    stop_line = ""
    tp_line   = ""
    try:
        import database as _db
        _trade = _db.get_open_trade_by_ticker(ticker)
        if _trade:
            _stop = _trade.get("atr_stop_price")
            if _stop:
                _stop_pct = (price - float(_stop)) / price * 100
                try:
                    from telegram_chat import _fmt_price as _fp2
                    stop_line = f"\n🛑  Stop Loss:      {_fp2(float(_stop))}  (-{_stop_pct:.1f}%)"
                    # Rough TP estimate: ~3× the stop distance
                    _tp = round(price + (price - float(_stop)) * 3, 2)
                    tp_line = f"\n🎯  יעד רווח:      {_fp2(_tp)}  (+{(_tp-price)/price*100:.1f}%)"
                except Exception:
                    stop_line = f"\n🛑  Stop Loss:      ${float(_stop):.2f}"
    except Exception:
        pass

    await send_message(
        f"🛒 <b>קנינו!</b>  🎉\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💹  <b>{ticker}</b>  ·  {qty_str} מניות\n\n"
        f"📌  מחיר קנייה:   {_price_str}\n"
        f"💸  סה״כ הושקע:  {_notional_str}"
        f"{stop_line}"
        f"{tp_line}\n\n"
        f"🎯  ציון:           <b>{score:.0f}/100</b>  {q}\n"
        f"🧠  סנטימנט:     {sentiment_score}/10  {sent_label}"
        f"{iceberg_line}"
        f"{id_line}\n\n"
        f"💡 <i>/stop {ticker}  |  /news {ticker}  |  /score {ticker}</i>"
    )


async def notify_trade_close(
    ticker:         str,
    qty:            float,
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
    result   = "🏆 רווח!" if win else "📉 הפסד"
    pct      = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
    dur_str  = _fmt_duration(duration_hours)
    reason_line = f"\n📌 סיבה: {reason}" if reason else ""
    id_line     = f"\n🔖 עסקה #{trade_id}" if trade_id else ""
    try:
        from telegram_chat import _fmt_price as _fp, _fmt_pnl as _fpnl
        _entry_str  = _fp(entry_price)
        _exit_str   = _fp(exit_price)
        _pnl_str    = _fpnl(pnl_gross)
        _net_str    = _fp(abs(pnl_net))
        _tax_str    = _fp(abs(tax_reserved))
    except Exception:
        _entry_str  = f"${entry_price:.2f}"
        _exit_str   = f"${exit_price:.2f}"
        _pnl_str    = f"{'+'  if win else '-'}${abs(pnl_gross):.2f}"
        _net_str    = f"${abs(pnl_net):.2f}"
        _tax_str    = f"${abs(tax_reserved):.2f}"

    await send_message(
        f"{'💰' if win else '📉'} <b>{'מכרנו ברווח!' if win else 'מכרנו בהפסד'}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💹  <b>{ticker}</b>  ·  {qty} מניות\n\n"
        f"📌  קנינו ב:      {_entry_str}\n"
        f"💵  מכרנו ב:    {_exit_str}\n"
        f"{'📈' if win else '📉'}  שינוי:          <b>{pct:+.2f}%</b>\n"
        f"⏱  זמן החזקה: {dur_str}\n\n"
        f"{'💚' if win else '❤️'}  {'רווח' if win else 'הפסד'}:       {_pnl_str}\n"
        f"💳  נטו אחרי מס: {_net_str}\n"
        f"🧾  מס שהופרש:  {_tax_str}"
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

    # Hebrew labels for known error types
    error_labels_he = {
        "api_timeout":        "פג זמן תגובה מ-API",
        "order_failed":       "ביצוע הזמנה נכשל",
        "insufficient_funds": "אין מספיק מזומן",
        "sentiment_fail":     "ניתוח סנטימנט נכשל",
        "loop_error":         "שגיאה כללית ברקע",
        "stop_loss_fail":     "עצירת הפסד נכשלה",
        "stop_loss":          "עצירת הפסד",
        "take_profit":        "רווח יעד הושג",
        "smart_sell":         "מכירה חכמה",
        "time_exit":          "יציאה לפי זמן",
        "emergency_exit":     "יציאת חירום",
        "stale_restart":      "פוזיציה ישנה — נוקתה",
    }
    error_label = error_labels_he.get(error_type, error_type.replace('_', ' '))

    ticker_line = f"  •  מניה: <b>{ticker}</b>" if ticker else ""
    detail_line = f"\n💬 {detail[:300]}"          if detail  else ""

    await send_message(
        f"⚠️ <b>שגיאה — {error_label}</b>"
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
    total_qty:    float,
    n_slices:     int,
    interval_sec: float,
) -> None:
    qty_str = f"{total_qty:.4f}" if total_qty != int(total_qty) else str(int(total_qty))
    slice_qty = round(total_qty / n_slices, 4) if n_slices else total_qty
    duration_min = (n_slices - 1) * interval_sec / 60
    await send_message(
        f"🧊 <b>פיצול הזמנה — {ticker}</b>\n"
        f"📦 {qty_str} מניות → {n_slices} חלקים × ~{slice_qty} מניות\n"
        f"⏱ מרווח: {interval_sec:.0f} שניות  |  משך משוער: ~{duration_min:.0f} דקות"
    )


async def notify_iceberg_done(
    ticker:     str,
    filled_qty: float,
    avg_price:  float,
    n_slices:   int,
    is_partial: bool,
) -> None:
    qty_str = f"{filled_qty:.4f}" if filled_qty != int(filled_qty) else str(int(filled_qty))
    status = "⚠️ בוצע חלקית" if is_partial else "✅ הושלם"
    await send_message(
        f"🧊 <b>פיצול הזמנה {status} — {ticker}</b>\n"
        f"📦 {qty_str} מניות בוצעו ב-{n_slices} חלקים\n"
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
    buys_today:         int   = 0,
) -> None:
    """Rich daily summary with per-trade breakdown, open positions, and ILS values."""
    from datetime import datetime, timezone, timedelta
    win_rate  = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl_emoji = "📈" if total_pnl >= 0 else "📉"

    # ILS formatting
    try:
        from telegram_chat import _fmt_price as _fp, _fmt_pnl as _fpnl
        equity_str  = _fp(equity)
        pnl_str     = _fpnl(total_pnl)
        net_str     = _fpnl(realized_pnl_net) if realized_pnl_net else ""
    except Exception:
        equity_str  = f"${equity:,.2f}"
        pnl_str     = f"${total_pnl:+.2f}"
        net_str     = f"${realized_pnl_net:+.2f}" if realized_pnl_net else ""

    lines = [
        f"🌙 <b>סיכום יומי</b>\n━━━━━━━━━━━━━━━━",
        f"🛒  קניות היום:    <b>{buys_today}</b>",
        f"💸  מכירות היום:  <b>{total_trades}</b>  (✅{wins}  ❌{losses})",
    ]
    if total_trades > 0:
        lines.append(f"🎯  Win Rate:        <b>{win_rate:.1f}%</b>")
    lines.append(f"{pnl_emoji}  רווח/הפסד:     <b>{pnl_str}</b>")
    if realized_pnl_net and net_str:
        lines.append(f"💳  נטו אחרי מס:  {net_str}")
    if tax_reserved > 0:
        lines.append(f"🧾  מס שהופרש:    ${tax_reserved:.2f}")

    # Per-trade breakdown (today's closed trades from DB)
    try:
        import database as _db
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_t = _db.get_trade_history(limit=30) or []
        today_closed = [x for x in all_t
                        if str(x.get("exit_time", ""))[:10] == today
                        and x.get("pnl_gross") is not None]
        if today_closed:
            lines.append("\n<b>עסקאות היום:</b>")
            for tr in today_closed[:5]:
                p     = float(tr.get("pnl_gross") or 0)
                ep    = float(tr.get("entry_price") or 0)
                xp    = float(tr.get("exit_price") or 0)
                pct   = (xp - ep) / ep * 100 if ep else 0
                icon  = "🟢" if p >= 0 else "🔴"
                lines.append(f"  {icon} <b>{tr['ticker']}</b>  {pct:+.1f}%  |  ${p:+.2f}")
    except Exception:
        pass

    lines.append(f"\n━━━━━━━━━━━━━━━━")
    lines.append(f"📂  פוזיציות פתוחות: <b>{open_positions}</b>")
    lines.append(f"💼  שווי תיק:           <b>{equity_str}</b>")

    # Open positions quick status
    try:
        import broker as _br, database as _db2
        open_trades = _db2.get_open_trades()
        if open_trades:
            for ot in open_trades[:3]:
                try:
                    pos = _br.get_position(ot["ticker"])
                    pct = float(pos.get("unrealized_plpc", 0)) * 100 if pos else 0
                    icon = "🟢" if pct >= 0 else "🔴"
                    lines.append(f"  {icon} <b>{ot['ticker']}</b>  {pct:+.1f}%")
                except Exception:
                    pass
    except Exception:
        pass

    await send_message("\n".join(lines))


async def notify_weekly_report(report_html: str) -> None:
    """Send pre-formatted weekly performance report HTML."""
    await send_message(report_html)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases  (keep old call-sites in heartbeat.py working)
# ─────────────────────────────────────────────────────────────────────────────

async def notify_buy(
    ticker:          str,
    qty:             float,
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
    Rich sell notification — pulls entry price + hold duration from DB
    so even the legacy heartbeat call-sites get a full breakdown.
    """
    win    = pnl_gross >= 0
    emoji  = "💰" if win else "📉"
    result = "מכרנו ברווח! 🎉" if win else "מכרנו בהפסד"

    # Pull entry details from DB for a richer message
    entry_price  = price   # fallback if DB lookup fails
    hold_str     = ""
    pct_str      = ""
    pnl_ils_str  = ""
    try:
        import database as _db
        trade = _db.get_open_trade_by_ticker(ticker)
        if not trade:
            # Already closed — try recent history
            hist = _db.get_trade_history(ticker=ticker, limit=1)
            trade = hist[0] if hist else None
        if trade:
            entry_price = float(trade.get("entry_price") or price)
            pct = (price - entry_price) / entry_price * 100 if entry_price else 0
            pct_str = f"{'📈' if pct >= 0 else '📉'}  שינוי:          <b>{pct:+.2f}%</b>\n"
            # Duration
            from datetime import datetime, timezone as _tz
            entry_ts = trade.get("entry_time")
            if entry_ts:
                try:
                    ed = datetime.strptime(str(entry_ts)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz.utc)
                    hrs = (datetime.now(_tz.utc) - ed).total_seconds() / 3600
                    hold_str = f"⏱  זמן החזקה:  {_fmt_duration(hrs)}\n"
                except Exception:
                    pass
        # ILS
        try:
            from telegram_chat import _fmt_price as _fp, _fmt_pnl as _fpnl
            entry_str = _fp(entry_price)
            exit_str  = _fp(price)
            pnl_ils   = _fpnl(pnl_gross)
        except Exception:
            entry_str = f"${entry_price:.2f}"
            exit_str  = f"${price:.2f}"
            pnl_ils   = f"${pnl_gross:+.2f}"
    except Exception:
        entry_str = f"${entry_price:.2f}"
        exit_str  = f"${price:.2f}"
        pnl_ils   = f"${pnl_gross:+.2f}"

    # Reason labels
    reason_map = {
        "take_profit":    "🎯 יעד רווח הושג",
        "stop_loss":      "🛑 סטופ לוס",
        "smart_sell":     "🧠 מכירה חכמה (ציון נפל)",
        "news_exit":      "📰 חדשות שליליות",
        "time_exit":      "⏱ יציאה לפי זמן",
        "manual":         "✋ יציאה ידנית",
        "eod_sweep":      "🌙 ניקוי סוף יום",
    }
    reason_label = reason_map.get(reason, f"📌 {reason}")

    await send_message(
        f"{emoji} <b>{result}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💹  <b>{ticker}</b>\n\n"
        f"📌  קנינו ב:       {entry_str}\n"
        f"💵  מכרנו ב:     {exit_str}\n"
        f"{pct_str}"
        f"{hold_str}\n"
        f"{'💚' if win else '❤️'}  {'רווח' if win else 'הפסד'}:        <b>{pnl_ils}</b>\n\n"
        f"{reason_label}"
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
