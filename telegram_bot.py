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


import re as _re

# HTML tags Telegram supports in HTML parse mode
_TELEGRAM_ALLOWED_TAGS = {
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "a", "code", "pre", "tg-spoiler",
}

# Pre-compiled pattern: matches allowed open/close tags exactly
_ALLOWED_TAG_RE = _re.compile(
    r"</?(b|strong|i|em|u|ins|s|strike|del|code|pre|tg-spoiler)"
    r"(\s[^>]*)?>|<a(\s[^>]*)?>|</a>",
    _re.IGNORECASE,
)

def _sanitize_html(text: str) -> str:
    """
    Robustly sanitize HTML for Telegram's strict HTML parser.

    Strategy (3-pass):
    1. Save all ALLOWED tags by replacing them with placeholders.
    2. Escape every remaining < and > character (comparison operators,
       unsupported tags like <ticker>, Hebrew tags, etc.).
    3. Restore the saved allowed tags.

    This prevents silent message drops caused by:
    - Comparison operators:  ציון 55 < min_score 70
    - Unsupported tags:      <ticker>, <score>, <מניה>
    - Broken/unclosed tags:  <anything without closing >
    """
    # Step 1 — extract and protect allowed tags
    placeholders: list[str] = []

    def _save_tag(m: _re.Match) -> str:
        placeholders.append(m.group(0))
        return f"\x00TAG{len(placeholders)-1}\x00"

    protected = _ALLOWED_TAG_RE.sub(_save_tag, text)

    # Step 2 — escape bare < and >
    protected = protected.replace("&", "&amp;")   # escape & first
    protected = protected.replace("<", "&lt;")
    protected = protected.replace(">", "&gt;")

    # Step 3 — restore allowed tags
    for i, tag in enumerate(placeholders):
        protected = protected.replace(f"\x00TAG{i}\x00", tag)

    return protected


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

def _build_progress_bar(pct: float, width: int = 10, fill: str = "🟩", empty: str = "⬜") -> str:
    """Build a visual progress bar. pct is 0-100."""
    filled = max(0, min(width, round(pct / 100 * width)))
    return fill * filled + empty * (width - filled)


def _build_pnl_chart(values: list[float], width: int = 20) -> str:
    """
    Tiny ASCII sparkline chart for P&L trend.
    Uses block characters: ▁▂▃▄▅▆▇█
    """
    if not values or len(values) < 2:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    result = ""
    for v in values[-width:]:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        result += blocks[max(0, min(len(blocks) - 1, idx))]
    return result


async def send_menu() -> bool:
    """
    Send a persistent reply keyboard with the most common commands.
    Enhanced version with more useful buttons and better layout.
    """
    if not _enabled():
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    keyboard = {
        "keyboard": [
            # Row 1: ⭐ Most important — positions with TV links
            [{"text": "📈 מניות + TradingView"}, {"text": "💰 P&L מהיר"}],
            # Row 2: Portfolio status
            [{"text": "📍 פוזיציות"},       {"text": "📊 ביצועים"},     {"text": "⚖️ סיכון"}],
            # Row 3: Market & analysis
            [{"text": "🌍 שוק עכשיו"},      {"text": "📰 חדשות"},       {"text": "🤖 AI ניתוח"}],
            # Row 4: Bot management
            [{"text": "🩺 בדיקה מלאה"},     {"text": "🔄 סיבוב תיק"},   {"text": "💰 Exits"}],
            # Row 5: Help
            [{"text": "📋 כל הפקודות"}],
        ],
        "resize_keyboard": True,
        "persistent": True,
    }
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": (
            "🤖 <b>מנהל ההשקעות שלך</b>\n"
            "━━━━━━━━━━━━━━━━\n"
            "📱 בחר מהתפריט או הקלד שאלה חופשית!"
        ),
        "parse_mode": "HTML",
        "reply_markup": keyboard,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def send_message_with_inline(
    text: str,
    buttons: list[list[dict]] | None = None,
) -> bool:
    """
    Send a message with optional inline keyboard buttons.

    buttons format: [[{"text": "לחץ כאן", "callback_data": "action:data"}]]
    """
    reply_markup = {"inline_keyboard": buttons} if buttons else None
    return await send_message(text, reply_markup=reply_markup)


async def edit_message(chat_id: str, message_id: int, new_text: str) -> bool:
    """Edit an existing Telegram message (for live updates)."""
    if not _enabled():
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": new_text[:4096],
        "parse_mode": "HTML",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def answer_callback_query(callback_id: str, text: str = "") -> bool:
    """Answer a Telegram callback query (removes the loading state on button)."""
    if not _enabled():
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    payload = {"callback_query_id": callback_id, "text": text[:200]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def send_message(text: str, reply_markup: dict | None = None,
                       force: bool = False) -> bool:
    """
    Send a message to Telegram AND Discord (if configured).
    Returns True if at least one channel succeeded.

    Args:
        text: HTML-formatted message body.
        reply_markup: Optional inline keyboard. Example:
            {"inline_keyboard": [[{"text": "💲 מחיר", "callback_data": "price:AAPL"}]]}
        force: If True, bypass anti-spam deduplication (for critical alerts).

    Note: Automatically translates English text to Hebrew via translation_service.

    Anti-spam: drops messages identical to one sent in the last 90 seconds
    (unless force=True). Prevents notification fatigue.
    """
    # ── ANTI-SPAM: dedupe identical messages within 90 seconds ────────────
    if not force and text:
        try:
            import hashlib
            import time as _spam_t
            _key = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
            if not hasattr(send_message, "_recent_hashes"):
                send_message._recent_hashes = {}
            _now = _spam_t.time()
            # Clean old
            send_message._recent_hashes = {
                k: v for k, v in send_message._recent_hashes.items()
                if _now - v < 90
            }
            if _key in send_message._recent_hashes:
                logger.debug(f"[ANTI-SPAM] Duplicate message dropped (sent {_now - send_message._recent_hashes[_key]:.0f}s ago)")
                return True   # pretend success — already shown to user
            send_message._recent_hashes[_key] = _now
        except Exception as _antispam_err:
            logger.debug(f"Anti-spam check failed: {_antispam_err}")

    # ── AUTO-TRANSLATE TO HEBREW ──────────────────────────────────────────
    # Smart translation: financial glossary first, then Google Translate
    # Preserves: HTML tags, tickers ($AAPL), numbers, percentages, URLs
    try:
        from translation_service import translate_message_smart, TRANSLATION_ENABLED
        if TRANSLATION_ENABLED:
            text = await translate_message_smart(text)
    except Exception as e:
        logger.debug(f"Translation skipped: {e}")
        # Continue with original text if translation fails

    # Send to Discord in parallel — only if a running event loop exists
    # Note: Discord gets the translated (Hebrew) version
    try:
        from discord_bot import send_discord as _send_discord
        import asyncio as _asyncio
        _loop = _asyncio.get_running_loop()   # raises RuntimeError if no loop
        task = _loop.create_task(_send_discord(text))
        task.add_done_callback(
            lambda t: logger.debug(f"Discord send failed: {t.exception()}")
            if not t.cancelled() and t.exception() else None
        )
    except RuntimeError:
        pass  # no running loop — Discord send skipped (non-critical)
    except Exception:
        pass

    if not _enabled():
        return False

    # Sanitize HTML — remove tags Telegram doesn't support (causes 400 + dropped msg)
    text = _sanitize_html(text)

    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text[:4096],   # Telegram max message length
        "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

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
    qty_str = f"{qty:.4f}".rstrip('0').rstrip('.')
    iceberg_line = f"\n🧊 פיצול הזמנה: {n_slices} חלקים" if is_iceberg else ""
    id_line = f"\n🔖 עסקה #{trade_id}" if trade_id else ""

    # Calculate stop & TP first (needed for formatting)
    stop_price = None
    tp_price   = None
    try:
        import database as _db2
        _t2 = _db2.get_open_trade_by_ticker(ticker)
        if _t2 and _t2.get("atr_stop_price"):
            stop_price = float(_t2["atr_stop_price"])
            tp_price   = round(price + (price - stop_price) * 3, 2)
    except Exception:
        pass
    if stop_price is None:
        from config import settings as _s
        stop_price = round(price * (1 - _s.STOP_LOSS_PCT  / 100), 2)
        tp_price   = round(price * (1 + _s.TAKE_PROFIT_PCT / 100), 2)
    stop_pct = (price - stop_price) / price * 100
    tp_pct   = (tp_price - price)   / price * 100

    try:
        from telegram_chat import _fmt_price as _fp
        _price_str    = _fp(price)
        _notional_str = _fp(notional)
        _stop_str     = _fp(stop_price)
        _tp_str       = _fp(tp_price)
    except Exception:
        _price_str    = f"${price:.2f}"
        _notional_str = f"${notional:,.2f}"
        _stop_str     = f"${stop_price:.2f}"
        _tp_str       = f"${tp_price:.2f}"
    # תווית איכות
    if score >= 75:   q = "🔥 מצוין"
    elif score >= 65: q = "✅ טוב"
    elif score >= 58: q = "⚠️ בסדר"
    else:             q = "📊 רגיל"

    # פס ציון ויזואלי
    _bar_filled = round(score / 10)
    _score_bar  = "🟩" * _bar_filled + "⬜" * (10 - _bar_filled)

    # Build risk/reward ratio
    rr_ratio = tp_pct / stop_pct if stop_pct > 0 else 0
    rr_str = f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "—"

    # Inline keyboard: quick actions
    inline_buttons = [
        [
            {"text": f"📊 פרטים על {ticker}", "callback_data": f"info:{ticker}"},
            {"text": "📍 כל הפוזיציות", "callback_data": "positions:all"},
        ],
        [
            {"text": "⚡ AI ניתוח", "callback_data": f"ai:{ticker}"},
            {"text": "📰 חדשות", "callback_data": f"news:{ticker}"},
        ],
    ]

    # TradingView link
    tv_url  = f"https://www.tradingview.com/chart/?symbol={ticker}"
    tv_line = f'\n🔗 <a href="{tv_url}">פתח גרף ב-TradingView</a>'

    # Partial exits plan
    partial_plan = (
        f"\n━━━━━━━━━━━━━━━━\n"
        f"💰 <b>תכנית יציאות חלקיות:</b>\n"
        f"  +5%  → מוכר 25% אוטומטית\n"
        f"  +10% → מוכר עוד 25%\n"
        f"  +18% → מוכר עוד 25%\n"
        f"  🏃 25% ירוץ עם Trailing Stop"
    )

    inline_buttons = [
        [
            {"text": f"📊 גרף {ticker}", "url": tv_url},
            {"text": "📍 כל הפוזיציות", "callback_data": "positions:all"},
        ],
        [
            {"text": f"⚡ AI ניתוח {ticker}", "callback_data": f"ai:{ticker}"},
            {"text": f"📰 חדשות {ticker}", "callback_data": f"news:{ticker}"},
        ],
        [
            {"text": "💰 P&L מהיר", "callback_data": "pnl:quick"},
            {"text": "⚖️ סיכון", "callback_data": "risk:check"},
        ],
    ]

    await send_message(
        f"🛒 <b>קניתי!</b>  <b>{ticker}</b>  {qty_str} מניות\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💵 מחיר כניסה:  <b>{_price_str}</b>\n"
        f"💰 סה״כ השקעה:  <b>{_notional_str}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🎯 Take Profit:  {_tp_str}  <b>(+{tp_pct:.1f}%)</b>\n"
        f"🛑 Stop Loss:    {_stop_str}  <b>(-{stop_pct:.1f}%)</b>\n"
        f"⚖️  Risk/Reward:  {rr_str}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 ציון:  {_score_bar}  <b>{score:.0f}/100</b>  {q}"
        f"{iceberg_line}{id_line}"
        f"{tv_line}"
        f"{partial_plan}",
        reply_markup={"inline_keyboard": inline_buttons},
    )

    # Also send to Discord with embed formatting
    try:
        from discord_bot import send_discord_trade_open as _send_discord_open
        import asyncio as _asyncio
        _loop = _asyncio.get_running_loop()
        task = _loop.create_task(_send_discord_open(ticker, qty, price, notional, stop_price, tp_price, score))
        task.add_done_callback(
            lambda t: logger.debug(f"Discord trade open send failed: {t.exception()}")
            if not t.cancelled() and t.exception() else None
        )
    except Exception:
        pass  # Discord send is non-critical


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

    _header = "╔══════════════════╗\n║  💰  <b>מכרנו ברווח!</b>    ║\n╚══════════════════╝" if win \
              else "╔══════════════════╗\n║  📉  <b>מכרנו בהפסד</b>    ║\n╚══════════════════╝"

    _arrow = "📈" if win else "📉"
    _pct_color = "🟢" if win else "🔴"

    title    = "💰 מכרתי ברווח! 🎉" if win else "📉 מכרתי בהפסד"
    pnl_icon = "💚" if win else "❤️"
    pnl_bar  = _build_progress_bar(min(100, max(0, 50 + pct * 5)), fill="🟩" if win else "🟥")

    # Reason emoji mapping
    reason_emoji = {
        "take_profit": "🎯 Take Profit",
        "stop_loss_hit": "🛑 Stop Loss",
        "smart_sell": "🤖 Smart Sell",
        "emergency_exit": "🚨 Emergency Exit",
        "time_exit": "⏰ זמן מקסימלי",
        "stale_restart": "🔄 Restart",
    }.get(reason, f"📌 {reason}" if reason else "")

    # TradingView link on sell too
    tv_url_sell = f"https://www.tradingview.com/chart/?symbol={ticker}"

    # Win streak emoji
    streak_line = ""
    try:
        import database as _dbs
        _hist = _dbs.get_trade_history(limit=5) or []
        _closed = [x for x in _hist if x.get("pnl_gross") is not None]
        if len(_closed) >= 3:
            _streak = 0
            for _t in _closed:
                if (float(_t.get("pnl_gross", 0)) >= 0) == win:
                    _streak += 1
                else:
                    break
            if _streak >= 3:
                streak_line = f"\n🔥 <b>רצף {_streak} {'ניצחונות' if win else 'הפסדים'} ברצף!</b>"
    except Exception:
        pass

    sell_buttons = [
        [
            {"text": f"📊 גרף {ticker}", "url": tv_url_sell},
            {"text": "📍 פוזיציות פתוחות", "callback_data": "positions:all"},
        ],
        [
            {"text": "📈 ביצועים", "callback_data": "performance:full"},
            {"text": "🔍 סרוק הזדמנויות", "callback_data": "scan:now"},
        ],
    ]

    await send_message(
        f"{title}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🏷️  <b>{ticker}</b>  |  {qty:.0f} מניות\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📥 קנייה:  {_entry_str}\n"
        f"📤 מכירה:  <b>{_exit_str}</b>\n"
        f"{_pct_color} שינוי:   <b>{pct:+.2f}%</b>\n"
        f"{pnl_bar}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{pnl_icon} P&L:  <b>{_pnl_str}</b>  (נטו: {_net_str})\n"
        f"🧾 מס:    {_tax_str}\n"
        f"⏱️  זמן:   {dur_str}"
        + (f"\n{reason_emoji}" if reason_emoji else "")
        + (f"\n🔖 עסקה #{trade_id}" if trade_id else "")
        + streak_line,
        reply_markup={"inline_keyboard": sell_buttons},
    )

    # Also send to Discord with embed formatting
    try:
        from discord_bot import send_discord_trade_close as _send_discord_close
        import asyncio as _asyncio
        _loop = _asyncio.get_running_loop()
        task = _loop.create_task(_send_discord_close(ticker, qty, entry_price, exit_price, pnl_gross, pnl_net, duration_hours))
        task.add_done_callback(
            lambda t: logger.debug(f"Discord trade close send failed: {t.exception()}")
            if not t.cancelled() and t.exception() else None
        )
    except Exception:
        pass  # Discord send is non-critical


async def notify_emergency(ticker: str, reason: str) -> None:
    """Emergency exit alert."""
    await send_message(
        f"🚨 <b>יציאת חירום — {ticker}</b>\n"
        f"⚠️ {reason}"
    )

    # Also send to Discord
    try:
        from discord_bot import send_discord_emergency as _send_discord_emerg
        import asyncio as _asyncio
        _loop = _asyncio.get_running_loop()
        task = _loop.create_task(_send_discord_emerg(ticker, reason))
        task.add_done_callback(
            lambda t: logger.debug(f"Discord emergency send failed: {t.exception()}")
            if not t.cancelled() and t.exception() else None
        )
    except Exception:
        pass


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

    # Only show detail if it contains Hebrew — suppress raw English exception text
    detail_line = ""
    if detail:
        heb_chars = sum(1 for c in detail if 'א' <= c <= 'ת')
        if heb_chars > 5:
            # Has meaningful Hebrew content — show it
            detail_line = f"\n💬 {detail[:200]}"
        else:
            # English/technical exception — log it but don't spam Telegram with English
            logger.debug(f"[NOTIFY_ERROR] Suppressed English detail: {detail[:200]}")

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

    # Also send to Discord
    try:
        from discord_bot import send_discord_circuit_breaker as _send_discord_cb
        import asyncio as _asyncio
        loss_pct = abs(daily_pnl / loss_limit * 100) if loss_limit else 0
        _loop = _asyncio.get_running_loop()
        task = _loop.create_task(_send_discord_cb(daily_pnl, loss_limit, loss_pct))
        task.add_done_callback(
            lambda t: logger.debug(f"Discord circuit breaker send failed: {t.exception()}")
            if not t.cancelled() and t.exception() else None
        )
    except Exception:
        pass


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


async def notify_stop_approaching(
    ticker: str,
    current_price: float,
    stop_price: float,
    entry_price: float,
    stop_distance_pct: float,
) -> None:
    """
    🚨 Alert when position is within 20% of stop loss distance.
    Rate-limited: once per ticker per 30 minutes.
    """
    key = f"stop_approaching:{ticker}"
    if _is_rate_limited(key):
        return
    _mark_sent(key)

    plpc = (current_price - entry_price) / entry_price * 100
    pnl_emoji = "🟢" if plpc >= 0 else "🔴"
    tv_url = f"https://www.tradingview.com/chart/?symbol={ticker}"

    await send_message(
        f"⚠️ <b>סטופ לוס מתקרב — {ticker}!</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📍 מחיר עכשיו:  <b>${current_price:.2f}</b>\n"
        f"🛑 Stop Loss:    <b>${stop_price:.2f}</b>\n"
        f"📏 מרחק לסטופ:  <b>{stop_distance_pct:.1f}%</b> בלבד!\n"
        f"{pnl_emoji} P&L:          <b>{plpc:+.1f}%</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f'🔗 <a href="{tv_url}">פתח גרף ב-TradingView</a>',
        reply_markup={"inline_keyboard": [[
            {"text": f"🚨 EXIT {ticker}", "callback_data": f"emergency:{ticker}"},
            {"text": "📊 גרף", "url": tv_url},
        ]]}
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
    qty_str   = f"{total_qty:.4f}".rstrip('0').rstrip('.')
    slice_qty = f"{round(total_qty / n_slices, 4):.4f}".rstrip('0').rstrip('.') if n_slices else qty_str
    duration_min = (n_slices - 1) * interval_sec / 60
    await send_message(
        f"🛒 <b>קונה בכמה פעימות</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🏷️ מניה: <b>{ticker}</b>\n"
        f"📦 כמות כוללת: {qty_str} מניות\n"
        f"✂️ קונה ב-{n_slices} פעימות (כדי לקבל מחיר טוב יותר)\n"
        f"🔢 כל פעימה: ~{slice_qty} מניות\n"
        f"⏱️ מרווח בין פעימות: {interval_sec:.0f} שניות\n"
        f"🕐 סיום משוער: בעוד ~{duration_min:.0f} דקות"
    )


async def notify_iceberg_done(
    ticker:       str,
    filled_qty:   float,
    avg_price:    float,
    n_slices:     int,
    is_partial:   bool,
    slice_results: list = None,
) -> None:
    qty_str = f"{filled_qty:.4f}".rstrip('0').rstrip('.')
    status  = "⚠️ הצלחתי לקנות רק חלק" if is_partial else "✅ סיימתי לקנות"
    try:
        from telegram_chat import _fmt_price as _fp
        price_str = _fp(avg_price)
    except Exception:
        price_str = f"${avg_price:.2f}"

    lines = [
        f"🛒 <b>{status}</b>",
        f"━━━━━━━━━━━━━━━━",
        f"🏷️ מניה: <b>{ticker}</b>",
        f"📦 קניתי בסה״כ: {qty_str} מניות",
        f"✂️ ב-{n_slices} פעימות",
    ]

    # פירוט כל פעימה — מחיר קנייה, יעד רווח, ועצירת הפסד
    if slice_results:
        from config import settings as _cfg
        lines.append(f"━━━━━━━━━━━━━━━━")
        for s in slice_results:
            sq  = f"{s['qty']:.4f}".rstrip('0').rstrip('.')
            sp  = s['price']
            tp  = round(sp * (1 + _cfg.TAKE_PROFIT_PCT / 100), 2)
            sl  = round(sp * (1 - _cfg.STOP_LOSS_PCT  / 100), 2)
            try:
                from telegram_chat import _fmt_price as _fp2
                sp_str = _fp2(sp)
                tp_str = _fp2(tp)
                sl_str = _fp2(sl)
            except Exception:
                sp_str = f"${sp:.2f}"
                tp_str = f"${tp:.2f}"
                sl_str = f"${sl:.2f}"
            lines.append(f"פעימה {s['slice']}: {sq} מניות")
            lines.append(f"   💵 קניתי ב: {sp_str}")
            lines.append(f"   ✅ אמכור ברווח ב: {tp_str} (+{_cfg.TAKE_PROFIT_PCT:.0f}%)")
            lines.append(f"   ❌ אמכור בהפסד ב: {sl_str} (-{_cfg.STOP_LOSS_PCT:.0f}%)")

    lines.append(f"━━━━━━━━━━━━━━━━")
    lines.append(f"💵 מחיר קנייה ממוצע: {price_str}")
    await send_message("\n".join(lines))


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

    # Win rate visual bar
    win_bar   = _build_progress_bar(win_rate, fill="🟩", empty="🟥")
    pnl_color = "🟢" if total_pnl >= 0 else "🔴"

    # Weekly P&L trend sparkline (last 7 days from DB)
    sparkline = ""
    try:
        import database as _dbsp
        from datetime import datetime, timezone, timedelta
        _week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        _daily = _dbsp.get_connection().execute("""
            SELECT date(exit_time) as d, SUM(pnl_gross) as dpnl
            FROM trade_log WHERE status IN ('stopped','sold')
            AND exit_time >= ?
            GROUP BY d ORDER BY d
        """, (_week_ago,)).fetchall()
        if len(_daily) >= 3:
            sparkline = " " + _build_pnl_chart([float(r[1]) for r in _daily], width=7)
    except Exception:
        pass

    from datetime import datetime, timezone, timedelta
    _now_il = datetime.now(timezone.utc) + timedelta(hours=3)
    date_str = _now_il.strftime("%d/%m/%Y")

    lines = [
        f"🌙 <b>סיכום יום {date_str}</b>\n━━━━━━━━━━━━━━━━",
        f"🛒  קניות:    <b>{buys_today}</b>  |  💸 מכירות: <b>{total_trades}</b>  (✅{wins} / ❌{losses})",
    ]
    if total_trades > 0:
        lines.append(f"🎯  Win Rate:  {win_bar}  <b>{win_rate:.1f}%</b>")
    lines.append(f"{pnl_color}  P&L יומי:  <b>{pnl_str}</b>{sparkline}")
    if realized_pnl_net and net_str:
        lines.append(f"💳  נטו אחרי מס:  {net_str}")
    if tax_reserved > 0:
        lines.append(f"🧾  מס שהופרש:    ${tax_reserved:.2f}")

    # Per-trade breakdown (today's closed trades from DB)
    try:
        import database as _db
        _today = _now_il.strftime("%Y-%m-%d")  # Israel date
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # UTC fallback
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
                    pos = await asyncio.to_thread(_br.get_position, ot["ticker"])
                    pct = float(pos.get("unrealized_plpc", 0)) * 100 if pos else 0
                    icon = "🟢" if pct >= 0 else "🔴"
                    lines.append(f"  {icon} <b>{ot['ticker']}</b>  {pct:+.1f}%")
                except Exception:
                    pass
    except Exception:
        pass

    # Quick-action inline keyboard on daily summary
    summary_buttons = [
        [
            {"text": "📍 פוזיציות פתוחות", "callback_data": "positions:all"},
            {"text": "📊 ניתוח מלא",       "callback_data": "performance:full"},
        ],
        [
            {"text": "🩺 מצב הבוט",        "callback_data": "health:check"},
            {"text": "🚨 אנומליות",         "callback_data": "anomalies:scan"},
        ],
    ]

    await send_message(
        "\n".join(lines),
        reply_markup={"inline_keyboard": summary_buttons},
    )


async def notify_weekly_report(report_html: str) -> None:
    """Send pre-formatted weekly performance report HTML."""
    await send_message(report_html)


async def notify_morning_briefing() -> None:
    """
    📅 Morning briefing — sent at market open (9:30 EST / 16:30 Israel).
    Shows: open positions P&L, today's plan, market context, top opportunities.
    """
    try:
        from datetime import datetime, timezone, timedelta
        _now_il = datetime.now(timezone.utc) + timedelta(hours=3)
        date_str = _now_il.strftime("%A %d/%m/%Y")
        day_heb = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"][_now_il.weekday()]

        lines = [
            f"🌅 <b>בוקר טוב! יום {day_heb}, {date_str}</b>",
            "━━━━━━━━━━━━━━━━",
            "",
        ]

        # 1. Current positions status
        try:
            import broker as _br, database as _db
            positions = await asyncio.to_thread(_br.get_positions)
            open_trades = await asyncio.to_thread(_db.get_open_trades)
            trade_map = {t["ticker"]: t for t in (open_trades or [])}

            if positions:
                total_pnl = sum(float(p.unrealized_pl) for p in positions)
                winners = sum(1 for p in positions if float(p.unrealized_pl) >= 0)
                pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
                lines.append(f"📍 <b>פוזיציות פתוחות: {len(positions)}</b>")
                lines.append(f"{pnl_emoji} רווח/הפסד: <b>${total_pnl:+,.2f}</b> | 🏆 {winners}/{len(positions)} ברווח")
                lines.append("")

                # Top 3 positions
                sorted_pos = sorted(positions, key=lambda x: float(x.unrealized_plpc), reverse=True)
                for pos in sorted_pos[:3]:
                    plpc = float(pos.unrealized_plpc) * 100
                    pl = float(pos.unrealized_pl)
                    em = "🟢" if pl >= 0 else "🔴"
                    tv_link = f'<a href="https://www.tradingview.com/chart/?symbol={pos.symbol}">{pos.symbol}</a>'
                    lines.append(f"  {em} {tv_link}  {plpc:+.1f}%  ${pl:+.2f}")
            else:
                lines.append("📭 אין פוזיציות פתוחות — הבוט מחפש הזדמנויות")
            lines.append("")
        except Exception:
            pass

        # 2. Budget status
        try:
            import budget as _bud
            b = await asyncio.to_thread(_bud.get_budget_status)
            cash = b.get("cash_available", 0)
            used_pct = b.get("budget_used_pct", 0)
            lines.append(f"💰 <b>מזומן פנוי: ${cash:,.0f}</b> ({100-used_pct:.0f}% פנוי)")
            lines.append("")
        except Exception:
            pass

        # 3. Market context (SPY/QQQ)
        try:
            import yfinance as _yf
            market_lines = []
            for sym, name in [("SPY", "S&P500"), ("QQQ", "Nasdaq")]:
                try:
                    t = _yf.Ticker(sym)
                    info = t.fast_info
                    chg = float(getattr(info, "three_month_change", 0) or 0) * 100
                    price = float(getattr(info, "last_price", 0) or 0)
                    em = "🟢" if chg >= 0 else "🔴"
                    market_lines.append(f"  {em} {name}: ${price:.0f}")
                except Exception:
                    pass
            if market_lines:
                lines.append("🌍 <b>מצב השוק:</b>")
                lines.extend(market_lines)
                lines.append("")
        except Exception:
            pass

        # 4. Today's action items
        lines.extend([
            "📋 <b>לבדוק היום:</b>",
            "  • /מניות — פוזיציות עם לינקים ל-TradingView",
            "  • /positions — מצב מפורט",
            "  • /risk — ניתוח סיכון",
            "",
            "🤖 <i>הבוט עובד אוטומטית — שפוי יום!</i>",
        ])

        morning_buttons = [[
            {"text": "📈 מניות שלי", "callback_data": "positions:all"},
            {"text": "💰 P&L", "callback_data": "pnl:quick"},
        ]]

        await send_message("\n".join(lines), reply_markup={"inline_keyboard": morning_buttons})

    except Exception as e:
        logger.error(f"Morning briefing failed: {e}")


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
# 🆕 Enhanced notifications (trending, goals, sentiment, market)
# ─────────────────────────────────────────────────────────────────────────────

async def notify_trending_tickers(tickers: list[str]) -> None:
    """📈 Alert about most-mentioned tickers in community today."""
    if not _enabled() or not tickers:
        return

    ticker_list = " • ".join(tickers[:5])
    await send_message(
        f"🔥 <b>מניות חמות היום</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"הקהילייה דיברה הכי הרבה על:\n"
        f"{ticker_list}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💡 <i>כדאי לשים עין על המניות האלו</i>"
    )


async def notify_daily_goal_progress(
    current_pnl: float,
    daily_target: float,
    trades_count: int,
) -> None:
    """🎯 Update on daily profit goal progress."""
    if not _enabled():
        return

    if daily_target <= 0:
        return

    progress_pct = (current_pnl / daily_target * 100) if daily_target > 0 else 0
    remaining = daily_target - current_pnl

    # Visual progress bar
    filled = int(progress_pct / 10)
    bar = "🟩" * filled + "⬜" * (10 - filled)

    emoji = "🎉" if current_pnl >= daily_target else "💪" if current_pnl > 0 else "🔄"
    status = "הגעת ליעד! 🏆" if current_pnl >= daily_target else f"עוד ${abs(remaining):.2f} עד היעד"

    await send_message(
        f"{emoji} <b>התקדמות יעד יומי</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"לאן נתאחדנו: ${current_pnl:+.2f}\n"
        f"היעד: ${daily_target:.2f}\n"
        f"{bar}  <b>{progress_pct:.0f}%</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 עסקאות סגורות היום: {trades_count}\n"
        f"💬 {status}"
    )


async def notify_sentiment_alert(
    ticker: str,
    sentiment_score: float,
    direction: str = "bullish",
) -> None:
    """📣 Alert when community sentiment shifts significantly."""
    if not _enabled():
        return

    emoji = "🟢" if sentiment_score >= 6 else "🔴" if sentiment_score <= 4 else "🟡"
    sentiment_text = "שורי" if sentiment_score >= 6 else "דובי" if sentiment_score <= 4 else "נייטרלי"

    await send_message(
        f"{emoji} <b>שינוי סנטימנט! {ticker}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"הקהילייה חושבת שהמניה: <b>{sentiment_text}</b>\n"
        f"ציון סנטימנט: <b>{sentiment_score:.1f}/10</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💡 שים עין על הדיונים בערוץ"
    )


async def notify_market_summary(
    market_status: str = "open",
    top_gainers: list[tuple[str, float]] = None,
    top_losers: list[tuple[str, float]] = None,
) -> None:
    """🌍 Market opening summary with gainers/losers."""
    if not _enabled():
        return

    lines = [
        f"🌍 <b>מצב השוק</b>",
        f"━━━━━━━━━━━━━━━━",
    ]

    if market_status == "open":
        lines.append("✅ השוק <b>פתוח</b> — זמן לסחור! 🚀")
    elif market_status == "closed":
        lines.append("🛑 השוק <b>סגור</b> — חזור מחר")
    else:
        lines.append(f"⏰ השוק: {market_status}")

    if top_gainers:
        lines.append("\n📈 <b>אצלים (Gainers):</b>")
        for ticker, pct in top_gainers[:3]:
            lines.append(f"  🟢 <b>{ticker}</b>: +{pct:.1f}%")

    if top_losers:
        lines.append("\n📉 <b>יורדים (Losers):</b>")
        for ticker, pct in top_losers[:3]:
            lines.append(f"  🔴 <b>{ticker}</b>: {pct:.1f}%")

    lines.append(f"\n━━━━━━━━━━━━━━━━")
    lines.append(f"💼 בואו נעשה כסף היום! 💪")

    await send_message("\n".join(lines))


async def notify_risk_metrics(
    sharpe_ratio: float | None,
    max_drawdown: float | None,
    win_rate: float | None,
) -> None:
    """📊 Daily risk and performance metrics. Tolerates None values gracefully."""
    if not _enabled():
        return

    # Defensive coding: any of the metrics can be None when there's
    # insufficient trade history yet. Format as "—" instead of crashing.
    def _fmt(value, suffix: str = "", sign: str = "", decimals: int = 2) -> str:
        if value is None:
            return "—"
        try:
            return f"{sign}{float(value):.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return "—"

    await send_message(
        f"📊 <b>מדדי סיכון וביצועים</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🎯 Sharpe Ratio: <b>{_fmt(sharpe_ratio, decimals=2)}</b>\n"
        f"📉 Max Drawdown: <b>{_fmt(max_drawdown, suffix='%', sign='-', decimals=1)}</b>\n"
        f"✅ Win Rate: <b>{_fmt(win_rate, suffix='%', decimals=1)}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>מדדים טובים = סחירות בטוחות יותר</i>"
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


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED ANALYTICS & INTERACTIVE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

async def notify_detailed_trade_analytics() -> None:
    """
    Send detailed trade analytics with:
    - P&L breakdown by ticker
    - Win rate by hour of day
    - Best and worst trades
    - Average holding time
    """
    if not _enabled():
        return

    try:
        import database
        conn = database.get_connection()

        # Get all closed trades from today
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades = conn.execute("""
            SELECT ticker, pnl_gross, entry_price, exit_price,
                   COALESCE(created_at, entry_time) as created_at, exit_time,
                   NULL as quality_score, sentiment_score as entry_sentiment_score
            FROM trade_log
            WHERE status != 'open'
            AND exit_time LIKE ?
            ORDER BY exit_time DESC
        """, (f"{today}%",)).fetchall()

        if not trades:
            await send_message("📊 עדיין אין עסקאות היום")
            return

        # Analyze by ticker
        by_ticker = {}
        for ticker, pnl, entry, exit_p, created, exit_t, quality, sentiment in trades:
            if ticker not in by_ticker:
                by_ticker[ticker] = {"count": 0, "pnl": 0, "wins": 0}
            by_ticker[ticker]["count"] += 1
            by_ticker[ticker]["pnl"] += pnl
            if pnl > 0:
                by_ticker[ticker]["wins"] += 1

        # Sort by P&L
        sorted_tickers = sorted(by_ticker.items(), key=lambda x: x[1]["pnl"], reverse=True)

        # Build message
        lines = [
            "📊 <b>ניתוח עסקאות מפורט</b>",
            "━━━━━━━━━━━━━━━━━━━",
            f"📅 {today}",
            f"📈 סה״כ עסקאות: {len(trades)}",
            "",
            "<b>💰 P&L לפי מניה:</b>",
        ]

        for ticker, stats in sorted_tickers[:10]:
            pnl = stats["pnl"]
            color = "🟢" if pnl >= 0 else "🔴"
            win_rate = (stats["wins"] / stats["count"] * 100) if stats["count"] > 0 else 0
            lines.append(
                f"{color} {ticker}: ${pnl:+.2f} | {stats['count']} עסקאות | {win_rate:.0f}% win"
            )

        # Find best and worst
        best_trade = max(trades, key=lambda x: x[1], default=None)
        worst_trade = min(trades, key=lambda x: x[1], default=None)

        if best_trade:
            lines.extend([
                "",
                f"🏆 <b>עסקה הטובה ביותר:</b> {best_trade[0]} ${best_trade[1]:+.2f}",
            ])

        if worst_trade:
            lines.append(
                f"📉 <b>עסקה הגרועה ביותר:</b> {worst_trade[0]} ${worst_trade[1]:+.2f}"
            )

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"Detailed trade analytics error: {e}")


async def notify_correlation_analysis() -> None:
    """
    Send analysis of current positions and their correlations.
    Warns if positions are too correlated.
    """
    if not _enabled():
        return

    try:
        from adaptive_trader import get_position_correlation_risk

        import broker
        positions = await asyncio.to_thread(broker.get_positions)

        if not positions:
            await send_message("✅ לא יש פוזיציות פתוחות")
            return

        tickers = [p.symbol for p in positions]
        correlation_risk = await get_position_correlation_risk(tickers)

        lines = [
            "📍 <b>ניתוח התאם בין פוזיציות</b>",
            "━━━━━━━━━━━━━━━━━",
        ]

        if correlation_risk["correlated"]:
            lines.append("⚠️ <b>הוזהרו קורלציות גבוהות:</b>")
            for pair in correlation_risk["pairs"]:
                lines.append(
                    f"  {pair['ticker1']} ↔️ {pair['ticker2']}: {pair['correlation']:.2f} ({pair['risk']})"
                )
        else:
            lines.append("✅ אין קורלציות גבוהות בין הפוזיציות")

        lines.append("")
        lines.append(f"📌 סה״כ פוזיציות: {len(tickers)}")

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")


async def notify_market_regime_analysis() -> None:
    """
    Send market regime analysis:
    - Volatility status
    - Sector leaders/laggards
    - Market breadth
    """
    if not _enabled():
        return

    try:
        from market_intelligence import detect_volatility_regime, analyze_sector_rotation, get_market_breadth

        # All three are async — must await directly, never wrap in to_thread
        vol_regime = await detect_volatility_regime()
        sectors    = await analyze_sector_rotation()
        breadth    = await get_market_breadth()

        lines = [
            "🌍 <b>ניתוח משטר שוק</b>",
            "━━━━━━━━━━━━━━",
            "",
            f"<b>📊 משטר תנודתיות:</b> {vol_regime.regime}",
            f"  Volatility 5d: {vol_regime.volatility_5d:.1f}%",
            f"  Volatility 20d: {vol_regime.volatility_20d:.1f}%",
            f"  {vol_regime.recommendation}",
            "",
            f"<b>📈 רוחב שוק:</b> {breadth.strength_indicator}",
            f"  Advances: {breadth.advances} | Declines: {breadth.declines}",
            f"  Breadth: {breadth.market_breadth_percent:.1f}%",
            "",
            "<b>🏆 סקטורים המובילים:</b>",
        ]

        for sector in sectors[:3]:
            lines.append(
                f"  {sector.rank}. {sector.sector}: +{sector.performance_pct:.2f}% {sector.recommendation}"
            )

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"Market regime analysis error: {e}")


async def notify_adaptive_parameters() -> None:
    """
    Send current adaptive trading parameters.
    Shows how the bot is adjusting for current market conditions.
    """
    if not _enabled():
        return

    try:
        from adaptive_trader import get_adaptive_trading_params
        from config import settings

        params = await get_adaptive_trading_params(
            base_quantity=1,
            base_min_buy_score=settings.MIN_BUY_SCORE,
            base_stop_loss_pct=settings.STOP_LOSS_PCT,
            base_take_profit_pct=settings.TAKE_PROFIT_PCT,
        )

        lines = [
            "🤖 <b>פרמטרים אדפטיביים נוכחיים</b>",
            "━━━━━━━━━━━━━━━━",
            "",
            "<b>📊 גודל פוזיציה:</b>",
            f"  Risk factor: {params['position_sizing']['risk_factor']:.2f}x",
            f"  Confidence: {params['position_sizing']['confidence_level']:.0%}",
            f"  Time factor: {params['position_sizing']['time_of_day_factor']:.2f}x",
            f"  Reason: {params['position_sizing']['reason']}",
            "",
            "<b>🎯 סף קנייה & RSI:</b>",
            f"  MIN_BUY_SCORE: {params['thresholds']['min_buy_score']:.1f}",
            f"  RSI Overbought: {params['thresholds']['rsi_overbought']:.0f}",
            f"  RSI Oversold: {params['thresholds']['rsi_oversold']:.0f}",
            "",
            "<b>🛑 Stop Loss & Take Profit:</b>",
            f"  Stop Loss: {params['stop_loss_tp']['stop_loss_pct']:.2f}%",
            f"  Take Profit: {params['stop_loss_tp']['take_profit_pct']:.2f}%",
            "",
            f"<b>🌡️ תנאי שוק:</b> {params['market_conditions']['volatility_level']}",
            f"  VIX-like: {params['market_conditions']['volatility']:.1f}",
        ]

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"Adaptive parameters error: {e}")


async def notify_performance_comparison() -> None:
    """
    Compare today's performance vs this week vs this month.
    """
    if not _enabled():
        return

    try:
        import database
        from datetime import datetime, timedelta
        conn = database.get_connection()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        month_ago = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

        # Today
        today_stats = conn.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END), SUM(pnl_gross)
            FROM trade_log
            WHERE status IN ('stopped', 'sold') AND exit_time LIKE ?
        """, (f"{today}%",)).fetchone()

        # This week
        week_stats = conn.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END), SUM(pnl_gross)
            FROM trade_log
            WHERE status IN ('stopped', 'sold') AND exit_time >= ?
        """, (week_ago,)).fetchone()

        # This month
        month_stats = conn.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END), SUM(pnl_gross)
            FROM trade_log
            WHERE status IN ('stopped', 'sold') AND exit_time >= ?
        """, (month_ago,)).fetchone()

        def format_period(label, count, wins, pnl):
            if not count:
                return f"{label}: 0 עסקאות"
            win_rate = (wins / count * 100) if count > 0 else 0
            return f"{label}: {count} עסקאות | {win_rate:.0f}% win | ${pnl or 0:+.2f}"

        lines = [
            "📈 <b>השוואת ביצועים</b>",
            "━━━━━━━━━━━━",
            format_period("📅 היום", today_stats[0] or 0, today_stats[1] or 0, today_stats[2]),
            format_period("📊 השבוע", week_stats[0] or 0, week_stats[1] or 0, week_stats[2]),
            format_period("📆 החודש", month_stats[0] or 0, month_stats[1] or 0, month_stats[2]),
        ]

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"Performance comparison error: {e}")


async def notify_ai_trading_insights() -> None:
    """
    Send AI-generated insights about trading patterns and recommendations.
    Uses continuous learner analysis.
    """
    if not _enabled():
        return

    try:
        from continuous_learner import learn_error_patterns, learn_sentiment_correlation

        errors = await asyncio.to_thread(learn_error_patterns)
        sentiments = await asyncio.to_thread(learn_sentiment_correlation)

        lines = [
            "💡 <b>AI Trading Insights</b>",
            "━━━━━━━━━━━━━━━",
            "",
        ]

        if errors:
            lines.append("<b>🔴 דפוסי טעויות מתחזרים:</b>")
            for error in errors[:3]:
                lines.append(
                    f"  • {error.error_type}: {error.frequency}x | "
                    f"הפסד ממוצע ${error.avg_loss:.2f}"
                )
                lines.append(f"    💡 הצעה: {error.suggested_fix}")

        if sentiments:
            lines.append("")
            lines.append("<b>💬 מניות עם התאם סנטימנט טוב:</b>")
            best_sentiment = sorted(sentiments.items(), key=lambda x: x[1].correlation_strength, reverse=True)[:3]
            for ticker, corr in best_sentiment:
                lines.append(
                    f"  • {ticker}: correlation={corr.correlation_strength:.2f} | {corr.recommendation}"
                )

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"AI insights error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE POSITIONS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

async def notify_live_positions() -> None:
    """
    Send a rich live snapshot of all open positions:
    Current price, P&L, stop-loss distance, AI score.
    """
    if not _enabled():
        return
    try:
        import broker, database
        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            await send_message("📊 <b>אין פוזיציות פתוחות כרגע</b>")
            return

        total_value = sum(float(p.market_value) for p in positions)
        total_pnl   = sum(float(p.unrealized_pl) for p in positions)

        lines = [
            "📍 <b>פוזיציות פתוחות</b>",
            "━━━━━━━━━━━━━━━━",
        ]

        for p in positions:
            try:
                mv   = float(p.market_value)
                pl   = float(p.unrealized_pl)
                plpc = float(p.unrealized_plpc) * 100
                cur  = float(p.current_price)
                avg  = float(p.avg_entry_price)
                qty  = float(p.qty)

                pl_emoji = "🟢" if pl >= 0 else "🔴"
                lines.append(
                    f"{pl_emoji} <b>{p.symbol}</b>  "
                    f"{qty:.0f}×${cur:.2f}  "
                    f"P&L: <b>${pl:+.2f}</b> ({plpc:+.1f}%)"
                )

                # Show stop from DB if available
                try:
                    open_trades = await asyncio.to_thread(database.get_open_trades)
                    trade_rec = next((t for t in open_trades if t["ticker"] == p.symbol), None)
                    if trade_rec and trade_rec.get("atr_stop_price"):
                        stop = trade_rec["atr_stop_price"]
                        dist = (cur - stop) / cur * 100
                        lines.append(f"  🛑 Stop ${stop:.2f} ({dist:.1f}% away)")
                except Exception:
                    pass

            except Exception as e:
                lines.append(f"  ⚠️ {p.symbol}: error ({e})")

        lines.extend([
            "━━━━━━━━━━━━━━━━",
            f"💼 Total Value: ${total_value:,.2f}",
            f"{'🟢' if total_pnl >= 0 else '🔴'} Total P&L: <b>${total_pnl:+,.2f}</b>",
        ])
        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"notify_live_positions error: {e}")


async def notify_score_enhancement(
    ticker: str,
    original_score: float,
    enhanced_score: float,
    adjustment: float,
    skip_trade: bool,
    skip_reason: str,
    signals: dict,
) -> None:
    """
    Notify when the AI score enhancer significantly changes a buy decision.
    Only fires when |adjustment| > 5 points or skip_trade is True.
    """
    if not _enabled():
        return
    if abs(adjustment) < 5 and not skip_trade:
        return

    if skip_trade:
        msg = (
            f"🚫 <b>AI blocked buy: {ticker}</b>\n"
            f"Base score {original_score:.1f} → SKIP\n"
            f"Reason: {skip_reason}"
        )
    elif adjustment > 0:
        msg = (
            f"⬆️ <b>AI boosted: {ticker}</b>\n"
            f"{original_score:.1f} → {enhanced_score:.1f} (+{adjustment:.1f})\n"
        )
    else:
        msg = (
            f"⬇️ <b>AI reduced: {ticker}</b>\n"
            f"{original_score:.1f} → {enhanced_score:.1f} ({adjustment:.1f})\n"
        )

    # Add signal breakdown
    for key, val in signals.items():
        if isinstance(val, dict) and "adjustment" in val and val["adjustment"] != 0:
            msg += f"  • {key}: {val['adjustment']:+.1f}pts\n"

    try:
        await send_message(msg)
    except Exception as e:
        logger.debug(f"Score enhancement notify failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RICH PORTFOLIO CARD  ← main daily overview
# ─────────────────────────────────────────────────────────────────────────────

async def send_portfolio_card() -> None:
    """
    Beautiful real-time portfolio card with:
    - Total value + daily P&L with sparkline
    - Per-position breakdown with mini-charts
    - Win rate bar + sector concentration
    - Inline buttons for quick actions
    """
    if not _enabled():
        return
    try:
        import broker, database
        from datetime import datetime, timezone, timedelta

        positions = await asyncio.to_thread(broker.get_positions)
        open_trades = await asyncio.to_thread(database.get_open_trades)

        if not positions:
            await send_message(
                "📊 <b>תיק ריק</b>\n"
                "━━━━━━━━━━━━━━━━\n"
                "🤖 הבוט מחפש הזדמנויות...\n"
                "📈 MIN_BUY_SCORE=65 | SMA50=חובה"
            )
            return

        total_val  = sum(float(p.market_value) for p in positions)
        total_pnl  = sum(float(p.unrealized_pl) for p in positions)
        total_cost = sum(float(p.cost_basis) for p in positions)
        total_pct  = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        pnl_emoji  = "🟢" if total_pnl >= 0 else "🔴"
        pnl_bar    = _build_progress_bar(min(100, max(0, 50 + total_pct * 3)))

        # Weekly P&L sparkline
        try:
            conn = database.get_connection()
            week_pnls = conn.execute("""
                SELECT COALESCE(SUM(pnl_gross),0) FROM trade_log
                WHERE status IN ('stopped','sold')
                AND exit_time >= datetime('now','-7 days')
                GROUP BY date(exit_time) ORDER BY date(exit_time)
            """).fetchall()
            spark = _build_pnl_chart([r[0] for r in week_pnls], width=7) if week_pnls else ""
        except Exception:
            spark = ""

        lines = [
            f"📊 <b>תיק מניות — עדכון חי</b>",
            f"━━━━━━━━━━━━━━━━",
            f"💼 שווי: <b>${total_val:,.2f}</b>",
            f"{pnl_emoji} P&L: <b>${total_pnl:+,.2f}</b> ({total_pct:+.2f}%)",
            f"{pnl_bar}{(' ' + spark) if spark else ''}",
            f"━━━━━━━━━━━━━━━━",
        ]

        # Per position with stop distance
        trade_map = {t["ticker"]: t for t in (open_trades or [])}
        for p in sorted(positions, key=lambda x: float(x.unrealized_plpc), reverse=True):
            pl   = float(p.unrealized_pl)
            plpc = float(p.unrealized_plpc) * 100
            cur  = float(p.current_price)
            icon = "🟢" if pl >= 0 else "🔴"
            trade = trade_map.get(p.symbol, {})
            stop  = trade.get("atr_stop_price")
            stop_str = f" | 🛑{((cur-stop)/cur*100):.1f}%↓" if stop else ""
            lines.append(
                f"{icon} <b>{p.symbol}</b>  "
                f"${cur:.2f}  {plpc:+.1f}%"
                f"{stop_str}"
            )

        lines.append("━━━━━━━━━━━━━━━━")
        lines.append(f"📊 {len(positions)} פוזיציות פתוחות")

        # Quick action buttons
        buttons = [
            [
                {"text": "🔄 רענן", "callback_data": "positions:all"},
                {"text": "📈 הכי טובות", "callback_data": "top:positions"},
            ],
            [
                {"text": "🩺 מצב בוט", "callback_data": "health:check"},
                {"text": "⚠️ סיכון", "callback_data": "risk:portfolio"},
            ],
        ]

        await send_message(
            "\n".join(lines),
            reply_markup={"inline_keyboard": buttons},
        )

    except Exception as e:
        logger.error(f"Portfolio card failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ALERT MANAGER  ← price alerts set via Telegram
# ─────────────────────────────────────────────────────────────────────────────

# In-memory price alerts: {ticker: [{"price": float, "direction": "above"|"below"}]}
_price_alerts: dict[str, list[dict]] = {}


def add_price_alert(ticker: str, target_price: float, direction: str = "above") -> str:
    """Add a price alert. direction = 'above' or 'below'."""
    ticker = ticker.upper()
    if ticker not in _price_alerts:
        _price_alerts[ticker] = []
    _price_alerts[ticker].append({
        "price": target_price,
        "direction": direction,
        "created_at": datetime.now(timezone.utc).isoformat() if True else "",
    })
    return f"✅ התראה נוספה: {ticker} {'מעל' if direction == 'above' else 'מתחת'} ${target_price:.2f}"


def remove_price_alert(ticker: str) -> str:
    """Remove all alerts for a ticker."""
    ticker = ticker.upper()
    if ticker in _price_alerts:
        count = len(_price_alerts.pop(ticker))
        return f"✅ הוסרו {count} התראות עבור {ticker}"
    return f"לא נמצאו התראות עבור {ticker}"


def list_price_alerts() -> str:
    """List all active price alerts."""
    if not _price_alerts:
        return "📋 אין התראות מחיר פעילות"
    lines = ["📋 <b>התראות מחיר פעילות:</b>"]
    for ticker, alerts in _price_alerts.items():
        for a in alerts:
            dir_str = "מעל" if a["direction"] == "above" else "מתחת"
            lines.append(f"  🔔 {ticker} {dir_str} ${a['price']:.2f}")
    return "\n".join(lines)


async def check_price_alerts(current_prices: dict[str, float]) -> None:
    """Check if any price alerts have been triggered."""
    fired = []
    for ticker, alerts in list(_price_alerts.items()):
        current = current_prices.get(ticker)
        if not current:
            continue
        remaining = []
        for alert in alerts:
            triggered = (
                (alert["direction"] == "above" and current >= alert["price"]) or
                (alert["direction"] == "below" and current <= alert["price"])
            )
            if triggered:
                fired.append((ticker, current, alert["price"], alert["direction"]))
            else:
                remaining.append(alert)
        if remaining:
            _price_alerts[ticker] = remaining
        else:
            _price_alerts.pop(ticker, None)

    for ticker, current, target, direction in fired:
        dir_str = "עלה מעל" if direction == "above" else "ירד מתחת"
        await send_message(
            f"🔔 <b>התראת מחיר! — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💵 מחיר עכשיו: <b>${current:.2f}</b>\n"
            f"🎯 {dir_str}: ${target:.2f}\n"
            f"⚡ בוצע!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SMART STATUS MESSAGE  ← /start shows this
# ─────────────────────────────────────────────────────────────────────────────

async def send_smart_welcome() -> None:
    """Rich welcome/status message shown on /start."""
    if not _enabled():
        return
    try:
        import broker, database
        from datetime import datetime, timezone

        # Quick stats
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            conn = database.get_connection()
            today_row = conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN pnl_gross>0 THEN 1 ELSE 0 END),
                       COALESCE(SUM(pnl_gross),0)
                FROM trade_log WHERE status IN ('stopped','sold') AND exit_time LIKE ?
            """, (f"{today}%",)).fetchone()
            trades_today = today_row[0] or 0
            wins_today   = today_row[1] or 0
            pnl_today    = today_row[2] or 0
        except Exception:
            trades_today = wins_today = 0
            pnl_today = 0

        market_open = False
        try:
            market_open = await asyncio.to_thread(broker.is_market_open)
        except Exception:
            pass

        mkt_str = "🟢 שוק פתוח" if market_open else "🔴 שוק סגור"
        pnl_str = f"${pnl_today:+.2f}" if pnl_today != 0 else "—"
        wr_str  = f"{wins_today/trades_today*100:.0f}%" if trades_today > 0 else "—"

        await send_message(
            f"🤖 <b>מנהל ההשקעות שלך</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{mkt_str}\n"
            f"📅 היום: {trades_today} עסקאות | WR:{wr_str} | {pnl_str}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💬 שאל אותי כל שאלה, או:\n"
            f"📍 /pos — פוזיציות פתוחות\n"
            f"📊 /performance — ביצועים\n"
            f"🩺 /doctor — בדיקה מקיפה\n"
            f"🔔 /alert AAPL 200 — התראת מחיר"
        )
    except Exception as e:
        logger.debug(f"Smart welcome failed: {e}")
        await send_menu()


from datetime import datetime, timezone  # ensure imported at module level
