"""
Telegram Two-Way Chat — Receive messages and reply intelligently.

The user sends a question to the bot in Telegram → the bot uses Groq LLM
+ live trading context to generate a Hebrew reply.

Flow:
  1. Telegram → POST /telegram/webhook (set up via Bot API setWebhook)
  2. webhook.py routes to handle_telegram_update()
  3. We extract the message, build context (cash, positions, recent trades)
  4. Pass to Groq LLM with system prompt in Hebrew
  5. Send reply back via telegram_bot.send_message()

Security:
  - Only respond to messages from the configured TELEGRAM_CHAT_ID
  - Ignore all other chats (prevents abuse if bot username leaks)
"""

import asyncio
import json
import logging
from openai import OpenAI

from config import settings
from telegram_bot import send_message
import broker
import budget
import database

logger = logging.getLogger(__name__)

# Lazy-init OpenAI/Groq client
_client = None


_usd_ils_cache: tuple[float, float] = (3.7, 0.0)  # (rate, timestamp)


def _get_usd_ils() -> float:
    """Get USD/ILS exchange rate. Cached 1 hour."""
    import time
    global _usd_ils_cache
    rate, ts = _usd_ils_cache
    if time.time() - ts < 3600 and rate > 0:
        return rate
    try:
        import yfinance as _yf
        t = _yf.Ticker("USDILS=X")
        info = t.fast_info
        r = float(getattr(info, "last_price", 0) or 0)
        if r > 0:
            _usd_ils_cache = (r, time.time())
            return r
    except Exception:
        pass
    return _usd_ils_cache[0] or 3.7  # fallback


def _fmt_price(usd: float) -> str:
    """Format price: $318.75 (₪1,178)"""
    try:
        ils = usd * _get_usd_ils()
        return f"${usd:,.2f} (₪{ils:,.0f})"
    except Exception:
        return f"${usd:,.2f}"


def _fmt_pnl(amount: float, show_label: bool = True) -> str:
    """Format P&L correctly in Hebrew RTL context"""
    label = "רווח 🟢" if amount >= 0 else "הפסד 🔴"
    # Use absolute value + label to avoid RTL sign rendering issues
    formatted = f"<b>${abs(amount):,.2f}</b>"
    if show_label:
        return f"{label}  {formatted}"
    return formatted


def _fmt_held(hours: float) -> str:
    """Format holding time: minutes / hours / days."""
    if hours < 1:
        mins = int(hours * 60)
        return f"{mins} דקות" if mins > 0 else "כמה שניות"
    if hours < 24:
        return f"{hours:.1f} שעות"
    return f"{hours/24:.1f} ימים"

# Context cache — avoid hammering broker API on every Telegram message
_context_cache: tuple[float, dict] = (0.0, {})
_CONTEXT_CACHE_TTL = 30   # seconds


def _get_client() -> OpenAI | None:
    global _client
    if _client is None and settings.GROQ_API_KEY:
        _client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


def _build_context() -> dict:
    """Gather FULL live trading context for the LLM — cached 30s to avoid broker spam."""
    import time as _t
    global _context_cache
    if _t.time() - _context_cache[0] < _CONTEXT_CACHE_TTL and _context_cache[1]:
        return _context_cache[1]

    from datetime import datetime, timezone as _tz
    try:
        status = budget.get_budget_status()
        open_trades = database.get_open_trades()
        history = database.get_trade_history(limit=10)
    except Exception as exc:
        logger.warning(f"[CHAT] Failed to build context: {exc}")
        status, open_trades, history = {}, [], []

    # ── Positions: merge DB trade log + live broker positions ─────────
    positions_summary = []

    if open_trades:
        for t in open_trades:
            ticker = t.get("ticker")
            try:
                pos = broker.get_position(ticker)
                cur = float(pos.get("current_price", t["entry_price"])) if pos else t["entry_price"]
                entry = t["entry_price"]
                qty = t["qty"]
                pct = (cur - entry) / entry * 100 if entry else 0
                pnl = (cur - entry) * qty
                val = cur * qty
                # Calculate how long position has been held
                held_hours = 0.0
                entry_time_str = t.get("entry_time")
                if entry_time_str:
                    try:
                        entry_dt = datetime.strptime(str(entry_time_str)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz.utc)
                        held_hours = round((datetime.now(_tz.utc) - entry_dt).total_seconds() / 3600, 1)
                    except Exception:
                        pass
                positions_summary.append({
                    "ticker":      ticker,
                    "qty":         round(qty, 4),
                    "entry":       round(entry, 2),
                    "current":     round(cur, 2),
                    "pct":         round(pct, 2),
                    "pnl":         round(pnl, 2),
                    "value":       round(val, 2),
                    "invested":    round(entry * qty, 2),
                    "atr_stop":    round(t.get("atr_stop_price") or 0, 2),
                    "held_hours":  held_hours,
                    "entry_time":  str(entry_time_str)[:16] if entry_time_str else "",
                    "sentiment":   t.get("sentiment_score"),
                })
            except Exception:
                pass
    else:
        # Fallback: read directly from broker (after DB wipe)
        try:
            broker_positions = broker.get_positions()
            for pos in broker_positions:
                ticker = pos.get("ticker", "")
                qty    = float(pos.get("qty", 0))
                cur    = float(pos.get("current_price", 0))
                entry  = float(pos.get("avg_entry_price", cur))
                pct    = float(pos.get("unrealized_plpc", 0)) * 100
                pnl    = float(pos.get("unrealized_pl", 0))
                val    = float(pos.get("market_value", cur * qty))
                if qty > 0:
                    positions_summary.append({
                        "ticker":  ticker,
                        "qty":     round(qty, 4),
                        "entry":   round(entry, 2),
                        "current": round(cur, 2),
                        "pct":     round(pct, 2),
                        "pnl":     round(pnl, 2),
                        "value":   round(val, 2),
                        "atr_stop": 0,
                    })
        except Exception as e:
            logger.warning(f"[CHAT] broker.get_positions failed: {e}")

    # ── Closed trades ─────────────────────────────────────────────────
    closed = []
    wins = 0
    for t in history:
        if t.get("status") and t["status"] != "open" and t.get("pnl_gross") is not None:
            closed.append({
                "ticker": t.get("ticker"),
                "pnl":    round(t.get("pnl_gross", 0) or 0, 2),
                "status": t.get("status"),
            })
            if (t.get("pnl_gross") or 0) > 0:
                wins += 1

    # ── Market conditions ─────────────────────────────────────────────
    vix = None
    market_open = False
    try:
        from indicators import get_vix
        vix = get_vix()
        market_open = broker.is_market_open()
    except Exception:
        pass

    # ── Circuit breaker ───────────────────────────────────────────────
    cb_tripped = False
    try:
        from circuit_breaker import check_circuit_breaker
        ok, _ = check_circuit_breaker()
        cb_tripped = not ok
    except Exception:
        pass

    # ── Recent headlines ──────────────────────────────────────────────
    news = []
    try:
        from news_service import get_general_headlines
        news = get_general_headlines(3)
    except Exception:
        pass

    # ── Trading hours (Israeli time) ─────────────────────────────────
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    # Israel is UTC+3 (IDT summer) or UTC+2 (IST winter)
    israel_offset = 3 if 3 <= now_utc.month <= 10 else 2
    now_il = now_utc + timedelta(hours=israel_offset)
    is_edt = 3 <= now_utc.month <= 10
    trading_hours = {
        "now_israel":          now_il.strftime("%H:%M (%A)"),
        "market_open_israel":  "16:30" if is_edt else "15:30",
        "market_close_israel": "23:00" if is_edt else "22:00",
        "premarket_israel":    "12:00" if is_edt else "11:00",
        "afterhours_israel":   "23:00-02:00" if is_edt else "22:00-01:00",
        "bot_scan_interval":   "כל 5 דקות בשעות מסחר",
        "morning_briefing":    "16:00 שעון ישראל (30 דקות לפני פתיחה)" if is_edt else "15:00 שעון ישראל",
        "daily_summary":       "23:05 שעון ישראל (אחרי סגירה)" if is_edt else "22:05 שעון ישראל",
        "weekly_report":       "ראשון 23:10 שעון ישראל",
        "season":              "קיץ (EDT)" if is_edt else "חורף (EST)",
    }

    # ── Bot settings ──────────────────────────────────────────────────
    from scoring import MIN_BUY_SCORE

    total_val = sum(p["value"] for p in positions_summary)
    win_rate = round(wins / len(closed) * 100, 1) if closed else 0

    _result = {
        # Portfolio
        "cash":                 round(status.get("cash_available", 0), 2),
        "equity":               round(status.get("equity", 0), 2),
        "open_pnl":             round(status.get("open_pnl", 0), 2),
        "realized_pnl_net":     round(status.get("realized_pnl_net", 0), 2),
        "realized_pnl_gross":   round(status.get("realized_pnl_gross", 0), 2),
        "max_budget":           round(status.get("total_budget", 1000), 2),
        "total_invested":       round(total_val, 2),
        # Positions
        "open_positions_count": len(open_trades),
        "open_positions":       positions_summary,
        # History
        "closed_trades":        closed,
        "total_closed":         len(closed),
        "win_rate":             win_rate,
        # Market
        "market_open":          market_open,
        "vix":                  vix,
        "circuit_breaker":      cb_tripped,
        # Bot config
        "min_buy_score":        MIN_BUY_SCORE,
        "max_positions":        settings.MAX_OPEN_POSITIONS,
        "stop_loss_pct":        settings.STOP_LOSS_PCT,
        "take_profit_pct":      settings.TAKE_PROFIT_PCT,
        "broker":               settings.ACTIVE_BROKER,
        # News
        "latest_news":          news,
        # Trading hours
        "trading_hours":        trading_hours,
    }
    _context_cache = (_t.time(), _result)   # type: ignore[assignment]
    return _result


def _generate_reply(user_message: str) -> str:
    """
    Fully dynamic reply — LLM analyzes ANY question and responds intelligently.
    No predefined keywords or fixed responses — pure AI.
    """
    client = _get_client()
    context = _build_context()

    if client:
        return _llm_reply(user_message, context)

    # ── Only if LLM unavailable — very simple fallback ──────────────────────
    return _simple_fallback(context)

def _llm_reply(user_message: str, context: dict) -> str:
    """Full LLM-powered dynamic reply — analyzes ANY question and provides the most relevant answer."""
    client = _get_client()
    if not client:
        return _simple_fallback(context)

    # Build detailed position summary
    positions = context.get("open_positions", [])
    pos_lines = []
    for p in positions:
        emoji = "🟢" if p["pct"] >= 0 else "🔴"
        stop = p.get("atr_stop") or 0
        held = p.get("held_hours", 0)
        held_str = _fmt_held(held)
        invested = p.get("invested") or round(p["entry"] * p["qty"], 2)
        pos_lines.append(
            f"{emoji} <b>{p['ticker']}</b>\n"
            f"   📦 {p['qty']} מניות  |  💵 הושקע {_fmt_price(invested)}\n"
            f"   📈 כניסה {_fmt_price(p['entry'])} → עכשיו {_fmt_price(p['current'])} ({p['pct']:+.1f}%)\n"
            f"   💰 {_fmt_pnl(p['pnl'])}  |  🛑 Stop: {_fmt_price(stop)}\n"
            f"   ⏱ הוחזק: {held_str}"
        )
    pos_text = "\n".join(pos_lines) if pos_lines else "אין פוזיציות פתוחות כרגע"

    # News
    news_text = " | ".join(context.get("latest_news", [])[:3]) or "אין חדשות"

    # Closed trades
    closed = context.get("closed_trades", [])
    closed_text = json.dumps(closed[-5:], ensure_ascii=False) if closed else "אין עסקאות סגורות"

    system_prompt = f"""⚠️ כלל ברזל: ענה אך ורק בעברית. שמות מניות (AAPL, TSLA) יישארו באנגלית.

אתה עוזר אישי של בוט מסחר. ענה בדיוק על מה שנשאל — לא יותר ולא פחות.

══ מדריך לסוגי שאלות ══

❓ "איזה מניות יש לי" / "מה יש בתיק" / "מה קניתי":
→ רשום כל מניה: שם, כמות, מחיר כניסה, מחיר עכשיו, רווח/הפסד ($ ו-%), כמה הושקע, stop loss

❓ "מה שווי התיק" / "כמה שווה התיק" / "מה ערך התיק":
→ ענה בשורות נפרדות:
💼 שווי תיק כולל: ${context.get('equity', 0):,.2f}
💰 מזומן: ${context.get('cash', 0):,.2f}
📈 מניות: ${context.get('total_invested', 0):,.2f}
💹 רווח/הפסד פתוח: ${context.get('open_pnl', 0):+,.2f}

❓ "כמה כסף יש לי" / "כמה מזומן":
→ ענה: 💰 מזומן פנוי: ${context.get('cash', 0):,.2f}

❓ "מה הרווח שלי" / "כמה הרווחתי" / "מה ההפסד":
→ ענה בשורות נפרדות:
📈 רווח/הפסד פתוח: ${context.get('open_pnl', 0):+,.2f}
{"💳 רווח ממומש: $" + f"{context.get('realized_pnl_net',0):+,.2f}" if context.get('realized_pnl_net',0) != 0 else "(אין עדיין רווח ממומש)"}

❓ שאלה על מניה ספציפית (לדוגמה "מה קורה עם AAPL"):
→ פרט רק את הנתונים על אותה מניה

❓ "מה המצב" / "תסביר מצב":
→ תן סיכום קצר של הכל: תיק, מניות, רווח, שוק

❓ "מי הברוקר" / "איזה ברוקר" / "דרך מי הוא קונה":
→ ענה: הברוקר הוא {context.get('broker', 'tv_paper')} — זהו ברוקר נייר (paper trading) שמדמה מסחר אמיתי עם כסף וירטואלי. הקניות מתבצעות דרך yfinance עם מחירים אמיתיים מהשוק

❓ "איך הבוט עובד" / "מה האסטרטגיה":
→ הסבר בפשטות: סורק כל 5 דקות, קונה מניות עם ציון ≥60/100, מוכר ב-Stop Loss או Take Profit

❓ כל שאלה אחרת — ענה בידידותיות בעברית

══ מצב התיק עכשיו ══
💰 מזומן פנוי: ${context.get('cash', 0):,.2f}
💼 שווי מניות: ${context.get('total_invested', 0):,.2f}
📊 תיק כולל: ${context.get('equity', 0):,.2f}
📈 רווח/הפסד פתוח: ${context.get('open_pnl', 0):+,.2f}
💳 רווח ממומש (נטו): ${context.get('realized_pnl_net', 0):+,.2f}
🔢 פוזיציות: {context.get('open_positions_count', 0)} פתוחות (מקסימום {context.get('max_positions', 4)})
📊 אחוז הצלחה: {context.get('win_rate', 0)}% ({context.get('total_closed', 0)} עסקאות סגורות)

══ פוזיציות פתוחות ══
{pos_text}

══ הגדרות בוט ══
🏦 ברוקר: {context.get('broker', 'tv_paper')} (paper trading — כסף וירטואלי)
ציון קנייה מינימלי: {context.get('min_buy_score', 60)}/100
Stop Loss: {context.get('stop_loss_pct', 5)}% | Take Profit: {context.get('take_profit_pct', 15)}%
🛑 Circuit Breaker: {'⚠️ פעיל — אין קניות!' if context.get('circuit_breaker') else '✅ תקין'}
🕐 שוק: {'🟢 פתוח' if context.get('market_open') else '🔴 סגור'}
📉 VIX: {context.get('vix') or 'N/A'}

══ שעות מסחר (ישראל) ══
עכשיו: {context.get('trading_hours', {}).get('now_israel', 'N/A')} | {context.get('trading_hours', {}).get('season', '')}
פתיחה: {context.get('trading_hours', {}).get('market_open_israel', '16:30')} | סגירה: {context.get('trading_hours', {}).get('market_close_israel', '23:00')}

══ עסקאות אחרונות ══
{closed_text}
"""

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=600,
            temperature=0.3,
        )
        reply = response.choices[0].message.content.strip()

        # Verify reply is in Hebrew — if mostly English chars, translate it
        hebrew_chars = sum(1 for c in reply if 'א' <= c <= 'ת')
        latin_chars = sum(1 for c in reply if c.isalpha() and c.isascii())
        if latin_chars > hebrew_chars * 2 and len(reply) > 20:
            # Too much English — ask LLM to translate
            logger.warning(f"[CHAT] Reply was in English, translating...")
            try:
                tr_resp = client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user",
                                "content": f"תרגם את הטקסט הבא לעברית בלבד:\n{reply}"}],
                    max_tokens=500, temperature=0.2,
                )
                reply = tr_resp.choices[0].message.content.strip()
            except Exception:
                pass

        logger.info(f"[CHAT] LLM reply generated ({len(reply)} chars)")
        return reply
    except Exception as exc:
        logger.warning(f"[CHAT] LLM failed: {exc} — using simple fallback")
        return _simple_fallback(context)




def _simple_fallback(ctx: dict) -> str:
    """Fallback when LLM is unavailable — still shows full position details."""
    cash     = ctx.get("cash", 0)
    equity   = ctx.get("equity", 0)
    pnl      = ctx.get("open_pnl", 0)
    realized = ctx.get("realized_pnl_net", 0)
    positions = ctx.get("open_positions", [])
    total_invested = ctx.get("total_invested", 0)

    pnl_note = " (הפוזיציות חדשות — עדיין לא זז המחיר)" if pnl == 0 and positions else ""

    lines = [f"🏦 <b>מצב התיק</b>\n━━━━━━━━━━━━━━━━"]
    lines.append(f"💎 <b>שווי כולל: ${equity:,.2f}</b>")
    if cash > 0:
        lines.append(f"💵 מזומן פנוי: ${cash:,.2f}")
    if total_invested > 0:
        lines.append(f"📊 מושקע במניות: ${total_invested:,.2f}")
    if pnl != 0:
        lines.append(f"💹 {_fmt_pnl(pnl)}{pnl_note}")
    if realized != 0:
        lines.append(f"🏆 רווח ממומש: <b>${realized:+.2f}</b>")

    if positions:
        lines.append(f"\n📂 <b>פוזיציות ({len(positions)}):</b>")
        for p in positions:
            invested = p.get("invested") or round(p["entry"] * p["qty"], 2)
            stop = p.get("atr_stop") or 0
            held = p.get("held_hours", 0)
            profit = p["pnl"] >= 0
            status_icon = "🟢📈" if profit else "🔴📉"
            pnl_label   = f"💰 {_fmt_pnl(p['pnl'])}"
            held_line = f"\n   ⏳ הוחזק: {_fmt_held(held)}" if held >= 0.5 else ""
            lines.append(
                f"\n{status_icon} <b>{p['ticker']}</b>\n"
                f"   🪙 {p['qty']} מניות  |  💸 הושקע: {_fmt_price(invested)}\n"
                f"   📌 כניסה: {_fmt_price(p['entry'])} ➜ {_fmt_price(p['current'])} ({p['pct']:+.1f}%)\n"
                f"   {pnl_label}"
                + (f"  |  🛡️ Stop: {_fmt_price(stop)}" if stop else "")
                + held_line
            )
    else:
        lines.append("\nאין פוזיציות פתוחות כרגע.")

    lines.append("\n━━━━━━━━━━━━━━━━")
    lines.append("⚡ הבוט זמין לשאלות — שלח כל שאלה ואענה!")
    return "\n".join(lines)


async def _send_typing(chat_id: str) -> None:
    """Send 'typing...' action to Telegram so user sees the bot is working."""
    try:
        import aiohttp as _aiohttp
        url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendChatAction"
        async with _aiohttp.ClientSession() as s:
            await s.post(url, json={"chat_id": chat_id, "action": "typing"},
                         timeout=_aiohttp.ClientTimeout(total=3))
    except Exception:
        pass  # typing indicator is best-effort


def _handle_command(text: str, context: dict) -> str | None:
    """
    Handle common questions directly — guaranteed formatting, no LLM needed.
    Returns reply string or None (→ fall through to LLM).
    """
    t = text.strip().lower()
    cmd = t.split()[0] if t else ""

    # ── /commands ──────────────────────────────────────────────────────────
    if cmd in ("/start", "/help", "עזרה"):
        return (
            "👋 <b>שלום! אני בוט המסחר שלך.</b>\n\n"
            "<b>שאל בעברית חופשית:</b>\n"
            "• איזה מניות יש לי?\n"
            "• כמה כסף יש לי?\n"
            "• מה הרווח שלי?\n"
            "• מה שווי התיק?\n"
            "• מתי השוק נפתח?\n\n"
            "<b>פקודות מהירות:</b>\n"
            "📊 /status — מצב מהיר\n"
            "📈 /sectors — דירוג סקטורים\n"
            "🌍 /market — מצב שוק\n"
            "⏸️ /pause — עצור קניות\n"
            "▶️ /resume — חדש קניות\n"
            "❓ /help — הודעה זו"
        )

    if cmd == "/status":
        return _simple_fallback(context)

    if cmd in ("/pause", "עצור", "עצור קניות", "pause"):
        import os as _os
        _os.environ["BOT_PAUSED"] = "true"
        return (
            "⏸️ <b>הבוט עצר קניות חדשות</b>\n"
            "הפוזיציות הקיימות ממשיכות להיות מנוטרות.\n"
            "לחידוש: שלח <b>/resume</b>"
        )

    if cmd in ("/resume", "המשך", "חדש", "resume"):
        import os as _os
        _os.environ.pop("BOT_PAUSED", None)
        return (
            "▶️ <b>הבוט חזר לפעולה</b>\n"
            "סורק מניות וקונה כרגיל ✅"
        )

    if cmd in ("/sectors", "סקטורים", "sectors", "מגזרים"):
        try:
            from sector_rotation import get_leading_sectors
            sectors = get_leading_sectors()
            if not sectors:
                return "❌ לא הצלחתי לקבל נתוני סקטורים"
            lines = ["📊 <b>סקטורים — דירוג מומנטום (20 יום)</b>\n━━━━━━━━━━━━━━━━"]
            medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟","1️⃣1️⃣"]
            for s in sectors:
                m = medals[s["rank"]-1] if s["rank"] <= len(medals) else "▪️"
                bar = "▓" * max(1, int((s["return_pct"] + 10) / 2)) if s["return_pct"] > -10 else "░"
                lines.append(f"{m} <b>{s['name']}</b>: {s['return_pct']:+.1f}%")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ שגיאה: {e}"

    if cmd in ("/market", "שוק", "market", "מצב שוק"):
        try:
            from indicators import get_market_conditions, get_fear_greed
            mkt = get_market_conditions()
            vix = mkt.get("vix")
            fg  = mkt.get("fear_greed") or get_fear_greed()
            spy = "📈 עולה" if mkt.get("spy_above_sma50") else "📉 יורד"
            vix_str = f"🌡️ VIX: {vix:.1f}" if vix else "🌡️ VIX: N/A"
            fg_label = ""
            if fg is not None:
                if fg <= 25:   fg_label = f"😨 פחד קיצוני ({fg})"
                elif fg <= 45: fg_label = f"😟 פחד ({fg})"
                elif fg <= 55: fg_label = f"😐 ניטרלי ({fg})"
                elif fg <= 75: fg_label = f"😏 חמדנות ({fg})"
                else:          fg_label = f"🤑 חמדנות קיצונית ({fg})"
            return (
                f"🌍 <b>מצב השוק</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 SPY: {spy}\n"
                f"{vix_str}\n"
                + (f"💭 Fear & Greed: {fg_label}\n" if fg_label else "")
                + f"{'✅ אפשר לקנות' if mkt.get('spy_above_sma50') and (vix or 20) < 28 else '⚠️ שוק לא אידיאלי לקנייה'}"
            )
        except Exception as e:
            return f"❌ לא הצלחתי לקבל מצב שוק: {e}"

    if cmd in ("/log", "לוג", "log"):
        return (
            "📋 <b>לוג סריקות אחרונות</b>\n"
            "כדי לראות את הלוגים המלאים:\n"
            "Render → tradebot → <b>Logs</b>"
        )

    # /sell TICKER — force sell a position
    if cmd == "/sell" or (cmd == "מכור" and len(t.split()) > 1):
        parts = t.split()
        ticker_to_sell = parts[1].upper() if len(parts) > 1 else ""
        if not ticker_to_sell:
            return "שימוש: /sell AAPL — לדוגמה"
        import database as _db
        trade = _db.get_open_trade_by_ticker(ticker_to_sell)
        if not trade:
            return f"❌ אין פוזיציה פתוחה עבור <b>{ticker_to_sell}</b>"
        import broker as _br
        import os as _os
        try:
            pos = _br.get_position(ticker_to_sell)
            cur = float(pos.get("current_price", trade["entry_price"])) if pos else trade["entry_price"]
            pnl = (cur - trade["entry_price"]) * trade["qty"]
            pct = (cur - trade["entry_price"]) / trade["entry_price"] * 100
            base_url = _os.getenv("RENDER_EXTERNAL_URL", "https://tradebot-yc8p.onrender.com").rstrip("/")
            return (
                f"⚠️ <b>אישור מכירה — {ticker_to_sell}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💵 מחיר עכשיו: ${cur:.2f}\n"
                f"{'🟢' if pnl >= 0 else '🔴'} רווח/הפסד: <b>${pnl:+.2f}</b> ({pct:+.1f}%)\n\n"
                f"למכירה — פתח בדפדפן ועם secret:\n"
                f"<code>{base_url}/emergency-exit/{ticker_to_sell}?secret=YOUR_SECRET</code>"
            )
        except Exception as e:
            return f"❌ שגיאה בבדיקת {ticker_to_sell}: {e}"

    # ── שאלות מניות/פוזיציות ───────────────────────────────────────────────
    stocks_keywords = ["מניות", "מניה", "פוזיציות", "מה יש", "מה קניתי", "מחזיק", "תיק שלי", "איזה"]
    if any(k in t for k in stocks_keywords):
        positions = context.get("open_positions", [])
        if not positions:
            return "אין פוזיציות פתוחות כרגע 📭"
        lines = [f"📂 <b>פוזיציות פתוחות ({len(positions)})</b>\n━━━━━━━━━━━━━━━━"]
        total_pnl = 0.0
        for p in positions:
            profit    = p["pnl"] >= 0
            status    = "🟢📈" if profit else "🔴📉"
            pnl_icon  = "🟢" if profit else "🔴"
            stop      = p.get("atr_stop") or 0
            held      = p.get("held_hours", 0)
            invested  = p.get("invested") or round(p["entry"] * p["qty"], 2)
            held_line = f"\n   ⏳ {_fmt_held(held)}" if held >= 0.5 else ""
            total_pnl += p["pnl"]
            lines.append(
                f"\n{status} <b>{p['ticker']}</b>\n"
                f"   🪙 {p['qty']} מניות  |  💸 הושקע: {_fmt_price(invested)}\n"
                f"   📌 {_fmt_price(p['entry'])} ➜ {_fmt_price(p['current'])} ({p['pct']:+.1f}%)\n"
                f"   💰 {_fmt_pnl(p['pnl'])}"
                + (f"  |  🛡️ Stop: {_fmt_price(stop)}" if stop else "")
                + held_line
            )
        total_icon = "🏆" if total_pnl >= 0 else "📉"
        lines.append(f"\n━━━━━━━━━━━━━━━━\n{total_icon} סה״כ: {_fmt_pnl(total_pnl)}")
        return "\n".join(lines)

    # ── שאלות רווח/הפסד ────────────────────────────────────────────────────
    profit_keywords = ["רווח", "הפסד", "כמה הרווחתי", "כמה הפסדתי", "p&l", "pnl"]
    if any(k in t for k in profit_keywords):
        positions = context.get("open_positions", [])
        realized  = context.get("realized_pnl_net", 0)
        total_pnl = context.get("open_pnl", 0)
        total_emoji = "📈" if total_pnl >= 0 else "📉"

        lines = ["💰 <b>רווח/הפסד</b>\n━━━━━━━━━━━━━━━━"]

        # Per-stock breakdown
        for p in positions:
            e = "🟢" if p["pct"] >= 0 else "🔴"
            lines.append(f"{e} <b>{p['ticker']}</b>: {_fmt_pnl(p['pnl'])} ({p['pct']:+.1f}%)")

        # Total
        lines.append(f"━━━━━━━━━━━━━━━━\n{total_emoji} סה״כ פתוח: {_fmt_pnl(total_pnl)}")
        if realized != 0:
            lines.append(f"💳 ממומש: {_fmt_pnl(realized)}")
        return "\n".join(lines)

    # ── שאלות שווי/תיק ─────────────────────────────────────────────────────
    portfolio_keywords = ["שווי", "ערך התיק", "שווה", "תיק", "portfolio"]
    if any(k in t for k in portfolio_keywords) and "מניות" not in t:
        cash      = context.get("cash", 0)
        equity    = context.get("equity", 0)
        invested  = context.get("total_invested", 0)
        pnl       = context.get("open_pnl", 0)
        realized  = context.get("realized_pnl_net", 0)
        lines = [
            f"💼 <b>שווי התיק</b>",
            f"━━━━━━━━━━━━━━━━",
            f"📊 סה״כ: <b>${equity:,.2f}</b>",
        ]
        if cash > 0:
            lines.append(f"💰 מזומן: ${cash:,.2f}")
        if invested > 0:
            lines.append(f"📈 מניות: {_fmt_price(invested)}")
        if pnl != 0:
            lines.append(f"💹 {_fmt_pnl(pnl)}")
        if realized != 0:
            lines.append(f"🏆 ממומש: {_fmt_pnl(realized)}")
        return "\n".join(lines)

    # ── שאלות מזומן ────────────────────────────────────────────────────────
    cash_keywords = ["כמה כסף", "כמה מזומן", "מזומן", "cash"]
    if any(k in t for k in cash_keywords):
        cash = context.get("cash", 0)
        equity = context.get("equity", 0)
        invested = context.get("total_invested", 0)
        pct_invested = round(invested / equity * 100, 1) if equity > 0 else 0
        lines = [f"💵 <b>מזומן פנוי: ${cash:,.2f}</b>"]
        if pct_invested > 0:
            lines.append(f"📊 {pct_invested}% מהתיק מושקע")
        if cash == 0:
            lines.append("⚠️ אין מזומן — ממתין למכירה לפני קנייה חדשה")
        return "\n".join(lines)

    # ── שאלות ביצועים ──────────────────────────────────────────────────────
    perf_keywords = ["ביצועים", "סטטיסטיקה", "כמה עסקאות", "win rate", "אחוז הצלחה"]
    if any(k in t for k in perf_keywords):
        closed = context.get("closed_trades", [])
        total  = len(closed)
        wins   = sum(1 for x in closed if x.get("pnl", 0) > 0)
        wr     = round(wins / total * 100, 1) if total > 0 else 0
        total_pnl = sum(x.get("pnl", 0) for x in closed)
        if total == 0:
            return "📊 <b>עדיין אין עסקאות סגורות</b>\nהבוט צריך לפחות 10 עסקאות לסטטיסטיקה"
        return (
            f"📊 <b>ביצועים</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🔢 עסקאות סגורות: {total}\n"
            f"✅ זכיות: {wins}  |  ❌ הפסדים: {total - wins}\n"
            f"🎯 אחוז הצלחה: <b>{wr}%</b>\n"
            f"💰 רווח כולל: <b>${total_pnl:+.2f}</b>"
        )

    return None  # let LLM handle everything else


async def handle_telegram_update(update: dict) -> dict:
    """
    Handle an incoming Telegram update.
    Returns a dict with status info (used for diagnostics).
    """
    message = update.get("message") or update.get("edited_message") or {}
    if not message:
        return {"status": "ignored", "reason": "no message in update"}

    chat = message.get("chat", {})
    chat_id = str(chat.get("id", ""))
    text = (message.get("text") or "").strip()

    # Security: only respond to the configured chat
    if not settings.TELEGRAM_CHAT_ID:
        return {"status": "ignored", "reason": "TELEGRAM_CHAT_ID not configured"}
    if chat_id != str(settings.TELEGRAM_CHAT_ID):
        logger.warning(f"[CHAT] Ignoring message from unauthorized chat {chat_id}")
        return {"status": "ignored", "reason": "unauthorized chat"}

    if not text:
        return {"status": "ignored", "reason": "empty message"}

    logger.info(f"[CHAT] Incoming: {text[:100]}")

    # Send typing indicator immediately so user knows bot is working
    await _send_typing(chat_id)

    # Generate reply — run in thread to avoid blocking the event loop during LLM call
    try:
        context = await asyncio.to_thread(_build_context)

        # Check for quick commands first (no LLM needed)
        reply = _handle_command(text, context)

        if reply is None:
            # Full LLM reply
            client = _get_client()
            if client:
                reply = await asyncio.to_thread(_llm_reply, text, context)
            else:
                reply = _simple_fallback(context)

    except Exception as exc:
        logger.error(f"[CHAT] Reply generation failed: {exc}")
        reply = "מצטער, נתקלתי בשגיאה. נסה שוב."

    # Send reply
    try:
        ok = await send_message(reply)
        return {
            "status": "replied" if ok else "send_failed",
            "incoming": text[:200],
            "reply": reply[:200],
        }
    except Exception as exc:
        logger.error(f"[CHAT] Failed to send reply: {exc}")
        return {"status": "error", "reason": str(exc)}
