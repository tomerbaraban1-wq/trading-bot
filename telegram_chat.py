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
            f"   📦 {p['qty']} מניות  |  💵 הושקע ${invested:,.2f}\n"
            f"   📈 כניסה ${p['entry']} → עכשיו ${p['current']} ({p['pct']:+.1f}%)\n"
            f"   💰 רווח/הפסד: <b>${p['pnl']:+.2f}</b>  |  🛑 Stop: ${stop}\n"
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
💳 רווח ממומש: ${context.get('realized_pnl_net', 0):+,.2f}

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

    lines = [f"📊 <b>מצב התיק</b>\n━━━━━━━━━━━━━━━━"]
    lines.append(f"💼 <b>שווי תיק כולל: ${equity:,.2f}</b>")
    lines.append(f"   💰 מזומן פנוי: ${cash:,.2f}  |  📈 מניות: ${total_invested:,.2f}")
    lines.append(f"   💹 רווח/הפסד פתוח: <b>${pnl:+.2f}</b>{pnl_note}")
    if realized != 0:
        lines.append(f"   💳 רווח ממומש: <b>${realized:+.2f}</b>")

    if positions:
        lines.append("\n<b>פוזיציות פתוחות:</b>")
        for p in positions:
            emoji = "🟢" if p["pct"] >= 0 else "🔴"
            invested = p.get("invested") or round(p["entry"] * p["qty"], 2)
            stop = p.get("atr_stop") or 0
            held = p.get("held_hours", 0)
            profit = p["pnl"] >= 0
            dir_emoji = "📈" if profit else "📉"
            pnl_tag   = "רווח 💚" if profit else "הפסד ❤️"
            held_line = f"\n   ⏱ {_fmt_held(held)}" if held >= 0.5 else ""
            lines.append(
                f"{dir_emoji} <b>{p['ticker']}</b> 📊  — {pnl_tag}\n"
                f"   📦 {p['qty']} מניות  |  💵 הושקע: <b>${invested:,.2f}</b>\n"
                f"   🔢 ${p['entry']} → ${p['current']} ({p['pct']:+.1f}%)\n"
                f"   💰 רווח/הפסד: <b>${p['pnl']:+.2f}</b>"
                + (f"  |  🛑 Stop: ${stop}" if stop else "")
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
            "שאל אותי כל שאלה בעברית חופשית:\n"
            "• איזה מניות יש לי?\n"
            "• כמה כסף יש לי?\n"
            "• מה הרווח שלי?\n"
            "• מה שווי התיק?\n"
            "• מתי השוק נפתח?\n\n"
            "📊 /status — מצב מהיר\n"
            "❓ /help — הודעה זו"
        )

    if cmd == "/status":
        return _simple_fallback(context)

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
            dir_emoji = "📈" if profit else "📉"
            pnl_tag   = "רווח 💚" if profit else "הפסד ❤️"
            stop      = p.get("atr_stop") or 0
            held      = p.get("held_hours", 0)
            invested  = p.get("invested") or round(p["entry"] * p["qty"], 2)
            held_line = f"\n   ⏱ {_fmt_held(held)}" if held >= 0.5 else ""
            total_pnl += p["pnl"]
            lines.append(
                f"\n{dir_emoji} <b>{p['ticker']}</b> 📊  — {pnl_tag}\n"
                f"   📦 {p['qty']} מניות  |  💵 הושקע: ${invested:,.2f}\n"
                f"   🔢 ${p['entry']} → ${p['current']} ({p['pct']:+.1f}%)\n"
                f"   💰 רווח/הפסד: <b>${p['pnl']:+.2f}</b>  |  🛑 Stop: ${stop}"
                f"{held_line}"
            )
        total_emoji = "📈" if total_pnl >= 0 else "📉"
        lines.append(f"\n━━━━━━━━━━━━━━━━\n{total_emoji} סה״כ רווח/הפסד: <b>${total_pnl:+.2f}</b>")
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
            lines.append(f"{e} <b>{p['ticker']}</b>: <b>${p['pnl']:+.2f}</b> ({p['pct']:+.1f}%)")

        # Total
        lines.append(f"━━━━━━━━━━━━━━━━\n{total_emoji} סה״כ פתוח: <b>${total_pnl:+.2f}</b>")
        if realized != 0:
            lines.append(f"💳 ממומש: <b>${realized:+.2f}</b>")
        return "\n".join(lines)

    # ── שאלות שווי/תיק ─────────────────────────────────────────────────────
    portfolio_keywords = ["שווי", "ערך התיק", "שווה", "תיק", "portfolio"]
    if any(k in t for k in portfolio_keywords) and "מניות" not in t:
        cash      = context.get("cash", 0)
        equity    = context.get("equity", 0)
        invested  = context.get("total_invested", 0)
        pnl       = context.get("open_pnl", 0)
        realized  = context.get("realized_pnl_net", 0)
        return (
            f"💼 <b>שווי התיק</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📊 סה״כ: <b>${equity:,.2f}</b>\n"
            f"💰 מזומן: ${cash:,.2f}\n"
            f"📈 מניות: ${invested:,.2f}\n"
            f"💹 רווח/הפסד פתוח: <b>${pnl:+.2f}</b>\n"
            f"💳 ממומש: ${realized:+.2f}"
        )

    # ── שאלות מזומן ────────────────────────────────────────────────────────
    cash_keywords = ["כמה כסף", "כמה מזומן", "מזומן", "cash"]
    if any(k in t for k in cash_keywords):
        cash = context.get("cash", 0)
        return f"💰 <b>מזומן פנוי: ${cash:,.2f}</b>"

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
