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
import re as _re
from openai import OpenAI

from config import settings
from telegram_bot import send_message
import broker
import budget
import database

logger = logging.getLogger(__name__)

# Ticker validation — only A-Z0-9 dots and dashes, 1-10 chars
_TICKER_RE = _re.compile(r'^[A-Z0-9.\-]{1,10}$')

def _safe_ticker(raw: str) -> str | None:
    """Return sanitized ticker or None if invalid."""
    t = raw.strip().upper()
    return t if _TICKER_RE.match(t) else None


def _score_bar(score: float, width: int = 10) -> str:
    """Visual progress bar: ██████░░░░  65/100"""
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _extract_ticker_from_text(text: str) -> str | None:
    """Try to find a stock ticker in free Hebrew text. e.g. 'מה ציון של AAPL?' → 'AAPL'"""
    words = text.upper().split()
    for w in words:
        clean = _re.sub(r'[^A-Z0-9.\-]', '', w)
        if _TICKER_RE.match(clean) and len(clean) >= 2:
            return clean
    return None

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
    """Format P&L correctly in Hebrew RTL context, with ILS."""
    label = "רווח 🟢" if amount >= 0 else "הפסד 🔴"
    formatted = f"<b>{_fmt_price(abs(amount))}</b>"
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
→ הסבר בפשטות: סורק כל 5 דקות, קונה מניות עם ציון ≥60/100, מוכר ב-Stop Loss או יעד רווח/הפסד

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
Stop Loss: {context.get('stop_loss_pct', 5)}% | יעד רווח/הפסד: {context.get('take_profit_pct', 15)}%
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
        # Suppress raw exception — Groq/OpenAI errors may contain auth headers or API keys
        logger.warning("[CHAT] LLM call failed (details suppressed for security) — using simple fallback")
        logger.debug(f"[CHAT] LLM error details: {type(exc).__name__}")
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
    lines.append(f"💎 <b>שווי כולל: {_fmt_price(equity)}</b>")
    if cash > 0:
        lines.append(f"💵 מזומן פנוי: {_fmt_price(cash)}")
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
            try:
                tp_pct = max(3.0, (stop / p["entry"] * 100) * 1.5) if stop and p["entry"] else 5.0
                tp_price = round(p["entry"] * (1 + tp_pct / 100), 2)
            except Exception:
                tp_price = 0
            lines.append(
                f"\n{status_icon} <b>{p['ticker']}</b>\n"
                f"   🔢 כמות: {p['qty']} מניות\n"
                f"   📌 מחיר קנייה: {_fmt_price(p['entry'])}\n"
                f"   📈 יעד רווח: {_fmt_price(tp_price) if tp_price else 'N/A'}\n"
                f"   📉 סטופ לוס: {_fmt_price(stop) if stop else 'N/A'}\n"
                f"   📍 עכשיו: {_fmt_price(p['current'])} ({p['pct']:+.1f}%)\n"
                f"   {pnl_label}\n"
                f"   ⏳ זמן החזקה: {_fmt_held(held) if held >= 0.5 else 'כמה דקות'}"
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
    if cmd in ("/start", "/help", "עזרה", "עזר", "פקודות", "מה אתה יכול"):
        return (
            "👋 <b>מנהל ההשקעות שלך — כל הפקודות</b>\n\n"
            "━━ 📊 <b>תיק ופוזיציות</b> ━━\n"
            "/status — מצב מלא של התיק\n"
            "/manioth — איזה מניות פתוחות\n"
            "/revach — רווח/הפסד פירוט\n"
            "/shovi — שווי התיק\n"
            "/mazon — מזומן פנוי\n"
            "/portfolio — הקצאה באחוזים\n"
            "/winners — פוזיציות ברווח\n"
            "/losers — פוזיציות בהפסד\n"
            "/risk — ניתוח סיכון\n\n"
            "━━ 📈 <b>ניתוח מניות</b> ━━\n"
            "/score AAPL — ניתוח עם Progress Bars\n"
            "/price AAPL — מחיר מיידי\n"
            "/news AAPL — חדשות + סנטימנט AI\n"
            "/newscheck — בדיקת חדשות לכל הפוזיציות\n"
            "/earnings AAPL — דוח רווחים\n"
            "/stop AAPL — מצב ה-Stop Loss\n"
            "/sector AAPL — איזה סקטור\n"
            "/compare AAPL MSFT — השוואה\n\n"
            "━━ 🌍 <b>מצב השוק</b> ━━\n"
            "/market — מצב כללי\n"
            "/sectors — דירוג סקטורים\n"
            "/vix — מדד הפחד VIX\n"
            "/fear — Fear & Greed Index\n"
            "/top — מניות עם ציון גבוה\n"
            "/watchlist — רשימת הסריקה\n\n"
            "━━ 📅 <b>היסטוריה וביצועים</b> ━━\n"
            "/today — מה קרה היום\n"
            "/history — עסקאות אחרונות\n"
            "/summary — סיכום 7 ימים\n"
            "/biztsuim — ביצועים ו-Win Rate\n"
            "/best — העסקה הטובה ביותר\n"
            "/worst — העסקה הגרועה ביותר\n"
            "/taxes — סיכום מס\n"
            "/backtest — למידה היסטורית\n\n"
            "━━ ⚙️ <b>שליטה</b> ━━\n"
            "/scan — הפעל סריקה מיידית\n"
            "/pause — עצור קניות חדשות\n"
            "/resume — חדש קניות\n"
            "/sell AAPL — מכור מניה עכשיו\n"
            "/alert AAPL 200 — הגדר התראת מחיר\n"
            "/alerts — ראה כל ההתראות הפעילות\n"
            "/budget — הגדרות הבוט\n"
            "/diagnose — למה הבוט לא קונה?\n"
            "/next — מתי השוק נפתח\n"
            "/uptime — כמה זמן הבוט רץ\n\n"
            "━━ 🧠 <b>AI</b> ━━\n"
            "/advice — ייעוץ AI על התיק\n"
            "/streak — רצף ניצחונות/הפסדות\n\n"
            "<i>💬 אפשר גם לשאול בעברית חופשית!</i>"
        )

    if cmd == "/status":
        positions = context.get("open_positions", [])
        cash      = context.get("cash", 0)
        equity    = context.get("equity", 0)
        pnl       = context.get("open_pnl", 0)
        realized  = context.get("realized_pnl_net", 0)
        vix       = context.get("vix")
        mkt_open  = context.get("market_open", False)
        cb        = context.get("circuit_breaker", False)
        n_pos     = len(positions)
        max_pos   = settings.MAX_OPEN_POSITIONS
        pnl_icon  = "📈" if pnl >= 0 else "📉"
        mkt_icon  = "🟢" if mkt_open else "🔴"
        lines = [f"📊 <b>סטטוס הבוט</b>\n━━━━━━━━━━━━━━━━"]
        lines.append(f"💼  שווי תיק:     <b>{_fmt_price(equity)}</b>")
        lines.append(f"💵  מזומן:          {_fmt_price(cash)}")
        lines.append(f"{pnl_icon}  רווח/הפסד פתוח:  {_fmt_pnl(pnl)}")
        if realized != 0:
            lines.append(f"🏆  ממומש:          {_fmt_pnl(realized)}")
        lines.append(f"📂  פוזיציות:      <b>{n_pos}/{max_pos}</b>")
        lines.append(f"{mkt_icon}  שוק:            {'פתוח' if mkt_open else 'סגור'}")
        if vix:
            vix_icon = "😌" if vix < 20 else ("😟" if vix < 28 else "😱")
            lines.append(f"{vix_icon}  VIX:              {vix:.1f}")
        if cb:
            lines.append(f"⛔  Circuit Breaker: פעיל — קניות מושהות")
        import os as _os
        if _os.getenv("BOT_PAUSED"):
            lines.append(f"⏸️  הבוט: מושהה — שלח /resume להמשך")
        else:
            lines.append(f"✅  הבוט: פעיל וסורק")
        if positions:
            lines.append(f"\n<b>פוזיציות:</b>")
            for p in positions:
                icon = "🟢" if p["pnl"] >= 0 else "🔴"
                lines.append(f"  {icon} <b>{p['ticker']}</b>  {p['pct']:+.1f}%  |  {_fmt_pnl(p['pnl'], False)}")
        return "\n".join(lines)

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
            medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟","1️⃣1️⃣"]
            lines  = ["📊 <b>סקטורים — מומנטום 20 יום</b>\n━━━━━━━━━━━━━━━━"]
            # Normalize bars: max return → full bar
            max_ret = max(abs(s["return_pct"]) for s in sectors) or 1
            for s in sectors:
                m    = medals[s["rank"]-1] if s["rank"] <= len(medals) else "▪️"
                ret  = s["return_pct"]
                fill = max(1, round(abs(ret) / max_ret * 10))
                bar  = ("█" if ret >= 0 else "░") * fill + ("░" if ret >= 0 else "█") * (10 - fill)
                icon = "🟢" if ret > 1 else ("🔴" if ret < -1 else "⚪")
                lines.append(f"{m} {icon} <b>{s['name']}</b>  {ret:+.1f}%\n   <code>{bar}</code>")
            # Tip: which sector the bot favors
            top = sectors[0]
            lines.append(f"\n━━━━━━━━━━━━━━━━\n🎯 <b>הבוט מעדיף:</b> {top['name']} ({top['return_pct']:+.1f}%)")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    if cmd in ("/market", "שוק", "market", "מצב שוק"):
        try:
            import yfinance as _yf
            from indicators import get_market_conditions, get_fear_greed
            mkt = get_market_conditions()
            vix = mkt.get("vix")
            fg  = mkt.get("fear_greed") or get_fear_greed()

            # Fetch SPY, QQQ, DIA daily change
            def _chg(sym):
                try:
                    fi = _yf.Ticker(sym).fast_info
                    c  = float(getattr(fi, "last_price", 0) or 0)
                    p  = float(getattr(fi, "previous_close", c) or c)
                    return (c - p) / p * 100 if p else 0
                except Exception:
                    return None
            spy_chg = _chg("SPY")
            qqq_chg = _chg("QQQ")
            dia_chg = _chg("DIA")

            def _idx_line(name, chg):
                if chg is None:
                    return f"  {name}: N/A"
                icon = "📈" if chg >= 0 else "📉"
                return f"  {icon} {name}: <b>{chg:+.2f}%</b>"

            # VIX bar
            vix_val  = vix or 20
            vix_bar  = "█" * min(20, round(vix_val / 2)) + "░" * max(0, 20 - round(vix_val / 2))
            vix_icon = "😌" if vix_val < 15 else ("🙂" if vix_val < 20 else ("😟" if vix_val < 28 else "😱"))

            # Fear & Greed
            if fg is not None:
                if fg <= 25:   fg_label = f"😨 פחד קיצוני ({fg})"
                elif fg <= 45: fg_label = f"😟 פחד ({fg})"
                elif fg <= 55: fg_label = f"😐 ניטרלי ({fg})"
                elif fg <= 75: fg_label = f"😏 חמדנות ({fg})"
                else:          fg_label = f"🤑 חמדנות קיצונית ({fg})"
            else:
                fg_label = "N/A"

            # Overall verdict
            spy_ok  = mkt.get("spy_above_sma50", True)
            vix_ok  = vix_val < 28
            fg_ok   = fg is None or fg >= 25
            buy_env = spy_ok and vix_ok
            if buy_env and (fg is None or fg <= 70):
                verdict = "✅ <b>סביבה טובה לקנייה</b>"
            elif not spy_ok:
                verdict = "⚠️ <b>SPY מתחת ל-SMA50 — הבוט לא קונה</b>"
            elif not vix_ok:
                verdict = "⚠️ <b>VIX גבוה — הבוט לא קונה</b>"
            else:
                verdict = "⚠️ <b>שוק חמדני — זהירות</b>"

            lines = [
                f"🌍 <b>מצב השוק</b>\n━━━━━━━━━━━━━━━━",
                f"<b>מדדים עיקריים:</b>",
                _idx_line("SPY (S&P500)", spy_chg),
                _idx_line("QQQ (נאסד\"ק)", qqq_chg),
                _idx_line("DIA (דאו)", dia_chg),
                f"\n<b>מדד הפחד:</b>",
                f"  {vix_icon} VIX: <b>{vix_val:.1f}</b>\n  <code>{vix_bar}</code>",
                f"\n💭 Fear & Greed: <b>{fg_label}</b>",
                f"\n━━━━━━━━━━━━━━━━\n{verdict}",
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/market] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /news TICKER
    if cmd in ("/news", "news", "חדשות") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /news AAPL"
        try:
            from news_service import get_headlines
            from sentiment import score_sentiment
            headlines = get_headlines(_ticker, limit=6)
            if not headlines:
                return f"📰 <b>חדשות — {_ticker}</b>\n\n😴 לא נמצאו חדשות עדכניות"
            # Get AI sentiment on the news
            sent = score_sentiment(_ticker)
            sc   = sent.score
            if sc <= 3:   sent_line = f"🔴 סנטימנט AI: {sc}/10 — שלילי מאוד"
            elif sc <= 5: sent_line = f"🟠 סנטימנט AI: {sc}/10 — שלילי"
            elif sc <= 7: sent_line = f"⚪ סנטימנט AI: {sc}/10 — ניטרלי"
            elif sc <= 8: sent_line = f"🟡 סנטימנט AI: {sc}/10 — חיובי"
            else:         sent_line = f"🟢 סנטימנט AI: {sc}/10 — חיובי מאוד"
            lines = [f"📰 <b>חדשות — {_ticker}</b>\n━━━━━━━━━━━━━━━━", sent_line, ""]
            for i, h in enumerate(headlines[:5], 1):
                lines.append(f"{i}. {h[:100]}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /price TICKER — quick current price check
    if cmd in ("/price", "price", "מחיר") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /price AAPL"
        try:
            import broker as _broker
            import yfinance as _yf
            # Try broker first (if position open), fallback to yfinance
            pos = _broker.get_position(_ticker)
            if pos:
                cur  = float(pos.get("current_price", 0))
                pct  = float(pos.get("unrealized_plpc", 0)) * 100
                held = pos.get("qty", 0)
                icon = "📈" if pct >= 0 else "📉"
                return (
                    f"💲 <b>מחיר — {_ticker}</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📍  מחיר עכשיו:  <b>{_fmt_price(cur)}</b>\n"
                    f"{icon}  שינוי בפוזיציה: <b>{pct:+.2f}%</b>\n"
                    f"🔢  כמות בידי:    {held}"
                )
            else:
                hist = _yf.Ticker(_ticker).fast_info
                cur = float(getattr(hist, "last_price", 0) or 0)
                prev = float(getattr(hist, "previous_close", cur) or cur)
                chg_pct = (cur - prev) / prev * 100 if prev else 0
                icon = "📈" if chg_pct >= 0 else "📉"
                if cur <= 0:
                    return f"❌ לא הצלחתי לקבל מחיר עבור {_ticker}"
                return (
                    f"💲 <b>מחיר — {_ticker}</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📍  מחיר עכשיו:  <b>{_fmt_price(cur)}</b>\n"
                    f"{icon}  שינוי יומי:      <b>{chg_pct:+.2f}%</b>"
                )
        except Exception as e:
            logger.error(f"[/price] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /alerts — show all active price alerts
    if cmd in ("/alerts", "alerts", "התראות", "התראות מחיר"):
        try:
            import os as _os
            raw = _os.getenv("USER_ALERTS", "")
            if not raw.strip():
                return "🔔 <b>התראות מחיר</b>\n━━━━━━━━━━━━━━━━\n😴 אין התראות פעילות\n\n💡 הגדר: /alert AAPL 200"
            parts = [a.strip() for a in raw.split(",") if ":" in a.strip()]
            if not parts:
                return "🔔 <b>התראות מחיר</b>\n━━━━━━━━━━━━━━━━\n😴 אין התראות פעילות"
            lines = [f"🔔 <b>התראות פעילות ({len(parts)})</b>\n━━━━━━━━━━━━━━━━"]
            for alert in parts:
                try:
                    tk, price_str = alert.split(":", 1)
                    lines.append(f"📌 <b>{tk.upper()}</b> → {_fmt_price(float(price_str))}")
                except Exception:
                    pass
            lines.append("\n💡 הסר הכל: /alerts clear")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/alerts] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /alerts clear — remove all price alerts
    if t in ("/alerts clear", "alerts clear", "נקה התראות", "מחק התראות"):
        try:
            import os as _os
            _os.environ["USER_ALERTS"] = ""
            return "🔕 <b>כל ההתראות נמחקו</b>"
        except Exception:
            return "❌ שגיאה פנימית"

    # /score TICKER
    if cmd in ("/score", "score", "ציון") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /score AAPL"
        try:
            from scoring import get_composite_score
            from sentiment import score_sentiment
            sent   = score_sentiment(_ticker)
            result = get_composite_score(_ticker, sent.score)
            sc     = result["composite_score"]
            tech   = result["scores"]["technicals"]
            mkt    = result["scores"]["market"]
            sent_n = round(sent.score / 10 * 100)   # normalize to 0-100
            vix    = result.get("vix", "N/A")
            buy    = result["should_buy"]
            # Quality label
            if sc >= 80:   qlabel = "🔥 חזק מאוד"
            elif sc >= 70: qlabel = "✅ טוב"
            elif sc >= 60: qlabel = "⚠️ גבולי"
            else:          qlabel = "❌ חלש"
            # Sentiment label
            if sent.score >= 8:   sl = "😍 מצוין"
            elif sent.score >= 6: sl = "🙂 חיובי"
            elif sent.score >= 4: sl = "😐 ניטרלי"
            else:                 sl = "😟 שלילי"
            return (
                f"🎯 <b>ניתוח — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⭐  ציון כולל:   <b>{sc:.0f}/100</b>  {qlabel}\n"
                f"    <code>{_score_bar(sc)}</code>\n\n"
                f"🔧  טכני:         {tech:.0f}/100\n"
                f"    <code>{_score_bar(tech)}</code>\n"
                f"🌍  שוק:           {mkt:.0f}/100\n"
                f"    <code>{_score_bar(mkt)}</code>\n"
                f"🧠  סנטימנט:   {sent.score}/10  {sl}\n"
                f"    <code>{_score_bar(sent_n)}</code>\n\n"
                f"🌡️  VIX: {vix}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"{'✅ <b>המלצה: קנה</b>' if buy else '⏭️ <b>המלצה: דלג</b>'}"
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /diagnose
    if cmd in ("/diagnose", "diagnose", "אבחון", "למה לא קונה"):
        try:
            import requests as _req
            import os as _os
            base = _os.getenv("RENDER_EXTERNAL_URL", "https://tradebot-yc8p.onrender.com").rstrip("/")
            resp = _req.get(f"{base}/diagnose", timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("summary", {})
                blockers = summary.get("blockers", [])
                verdict = summary.get("verdict", "")
                lines = [f"🔍 <b>אבחון</b>\n━━━━━━━━━━━━━━━━"]
                lines.append(verdict)
                if blockers:
                    lines.append("\n<b>חסמים:</b>")
                    for b in blockers:
                        lines.append(f"⛔ {b}")
                return "\n".join(lines)
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /backtest
    if cmd in ("/backtest", "backtest", "למידה היסטורית"):
        try:
            from backtest_learner import get_insights
            insights = get_insights()
            if insights.get("status") == "not_run_yet":
                return "⏳ הלמידה עוד לא רצה — תופעל בראשון הקרוב בערב"
            wr = insights.get("overall_win_rate", 0)
            total = insights.get("total_signals", 0)
            optimal = insights.get("optimal_min_score", 58)
            computed = insights.get("computed_at", "")
            return (
                f"🧠 <b>למידה היסטורית</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 סיגנלים שנותחו: {total}\n"
                f"✅ אחוז הצלחה: <b>{wr:.1f}%</b>\n"
                f"🎯 ציון אופטימלי: <b>{optimal}</b>\n"
                f"🕐 עודכן: {computed}"
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /top — top scoring stocks right now
    if cmd in ("/top", "top", "הכי טובים", "מועמדים"):
        try:
            from scanner import get_watchlist
            from scoring import get_composite_score, MIN_BUY_SCORE
            from sentiment import score_sentiment
            import random
            wl      = get_watchlist()
            sample  = random.sample(wl, min(15, len(wl)))  # scan 15 instead of 8
            results = []
            for _tk in sample:
                try:
                    sent = score_sentiment(_tk)
                    comp = get_composite_score(_tk, sent.score)
                    results.append((_tk, comp["composite_score"], comp["should_buy"], sent.score))
                except Exception:
                    continue
            results.sort(key=lambda x: x[1], reverse=True)
            lines = [f"🏆 <b>מובילי הסריקה — {len(results)} מניות</b>\n━━━━━━━━━━━━━━━━"]
            buys  = [(t,s,b,ss) for t,s,b,ss in results if b]
            skips = [(t,s,b,ss) for t,s,b,ss in results if not b]
            if buys:
                lines.append("✅ <b>מעל סף קנייה:</b>")
                for tk, sc, _, ss in buys[:5]:
                    lines.append(f"  🟢 <b>{tk}</b>  {sc:.0f}/100  {_score_bar(sc, 8)}  🧠{ss}/10")
            if skips:
                lines.append("\n⏭️ <b>מתחת לסף:</b>")
                for tk, sc, _, ss in skips[:3]:
                    lines.append(f"  ⚪ <b>{tk}</b>  {sc:.0f}/100")
            lines.append(f"\n━━━━━━━━━━━━━━━━\n🎯 סף קנייה: <b>{MIN_BUY_SCORE}</b>")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /history — recent trade history
    if cmd in ("/history", "history", "היסטוריה", "עסקאות"):
        try:
            import database as _db
            trades = _db.get_trade_history(limit=8) or []
            closed = [t for t in trades if t.get("status") != "open"]
            if not closed:
                return "📋 אין עסקאות סגורות עדיין"
            lines = [f"📋 <b>עסקאות אחרונות ({len(closed)})</b>\n━━━━━━━━━━━━━━━━"]
            total_pnl = 0.0
            for _t in closed[:7]:
                pnl  = float(_t.get("pnl_gross") or 0)
                ep   = float(_t.get("entry_price") or 0)
                xp   = float(_t.get("exit_price") or 0)
                pct  = (xp - ep) / ep * 100 if ep else 0
                icon = "🟢" if pnl >= 0 else "🔴"
                date = str(_t.get("exit_time") or "")[:10]
                reason = _t.get("status", "")
                reason_map = {
                    "take_profit": "🎯יעד", "stop_loss": "🛑סטופ",
                    "smart_sell": "🧠AI", "news_exit": "📰חדשות",
                    "time_exit": "⏱זמן", "manual": "✋ידני"
                }
                reason_str = reason_map.get(reason, reason[:6] if reason else "")
                total_pnl += pnl
                lines.append(
                    f"\n{icon} <b>{_t.get('ticker','?')}</b>  {pct:+.1f}%  |  {_fmt_pnl(pnl, False)}\n"
                    f"   📌 {_fmt_price(ep)} → {_fmt_price(xp)}  ·  {reason_str}  ·  {date}"
                )
            tot_icon = "💰" if total_pnl >= 0 else "📉"
            lines.append(f"\n━━━━━━━━━━━━━━━━\n{tot_icon} סה״כ: {_fmt_pnl(total_pnl)}")
            return "\n".join(lines)
        except Exception as _e:
            logger.error(f"[/history] Error: {_e}")
            return "📋 <b>עסקאות אחרונות</b>\n⚠️ לא הצלחתי לטעון היסטוריה כרגע."

    # /fear — Fear & Greed Index
    if cmd in ("/fear", "fear", "פחד", "חמדנות", "fear greed"):
        try:
            from indicators import get_fear_greed, get_vix
            fg = get_fear_greed()
            vix = get_vix()
            if fg is None:
                return "❌ לא הצלחתי לקבל Fear & Greed"
            if fg <= 25:   label = "😨 פחד קיצוני"
            elif fg <= 45: label = "😟 פחד"
            elif fg <= 55: label = "😐 ניטרלי"
            elif fg <= 75: label = "😏 חמדנות"
            else:          label = "🤑 חמדנות קיצונית"
            tip = "✅ הזדמנות קנייה!" if fg <= 30 else ("⚠️ שוק חמדני — היזהר" if fg >= 70 else "")
            return (
                f"😨 <b>Fear & Greed Index</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 ציון: <b>{fg}/100</b>\n"
                f"💭 מצב: {label}\n"
                f"🌡️ VIX: {vix or 'N/A'}\n"
                + (f"\n{tip}" if tip else "")
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /earnings TICKER
    if cmd in ("/earnings", "earnings", "דוחות") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /earnings AAPL"
        try:
            from earnings import check_earnings_risk, get_earnings_impact
            risky, reason, days = check_earnings_risk(_ticker)
            impact = get_earnings_impact(_ticker)
            lines = [f"📅 <b>דוחות — {_ticker}</b>\n━━━━━━━━━━━━━━━━"]
            if days is not None:
                if risky:
                    lines.append(f"⛔ <b>Blackout: {days} ימים לדוח</b>")
                else:
                    lines.append(f"✅ הדוח הבא: בעוד {days} ימים")
            beat = impact.get("beat_rate", 0)
            avg_move = impact.get("avg_move_pct", 0)
            quarters = impact.get("quarters_analyzed", 0)
            if quarters > 0:
                lines.append(f"🎯 Beat rate: <b>{beat*100:.0f}%</b> ({quarters} רבעונים)")
                lines.append(f"📊 תנועה ממוצעת: <b>{avg_move:.1f}%</b>")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /stop [TICKER] — stop loss for one or all positions
    if cmd in ("/stop", "stop", "סטופ", "עצירה"):
        import database as _db
        parts = t.split()
        _ticker = _safe_ticker(parts[1]) if len(parts) > 1 else None
        try:
            if _ticker:
                # Single ticker
                trade = _db.get_open_trade_by_ticker(_ticker)
                if not trade:
                    return f"❌ אין פוזיציה פתוחה עבור <b>{_ticker}</b>"
                trades = [trade]
            else:
                # All open positions
                trades = _db.get_open_trades() or []
                if not trades:
                    return "📭 אין פוזיציות פתוחות"

            lines = [f"🛡️ <b>Stop Loss {'— ' + _ticker if _ticker else '— כל הפוזיציות'}</b>\n━━━━━━━━━━━━━━━━"]
            for trade in trades:
                tk    = trade.get("ticker", "?")
                stop  = trade.get("atr_stop_price")
                entry = float(trade.get("entry_price") or 0)
                wm    = float(trade.get("high_watermark") or entry)
                if not stop or not entry:
                    lines.append(f"⚪ <b>{tk}</b>: סטופ לא מוגדר")
                    continue
                stop = float(stop)
                # Get current price for distance
                try:
                    pos = broker.get_position(tk)
                    cur = float(pos.get("current_price", entry)) if pos else entry
                except Exception:
                    cur = entry
                dist_from_cur  = (cur - stop) / cur * 100 if cur else 0
                dist_from_entry = (entry - stop) / entry * 100 if entry else 0
                in_profit = stop > entry
                stop_status = "💚 ברווח" if in_profit else "❤️ בהפסד"
                lines.append(
                    f"\n{'🟢' if cur >= entry else '🔴'} <b>{tk}</b>\n"
                    f"   📍 עכשיו:    {_fmt_price(cur)}\n"
                    f"   📌 כניסה:    {_fmt_price(entry)}\n"
                    f"   🛑 סטופ:     {_fmt_price(stop)}  ({stop_status})\n"
                    f"   📏 מרחק:    <b>{dist_from_cur:.1f}%</b> מהמחיר\n"
                    f"   🏆 שיא:      {_fmt_price(wm)}"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/stop] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /compare AAPL MSFT — compare two stocks
    if cmd in ("/compare", "compare", "השווה") and len(t.split()) >= 3:
        parts = t.split()
        t1 = _safe_ticker(parts[1]) or ""
        t2 = _safe_ticker(parts[2]) or ""
        if not t1 or not t2:
            return "❌ טיקר לא חוקי — דוגמה: /compare AAPL MSFT"
        try:
            from scoring import get_composite_score
            from sentiment import score_sentiment
            import yfinance as _yf
            sent1 = score_sentiment(t1)
            sent2 = score_sentiment(t2)
            r1    = get_composite_score(t1, sent1.score)
            r2    = get_composite_score(t2, sent2.score)
            s1, s2 = r1["composite_score"], r2["composite_score"]
            tech1, tech2 = r1["scores"]["technicals"], r2["scores"]["technicals"]
            mkt1,  mkt2  = r1["scores"]["market"],     r2["scores"]["market"]
            winner = t1 if s1 >= s2 else t2
            diff   = abs(s1 - s2)
            # Quick price change
            def _day_chg(sym):
                try:
                    fi = _yf.Ticker(sym).fast_info
                    cur  = float(getattr(fi, "last_price", 0) or 0)
                    prev = float(getattr(fi, "previous_close", cur) or cur)
                    return (cur - prev) / prev * 100 if prev else 0
                except Exception:
                    return 0
            chg1 = _day_chg(t1)
            chg2 = _day_chg(t2)
            b1 = "✅ קנה" if r1["should_buy"] else "❌ דלג"
            b2 = "✅ קנה" if r2["should_buy"] else "❌ דלג"
            return (
                f"⚔️ <b>{t1}  vs  {t2}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>{t1}</b>\n"
                f"  ⭐  ציון:  {s1:.0f}/100  <code>{_score_bar(s1, 8)}</code>\n"
                f"  🔧  טכני:  {tech1:.0f}  |  🌍 שוק: {mkt1:.0f}  |  🧠 {sent1.score}/10\n"
                f"  📅  יום:   {chg1:+.2f}%  |  {b1}\n\n"
                f"<b>{t2}</b>\n"
                f"  ⭐  ציון:  {s2:.0f}/100  <code>{_score_bar(s2, 8)}</code>\n"
                f"  🔧  טכני:  {tech2:.0f}  |  🌍 שוק: {mkt2:.0f}  |  🧠 {sent2.score}/10\n"
                f"  📅  יום:   {chg2:+.2f}%  |  {b2}\n\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🏆 <b>עדיף: {winner}</b>  (פער {diff:.0f} נקודות)"
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /correlation — portfolio correlation
    if cmd in ("/correlation", "correlation", "קורלציה", "מתאם"):
        try:
            import requests as _req, os as _os
            base   = _os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
            secret = settings.WEBHOOK_SECRET
            if not base:
                return "⚙️ RENDER_EXTERNAL_URL לא מוגדר"
            r = _req.get(f"{base}/correlation",
                         headers={"X-Webhook-Secret": secret}, timeout=15)
            data = r.json()
            matrix = data.get("matrix", {})
            max_corr = data.get("max_correlation", 0)
            max_pair = data.get("max_pair", [])
            if not matrix:
                return "📊 צריך לפחות 2 פוזיציות לחישוב קורלציה"
            lines = [f"📊 <b>קורלציה בתיק</b>\n━━━━━━━━━━━━━━━━"]
            for t_a, row in matrix.items():
                for t_b, corr in row.items():
                    if t_a < t_b:
                        icon = "🔴" if abs(corr) > 0.7 else ("🟡" if abs(corr) > 0.4 else "🟢")
                        lines.append(f"{icon} {t_a}↔{t_b}: {corr:.2f}")
            if max_pair:
                lines.append(f"━━━━━━━━━━━━━━━━\n⚠️ הכי מתואמות: {max_pair[0]}↔{max_pair[1]} ({max_corr:.2f})")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /next — next market open + weekly schedule
    if cmd in ("/next", "next", "מתי שוק", "מתי נפתח", "פתיחה"):
        try:
            import broker as _br
            from datetime import datetime, timezone, timedelta
            clock    = _br.get_clock()
            now_utc  = datetime.now(timezone.utc)
            is_edt   = 3 <= now_utc.month <= 10
            il_off   = timedelta(hours=3 if is_edt else 2)
            open_il  = "16:30" if is_edt else "15:30"
            close_il = "23:00" if is_edt else "22:00"
            now_il   = now_utc + il_off

            if clock and clock.get("is_open"):
                next_close = clock.get("next_close", "")
                mins_left  = 0
                if next_close:
                    dt_close  = datetime.fromisoformat(str(next_close).replace("Z", "+00:00"))
                    mins_left = int((dt_close - now_utc).total_seconds() / 60)
                h, m = divmod(mins_left, 60)
                header = (
                    f"🟢 <b>השוק פתוח עכשיו!</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"🕐  שעה: {now_il.strftime('%H:%M')} ישראל\n"
                    f"⏳  נסגר בעוד: <b>{h}ש' {m}ד'</b>  ({close_il})"
                )
            else:
                next_open = clock.get("next_open", "") if clock else ""
                if next_open:
                    dt_open = datetime.fromisoformat(str(next_open).replace("Z", "+00:00"))
                    dt_il   = dt_open + il_off
                    mins    = int((dt_open - now_utc).total_seconds() / 60)
                    h, m    = divmod(mins, 60)
                    header  = (
                        f"🔴 <b>השוק סגור</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"🕐  עכשיו: {now_il.strftime('%H:%M')} ישראל\n"
                        f"🟢  פתיחה: <b>{dt_il.strftime('%H:%M')}</b>  בעוד {h}ש' {m}ד'"
                    )
                else:
                    header = f"🔴 <b>השוק סגור</b>"

            # Weekly schedule
            day_names = {0:"שני",1:"שלישי",2:"רביעי",3:"חמישי",4:"שישי",5:"שבת",6:"ראשון"}
            sched = [f"\n<b>לוח שבועי (ישראל):</b>"]
            for offset in range(7):
                d     = now_utc + timedelta(days=offset)
                wday  = d.weekday()
                name  = day_names.get(wday, "")
                d_str = d.strftime("%d/%m")
                if wday < 5:   # Mon-Fri
                    is_today = (d.date() == now_utc.date())
                    marker   = " ◀" if is_today else ""
                    sched.append(f"  📅 {name} {d_str}:  {open_il}–{close_il}{marker}")
                else:
                    sched.append(f"  🛌 {name} {d_str}:  סגור")

            return header + "\n" + "\n".join(sched)
        except Exception as e:
            logger.error(f"[/next] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /portfolio — full allocation breakdown with visual bars
    if cmd in ("/portfolio", "portfolio", "הקצאה", "פיזור"):
        positions = context.get("open_positions", [])
        equity    = context.get("equity", 1) or 1
        cash      = context.get("cash", 0)
        if not positions and cash == 0:
            return "📭 התיק ריק כרגע"
        lines = [f"📊 <b>הקצאת תיק</b>\n━━━━━━━━━━━━━━━━"]
        # Cash row
        cash_pct = cash / equity * 100 if equity else 0
        cash_bar = "█" * max(1, round(cash_pct / 5)) + "░" * max(0, 20 - round(cash_pct / 5))
        lines.append(f"💵 <b>מזומן</b>  {cash_pct:.1f}%\n   <code>{cash_bar}</code>  {_fmt_price(cash)}")
        # Positions sorted by allocation
        sorted_pos = sorted(positions, key=lambda p: p.get("value", 0), reverse=True)
        for p in sorted_pos:
            val  = p.get("value", p["qty"] * p["current"])
            pct  = val / equity * 100 if equity else 0
            bar  = "█" * max(1, round(pct / 5)) + "░" * max(0, 20 - round(pct / 5))
            icon = "🟢" if p["pnl"] >= 0 else "🔴"
            lines.append(
                f"\n{icon} <b>{p['ticker']}</b>  {pct:.1f}%  ({p['pct']:+.1f}%)\n"
                f"   <code>{bar}</code>  {_fmt_price(val)}"
            )
        lines.append(f"\n━━━━━━━━━━━━━━━━\n💎 <b>סה״כ:</b>  {_fmt_price(equity)}")
        return "\n".join(lines)

    # /summary — weekly performance summary with mini-chart
    if cmd in ("/summary", "summary", "סיכום שבועי", "שבוע"):
        try:
            import database as _db
            from datetime import datetime, timezone, timedelta
            now_utc  = datetime.now(timezone.utc)
            week_ago = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
            trades   = _db.get_trade_history(limit=150) or []
            weekly   = [t for t in trades
                        if str(t.get("exit_time",""))[:10] >= week_ago
                        and t.get("pnl_gross") is not None]
            if not weekly:
                return f"📅 <b>סיכום 7 ימים</b>\n━━━━━━━━━━━━━━━━\n😴 לא היו עסקאות השבוע"

            wins      = [t for t in weekly if float(t.get("pnl_gross") or 0) > 0]
            total_pnl = sum(float(t.get("pnl_gross") or 0) for t in weekly)
            wr        = round(len(wins) / len(weekly) * 100, 1) if weekly else 0

            # Daily P&L mini-chart (last 7 days)
            daily_pnl: dict[str, float] = {}
            for tr in weekly:
                day = str(tr.get("exit_time",""))[:10]
                daily_pnl[day] = daily_pnl.get(day, 0) + float(tr.get("pnl_gross") or 0)
            days_sorted = sorted(daily_pnl.keys())[-7:]
            bars = ["▁","▂","▃","▄","▅","▆","▇","█"]
            pnls = [daily_pnl[d] for d in days_sorted]
            mn, mx = min(pnls), max(pnls)
            rng = mx - mn or 1
            chart = "".join(bars[round((v - mn) / rng * 7)] for v in pnls)
            chart_line = f"\n<code>P&L: {chart}</code>  ({len(days_sorted)} ימים)"

            # Per-day breakdown
            day_lines = []
            for day in days_sorted[-5:]:
                p = daily_pnl[day]
                icon = "🟢" if p >= 0 else "🔴"
                day_lines.append(f"  {icon} {day[5:]}:  ${p:+.2f}")

            return (
                f"📅 <b>סיכום 7 ימים</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🔢  עסקאות: <b>{len(weekly)}</b>  (✅{len(wins)} ❌{len(weekly)-len(wins)})\n"
                f"🎯  Win Rate: <b>{wr}%</b>\n"
                f"💰  רווח שבועי: {_fmt_pnl(total_pnl)}"
                f"{chart_line}\n\n"
                f"<b>יומי:</b>\n" + "\n".join(day_lines)
            )
        except Exception as e:
            logger.error(f"[/summary] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /best — best performing position ever
    if cmd in ("/best", "best", "הכי טוב", "הצלחה"):
        try:
            import database as _db
            wins = _db.get_win_trades(limit=50)
            if not wins:
                return "📭 אין עסקאות רווחיות עדיין"
            best = max(wins, key=lambda t: t.get("pnl_gross") or 0)
            pnl = best.get("pnl_gross", 0)
            return (
                f"🏆 <b>העסקה הטובה ביותר</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 מניה: <b>{best['ticker']}</b>\n"
                f"💵 כניסה: {_fmt_price(best['entry_price'])}\n"
                f"💵 יציאה: {_fmt_price(best.get('exit_price', 0))}\n"
                f"💰 רווח: <b>{_fmt_pnl(pnl)}</b>\n"
                f"📅 תאריך: {str(best.get('exit_time', ''))[:10]}"
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /worst — worst losing trade
    if cmd in ("/worst", "worst", "הכי גרוע", "הפסד גדול"):
        try:
            import database as _db
            losses = _db.get_loss_trades(limit=50)
            if not losses:
                return "✅ אין עסקאות מפסידות עדיין!"
            worst = min(losses, key=lambda t: t.get("pnl_gross") or 0)
            pnl = worst.get("pnl_gross", 0)
            return (
                f"📉 <b>העסקה הגרועה ביותר</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 מניה: <b>{worst['ticker']}</b>\n"
                f"💵 כניסה: {_fmt_price(worst['entry_price'])}\n"
                f"💵 יציאה: {_fmt_price(worst.get('exit_price', 0))}\n"
                f"💸 הפסד: <b>{_fmt_pnl(pnl)}</b>\n"
                f"📅 תאריך: {str(worst.get('exit_time', ''))[:10]}"
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /uptime — bot running time
    if cmd in ("/uptime", "uptime", "כמה זמן רץ", "זמן פעילות"):
        try:
            import time as _t
            from main import START_TIME as _st
            secs = int(_t.time() - _st)
            h, rem = divmod(secs, 3600)
            m = rem // 60
            import os as _os
            paused = bool(_os.getenv("BOT_PAUSED"))
            status_str = "⏸️ מושהה" if paused else "✅ פעיל וסורק"
            return (
                f"🤖 <b>זמן פעילות הבוט</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⏱  פעיל: <b>{h} שעות ו-{m} דקות</b>\n"
                f"🔄  מצב: {status_str}"
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /taxes — tax summary
    if cmd in ("/taxes", "taxes", "מס", "מיסים"):
        try:
            import database as _db
            tax = _db.get_tax_summary() or {}
            reserved = float(tax.get("tax_reserved") or 0)
            credit   = float(tax.get("tax_credit") or 0)
            net      = max(reserved - credit, 0)
            gross    = float(tax.get("realized_pnl_gross") or 0)
            lines    = [f"🧾 <b>סיכום מס</b>\n━━━━━━━━━━━━━━━━"]
            lines.append(f"💵  רווח ממומש:   {_fmt_price(gross)}")
            lines.append(f"🧾  מס שהופרש:   {_fmt_price(reserved)}")
            if credit > 0:
                lines.append(f"🎁  זיכוי מס:      {_fmt_price(credit)}")
            lines.append(f"💳  חוב מס נטו:  <b>{_fmt_price(net)}</b>")
            if gross == 0:
                lines.append("\n💡 עדיין לא ממשת רווחים — אין חבות מס")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/taxes] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /risk — portfolio risk analysis
    if cmd in ("/risk", "risk", "סיכון", "ניתוח סיכון"):
        positions = context.get("open_positions", [])
        equity    = context.get("equity", 1) or 1
        vix       = context.get("vix") or 20
        if not positions:
            return "✅ אין פוזיציות פתוחות — אפס סיכון"
        lines = [f"⚠️ <b>ניתוח סיכון</b>\n━━━━━━━━━━━━━━━━"]
        total_at_risk = 0.0
        total_worst   = 0.0
        for p in positions:
            stop  = p.get("atr_stop") or 0
            entry = p.get("entry", 0)
            cur   = p.get("current", entry)
            qty   = p.get("qty", 0)
            val   = p.get("value", cur * qty)
            # Risk to stop
            if stop and entry:
                at_risk  = (cur - stop) * qty
                risk_pct = at_risk / equity * 100 if equity else 0
                total_at_risk += max(at_risk, 0)
            else:
                at_risk  = val * 0.05   # assume 5% risk if no stop
                risk_pct = at_risk / equity * 100
            # Worst-case: -20% from current (market crash scenario)
            worst    = val * 0.20
            total_worst += worst
            icon = "🟢" if p["pnl"] >= 0 else "🔴"
            lines.append(
                f"{icon} <b>{p['ticker']}</b>\n"
                f"   🛑 סיכון לסטופ:  <b>{_fmt_price(at_risk)}</b>  ({risk_pct:.1f}%)\n"
                f"   💥 תרחיש קיצון: {_fmt_price(worst)}"
            )
        lines.append(f"\n━━━━━━━━━━━━━━━━")
        total_risk_pct  = total_at_risk / equity * 100
        total_worst_pct = total_worst   / equity * 100
        risk_icon = "🟢" if total_risk_pct < 5 else ("🟡" if total_risk_pct < 10 else "🔴")
        lines.append(f"{risk_icon} סיכון לסטופ:  <b>{_fmt_price(total_at_risk)} ({total_risk_pct:.1f}%)</b>")
        lines.append(f"💥 תרחיש -20%:  <b>{_fmt_price(total_worst)} ({total_worst_pct:.1f}%)</b>")
        # VIX context
        if vix >= 28:
            lines.append(f"\n🚨 VIX={vix:.1f} — שוק תנודתי, הסיכון גבוה מהרגיל!")
        elif vix >= 20:
            lines.append(f"\n⚠️ VIX={vix:.1f} — תנודתיות מוגברת")
        else:
            lines.append(f"\n✅ VIX={vix:.1f} — שוק רגוע")
        return "\n".join(lines)

    # /sector TICKER
    if cmd in ("/sector", "sector", "סקטור") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /sector AAPL"
        try:
            from sector_rotation import get_sector_for_ticker, SECTOR_ETFS, get_leading_sectors
            etf = get_sector_for_ticker(_ticker)
            if not etf:
                return f"❌ לא מצאתי סקטור עבור <b>{_ticker}</b>"
            name = SECTOR_ETFS.get(etf, etf)
            sectors = get_leading_sectors()
            rank = next((s["rank"] for s in sectors if s["etf"] == etf), "?")
            ret = next((s["return_pct"] for s in sectors if s["etf"] == etf), 0)
            return (
                f"📊 <b>סקטור — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🏢 סקטור: <b>{name}</b> ({etf})\n"
                f"🏆 דירוג: <b>#{rank}</b> מתוך 11\n"
                f"📈 מומנטום 20 יום: <b>{ret:+.1f}%</b>"
            )
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /watchlist — show watchlist grouped by sector
    if cmd in ("/watchlist", "watchlist", "רשימה", "מניות לסריקה"):
        try:
            from scanner import get_watchlist
            from sector_rotation import get_sector_for_ticker, SECTOR_ETFS
            wl    = get_watchlist()
            total = len(wl)
            # Group by sector
            sectors_map: dict[str, list[str]] = {}
            no_sector = []
            for tk in wl:
                etf = get_sector_for_ticker(tk)
                if etf:
                    name = SECTOR_ETFS.get(etf, etf)
                    sectors_map.setdefault(name, []).append(tk)
                else:
                    no_sector.append(tk)
            lines = [
                f"👁️ <b>רשימת הסריקה</b>\n━━━━━━━━━━━━━━━━",
                f"📊 סה״כ: <b>{total} מניות</b>  |  סריקה כל 5 דקות",
                "",
            ]
            for sec_name, tickers in sorted(sectors_map.items(), key=lambda x: -len(x[1])):
                lines.append(f"<b>{sec_name}:</b>  {' · '.join(tickers[:8])}")
            if no_sector:
                lines.append(f"<b>אחר:</b>  {' · '.join(no_sector[:8])}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /vix — VIX level
    if cmd in ("/vix", "vix"):
        try:
            from indicators import get_vix
            vix = get_vix()
            if vix is None:
                return "❌ לא הצלחתי לקבל VIX"
            if vix < 15:   label = "😌 רגיעה מוחלטת"
            elif vix < 20: label = "🙂 שוק רגוע"
            elif vix < 25: label = "😐 מעט מוגבר"
            elif vix < 30: label = "😟 פחד"
            else:          label = "😱 פחד קיצוני"
            return f"🌡️ <b>VIX — מדד הפחד</b>\n━━━━━━━━━━━━━━━━\n📊 רמה: <b>{vix:.1f}</b>\n💭 מצב: {label}"
        except Exception as e:
            logger.error(f"[CHAT CMD] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /budget — budget settings + capacity
    if cmd in ("/budget", "budget", "תקציב", "הגדרות"):
        try:
            cash     = context.get("cash", 0)
            equity   = context.get("equity", 0) or settings.MAX_BUDGET
            n_open   = context.get("open_positions_count", 0)
            max_pos  = settings.MAX_OPEN_POSITIONS
            capacity = max_pos - n_open
            # Kelly info
            kelly_line = ""
            try:
                from budget import _kelly_fraction
                kf = _kelly_fraction()
                if kf and kf > 0:
                    kelly_line = f"\n📐  Kelly Fraction:  <b>{kf*100:.1f}%</b>"
            except Exception:
                pass
            # How much per new position
            pos_budget = equity * settings.MAX_POSITION_PCT / 100
            return (
                f"⚙️ <b>הגדרות הבוט</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰  תקציב:              ${settings.MAX_BUDGET:,.0f}\n"
                f"💵  מזומן פנוי:         {_fmt_price(cash)}\n"
                f"📂  פוזיציות:           <b>{n_open}/{max_pos}</b>  "
                f"({'✅ יש מקום ל-' + str(capacity) if capacity > 0 else '⛔ מלא'})\n"
                f"📏  גודל פוזיציה מקס:  {settings.MAX_POSITION_PCT}%  (~{_fmt_price(pos_budget)})\n"
                f"🛑  Stop Loss:          {settings.STOP_LOSS_PCT}%\n"
                f"🎯  יעד רווח:           {settings.TAKE_PROFIT_PCT}%"
                f"{kelly_line}\n"
                f"🤖  ברוקר:             {settings.ACTIVE_BROKER}"
            )
        except Exception as e:
            logger.error(f"[/budget] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /newscheck — force immediate news sentiment check on all open positions
    if cmd in ("/newscheck", "newscheck", "בדוק חדשות", "חדשות חדשות"):
        try:
            import database as _db
            open_trades = _db.get_open_trades() or []
            if not open_trades:
                return "📭 אין פוזיציות פתוחות לבדיקה"
            from sentiment import score_sentiment_live as _sl
            lines = [f"📰 <b>בדיקת חדשות בזמן אמת</b>\n━━━━━━━━━━━━━━━━"]
            for tr in open_trades:
                tk = tr.get("ticker", "")
                if not tk:
                    continue
                try:
                    sent = _sl(tk)
                    sc   = sent.score
                    if sc <= 3:   icon, label = "🔴", "שלילי מאוד"
                    elif sc <= 4: icon, label = "🟠", "שלילי"
                    elif sc <= 6: icon, label = "⚪", "ניטרלי"
                    elif sc <= 8: icon, label = "🟡", "חיובי"
                    else:         icon, label = "🟢", "חיובי מאוד"
                    top_hl = sent.headlines[0][:70] if sent.headlines else "אין כותרות"
                    lines.append(
                        f"\n{icon} <b>{tk}</b>  ציון: {sc}/10  ({label})\n"
                        f"   📰 {top_hl}"
                    )
                except Exception:
                    lines.append(f"\n⚪ <b>{tk}</b>: לא הצלחתי לבדוק")
            lines.append(f"\n━━━━━━━━━━━━━━━━\n💡 בדיקה אוטומטית כל 10 דקות")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/newscheck] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /today — what happened today
    if cmd in ("/today", "today", "היום", "מה היה היום"):
        try:
            import database as _db
            from datetime import datetime, timezone
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            all_trades = _db.get_trade_history(limit=100) or []
            opened = [t for t in all_trades if str(t.get("entry_time") or "")[:10] == today_str]
            closed = [t for t in all_trades if str(t.get("exit_time") or "")[:10] == today_str]
            total_pnl = sum(float(t.get("pnl_gross") or 0) for t in closed)
            lines = [f"📅 <b>סיכום היום — {today_str}</b>\n━━━━━━━━━━━━━━━━"]
            lines.append(f"🛒  קניות היום:  <b>{len(opened)}</b>")
            lines.append(f"💸  מכירות היום:  <b>{len(closed)}</b>")
            if closed:
                lines.append(f"💰  רווח/הפסד:  {_fmt_pnl(total_pnl)}")
                # Show each closed trade
                for _ct in closed:
                    _sym  = _ct.get("ticker", "?")
                    _pnl  = float(_ct.get("pnl_gross") or 0)
                    _pct  = float(_ct.get("pnl_pct") or 0)
                    _icon = "🟢" if _pnl >= 0 else "🔴"
                    lines.append(f"  {_icon} <b>{_sym}</b>  {_pct:+.1f}%  |  {_fmt_pnl(_pnl, False)}")
            if opened:
                lines.append(f"\n📂  פתוחות היום:")
                for _ot in opened:
                    lines.append(f"  📌 <b>{_ot.get('ticker','?')}</b>  @ {_fmt_price(_ot.get('entry_price', 0))}")
            if not opened and not closed:
                lines.append("\n😴  לא היו עסקאות היום")
            return "\n".join(lines)
        except Exception as _e:
            logger.error(f"[/today] Error: {_e}")
            return f"📅 <b>סיכום היום</b>\n⚠️ לא הצלחתי לטעון נתוני היום כרגע.\nנסה שוב בעוד רגע."

    # /alert TICKER PRICE — price alert (stored in memory)
    if cmd in ("/alert", "alert", "התראה") and len(t.split()) >= 3:
        parts = t.split()
        _ticker = _safe_ticker(parts[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /alert AAPL 200"
        try:
            _price = float(parts[2])
            import os as _os
            _alerts = _os.environ.get("USER_ALERTS", "")
            _new = f"{_ticker}:{_price}"
            _os.environ["USER_ALERTS"] = (_alerts + "," + _new).strip(",")
            return f"🔔 <b>התראה נוצרה!</b>\nכשמחיר <b>{_ticker}</b> יגיע ל-{_fmt_price(_price)} — תקבל הודעה ✅"
        except Exception:
            return "שימוש: /alert AAPL 200 (טיקר + מחיר יעד)"

    # /winners — show all winning positions
    if cmd in ("/winners", "winners", "מרוויחים", "רווחים"):
        positions = context.get("open_positions", [])
        winners = [p for p in positions if p["pnl"] > 0]
        if not winners:
            return "😔 אין פוזיציות ברווח כרגע"
        lines = [f"🏆 <b>פוזיציות ברווח ({len(winners)})</b>\n━━━━━━━━━━━━━━━━"]
        for p in sorted(winners, key=lambda x: x["pct"], reverse=True):
            lines.append(f"🟢 <b>{p['ticker']}</b>: {p['pct']:+.1f}% | {_fmt_pnl(p['pnl'], False)}")
        return "\n".join(lines)

    # /losers — show losing positions
    if cmd in ("/losers", "losers", "מפסידים", "הפסדים"):
        positions = context.get("open_positions", [])
        losers = [p for p in positions if p["pnl"] < 0]
        if not losers:
            return "✅ אין פוזיציות בהפסד כרגע!"
        lines = [f"📉 <b>פוזיציות בהפסד ({len(losers)})</b>\n━━━━━━━━━━━━━━━━"]
        for p in sorted(losers, key=lambda x: x["pct"]):
            lines.append(f"🔴 <b>{p['ticker']}</b>: {p['pct']:+.1f}% | {_fmt_pnl(p['pnl'], False)}")
        return "\n".join(lines)

    # /scan — trigger immediate buy scan
    if cmd in ("/scan", "scan", "סרוק", "סריקה"):
        try:
            import requests as _req, os as _os
            base = _os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
            secret = settings.WEBHOOK_SECRET
            if not base or not secret:
                return "⚙️ RENDER_EXTERNAL_URL לא מוגדר"
            r = _req.post(f"{base}/scan/now",
                          headers={"X-Webhook-Secret": secret}, timeout=60)
            if r.status_code == 200:
                data = r.json()
                bought = data.get("bought", 0)
                scanned = data.get("scanned", 0)
                cash = data.get("remaining_cash", 0)
                if bought > 0:
                    return (
                        f"🔍 <b>סריקה הושלמה</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"✅  נקנו: <b>{bought} מניות</b>\n"
                        f"🔎  נסרקו: {scanned}\n"
                        f"💵  מזומן נותר: ${cash:,.2f}"
                    )
                else:
                    reason = data.get("reason", "לא נמצאו הזדמנויות")
                    return (
                        f"🔍 <b>סריקה הושלמה</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"⏭️  {reason}\n"
                        f"🔎  נסרקו: {scanned} מניות"
                    )
            else:
                return f"⚠️ סריקה נכשלה (קוד {r.status_code})"
        except Exception as e:
            logger.error(f"[/scan] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /streak — current win/loss streak
    if cmd in ("/streak", "streak", "רצף", "כמה ברצף"):
        try:
            import database as _db
            trades = _db.get_trade_history(limit=30) or []
            closed = [t for t in trades if t.get("status") != "open" and t.get("pnl_gross") is not None]
            if not closed:
                return "📊 <b>רצף</b>\n━━━━━━━━━━━━━━━━\nאין עסקאות סגורות עדיין"
            # Count current streak
            current_streak = 0
            current_type = None
            for tr in closed:
                win = float(tr.get("pnl_gross") or 0) > 0
                if current_type is None:
                    current_type = win
                    current_streak = 1
                elif win == current_type:
                    current_streak += 1
                else:
                    break
            # Overall stats
            wins  = sum(1 for t in closed if float(t.get("pnl_gross") or 0) > 0)
            total = len(closed)
            wr    = round(wins / total * 100, 1) if total else 0
            if current_type:
                streak_line = f"🔥 רצף ניצחונות: <b>{current_streak}</b>"
                streak_icon = "🏆"
            else:
                streak_line = f"❌ רצף הפסדות: <b>{current_streak}</b>"
                streak_icon = "📉"
            # Visual streak dots
            dots = ""
            for tr in reversed(closed[:10]):
                dots += "🟢" if float(tr.get("pnl_gross") or 0) > 0 else "🔴"
            return (
                f"{streak_icon} <b>רצף נוכחי</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"{streak_line}\n\n"
                f"📊 10 עסקאות אחרונות:\n{dots}\n\n"
                f"🎯 Win Rate: <b>{wr}%</b>  ({wins}/{total})"
            )
        except Exception as e:
            logger.error(f"[/streak] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /advice — AI advice on current portfolio
    if cmd in ("/advice", "advice", "ייעוץ", "מה לעשות", "המלצה"):
        try:
            client = _get_client()
            if not client:
                return "⚙️ Groq API לא מוגדר — לא ניתן לקבל ייעוץ AI"
            positions = context.get("open_positions", [])
            cash      = context.get("cash", 0)
            equity    = context.get("equity", 0)
            vix       = context.get("vix", "N/A")
            realized  = context.get("realized_pnl_net", 0)
            pos_text  = ""
            for p in positions:
                pos_text += f"- {p['ticker']}: {p['pct']:+.1f}% ({p['pnl']:+.2f}$)\n"
            prompt = (
                f"אתה יועץ השקעות. ענה בעברית קצרה (עד 120 מילה). "
                f"תיק: equity=${equity:.0f}, cash=${cash:.0f}, VIX={vix}\n"
                f"פוזיציות פתוחות:\n{pos_text or 'אין'}\n"
                f"רווח ממומש: ${realized:+.2f}\n\n"
                f"שאלה: {text}\n\n"
                f"תן המלצה קצרה, ספציפית ומעשית."
            )
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250, temperature=0.4,
            )
            advice = resp.choices[0].message.content.strip()
            return f"🤖 <b>ייעוץ AI</b>\n━━━━━━━━━━━━━━━━\n{advice}"
        except Exception as e:
            logger.error(f"[/advice] Error: {type(e).__name__}")
            return "❌ שגיאה פנימית — נסה שוב"

    if cmd in ("/log", "לוג", "log"):
        return (
            "📋 <b>לוג סריקות אחרונות</b>\n"
            "כדי לראות את הלוגים המלאים:\n"
            "Render → tradebot → <b>Logs</b>"
        )

    # /sell TICKER — force sell a position
    if cmd == "/sell" or (cmd == "מכור" and len(t.split()) > 1):
        parts = t.split()
        ticker_to_sell = _safe_ticker(parts[1]) if len(parts) > 1 else ""
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

    # ── Explicit command aliases (faster path than keyword scan) ──────────
    if cmd in ("/manioth", "/revach", "/shovi", "/mazon"):
        t = cmd[1:]  # strip "/" so keyword scan below picks it up
    if cmd in ("/biztsuim",):
        t = "biztsuim"

    # ── שאלות מניות/פוזיציות ───────────────────────────────────────────────
    stocks_keywords = ["מניות", "מניה", "פוזיציות", "מה יש", "מה קניתי", "מחזיק", "תיק שלי", "איזה", "manioth", "/manioth"]
    if any(k in t for k in stocks_keywords):
        positions = context.get("open_positions", [])
        if not positions:
            return "אין פוזיציות פתוחות כרגע 📭"
        lines = [f"📂 <b>פוזיציות פתוחות ({len(positions)})</b>\n━━━━━━━━━━━━━━━━"]
        total_pnl = 0.0
        for p in positions:
            profit   = p["pnl"] >= 0
            status   = "🟢" if profit else "🔴"
            stop     = p.get("atr_stop") or 0
            held     = p.get("held_hours", 0)
            held_str = _fmt_held(held) if held >= 0.5 else "כמה דקות"
            # Estimate take-profit (~4×ATR above entry, min 3%)
            try:
                tp_pct = max(3.0, (stop / p["entry"] * 100) * 1.5) if stop and p["entry"] else 5.0
                tp_price = round(p["entry"] * (1 + tp_pct / 100), 2)
            except Exception:
                tp_price = 0
            total_pnl += p["pnl"]
            lines.append(
                f"\n{'🟢' if profit else '🔴'} <b>{p['ticker']}</b>\n"
                f"   🔢 כמות:                    {p['qty']} מניות\n"
                f"   📌 מחיר קנייה:         {_fmt_price(p['entry'])}\n"
                f"   📍 מחיר עכשיו:          {_fmt_price(p['current'])} ({p['pct']:+.1f}%)\n"
                f"   📈 יעד רווח:        {_fmt_price(tp_price) if tp_price else 'N/A'}\n"
                f"   📉 סטופ לוס:      {_fmt_price(stop) if stop else 'N/A'}\n"
                f"   ⏳ זמן החזקה:          {held_str}\n"
                f"   {'💚' if profit else '❤️'} {'רווח' if profit else 'הפסד'}:              {_fmt_pnl(p['pnl'], False)}"
            )
        total_icon = "🏆" if total_pnl >= 0 else "📉"
        lines.append(f"\n━━━━━━━━━━━━━━━━\n{total_icon} סה״כ: {_fmt_pnl(total_pnl)}")
        return "\n".join(lines)

    # ── שאלות רווח/הפסד ────────────────────────────────────────────────────
    profit_keywords = ["רווח", "הפסד", "כמה הרווחתי", "כמה הפסדתי", "p&l", "pnl", "revach", "/revach"]
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
    portfolio_keywords = ["שווי", "ערך התיק", "שווה", "תיק", "portfolio", "shovi", "/shovi"]
    if any(k in t for k in portfolio_keywords) and "מניות" not in t:
        cash      = context.get("cash", 0)
        equity    = context.get("equity", 0)
        invested  = context.get("total_invested", 0)
        pnl       = context.get("open_pnl", 0)
        realized  = context.get("realized_pnl_net", 0)
        lines = [
            f"💼 <b>שווי התיק</b>",
            f"━━━━━━━━━━━━━━━━",
            f"📊 סה״כ: <b>{_fmt_price(equity)}</b>",
        ]
        if cash > 0:
            lines.append(f"💰 מזומן: {_fmt_price(cash)}")
        if invested > 0:
            lines.append(f"📈 מניות: {_fmt_price(invested)}")
        if pnl != 0:
            lines.append(f"💹 {_fmt_pnl(pnl)}")
        if realized != 0:
            lines.append(f"🏆 ממומש: {_fmt_pnl(realized)}")
        return "\n".join(lines)

    # ── שאלות מזומן ────────────────────────────────────────────────────────
    cash_keywords = ["כמה כסף", "כמה מזומן", "מזומן", "cash", "mazon", "/mazon"]
    if any(k in t for k in cash_keywords):
        cash = context.get("cash", 0)
        equity = context.get("equity", 0)
        invested = context.get("total_invested", 0)
        pct_invested = round(invested / equity * 100, 1) if equity > 0 else 0
        lines = [f"💵 <b>מזומן פנוי: {_fmt_price(cash)}</b>"]
        if pct_invested > 0:
            lines.append(f"📊 {pct_invested}% מהתיק מושקע")
        if cash == 0:
            lines.append("⚠️ אין מזומן — ממתין למכירה לפני קנייה חדשה")
        return "\n".join(lines)

    # ── שאלות ביצועים ──────────────────────────────────────────────────────
    perf_keywords = ["ביצועים", "סטטיסטיקה", "כמה עסקאות", "win rate", "אחוז הצלחה", "biztsuim", "/biztsuim"]
    if any(k in t for k in perf_keywords):
        try:
            import database as _db
            all_closed = _db.get_trade_history(limit=200) or []
            closed = [x for x in all_closed if x.get("status") != "open" and x.get("pnl_gross") is not None]
            total  = len(closed)
            if total == 0:
                return "📊 <b>ביצועים</b>\n━━━━━━━━━━━━━━━━\n😴 אין עסקאות סגורות עדיין"
            wins   = [x for x in closed if float(x.get("pnl_gross") or 0) > 0]
            losses = [x for x in closed if float(x.get("pnl_gross") or 0) <= 0]
            wr     = round(len(wins) / total * 100, 1)
            total_pnl  = sum(float(x.get("pnl_gross") or 0) for x in closed)
            avg_win    = sum(float(x.get("pnl_gross") or 0) for x in wins) / len(wins) if wins else 0
            avg_loss   = sum(float(x.get("pnl_gross") or 0) for x in losses) / len(losses) if losses else 0
            rr_ratio   = abs(avg_win / avg_loss) if avg_loss else 0
            best_trade = max(closed, key=lambda x: float(x.get("pnl_gross") or 0))
            worst_trade= min(closed, key=lambda x: float(x.get("pnl_gross") or 0))
            # Avg hold time
            hold_times = []
            for x in closed:
                try:
                    from datetime import datetime, timezone as _tz2
                    ed = datetime.strptime(str(x.get("entry_time",""))[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz2.utc)
                    xd = datetime.strptime(str(x.get("exit_time",""))[:19],  "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz2.utc)
                    hold_times.append((xd - ed).total_seconds() / 3600)
                except Exception:
                    pass
            avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0
            hold_str = f"{avg_hold:.1f}ש'" if avg_hold < 24 else f"{avg_hold/24:.1f} ימים"
            # WR bar
            wr_bar = "🟢" * round(wr / 10) + "⚪" * (10 - round(wr / 10))
            return (
                f"📊 <b>ביצועים מלאים</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🔢  עסקאות:         <b>{total}</b>  (✅{len(wins)} ❌{len(losses)})\n"
                f"🎯  Win Rate:       <b>{wr}%</b>\n"
                f"    {wr_bar}\n\n"
                f"💰  רווח כולל:       {_fmt_pnl(total_pnl)}\n"
                f"📈  ממוצע ניצחון:  <b>${avg_win:+.2f}</b>\n"
                f"📉  ממוצע הפסד:   <b>${avg_loss:+.2f}</b>\n"
                f"⚖️   R:R Ratio:      <b>{rr_ratio:.2f}</b>\n"
                f"⏱️   זמן ממוצע:     <b>{hold_str}</b>\n\n"
                f"🏆  הכי טוב:  <b>{best_trade['ticker']}</b>  ${float(best_trade.get('pnl_gross',0)):+.2f}\n"
                f"📉  הכי גרוע: <b>{worst_trade['ticker']}</b>  ${float(worst_trade.get('pnl_gross',0)):+.2f}"
            )
        except Exception as e:
            logger.error(f"[/biztsuim] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # ── Auto-detect ticker in free Hebrew text ─────────────────────────────────
    # e.g. "מה ציון של AAPL?" → run /score AAPL
    # e.g. "מה חדשות על TSLA" → run /news TSLA
    _auto_ticker = _extract_ticker_from_text(text)
    if _auto_ticker:
        _score_kws = ["ציון", "score", "כמה", "כדאי", "לקנות", "לא לקנות", "מה דעתך"]
        _news_kws  = ["חדשות", "news", "כותרות", "מה קורה", "מה היה"]
        _price_kws = ["מחיר", "price", "שווה", "עולה", "יורד"]
        if any(k in t for k in _score_kws):
            # Simulate /score TICKER
            return _handle_command(f"/score {_auto_ticker}", context)
        if any(k in t for k in _news_kws):
            return _handle_command(f"/news {_auto_ticker}", context)
        if any(k in t for k in _price_kws):
            return _handle_command(f"/price {_auto_ticker}", context)

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
        # Run in thread — _handle_command makes blocking DB/broker calls
        reply = await asyncio.to_thread(_handle_command, text, context)

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
