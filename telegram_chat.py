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


def _get_client() -> OpenAI | None:
    global _client
    if _client is None and settings.GROQ_API_KEY:
        _client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


def _build_context() -> dict:
    """Gather live trading context for the LLM."""
    try:
        status = budget.get_budget_status()
        open_trades = database.get_open_trades()
        history = database.get_trade_history(limit=5)
    except Exception as exc:
        logger.warning(f"[CHAT] Failed to build full context: {exc}")
        status, open_trades, history = {}, [], []

    positions_summary = []
    for t in open_trades:
        ticker = t.get("ticker")
        try:
            pos = broker.get_position(ticker)
            if pos:
                cur = float(pos.get("current_price", t["entry_price"]))
                pct = (cur - t["entry_price"]) / t["entry_price"] * 100 if t["entry_price"] else 0
                positions_summary.append({
                    "ticker": ticker,
                    "qty": t["qty"],
                    "entry": round(t["entry_price"], 2),
                    "current": round(cur, 2),
                    "pct": round(pct, 2),
                })
        except Exception:
            pass

    recent = []
    for t in history:
        if t.get("status") and t["status"] != "open":
            recent.append({
                "ticker": t.get("ticker"),
                "pnl": round(t.get("pnl_gross", 0) or 0, 2),
                "status": t.get("status"),
            })

    return {
        "cash": status.get("cash_available", 0),
        "equity": status.get("equity", 0),
        "open_pnl": status.get("open_pnl", 0),
        "realized_pnl_net": status.get("realized_pnl_net", 0),
        "max_budget": status.get("total_budget", 1000),
        "open_positions_count": len(open_trades),
        "open_positions": positions_summary,
        "recent_closed_trades": recent,
        "broker": settings.ACTIVE_BROKER,
    }


def _generate_reply(user_message: str) -> str:
    """Use Groq LLM to generate a Hebrew reply based on the question + bot context."""
    client = _get_client()
    context = _build_context()

    if not client:
        return _fallback_reply(user_message, context)

    positions = context.get("open_positions", [])
    pos_text = ""
    if positions:
        pos_text = "\nפוזיציות פתוחות:\n"
        for p in positions:
            emoji = "🟢" if p["pct"] >= 0 else "🔴"
            pos_text += (
                f"  {emoji} {p['ticker']}: {p['qty']} מניות | "
                f"כניסה ${p['entry']} → עכשיו ${p['current']} | "
                f"שינוי {p['pct']:+.1f}% | "
                f"שווי ${round(p['current'] * p['qty'], 2)}\n"
            )
    else:
        pos_text = "\nאין פוזיציות פתוחות כרגע.\n"

    total_portfolio_value = sum(
        p["current"] * p["qty"] for p in positions
    )

    system_prompt = (
        "אתה עוזר חכם לבוט מסחר אוטומטי. ענה **תמיד בעברית**.\n"
        "תן תשובות מפורטות ומדויקות בהתבסס על נתוני התיק האמיתיים.\n"
        "כשמישהו שואל על מניות — ציין את שם המניה, הכמות, המחיר הנוכחי, השינוי %, והשווי.\n"
        "כשמישהו שואל על שווי — ציין את הסכום בדולרים.\n"
        "כשמישהו שואל על רווח/הפסד — ציין גם רווח פתוח וגם ממומש.\n"
        "תהיה ידידותי וברור. השתמש באימוג'ים מתאימים.\n\n"
        f"📊 נתוני התיק:\n"
        f"💵 מזומן זמין: ${context.get('cash', 0):,.2f}\n"
        f"💼 שווי כל המניות: ${total_portfolio_value:,.2f}\n"
        f"📈 שווי תיק כולל: ${context.get('equity', 0):,.2f}\n"
        f"💰 רווח/הפסד פתוח: ${context.get('open_pnl', 0):+,.2f}\n"
        f"💳 רווח ממומש (נטו): ${context.get('realized_pnl_net', 0):+,.2f}\n"
        f"🔢 מספר פוזיציות: {context.get('open_positions_count', 0)}\n"
        f"{pos_text}"
        f"\nעסקאות אחרונות: {json.dumps(context.get('recent_closed_trades', []), ensure_ascii=False)}"
    )

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=400,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning(f"[CHAT] LLM failed: {exc} — using fallback")
        return _fallback_reply(user_message, context)


def _fallback_reply(user_message: str, ctx: dict) -> str:
    """Smart keyword-matching reply when LLM is unavailable."""
    msg = user_message.lower()
    positions = ctx.get("open_positions", [])
    total_val = sum(p["current"] * p["qty"] for p in positions)

    # ── מצב כללי ──────────────────────────────────────────────────────
    if any(w in msg for w in ["מצב", "סטטוס", "status", "תיק", "סיכום"]):
        lines = [
            "📊 <b>מצב התיק</b>",
            f"💵 מזומן: ${ctx.get('cash', 0):,.2f}",
            f"📈 שווי מניות: ${total_val:,.2f}",
            f"💼 שווי כולל: ${ctx.get('equity', 0):,.2f}",
            f"💰 רווח/הפסד פתוח: ${ctx.get('open_pnl', 0):+.2f}",
            f"💳 רווח ממומש: ${ctx.get('realized_pnl_net', 0):+.2f}",
            f"🔢 פוזיציות פתוחות: {ctx.get('open_positions_count', 0)}",
        ]
        return "\n".join(lines)

    # ── רשימת מניות ───────────────────────────────────────────────────
    if any(w in msg for w in ["מניות", "פוזיציות", "מחזיק", "יש לי", "positions"]):
        if not positions:
            return "📭 אין פוזיציות פתוחות כרגע."
        lines = [f"📈 <b>{len(positions)} פוזיציות פתוחות:</b>"]
        for p in positions:
            emoji = "🟢" if p["pct"] >= 0 else "🔴"
            val = round(p["current"] * p["qty"], 2)
            lines.append(
                f"{emoji} <b>{p['ticker']}</b>: {p['qty']} מניות | "
                f"${p['current']} ({p['pct']:+.1f}%) | שווי ${val:,.2f}"
            )
        lines.append(f"\n💼 <b>סה״כ שווי מניות: ${total_val:,.2f}</b>")
        return "\n".join(lines)

    # ── שווי ──────────────────────────────────────────────────────────
    if any(w in msg for w in ["שווי", "כמה שווה", "ערך", "value"]):
        lines = [
            "💼 <b>שווי התיק</b>",
            f"📈 שווי מניות: ${total_val:,.2f}",
            f"💵 מזומן: ${ctx.get('cash', 0):,.2f}",
            f"💰 סה״כ: ${ctx.get('equity', 0):,.2f}",
        ]
        if positions:
            lines.append("\n<b>פירוט:</b>")
            for p in positions:
                val = round(p["current"] * p["qty"], 2)
                lines.append(f"  • {p['ticker']}: ${val:,.2f}")
        return "\n".join(lines)

    # ── מזומן / כסף פנוי ──────────────────────────────────────────────
    if any(w in msg for w in ["מזומן", "כסף פנוי", "כמה כסף", "כמה מזומן", "תקציב"]):
        cash = ctx.get("cash", 0)
        budget = ctx.get("max_budget", 1000)
        used_pct = round((1 - cash / budget) * 100, 1) if budget > 0 else 0
        return (
            f"💵 <b>מזומן זמין</b>\n"
            f"💰 פנוי: ${cash:,.2f}\n"
            f"📊 תקציב כולל: ${budget:,.2f}\n"
            f"📈 בשימוש: {used_pct}%"
        )

    # ── הבוט עושה מה ─────────────────────────────────────────────────
    if any(w in msg for w in ["עושה", "מה קורה", "פעיל", "עובד", "רץ", "מה הבוט"]):
        n = ctx.get("open_positions_count", 0)
        return (
            f"🤖 <b>מה הבוט עושה עכשיו?</b>\n"
            f"✅ רץ ומסרק מניות כל 5 דקות\n"
            f"📈 קונה מניות עם ציון 60+\n"
            f"👁️ מנטר {n} פוזיציות פתוחות\n"
            f"📰 קורא חדשות 24/7\n"
            f"🛡️ ATR Trailing Stop פעיל"
        )

    # ── עסקאות אחרונות ───────────────────────────────────────────────
    if any(w in msg for w in ["עסקאות", "קניות", "מכירות", "היסטוריה", "אחרון"]):
        recent = ctx.get("recent_closed_trades", [])
        if not recent:
            return "📭 אין עסקאות אחרונות."
        lines = ["📋 <b>עסקאות אחרונות:</b>"]
        for t in recent[:5]:
            emoji = "✅" if t["pnl"] >= 0 else "❌"
            lines.append(f"{emoji} {t['ticker']}: ${t['pnl']:+.2f} ({t['status']})")
        return "\n".join(lines)

    # ── מניה ספציפית ─────────────────────────────────────────────────
    for p in positions:
        if p["ticker"].upper() in msg.upper():
            val = round(p["current"] * p["qty"], 2)
            pnl = round((p["current"] - p["entry"]) * p["qty"], 2)
            emoji = "🟢" if p["pct"] >= 0 else "🔴"
            return (
                f"{emoji} <b>{p['ticker']}</b>\n"
                f"📦 כמות: {p['qty']} מניות\n"
                f"💵 מחיר כניסה: ${p['entry']}\n"
                f"💰 מחיר עכשיו: ${p['current']}\n"
                f"📊 שינוי: {p['pct']:+.1f}%\n"
                f"💼 שווי: ${val:,.2f}\n"
                f"💹 רווח/הפסד: ${pnl:+.2f}"
            )

    # ── רווח/הפסד ─────────────────────────────────────────────────────
    if any(w in msg for w in ["רווח", "הפסד", "כסף", "pnl", "הרווחתי", "הפסדתי"]):
        lines = [
            "💰 <b>רווח/הפסד</b>",
            f"📊 פתוח (לא ממומש): ${ctx.get('open_pnl', 0):+.2f}",
            f"💳 ממומש (נטו אחרי מס): ${ctx.get('realized_pnl_net', 0):+.2f}",
        ]
        if positions:
            lines.append("\n<b>לפי מניה:</b>")
            for p in positions:
                emoji = "📈" if p["pct"] >= 0 else "📉"
                pnl = round((p["current"] - p["entry"]) * p["qty"], 2)
                lines.append(f"  {emoji} {p['ticker']}: ${pnl:+.2f} ({p['pct']:+.1f}%)")
        return "\n".join(lines)

    # ── מחיר מניה ────────────────────────────────────────────────────
    if any(w in msg for w in ["מחיר", "price", "כמה עולה", "כמה שווה"]) and len(msg.split()) >= 2:
        words = msg.upper().split()
        tickers_to_check = [w for w in words if 2 <= len(w) <= 6 and w.isalpha() and w not in
                            ["PRICE", "מחיר", "כמה", "עולה", "שווה", "מה", "של", "ה"]]
        if tickers_to_check:
            ticker = tickers_to_check[0]
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                info = t.fast_info
                price = float(getattr(info, "last_price", 0) or getattr(info, "previous_close", 0) or 0)
                if price > 0:
                    return f"💵 <b>{ticker}</b>: ${price:.2f}"
                else:
                    return f"לא מצאתי מחיר ל-{ticker} 🤔"
            except Exception:
                return f"לא מצאתי מחיר ל-{ticker} 🤔"

    # ── הפסד מקסימלי יומי ────────────────────────────────────────────
    if any(w in msg for w in ["circuit", "breaker", "מגבלה", "הפסד יומי", "עצירה"]):
        from circuit_breaker import get_status as _cb_status
        cb = _cb_status()
        tripped = cb.get("tripped", False)
        emoji = "🔴" if tripped else "🟢"
        return (
            f"{emoji} <b>Circuit Breaker</b>\n"
            f"{'🚨 פעיל — אין קניות!' if tripped else '✅ תקין — הבוט קונה'}\n"
            f"📉 הפסד יומי: ${cb.get('daily_pnl', 0):+.2f}\n"
            f"🚫 מגבלה: ${cb.get('max_daily_loss', 0):.2f}"
        )

    # ── ביצועים שבועיים ──────────────────────────────────────────────
    if any(w in msg for w in ["שבוע", "שבועי", "ביצועים", "סטטיסטיקות"]):
        import database as _db
        trades = _db.get_trade_history(limit=50)
        closed = [t for t in trades if t.get("status") and t["status"] != "open" and t.get("pnl_gross")]
        if not closed:
            return "📊 אין עסקאות סגורות עדיין."
        wins = [t for t in closed if (t.get("pnl_gross") or 0) > 0]
        total_pnl = sum(t.get("pnl_gross") or 0 for t in closed)
        win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0
        best = max(closed, key=lambda t: t.get("pnl_gross") or 0)
        worst = min(closed, key=lambda t: t.get("pnl_gross") or 0)
        return (
            f"📊 <b>ביצועים</b>\n"
            f"🔢 עסקאות: {len(closed)}\n"
            f"🎯 אחוז הצלחה: {win_rate}%\n"
            f"💰 רווח כולל: ${total_pnl:+.2f}\n"
            f"🏆 הטובה: {best.get('ticker')} ${(best.get('pnl_gross') or 0):+.2f}\n"
            f"💀 הגרועה: {worst.get('ticker')} ${(worst.get('pnl_gross') or 0):+.2f}"
        )

    # ── כמה פוזיציות נוספות אפשר לפתוח ─────────────────────────────
    if any(w in msg for w in ["כמה אפשר", "כמה עוד", "מקום", "פנוי"]):
        from config import settings as _s
        open_count = ctx.get("open_positions_count", 0)
        max_pos = _s.MAX_OPEN_POSITIONS
        remaining = max_pos - open_count
        return (
            f"📊 <b>קיבולת תיק</b>\n"
            f"📈 פוזיציות פתוחות: {open_count}/{max_pos}\n"
            f"✅ אפשר לפתוח עוד: {remaining} פוזיציות"
        )

    # ── ציון קנייה ──────────────────────────────────────────────────
    if any(w in msg for w in ["ציון", "score", "סף", "60", "קנייה"]):
        from scoring import MIN_BUY_SCORE
        return (
            f"🎯 <b>ציון קנייה</b>\n"
            f"הבוט קונה מניות עם ציון <b>{MIN_BUY_SCORE}+</b> מתוך 100\n"
            f"הציון מורכב מ:\n"
            f"  • 60% טכני (RSI, MACD, Bollinger)\n"
            f"  • 20% שוק (VIX, SPY)\n"
            f"  • 20% סנטימנט (חדשות AI)"
        )

    # ── האם השוק פתוח ───────────────────────────────────────────────
    if any(w in msg for w in ["שוק", "פתוח", "סגור", "שעות", "market"]):
        import broker as _broker
        try:
            is_open = _broker.is_market_open()
            status_text = "🟢 <b>השוק פתוח!</b>" if is_open else "🔴 <b>השוק סגור</b>"
            return (
                f"{status_text}\n"
                f"⏰ שעות מסחר: 16:30-23:00 שעון ישראל\n"
                f"📅 ימי מסחר: ב׳-ו׳"
            )
        except Exception:
            return "לא הצלחתי לבדוק את שעות השוק 🤔"

    # ── הגדרות הבוט ─────────────────────────────────────────────────
    if any(w in msg for w in ["הגדרות", "settings", "stop loss", "take profit", "אחוז"]):
        from config import settings as _s
        return (
            f"⚙️ <b>הגדרות הבוט</b>\n"
            f"🛑 Stop Loss: {_s.STOP_LOSS_PCT}%\n"
            f"✅ Take Profit: {_s.TAKE_PROFIT_PCT}%\n"
            f"🎯 ציון מינימום: {_s.MAX_OPEN_POSITIONS} פוזיציות מקס'\n"
            f"💰 תקציב: ${_s.MAX_BUDGET:,.2f}\n"
            f"📊 גודל פוזיציה מקסימלי: {_s.MAX_POSITION_PCT}%"
        )

    # ── ברכה ──────────────────────────────────────────────────────────
    if any(w in msg for w in ["שלום", "היי", "הי", "hello", "hi", "בוקר", "ערב"]):
        return (
            "שלום! 👋 אני הבוט שלך.\n\n"
            "<b>שאלות שאפשר לשאול:</b>\n"
            "• מה המצב? — סיכום התיק\n"
            "• אילו מניות יש לי? — כל הפוזיציות\n"
            "• כמה שווה התיק? — שווי כולל\n"
            "• כמה הרווחתי? — רווח/הפסד\n"
            "• כמה מזומן יש לי?\n"
            "• מה הבוט עושה עכשיו?\n"
            "• עסקאות אחרונות?\n"
            "• AAPL (שם מניה) — פרטים על מניה ספציפית"
        )

    # ── עזרה ──────────────────────────────────────────────────────────
    if any(w in msg for w in ["עזרה", "help", "מה אפשר", "פקודות", "תפריט"]):
        return (
            "📋 <b>כל מה שאפשר לשאול:</b>\n\n"
            "💼 <b>תיק:</b>\n"
            "  • מה המצב?\n"
            "  • כמה שווה התיק?\n"
            "  • כמה הרווחתי?\n"
            "  • כמה מזומן יש לי?\n"
            "  • כמה פוזיציות אפשר לפתוח?\n\n"
            "📈 <b>מניות:</b>\n"
            "  • אילו מניות יש לי?\n"
            "  • AAPL (שם מניה — פרטים)\n"
            "  • מחיר TSLA (מחיר עכשיו)\n\n"
            "📊 <b>ביצועים:</b>\n"
            "  • ביצועים שבועיים\n"
            "  • עסקאות אחרונות?\n\n"
            "🤖 <b>בוט:</b>\n"
            "  • מה הבוט עושה עכשיו?\n"
            "  • האם השוק פתוח?\n"
            "  • הגדרות הבוט\n"
            "  • ציון קנייה\n"
            "  • circuit breaker"
        )

    # ── ברירת מחדל ────────────────────────────────────────────────────
    return (
        "🤔 לא הבנתי.\n\n"
        "כתוב <b>עזרה</b> לרשימת כל הפקודות."
    )


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

    # Generate reply
    try:
        reply = _generate_reply(text)
    except Exception as exc:
        logger.error(f"[CHAT] Reply generation failed: {exc}")
        reply = "מצטער, נתקלתי בשגיאה בעיבוד השאלה. נסה שוב."

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
