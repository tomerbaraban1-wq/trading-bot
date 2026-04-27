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
        # Fallback: simple keyword-based answers in Hebrew
        return _fallback_reply(user_message, context)

    system_prompt = (
        "אתה עוזר אישי לבוט מסחר אוטומטי. ענה תמיד בעברית בקצרה (עד 3 שורות).\n"
        "ענה רק על שאלות שקשורות לבוט, למסחר או למצב התיק.\n"
        "אם המשתמש מנסה לשאול אותך משהו אחר — תזכיר שאתה עוזר לבוט מסחר.\n"
        "תהיה ידידותי ועניני. אל תשתמש באימוג'ים מוגזמים — מקסימום 1-2 לכל תשובה.\n"
        "השתמש במידע על התיק כדי לענות על שאלות כמו 'מה המצב?', 'כמה רווח יש לי?',\n"
        "'איזה מניות יש לי?', 'מה הבוט עושה עכשיו?' וכו'.\n\n"
        f"מצב הבוט עכשיו (JSON):\n{json.dumps(context, ensure_ascii=False, indent=2)}"
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
    """Simple keyword-matching reply when LLM is unavailable."""
    msg = user_message.lower()

    if any(w in msg for w in ["מצב", "סטטוס", "status", "כמה", "תיק"]):
        return (
            f"📊 מצב התיק:\n"
            f"💵 מזומן: ${ctx.get('cash', 0):,.2f}\n"
            f"💼 שווי תיק: ${ctx.get('equity', 0):,.2f}\n"
            f"📈 פוזיציות פתוחות: {ctx.get('open_positions_count', 0)}\n"
            f"💰 רווח/הפסד פתוח: ${ctx.get('open_pnl', 0):+.2f}"
        )

    if any(w in msg for w in ["פוזיציות", "מניות", "positions"]):
        positions = ctx.get("open_positions", [])
        if not positions:
            return "אין פוזיציות פתוחות כרגע 📭"
        lines = [f"📈 {len(positions)} פוזיציות פתוחות:"]
        for p in positions[:6]:
            emoji = "🟢" if p["pct"] >= 0 else "🔴"
            lines.append(f"{emoji} {p['ticker']}: {p['pct']:+.1f}% (${p['current']})")
        return "\n".join(lines)

    if any(w in msg for w in ["רווח", "הפסד", "כסף", "pnl"]):
        return (
            f"💰 רווח/הפסד:\n"
            f"📊 פתוח: ${ctx.get('open_pnl', 0):+.2f}\n"
            f"💳 ממומש (נטו): ${ctx.get('realized_pnl_net', 0):+.2f}"
        )

    if any(w in msg for w in ["שלום", "היי", "הי", "hello", "hi"]):
        return "שלום! 👋 איך אפשר לעזור? תוכל לשאול אותי על מצב התיק, פוזיציות, רווחים וכו'."

    return (
        "מצטער, לא הבנתי 🤔\n"
        "תוכל לשאול אותי דברים כמו:\n"
        "• מה המצב?\n"
        "• אילו פוזיציות יש לי?\n"
        "• כמה רווח יש לי?"
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
