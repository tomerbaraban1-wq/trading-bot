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
    # Known single-letter tickers (V=Visa, C=Citi, F=Ford, X=US Steel, T=AT&T, etc.)
    _SINGLE_LETTER_TICKERS = {"V", "C", "F", "X", "T", "B", "K", "D", "R", "S", "W"}
    # Common abbreviations that should NOT be treated as tickers
    _NOT_TICKERS = {
        "VIX", "RSI", "ATR", "SMA", "EMA", "ETF", "ETH", "BTC", "GDP", "BOT",
        "API", "URL", "CEO", "CFO", "IPO", "SEC", "FED", "CPI", "NFP", "THE",
        "FOR", "ARE", "AND", "BUT", "NOT", "USD", "ILS", "EUR", "GBP",
    }
    words = text.upper().split()
    for w in words:
        # Strip trailing punctuation (AAPL. → AAPL)
        clean = _re.sub(r'[^A-Z0-9\-]', '', _re.sub(r'[.\-]+$', '', w))
        if not clean or not _TICKER_RE.match(clean):
            continue
        if clean in _NOT_TICKERS:
            continue
        # Reject pure numbers
        if clean.isdigit():
            continue
        if len(clean) >= 2:
            return clean
        # Single letter — only accept known tickers
        if clean in _SINGLE_LETTER_TICKERS:
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
_CONTEXT_CACHE_TTL = 300  # 5 minutes — reduces slow yfinance/broker calls

# Conversation memory — remember last N exchanges for follow-up questions
# Keyed by chat_id (str). Each entry: list of (role, content) tuples.
_conversation_history: dict[str, list] = {}
_MAX_HISTORY_TURNS = 5   # remember last 5 user+bot exchanges (10 messages)
_HISTORY_LOCK = __import__('threading').Lock()


def _remember(chat_id: str, role: str, content: str) -> None:
    """Save a message to conversation memory (keeps last 10 messages per chat)."""
    if not chat_id or not content:
        return
    with _HISTORY_LOCK:
        hist = _conversation_history.setdefault(chat_id, [])
        hist.append({"role": role, "content": content[:500]})  # cap each msg
        # Keep last 2*MAX_HISTORY_TURNS messages (5 user + 5 bot)
        if len(hist) > _MAX_HISTORY_TURNS * 2:
            del hist[: len(hist) - _MAX_HISTORY_TURNS * 2]


def _get_history(chat_id: str) -> list:
    """Return previous conversation messages for context."""
    with _HISTORY_LOCK:
        return list(_conversation_history.get(chat_id, []))


def _detect_ticker(text: str) -> str | None:
    """
    זיהוי אם המשתמש שלח רק טיקר (לדוגמה 'NVDA' או 'aapl').
    מחזיר את הטיקר באותיות גדולות, או None אם לא טיקר.
    """
    t = (text or "").strip().upper()
    # 1-5 letters, optionally with . or -
    import re as _re
    if _re.fullmatch(r"[A-Z]{1,5}([.-][A-Z]{1,3})?", t):
        # Exclude common words
        if t in ("OK", "YES", "NO", "HI", "HELP", "YO", "OY"):
            return None
        return t
    return None


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
        # DB calls are fast; broker calls are slow — avoid broker here
        open_trades = database.get_open_trades()
        history     = database.get_trade_history(limit=10)
        # Budget from DB directly (faster than broker API)
        try:
            status = budget.get_budget_status()
        except Exception:
            status = {}
    except Exception as exc:
        logger.warning(f"[CHAT] Failed to build context: {exc}")
        status, open_trades, history = {}, [], []

    # ── Positions: merge DB trade log + live broker positions ─────────
    positions_summary = []

    if open_trades:
        # Batch price fetch via yfinance — much faster than per-ticker broker API calls
        _tickers_list = [t.get("ticker") for t in open_trades if t.get("ticker")]
        _yf_prices: dict = {}
        try:
            import yfinance as _yf_ctx
            if _tickers_list:
                _hist = _yf_ctx.download(_tickers_list, period="1d",
                                         progress=False, auto_adjust=True)
                if not _hist.empty:
                    if len(_tickers_list) == 1:
                        # Single ticker: yf.download returns flat columns → "Close" is a Series
                        if "Close" in _hist.columns:
                            try:
                                _yf_prices[_tickers_list[0]] = float(_hist["Close"].dropna().iloc[-1])
                            except Exception:
                                pass
                    else:
                        # Multiple tickers: MultiIndex columns → "Close" is a DataFrame
                        _cols = _hist.columns.get_level_values(0) if hasattr(_hist.columns, "get_level_values") else _hist.columns
                        if "Close" in _cols:
                            _close = _hist["Close"]
                            for _tk in _tickers_list:
                                try:
                                    _yf_prices[_tk] = float(_close[_tk].dropna().iloc[-1])
                                except Exception:
                                    pass
        except Exception:
            pass

        for t in open_trades:
            ticker = t.get("ticker")
            try:
                # Use yfinance price (fast, cached) — fall back to entry_price
                cur = _yf_prices.get(ticker, t["entry_price"])
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
        # Trade counts (full lifetime stats)
        "trade_counts":         (lambda: __import__('database').get_total_trades_count())(),
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

def _llm_reply(user_message: str, context: dict, history: list | None = None) -> str:
    """Full LLM-powered dynamic reply — analyzes ANY question and provides the most relevant answer.

    If `history` is provided, includes previous exchanges so the LLM can answer
    follow-up questions naturally (e.g. "and what about NVDA?" referring to
    a previous AAPL discussion).
    """
    client = _get_client()
    if not client:
        return _simple_fallback(context)

    # Build detailed position summary — include Buffett quality score
    positions = context.get("open_positions", [])
    pos_lines = []
    for p in positions:
        emoji = "🟢" if p["pct"] >= 0 else "🔴"
        stop = p.get("atr_stop") or 0
        held = p.get("held_hours", 0)
        held_str = _fmt_held(held)
        invested = p.get("invested") or round(p["entry"] * p["qty"], 2)
        # Add Buffett quality (cached - fast)
        _buf_info = ""
        try:
            from buffett_analysis import get_buffett_analysis
            _ba = get_buffett_analysis(p['ticker'])
            _bs = _ba.get("score", 0)
            _moat = _ba.get("moat", "?")
            _buf_info = f"  |  🎩 איכות={_bs:.0f}/100 (moat:{_moat})"
        except Exception:
            pass

        pos_lines.append(
            f"{emoji} <b>{p['ticker']}</b>\n"
            f"   📦 {p['qty']} מניות  |  💵 הושקע {_fmt_price(invested)}{_buf_info}\n"
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

    ils_rate = _get_usd_ils()
    def _ils(usd): return f"${usd:,.2f} (₪{usd*ils_rate:,.0f})"

    system_prompt = f"""⚠️ כלל ברזל: ענה אך ורק בעברית. שמות מניות (AAPL, TSLA) נשארים באנגלית. הצג מחירים גם בדולר וגם בשקל.

אתה מנהל ההשקעות האישי של המשתמש — אנליסט ברמת וורן באפט.
אתה חושב לטווח ארוך, מחפש איכות, מבין את ה-moat (יתרון תחרותי), ולא ממליץ על דברים מבלי לנתח.

🎯 העקרונות שלך (כמו של וורן באפט):
1. <b>איכות לפני מחיר</b> — קונה רק חברות עם ROE>15%, שולי רווח טובים, חוב נמוך
2. <b>Moat קודם</b> — מה היתרון התחרותי? איך החברה מגנה על השוק שלה?
3. <b>טווח ארוך</b> — "התקופה האידיאלית להחזיק מניה היא לנצח"
4. <b>הימנע מהיפ</b> — P/E מעל 40 הוא דגל אדום
5. <b>חברות שאתה מבין</b> — לא קונה משהו שלא מבין מה הם עושים
6. <b>שולי בטחון</b> — קונה כשהמחיר נמוך מהשווי האמיתי
7. <b>תזרים מזומנים</b> — חשוב יותר מרווח חשבונאי

אתה יודע לענות על כל שאלה: על התיק, על מניות ספציפיות, על השוק, על אסטרטגיה, וגם שאלות כלליות על השקעות.

שער דולר/שקל עכשיו: 1$ = ₪{ils_rate:.2f}

══ סטטיסטיקת עסקאות (סך הכל מאז ההתחלה) ══
📊 סה"כ עסקאות:    {context.get('trade_counts', {}).get('total', 0)}
📂 פתוחות:           {context.get('trade_counts', {}).get('open', 0)}
✔️ סגורות:           {context.get('trade_counts', {}).get('closed', 0)}
✅ ברווח:            {context.get('trade_counts', {}).get('wins', 0)}
❌ בהפסד:             {context.get('trade_counts', {}).get('losses', 0)}
📅 היום:             {context.get('trade_counts', {}).get('today', 0)}

══ הגדרות הבוט הנוכחיות (חשוב!) ══
🎯 ציון קנייה מינימלי: {context.get('min_buy_score', 60)}/100
🎩 איכות באפט מינימלית: 50/100 (אחרת לא קונה)
🛡️ הגנה Drawdown: 10% מהשיא ההיסטורי
🚦 Take Profit (TP): 15% | Stop Loss: 3.5%
📉 פילטר VIX: לא קונה אם VIX > 22 (פאניקה)
⏰ שעות סחר: 10:00-15:30 ET (חוץ מ-15 דק' פתיחה/סגירה)
📈 Stage 1 (נעילת רווח): +1.5%-2% — מוכר 50%
📈 Stage 2: +4%-6% — מוכר עוד 50% מהנותר
🛡️ Break-Even: ב-+1% הסטופ עולה לכניסה

══ מצב התיק עכשיו ══
💰 מזומן פנוי:       {_ils(context.get('cash', 0))}
💼 מושקע במניות:   {_ils(context.get('total_invested', 0))}
📊 תיק כולל:          {_ils(context.get('equity', 0))}
📈 רווח/הפסד פתוח: {_ils(context.get('open_pnl', 0))}
💳 רווח ממומש:      {_ils(context.get('realized_pnl_net', 0))}
🔢 פוזיציות:           {context.get('open_positions_count', 0)} פתוחות
🎯 אחוז הצלחה:       {context.get('win_rate', 0)}% ({context.get('total_closed', 0)} עסקאות)
🌡️ VIX:                {context.get('vix') or '—'}
🕐 שוק:               {'🟢 פתוח' if context.get('market_open') else '🔴 סגור'}

══ פוזיציות פתוחות ══
{pos_text}

══ הגדרות הבוט ══
ציון קנייה מינימלי: {context.get('min_buy_score', 60)}/100
Stop Loss: {context.get('stop_loss_pct', 5)}% | יעד רווח: {context.get('take_profit_pct', 15)}%
Circuit Breaker: {'⚠️ פעיל' if context.get('circuit_breaker') else '✅ תקין'}

══ עסקאות אחרונות ══
{closed_text}

══ הנחיות מענה ══
• שאלות על מספר עסקאות (כמה עסקאות עשית?): השתמש בנתון "trade_counts" — ענה במספרים מדויקים
• שאלות על תיק: ענה עם מחירים בדולר + שקל
• שאלות "האם כדאי לקנות X?" — תמיד תחשוב כמו באפט: ROE? Moat? חוב? תמחור? תן המלצה מנומקת
• שאלות "מה קורה עם X?" — נתח: מחיר, מגמה, fundamentals, סיכונים, הזדמנויות
• שאלות על השוק: הסתמך על VIX ומצב השוק למעלה
• שאלות על אסטרטגיה: סורק כל 5 דקות, ציון ≥{context.get('min_buy_score',60)}, ATR stop
• שאלות כלליות: ענה כמנהל השקעות מקצועי שלמד מספריו של באפט
• אם שאלה לא ברורה: בקש הבהרה קצרה
• ענה ממוקד — 4-8 שורות בדרך כלל
• אל תאמר "אני לא יכול" — תמיד נסה לעזור
• המלץ למשתמש להריץ /buffett TICKER לניתוח עמוק של מניה ספציפית
"""

    try:
        # Detect simple vs complex question for token optimization
        _msg_len = len(user_message)
        _is_simple = (
            _msg_len < 40                          # short question
            or user_message.startswith("/")        # command
            or any(w in user_message for w in ["כמה", "מה המחיר", "מה הרווח", "כן", "לא"])
        )
        _max_tokens = 180 if _is_simple else 450
        _temp       = 0.2 if _is_simple else 0.4

        # Build message chain with conversation history (if any)
        messages = [{"role": "system", "content": system_prompt}]
        if history and not _is_simple:
            # Include history only for complex questions (saves tokens on simple ones)
            for h in history[-6:]:   # last 3 turns
                if h.get("role") in ("user", "assistant"):
                    messages.append({"role": h["role"], "content": h.get("content", "")[:300]})
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=_max_tokens,
            temperature=_temp,
        )
        reply = response.choices[0].message.content.strip()

        # Skip translation — system prompt enforces Hebrew. Only translate if >80% English.
        hebrew_chars = sum(1 for c in reply if 'א' <= c <= 'ת')
        latin_chars  = sum(1 for c in reply if c.isalpha() and c.isascii())
        if latin_chars > hebrew_chars * 4 and len(reply) > 30:
            logger.warning("[CHAT] Reply mostly English — translating")
            try:
                tr_resp = client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user",
                                "content": f"תרגם לעברית בלבד:\n{reply}"}],
                    max_tokens=300, temperature=0.1,
                )
                translated = tr_resp.choices[0].message.content.strip()
                if translated:
                    reply = translated
            except Exception:
                reply = _simple_fallback(context)  # fallback if translation fails

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
                tp_pct = max(3.0, ((p["entry"] - stop) / p["entry"] * 100) * 1.5) if stop and p["entry"] else 5.0
                tp_price = round(p["entry"] * (1 + tp_pct / 100), 2)
            except Exception:
                tp_price = 0
            lines.append(
                f"\n{status_icon} <b>{p['ticker']}</b>\n"
                f"   🔢 כמות: {p['qty']} מניות\n"
                f"   📌 מחיר קנייה: {_fmt_price(p['entry'])}\n"
                f"   📈 יעד רווח: {_fmt_price(tp_price) if tp_price else '—'}\n"
                f"   📉 סטופ לוס: {_fmt_price(stop) if stop else '—'}\n"
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
    args = " ".join(t.split()[1:]) if len(t.split()) > 1 else ""

    # ── Advanced commands via telegram_commands.py ────────────────────────
    # Map /command → handler  (lazy loaded to avoid circular import)
    advanced_commands = {
        # English commands
        "/health": "health",
        "/performance": "performance",
        "/news": "news",
        "/risk": "risk",
        "/confluence": "confluence",
        "/forecast": "forecast",
        "/ai_decision": "ai_decision",
        "/ai": "ai_decision",
        "/backtest": "backtest",
        "/bt": "backtest",
        "/doctor": "doctor",
        "/anomalies": "anomalies",
        # Hebrew aliases (use English in commands - Telegram requirement)
        "/briut": "health",         # בריאות
        "/betzuim": "performance",  # ביצועים
        "/hadashot": "news",        # חדשות
        "/sikun": "risk",           # סיכון
        "/tachzit": "forecast",     # תחזית
        "/khlita": "ai_decision",   # החלטה
        "/bdika": "backtest",       # בדיקה
        "/rofeh": "doctor",         # רופא
        "/anomaliot": "anomalies",  # אנומליות
        # Hebrew text shortcuts (when user types Hebrew without /)
        "בריאות": "health",
        "ביצועים": "performance",
        "חדשות": "news",
        "סיכון": "risk",
        "תחזית": "forecast",
        "רופא": "doctor",
        "אנומליות": "anomalies",
        "בדיקה מקיפה": "doctor",
    }

    if cmd in advanced_commands:
        try:
            import asyncio as _asyncio
            from telegram_commands import route_command
            handler_name = advanced_commands[cmd]

            # Run async command in current event loop
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    # Create task and wait for result
                    future = _asyncio.run_coroutine_threadsafe(
                        route_command(handler_name, args),
                        loop
                    )
                    result = future.result(timeout=60)
                    if result:
                        return result
                else:
                    result = loop.run_until_complete(route_command(handler_name, args))
                    if result:
                        return result
            except Exception as e:
                return f"⚠️ שגיאה בפקודה {cmd}: {e}"
        except Exception as e:
            return f"⚠️ {cmd} זמנית לא זמין: {e}"

    # ── /commands ──────────────────────────────────────────────────────────
    if cmd == "/start":
        # Send the persistent keyboard menu
        import asyncio as _asyncio
        try:
            from telegram_bot import send_menu as _send_menu
            _asyncio.ensure_future(_send_menu())
        except Exception:
            pass
        return (
            "👋 <b>שלום! אני מנהל ההשקעות שלך</b>\n"
            "━━━━━━━━━━━━━━━━\n"
            "📱 כפתורי תפריט הופיעו למטה ↓\n\n"
            "💬 <b>אפשר לשאול אותי כל שאלה חופשית!</b>\n"
            "לדוגמה:\n"
            "  • <i>מה קורה עם AAPL?</i>\n"
            "  • <i>האם השוק עולה או יורד?</i>\n"
            "  • <i>כמה הרווחתי החודש?</i>\n"
            "  • <i>מה דעתך על התיק שלי?</i>\n\n"
            "📋 /help — כל הפקודות"
        )

    if cmd in ("/help", "עזרה", "עזר", "פקודות", "מה אתה יכול"):
        return (
            "👋 <b>מנהל ההשקעות שלך</b>\n\n"
            "💬 <b>אפשר לשאול כל שאלה חופשית!</b>\n"
            "לדוגמה: <i>\"מה קורה עם AAPL?\"</i> או <i>\"מה דעתך על התיק שלי?\"</i>\n\n"
            "━━ 📊 <b>תיק ופוזיציות</b> ━━\n"
            "/status — מצב מלא של התיק\n"
            "/manioth — איזה מניות פתוחות\n"
            "/revach — רווח/הפסד פירוט\n"
            "/shovi — שווי התיק\n"
            "/mazon — מזומן פנוי\n"
            "/portfolio — הקצאה באחוזים\n"
            "/winners — פוזיציות ברווח\n"
            "/losers — פוזיציות בהפסד\n"
            "/risk — ניתוח סיכון\n"
            "/health — בריאות כל הפוזיציות\n"
            "/correlation — קורלציה בין פוזיציות\n"
            "/pnl — רווח/הפסד מהיר\n\n"
            "━━ 📈 <b>ניתוח מניות</b> ━━\n"
            "/score AAPL — ניתוח עם גרפי ציון\n"
            "/chart AAPL — גרף ASCII 30 ימים\n"
            "/fundamental AAPL — P/E, הכנסות, מרווחים\n"
            "/buffett AAPL — ניתוח מלא ברמת וורן באפט 🎯\n"
            "/dividend AAPL — דיבידנד ותשואה\n"
            "/52week AAPL — מיקום ב-52 שבועות\n"
            "/price AAPL — מחיר מיידי\n"
            "/news AAPL — חדשות + סנטימנט AI\n"
            "/newscheck — בדיקת חדשות לכל הפוזיציות\n"
            "/earnings AAPL — דוח רווחים\n"
            "/stop AAPL — מצב ה-Stop Loss\n"
            "/target AAPL 210 — הגדר יעד רווח ידני\n"
            "/volume AAPL — ניתוח נפח מסחר\n"
            "/sector AAPL — איזה סקטור\n"
            "/compare AAPL MSFT — השוואה\n\n"
            "━━ 🌍 <b>מצב השוק</b> ━━\n"
            "/market — מצב כללי (SPY/QQQ/DIA)\n"
            "/trending — מניות בתנופה חזקה\n"
            "/gainers — מניות מובילות היום\n"
            "/exposure — חשיפת תיק לסקטורים\n"
            "/volatility AAPL — ATR ו-Beta\n"
            "/sectors — דירוג סקטורים\n"
            "/macro — אירועים כלכליים קרובים\n"
            "/vix — מדד הפחד VIX\n"
            "/fear — פחד וחמדנות\n"
            "/top — מניות עם ציון גבוה\n"
            "/watchlist — רשימת הסריקה\n"
            "/quick AAPL — סקירה מהירה (מחיר+ציון+52W)\n"
            "/position AAPL — פוזיציה מפורטת\n"
            "/watchadd AAPL — הוסף מניה לרשימה\n"
            "/watchremove AAPL — הסר מניה מהרשימה\n"
            "/signals — הזדמנויות קנייה עכשיו\n\n"
            "━━ 📅 <b>היסטוריה וביצועים</b> ━━\n"
            "/today — מה קרה היום\n"
            "/history — עסקאות אחרונות\n"
            "/summary — סיכום 7 ימים\n"
            "/monthly — סיכום 30 ימים\n"
            "/biztsuim — ביצועים ו-אחוז הצלחה\n"
            "/best — העסקה הטובה ביותר\n"
            "/worst — העסקה הגרועה ביותר\n"
            "/taxes — סיכום מס\n"
            "/backtest — למידה היסטורית\n\n"
            "━━ ⚙️ <b>שליטה</b> ━━\n"
            "/scan — הפעל סריקה מיידית\n"
            "/morning — תדרוך בוקר עכשיו\n"
            "/quiet — מצב שקט (פחות התראות)\n"
            "/loud — כל ההתראות\n"
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
            "/ask מה לעשות עם AAPL? — שאלה חופשית\n"
            "/review — AI סוקר את כל הפוזיציות\n"
            "/journal — יומן עסקאות אישי\n"
            "/whatsnew — 5 הפעולות האחרונות\n"
            "/levels AAPL — תמיכה/תנגדות\n"
            "/remind 17:30 לבדוק TSLA — תזכורת\n"
            "/advice — ייעוץ AI על התיק\n"
            "/explain RSI — הסבר מונח פיננסי\n"
            "/streak — רצף ניצחונות/הפסדות\n\n"
            "<i>💬 אפשר גם לשאול בעברית חופשית!</i>"
        )

    # /count — quick trade statistics summary
    # Match full text for Hebrew variations + common typos
    _trade_count_triggers = (
        "כמה עסק", "כמה סק", "כמה הסק", "כמה הסכ", "כמה סכ",  # שגיאות כתיב נפוצות
        "כמה הקנייות", "כמה קניות", "כמה מכרת", "כמה קנית",
        "ספירה", "ספור", "סטטיסטיק", "סיכום עסק",
        "כמה ברווח", "כמה בהפסד", "כמה עשית",
    )
    if (cmd in ("/count", "count") or
        any(tr in t for tr in _trade_count_triggers)):
        tc = context.get("trade_counts", {}) or {}
        total   = tc.get("total", 0)
        open_   = tc.get("open", 0)
        closed  = tc.get("closed", 0)
        wins    = tc.get("wins", 0)
        losses  = tc.get("losses", 0)
        today   = tc.get("today", 0)
        wr      = (wins / closed * 100) if closed else 0
        # Get realized PnL
        realized = context.get("realized_pnl_net", 0)
        realized_str = _fmt_pnl(realized) if realized else "—"
        wr_icon = "🟢" if wr >= 55 else ("🟡" if wr >= 45 else "🔴") if closed else "⚪"

        lines = [
            f"🔢 <b>סיכום עסקאות</b>",
            f"━━━━━━━━━━━━━━━━",
            f"📊 סה\"כ עסקאות שעשיתי: <b>{total}</b>",
            f"📂 פתוחות עכשיו:        {open_}",
            f"✔️ סגורות:                {closed}",
            f"━━━━━━━━━━━━━━━━",
            f"✅ ברווח:                <b>{wins}</b>",
            f"❌ בהפסד:                 <b>{losses}</b>",
            f"{wr_icon} אחוז הצלחה:       <b>{wr:.0f}%</b>" + (" — " + ("מעולה" if wr >= 55 else "בסדר" if wr >= 45 else "צריך לשפר") if closed else " (אין נתונים עדיין)"),
            f"━━━━━━━━━━━━━━━━",
            f"💰 רווח ממומש כולל:    {realized_str}",
        ]
        if today:
            lines.append(f"📅 נפתחו היום: <b>{today}</b>")
        return "\n".join(lines)

    # /risk_score — overall portfolio risk metric
    if cmd in ("/risk_score", "סיכון", "ניקוד סיכון", "סיכון תיק"):
        positions = context.get("open_positions", [])
        if not positions:
            return "📊 אין פוזיציות פתוחות — אין סיכון כרגע 🟢"
        cash = context.get("cash", 0)
        equity = context.get("equity", 1)
        budget_used = context.get("budget_used_pct", 0)
        n_pos = len(positions)

        # Compute risk score 0-100
        risk = 0
        risk_factors = []

        # Concentration risk (how exposed are we)
        if budget_used > 80:
            risk += 30; risk_factors.append(f"🔴 חשיפה גבוהה: {budget_used:.0f}% מהתקציב")
        elif budget_used > 60:
            risk += 15; risk_factors.append(f"🟡 חשיפה בינונית: {budget_used:.0f}%")
        else:
            risk += 5; risk_factors.append(f"🟢 חשיפה נמוכה: {budget_used:.0f}%")

        # Losing positions risk
        losing = [p for p in positions if p.get("pct", 0) < 0]
        if len(losing) >= n_pos * 0.5:
            risk += 25; risk_factors.append(f"🔴 {len(losing)}/{n_pos} פוזיציות בהפסד")
        elif losing:
            risk += 10; risk_factors.append(f"🟡 {len(losing)}/{n_pos} פוזיציות בהפסד")
        else:
            risk_factors.append(f"🟢 כל הפוזיציות ברווח")

        # Deep losers risk
        deep_losers = [p for p in positions if p.get("pct", 0) < -3]
        if deep_losers:
            risk += 20; risk_factors.append(f"🔴 {len(deep_losers)} פוזיציות בהפסד עמוק (>3%)")

        # Concentration: largest position
        if positions:
            largest = max(positions, key=lambda p: (p.get("entry", 0) * p.get("qty", 0)))
            largest_val = largest.get("entry", 0) * largest.get("qty", 0)
            concentration_pct = (largest_val / equity * 100) if equity else 0
            if concentration_pct > 30:
                risk += 20; risk_factors.append(f"🔴 ריכוז גבוה: {largest['ticker']} = {concentration_pct:.0f}% מהתיק")
            elif concentration_pct > 20:
                risk += 10; risk_factors.append(f"🟡 ריכוז בינוני: {largest['ticker']} = {concentration_pct:.0f}%")

        # VIX risk
        vix = context.get("vix", 0)
        if vix and vix > 25:
            risk += 15; risk_factors.append(f"🔴 VIX גבוה ({vix:.0f}) — שוק תנודתי")
        elif vix and vix > 18:
            risk += 5; risk_factors.append(f"🟡 VIX בינוני ({vix:.0f})")

        risk = min(100, risk)

        # Verdict
        if risk < 25:
            verdict = "🟢 <b>נמוך</b> — התיק יציב"
        elif risk < 50:
            verdict = "🟡 <b>בינוני</b> — לעקוב"
        elif risk < 75:
            verdict = "🟠 <b>גבוה</b> — שקול הקטנת חשיפה"
        else:
            verdict = "🔴 <b>גבוה מאוד</b> — מומלץ למכור חלק"

        bar = "🟥" * (risk // 10) + "⬜" * (10 - risk // 10)
        lines = [
            f"⚠️ <b>ניקוד סיכון תיק</b>",
            f"━━━━━━━━━━━━━━━━",
            f"📊 ציון: <b>{risk}/100</b>",
            f"{bar}",
            f"{verdict}",
            f"━━━━━━━━━━━━━━━━",
            f"<b>גורמי סיכון:</b>",
        ]
        for r in risk_factors:
            lines.append(f"   {r}")
        return "\n".join(lines)

    # /tomorrow — what to expect tomorrow + watchlist
    if cmd in ("/tomorrow", "tomorrow", "מחר", "מה מחר", "תוכנית מחר"):
        try:
            import datetime as _dt_tm
            now = _dt_tm.datetime.now()
            weekday = now.weekday()
            day_names = ['שני','שלישי','רביעי','חמישי','שישי','שבת','ראשון']
            tomorrow_day = (weekday + 1) % 7
            tomorrow_name = day_names[tomorrow_day]
            tomorrow_date = (now + _dt_tm.timedelta(days=1)).strftime("%d-%m-%Y")

            # Will market be open tomorrow?
            if tomorrow_day in (5, 6):   # Sat=5, Sun=6
                market_status = "🔴 שוק סגור (סוף שבוע)"
            else:
                market_status = "🟢 שוק פתוח 16:30-23:00 ישראל"

            # Top 3 Buffett picks for tomorrow
            picks_lines = ["⏳ מחשב Top 3 איכותיות..."]
            try:
                from scanner import get_watchlist as _gwl_t
                from buffett_analysis import get_buffett_analysis as _ba_t
                tickers = _gwl_t()[:10]
                picks = []
                for tk in tickers:
                    try:
                        a = _ba_t(tk)
                        if a.get("score", 0) >= 65:
                            picks.append((tk, a.get("score", 0), a.get("moat", "?")))
                    except Exception:
                        continue
                if picks:
                    picks.sort(key=lambda x: x[1], reverse=True)
                    top3 = picks[:3]
                    picks_lines = []
                    for tk, s, m in top3:
                        moat_icon = {"strong": "💪", "medium": "🛡️", "weak": "⚠️"}.get(m, "?")
                        picks_lines.append(f"   {moat_icon} <b>{tk}</b> — איכות {s:.0f}/100")
                else:
                    picks_lines = ["   (אין מניות איכותיות בקריטריונים הנוכחיים)"]
            except Exception:
                picks_lines = ["   (לא הצלחתי לחשב — נסה מאוחר יותר)"]

            # Check for earnings tomorrow
            earnings_alert = ""
            try:
                positions = context.get("open_positions", [])
                from earnings import check_earnings_risk
                tomorrow_earnings = []
                for p in positions:
                    risky, _, days = check_earnings_risk(p["ticker"])
                    if days is not None and days <= 1:
                        tomorrow_earnings.append(p["ticker"])
                if tomorrow_earnings:
                    earnings_alert = (
                        f"\n📑 <b>דוחות מחר:</b>\n"
                        + "\n".join(f"   ⚠️ {t}" for t in tomorrow_earnings)
                    )
            except Exception:
                pass

            return (
                f"📅 <b>תוכנית מחר — יום {tomorrow_name} ({tomorrow_date})</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"{market_status}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Top 3 לבדיקה:</b>\n"
                + "\n".join(picks_lines)
                + earnings_alert
                + f"\n━━━━━━━━━━━━━━━━\n"
                + f"💡 אני אסרוק אוטומטית בפתיחה ואקנה את האיכותיות"
            )
        except Exception as e:
            logger.error(f"[/tomorrow] Error: {e}")
            return "❌ שגיאה ביצירת תוכנית מחר"

    # /digest — mega daily summary (everything in one place)
    if cmd in ("/digest", "digest", "סיכום מלא", "מעדכן", "תקציר", "סיכום יומי"):
        try:
            equity   = context.get("equity", 0)
            cash     = context.get("cash", 0)
            pos_pnl  = context.get("open_pnl", 0)
            realized = context.get("realized_pnl_net", 0)
            positions = context.get("open_positions", [])
            tc       = context.get("trade_counts", {})
            vix      = context.get("vix")
            market_open = context.get("market_open", False)

            wr = (tc.get("wins", 0) / tc.get("closed", 1) * 100) if tc.get("closed") else 0
            n_pos = len(positions)

            # Portfolio change vs initial
            initial = float(__import__("os").getenv("MAX_BUDGET", "10000"))
            total_return = (equity - initial) / initial * 100 if initial else 0

            # Top performer + worst
            winners = sorted([p for p in positions if p.get("pct", 0) > 0], key=lambda p: -p.get("pct", 0))
            losers  = sorted([p for p in positions if p.get("pct", 0) < 0], key=lambda p: p.get("pct", 0))

            lines = [
                f"📋 <b>תקציר יומי</b>",
                f"━━━━━━━━━━━━━━━━",
                f"💼 שווי תיק:    <b>${equity:,.2f}</b>",
                f"📊 תשואה כוללת: <b>{total_return:+.2f}%</b>",
                f"💵 מזומן:        ${cash:,.2f}",
                f"📈 רווח פתוח:   ${pos_pnl:+.2f}",
                f"💰 רווח ממומש:  ${realized:+.2f}",
                f"━━━━━━━━━━━━━━━━",
                f"📂 פוזיציות: <b>{n_pos}</b>",
            ]
            if winners:
                top = winners[0]
                lines.append(f"   🏆 הכי טובה: <b>{top['ticker']}</b> ({top['pct']:+.1f}%)")
            if losers:
                worst = losers[0]
                lines.append(f"   📉 הכי חלשה: <b>{worst['ticker']}</b> ({worst['pct']:+.1f}%)")

            lines.append(f"━━━━━━━━━━━━━━━━")
            lines.append(f"🎯 עסקאות: {tc.get('total', 0)} (אחוז הצלחה {wr:.0f}%)")
            lines.append(f"   ✅ {tc.get('wins', 0)} ניצחונות  |  ❌ {tc.get('losses', 0)} הפסדים")

            # Market state
            lines.append(f"━━━━━━━━━━━━━━━━")
            mkt_emoji = "🟢" if market_open else "🔴"
            lines.append(f"{mkt_emoji} שוק: {'פתוח' if market_open else 'סגור'}")
            if vix:
                vix_emoji = "🟢" if vix < 20 else "🟡" if vix < 27 else "🔴"
                lines.append(f"{vix_emoji} VIX: {vix:.1f}")

            # Suggestions
            lines.append(f"━━━━━━━━━━━━━━━━")
            lines.append(f"💡 <b>צעדים שאני ממליץ:</b>")
            suggestions = []
            if n_pos == 0:
                suggestions.append("• /best — לראות הזדמנויות עכשיו")
            if cash / equity > 0.5 if equity else False:
                suggestions.append("• /best — יש לך הרבה מזומן זמין")
            if losers and any(p.get("pct", 0) < -3 for p in losers):
                suggestions.append("• /risk_score — לבדוק סיכון")
            if not suggestions:
                suggestions.append("• /journal — לראות יומן עסקאות")
                suggestions.append("• /alpha — ביצוע מול השוק")
            for s in suggestions[:3]:
                lines.append(s)
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/digest] Error: {e}")
            return "❌ שגיאה בייצור תקציר"

    # /diversity — quick portfolio diversity check
    if cmd in ("/diversity", "diversity", "פיזור", "ריכוז"):
        positions = context.get("open_positions", [])
        if not positions:
            return "📊 אין פוזיציות — אין מה לנתח 🟢"
        try:
            from sector_rotation import get_sector_for_ticker
            sectors = {}
            for p in positions:
                s = get_sector_for_ticker(p["ticker"]) or "אחר"
                sectors[s] = sectors.get(s, 0) + 1
            total = len(positions)
            lines = ["🎯 <b>פיזור התיק שלך</b>", "━━━━━━━━━━━━━━━━"]
            max_sec = max(sectors.values()) if sectors else 0
            for sec, count in sorted(sectors.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                icon = "🔴" if pct > 50 else ("🟡" if pct > 30 else "🟢")
                lines.append(f"{icon} <b>{sec}</b>: {count}/{total} ({pct:.0f}%)")
            lines.append("━━━━━━━━━━━━━━━━")
            if max_sec / total > 0.5:
                lines.append("⚠️ <b>ריכוז גבוה</b> — שקול לפזר יותר")
                lines.append("💡 רוב הסיכון בסקטור אחד")
            elif max_sec / total > 0.3:
                lines.append("🟡 ריכוז בינוני — סביר")
            else:
                lines.append("🟢 פיזור טוב — סיכון מוגבל")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/diversity] Error: {e}")
            return "❌ שגיאה בניתוח"

    # /alpha — bot performance vs S&P 500 (alpha = outperformance)
    if cmd in ("/alpha", "alpha", "אלפא", "מול שוק", "מול sp500"):
        try:
            equity = context.get("equity", 10000)
            initial = float(__import__("os").getenv("MAX_BUDGET", "10000"))
            bot_return_pct = (equity - initial) / initial * 100 if initial else 0
            # Get SPY return since bot inception
            try:
                history = database.get_trade_history(limit=200)
                first_trade = next(
                    (t for t in reversed(history) if t.get("entry_time")), None
                )
                start_date = first_trade.get("entry_time", "")[:10] if first_trade else None
            except Exception:
                start_date = None

            spy_return_pct = None
            if start_date:
                try:
                    import yfinance as _yf_a
                    from datetime import datetime as _dt_a, timedelta as _td_a
                    start_dt = _dt_a.strptime(start_date, "%Y-%m-%d")
                    end_dt   = _dt_a.now()
                    spy_data = _yf_a.Ticker("SPY").history(
                        start=start_dt - _td_a(days=1),
                        end=end_dt + _td_a(days=1),
                        auto_adjust=True,
                    )
                    if not spy_data.empty:
                        spy_start = float(spy_data["Close"].iloc[0])
                        spy_end   = float(spy_data["Close"].iloc[-1])
                        spy_return_pct = (spy_end - spy_start) / spy_start * 100
                except Exception:
                    pass

            alpha = bot_return_pct - (spy_return_pct or 0)
            bot_icon = "📈" if bot_return_pct >= 0 else "📉"
            spy_icon = "📈" if (spy_return_pct or 0) >= 0 else "📉"

            if alpha > 5:
                verdict = "🏆 <b>הבוט מנצח את השוק בגדול!</b>"
            elif alpha > 0:
                verdict = "✅ <b>הבוט מנצח את השוק</b>"
            elif alpha > -5:
                verdict = "⚪ קרוב לשוק — לא רחוק"
            else:
                verdict = "📉 השוק מנצח — צריך לשפר"

            lines = [
                f"📊 <b>ביצוע מול S&P 500</b>",
                f"━━━━━━━━━━━━━━━━",
                f"{bot_icon} <b>הבוט שלך</b>: {bot_return_pct:+.2f}%",
            ]
            if spy_return_pct is not None:
                lines.append(f"{spy_icon} <b>SPY</b>:        {spy_return_pct:+.2f}%")
                lines.append(f"━━━━━━━━━━━━━━━━")
                lines.append(f"🎯 <b>אלפא (יתרון על השוק)</b>: <b>{alpha:+.2f}%</b>")
                lines.append(verdict)
            else:
                lines.append("⚠️ אין מספיק היסטוריה לחישוב אלפא")
            if start_date:
                lines.append(f"━━━━━━━━━━━━━━━━")
                lines.append(f"📅 מתאריך: {start_date}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/alpha] Error: {e}")
            return "❌ שגיאה בחישוב אלפא"

    # /new — show new commands added recently
    if cmd in ("/new", "new", "חדש", "פקודות חדשות"):
        return (
            "🆕 <b>פקודות חדשות חכמות</b>\n"
            "━━━━━━━━━━━━━━━━\n"
            "🤖 <b>אנליטיקה חכמה:</b>\n"
            "/best — Top 3 הזדמנויות עכשיו\n"
            "/buffett AAPL — ניתוח באפט מלא\n"
            "/why AAPL — למה הבוט קונה/לא?\n"
            "/cheap AAPL — האם המניה זולה (52w)?\n"
            "/risk_score — סיכון תיק\n"
            "/journal — יומן עסקאות\n"
            "/count — סיכום עסקאות\n"
            "/activity_now — מה הבוט עושה עכשיו\n"
            "━━━━━━━━━━━━━━━━\n"
            "🛡️ <b>הגנות אוטומטיות:</b>\n"
            "• הגנת Drawdown 10%\n"
            "• Pre-Earnings (סגירה 2 ימים לפני)\n"
            "• Progressive Stop (0.5×ATR ב-+15%)\n"
            "• Break-Even ב-+1%\n"
            "• Self-improvement (5 הפסדים → זהיר)\n"
            "━━━━━━━━━━━━━━━━\n"
            "📱 <b>התראות אקטיביות:</b>\n"
            "• Profit milestones (+2/5/10/15%)\n"
            "• Rapid moves (±2% ב-10 דק')\n"
            "• Golden opportunities\n"
            "• Earnings (Beat/Miss)\n"
            "• Re-entry suggestions\n"
            "━━━━━━━━━━━━━━━━\n"
            "💡 גם שאלות חופשיות עובדות!\n"
            "<i>\"מה דעתך על AAPL?\"</i>\n"
            "<i>\"כמה הרווחתי השבוע?\"</i>"
        )

    # /journal — recent trade journal with lessons
    if cmd in ("/journal", "journal", "יומן", "סיכום עסקאות"):
        try:
            history = database.get_trade_history(limit=10)
            closed = [t for t in history if t.get("exit_time")]
            if not closed:
                return "📔 <b>יומן עסקאות</b>\n━━━━━━━━━━━━━━━━\n😴 עדיין אין עסקאות סגורות"
            lines = ["📔 <b>יומן עסקאות אחרונות</b>", "━━━━━━━━━━━━━━━━"]
            best_pct = -999.0; worst_pct = 999.0
            best_trade = worst_trade = None
            total_pnl = 0.0
            for t in closed[:5]:
                tk = t.get("ticker")
                entry = t.get("entry_price", 0)
                exit_p = t.get("exit_price", 0)
                if entry and exit_p:
                    pct = (exit_p - entry) / entry * 100
                    pnl = t.get("pnl_gross", 0) or 0
                    total_pnl += pnl
                    status = t.get("status", "?")
                    icon = "✅" if pnl > 0 else "❌"
                    reason_map = {
                        "take_profit": "יעד רווח", "stop_loss": "סטופ",
                        "smart_sell": "מכירה חכמה", "news_exit": "חדשות שליליות",
                        "earnings_miss": "פספוס דוח", "time_exit": "פג זמן",
                        "momentum_exit": "אובדן מומנטום", "partial_tp": "חלק מהיעד",
                        "closed": "סגור", "stale_restart": "restart",
                    }
                    reason = reason_map.get(status, status)
                    lines.append(f"{icon} <b>{tk}</b>: {pct:+.2f}% (${pnl:+.2f}) — {reason}")
                    if pct > best_pct:
                        best_pct = pct; best_trade = tk
                    if pct < worst_pct:
                        worst_pct = pct; worst_trade = tk
            # Summary
            lines.append("━━━━━━━━━━━━━━━━")
            lines.append(f"💰 רווח כולל מ-5 העסקאות: <b>${total_pnl:+.2f}</b>")
            if best_trade:
                lines.append(f"🏆 הטובה: <b>{best_trade}</b> ({best_pct:+.1f}%)")
            if worst_trade and worst_trade != best_trade:
                lines.append(f"📉 הגרועה: <b>{worst_trade}</b> ({worst_pct:+.1f}%)")
            # Insights
            wins = sum(1 for t in closed[:5] if (t.get("pnl_gross") or 0) > 0)
            wr5 = wins / min(5, len(closed)) * 100
            if wr5 >= 70:
                lines.append(f"💡 כושר טוב — {wr5:.0f}% הצלחה ב-5 האחרונות")
            elif wr5 >= 50:
                lines.append(f"💡 ביצוע סביר — {wr5:.0f}% הצלחה")
            else:
                lines.append(f"💡 צריך לעבוד — רק {wr5:.0f}% הצלחה (הבוט יחמיר קריטריונים)")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/journal] Error: {e}")
            return "❌ שגיאה בייצור יומן"

    # /why TICKER — explain why bot bought/didn't buy this stock
    if cmd in ("/why", "why", "למה") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /why AAPL"
        try:
            from scoring import get_composite_score as _gcs_w
            from sentiment import score_sentiment as _ss_w
            from buffett_analysis import get_buffett_analysis as _ba_w

            sent = _ss_w(_ticker)
            sc = _gcs_w(_ticker, sent.score)
            tech = sc.get("composite_score", 0)
            should_buy = sc.get("should_buy", False)
            min_score = sc.get("min_score", 60)

            ba = _ba_w(_ticker)
            buf = ba.get("score", 0)
            verdict = ba.get("verdict_he", "")

            # Reasons
            reasons_buy = []
            reasons_skip = []
            if tech >= min_score:
                reasons_buy.append(f"✅ ציון טכני גבוה ({tech:.0f}/{min_score})")
            else:
                reasons_skip.append(f"❌ ציון טכני נמוך ({tech:.0f}<{min_score})")
            if buf >= 50:
                reasons_buy.append(f"✅ איכות באפט טובה ({buf:.0f}/100)")
            else:
                reasons_skip.append(f"❌ איכות באפט נמוכה ({buf:.0f}<50)")
            if sent.score >= 5:
                reasons_buy.append(f"✅ סנטימנט חיובי ({sent.score}/10)")
            else:
                reasons_skip.append(f"❌ סנטימנט שלילי ({sent.score}<5)")

            # Earnings check
            try:
                from earnings import check_earnings_risk
                risky, _, days = check_earnings_risk(_ticker)
                if risky and days is not None:
                    reasons_skip.append(f"⚠️ דוח רווחים בעוד {days} ימים")
            except Exception:
                pass

            decision = "✅ <b>הבוט היה קונה</b>" if (should_buy and buf >= 50) else "❌ <b>הבוט לא היה קונה</b>"
            lines = [
                f"🤔 <b>למה {_ticker}?</b>",
                f"━━━━━━━━━━━━━━━━",
                f"{verdict}",
                f"",
                decision,
                f"━━━━━━━━━━━━━━━━",
            ]
            if reasons_buy:
                lines.append("<b>סיבות חיוביות:</b>")
                for r in reasons_buy:
                    lines.append(f"   {r}")
            if reasons_skip:
                lines.append("<b>סיבות שליליות:</b>")
                for r in reasons_skip:
                    lines.append(f"   {r}")
            lines.append("━━━━━━━━━━━━━━━━")
            lines.append(f"📋 פרטים נוספים: /buffett {_ticker}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/why] Error: {e}")
            return f"❌ שגיאה בניתוח {_ticker}"

    # /cheap TICKER — is this stock cheap vs 52-week range?
    if cmd in ("/cheap", "cheap", "זול", "זולה") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /cheap AAPL"
        try:
            import yfinance as _yf_c
            info = _yf_c.Ticker(_ticker).info
            cur = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
            low_52 = float(info.get("fiftyTwoWeekLow") or 0)
            high_52 = float(info.get("fiftyTwoWeekHigh") or 0)
            if not (cur and low_52 and high_52 and high_52 > low_52):
                return f"❌ אין נתוני 52w עבור {_ticker}"
            # Position in 52w range (0 = at low, 100 = at high)
            position = (cur - low_52) / (high_52 - low_52) * 100
            from_high = (cur - high_52) / high_52 * 100
            from_low  = (cur - low_52) / low_52 * 100

            if position < 20:
                verdict = "🟢 מאוד זולה! קרוב לנקודה הנמוכה של השנה"
            elif position < 35:
                verdict = "🟡 זולה יחסית"
            elif position < 65:
                verdict = "⚪ בטווח אמצעי"
            elif position < 85:
                verdict = "🟠 קרוב לשיא — יקרה"
            else:
                verdict = "🔴 בשיא או קרוב מאוד — יקרה מאוד"

            bar_filled = int(position / 10)
            bar = "🟩" * bar_filled + "⬜" * (10 - bar_filled)
            return (
                f"📏 <b>ניתוח 52 שבועות — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💵 מחיר נוכחי: <b>${cur:.2f}</b>\n"
                f"📉 שפל שנתי: ${low_52:.2f} ({from_low:+.1f}%)\n"
                f"📈 שיא שנתי: ${high_52:.2f} ({from_high:+.1f}%)\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 מיקום: <b>{position:.0f}%</b> בטווח\n"
                f"{bar}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💡 {verdict}"
            )
        except Exception as e:
            logger.error(f"[/cheap] Error: {e}")
            return f"❌ שגיאה בניתוח {_ticker}"

    # /best — Top 3 best opportunities right now (combines technical + Buffett)
    if cmd in ("/best", "best", "הכי טוב", "הזדמנויות", "מה הכי טוב", "הטובים"):
        try:
            from scanner import get_watchlist as _gwl_b
            from buffett_analysis import get_buffett_analysis as _ba_b
            from scoring import get_composite_score as _gcs_b
            from sentiment import score_sentiment as _ss_b
            tickers = _gwl_b()[:12]   # check top 12 from watchlist
            scored = []
            for tk in tickers:
                try:
                    sent = _ss_b(tk)
                    sc = _gcs_b(tk, sent.score)
                    tech = sc.get("composite_score", 0)
                    ba = _ba_b(tk)
                    buf = ba.get("score", 0)
                    combined = tech * 0.6 + buf * 0.4
                    scored.append((tk, tech, buf, combined, ba.get("moat", "?")))
                except Exception:
                    continue
            if not scored:
                return "😴 לא הצלחתי לחשב הזדמנויות כרגע — נסה שוב מאוחר יותר"
            scored.sort(key=lambda x: x[3], reverse=True)
            top3 = scored[:3]
            lines = [
                "🏆 <b>3 הזדמנויות הטובות עכשיו</b>",
                "━━━━━━━━━━━━━━━━",
            ]
            for i, (tk, tech, buf, comb, moat) in enumerate(top3, 1):
                moat_icon = {"strong": "💪", "medium": "🛡️", "weak": "⚠️"}.get(moat, "?")
                lines.append(
                    f"{i}. {moat_icon} <b>{tk}</b>\n"
                    f"   ⭐ ציון משולב: <b>{comb:.0f}/100</b>\n"
                    f"   🔧 טכני: {tech:.0f}  |  🎩 באפט: {buf:.0f}\n"
                    f"   📋 לפרטים: /buffett {tk}"
                )
            lines.append("━━━━━━━━━━━━━━━━")
            lines.append("💡 משוקלל: 60% טכני + 40% איכות")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/best] Error: {e}")
            return "❌ שגיאה בחישוב — נסה שוב"

    # /activity_now — מה הבוט עושה ברגע זה
    if cmd in ("/activity_now", "/now", "מה אתה עושה", "מה הבוט עושה", "מה אתה עושה עכשיו"):
        try:
            import requests as _req
            import os as _os
            base = _os.getenv("RENDER_EXTERNAL_URL", "https://trading-bot-e66l.onrender.com").rstrip("/")
            r = _req.get(f"{base}/activity", timeout=8)
            recent = r.json()[:5] if r.ok else []
        except Exception:
            recent = []

        if not recent:
            return "🤖 הבוט פעיל אבל אין פעילות אחרונה לדווח עליה"

        lines = ["🤖 <b>מה אני עושה עכשיו</b>", "━━━━━━━━━━━━━━━━"]
        for item in recent:
            icon = item.get("icon", "•")
            text = item.get("text", "")
            ts   = item.get("ts", "")[-8:]   # show HH:MM:SS only
            lines.append(f"{icon} {text}  <i>({ts})</i>")
        lines.append("━━━━━━━━━━━━━━━━")
        # Also show market status
        if context.get("market_open"):
            lines.append("🟢 השוק פתוח — סורק ומחפש הזדמנויות")
        else:
            lines.append("💤 השוק סגור — מתאמן וקורא חדשות")
        return "\n".join(lines)

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
        lines = [f"📊 <b>מצב התיק</b>", "━━━━━━━━━━━━━━━━"]
        lines.append(f"💼 שווי תיק:   <b>{_fmt_price(equity)}</b>")
        lines.append(f"💵 מזומן:       {_fmt_price(cash)}")
        lines.append(f"{pnl_icon} רווח/הפסד:  {_fmt_pnl(pnl)}")
        if realized != 0:
            lines.append(f"🏆 ממומש:       {_fmt_pnl(realized)}")
        lines.append(f"📂 פוזיציות:   <b>{n_pos}/{max_pos}</b>")
        lines.append(f"{mkt_icon} שוק:          {'פתוח ✅' if mkt_open else 'סגור 🔴'}")
        if vix:
            vix_icon = "😌" if vix < 20 else ("😟" if vix < 28 else "😱")
            lines.append(f"{vix_icon} VIX:          {vix:.1f}")
        if cb:
            lines.append(f"⛔ Circuit Breaker: פעיל")
        import os as _os
        if _os.getenv("BOT_PAUSED"):
            lines.append(f"⏸️ הבוט:       מושהה")
        else:
            lines.append(f"✅ הבוט:       פעיל וסורק")
        # Trade counts (lifetime stats)
        tc = context.get("trade_counts", {}) or {}
        if tc.get("total"):
            wr = (tc.get("wins", 0) / tc.get("closed", 1) * 100) if tc.get("closed") else 0
            lines.append(f"━━━━━━━━━━━━━━━━")
            lines.append(f"📊 סה\"כ עסקאות שעשיתי: <b>{tc.get('total', 0)}</b>"
                         + (f"  (היום: {tc.get('today', 0)})" if tc.get('today') else ""))
            lines.append(f"   ✅ ברווח: {tc.get('wins', 0)}  |  ❌ בהפסד: {tc.get('losses', 0)}"
                         + (f"  |  הצלחה: {wr:.0f}%" if tc.get('closed') else ""))

        if positions:
            total_pnl_open = sum(p["pnl"] for p in positions)
            total_invested = sum(p.get("invested") or round(p["entry"] * p["qty"], 2) for p in positions)
            lines.append(f"")
            lines.append(f"<b>📂 פוזיציות ({n_pos}):</b>")
            lines.append(f"━━━━━━━━━━━━━━━━")
            for p in positions:
                icon = "🟢" if p["pnl"] >= 0 else "🔴"
                qty  = f"{p['qty']:.4f}".rstrip('0').rstrip('.')
                lines.append(f"{icon} <b>{p['ticker']}</b>")
                lines.append(f"   🔢 כמות:        {qty} מניות")
                lines.append(f"   💵 כניסה:       {_fmt_price(p['entry'])}")
                lines.append(f"   📍 עכשיו:       {_fmt_price(p['current'])} ({p['pct']:+.1f}%)")
                lines.append(f"   💰 רווח/הפסד:  {_fmt_pnl(p['pnl'], False)}")
                lines.append(f"   💼 שווי:        {_fmt_price(p['current'] * p['qty'])}")
                lines.append(f"")
            lines.append(f"📊 סה\"כ הושקע: <b>{_fmt_price(total_invested)}</b>")
            if total_pnl_open != 0:
                lines.append(f"{'📈' if total_pnl_open >= 0 else '📉'} סה\"כ רווח: <b>{_fmt_pnl(total_pnl_open, False)}</b>")

        # Smart action suggestion based on state
        lines.append(f"\n━━━━━━━━━━━━━━━━\n💡 <i>")
        if not positions and cash > 0 and not context.get("circuit_breaker"):
            lines.append("אין פוזיציות — נסה /top לראות הזדמנויות</i>")
        elif positions and len(positions) >= max_pos:
            best_win = max(positions, key=lambda p: p["pct"]) if positions else None
            if best_win and best_win["pct"] >= 5:
                lines.append(f"ברווח יפה ב-{best_win['ticker']} — שקול /stop {best_win['ticker']}</i>")
            else:
                lines.append("תיק מלא — /risk לניתוח סיכון | /newscheck לחדשות</i>")
        elif context.get("circuit_breaker"):
            lines.append("Circuit Breaker פעיל — /diagnose לבדיקה</i>")
        else:
            lines.append("/top לסריקה | /market למצב שוק | /risk לניתוח</i>")

        return "\n".join(lines)

    # /watchadd TICKER — add ticker to watchlist
    if cmd in ("/watchadd", "watchadd", "הוסף מניה", "הוסף לרשימה") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /watchadd AAPL"
        try:
            import os as _os, yfinance as _yf
            # Validate ticker exists
            info = _yf.Ticker(_ticker).fast_info
            if not getattr(info, "last_price", 0):
                return f"❌ לא מצאתי מניה בשם <b>{_ticker}</b>"
            existing = [t.strip().upper() for t in _os.getenv("USER_WATCHLIST", "").split(",") if t.strip()]
            if _ticker in existing:
                return f"✅ <b>{_ticker}</b> כבר ברשימה!"
            existing.append(_ticker)
            _os.environ["USER_WATCHLIST"] = ",".join(existing)
            # Also remove from REMOVE list if was there
            removes = [t.strip().upper() for t in _os.getenv("USER_WATCHLIST_REMOVE", "").split(",") if t.strip()]
            if _ticker in removes:
                removes.remove(_ticker)
                _os.environ["USER_WATCHLIST_REMOVE"] = ",".join(removes)
            return (
                f"➕ <b>{_ticker} נוסף לרשימה!</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 רשימה מותאמת: {len(existing)} מניות\n"
                f"🔍 הבוט יסרוק את <b>{_ticker}</b> בסבב הבא\n\n"
                f"💡 להסרה: /watchremove {_ticker}"
            )
        except Exception as e:
            logger.error(f"[/watchadd] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /watchremove TICKER — remove ticker from watchlist
    if cmd in ("/watchremove", "watchremove", "הסר מניה", "הסר מרשימה") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /watchremove AAPL"
        try:
            import os as _os
            # Remove from user additions
            existing = [x.strip().upper() for x in _os.getenv("USER_WATCHLIST", "").split(",") if x.strip()]
            if _ticker in existing:
                existing.remove(_ticker)
                _os.environ["USER_WATCHLIST"] = ",".join(existing)
            # Add to remove list
            removes = [x.strip().upper() for x in _os.getenv("USER_WATCHLIST_REMOVE", "").split(",") if x.strip()]
            if _ticker not in removes:
                removes.append(_ticker)
                _os.environ["USER_WATCHLIST_REMOVE"] = ",".join(removes)
            return (
                f"➖ <b>{_ticker} הוסר מהרשימה!</b>\n"
                f"הבוט לא יסרוק אותה יותר.\n\n"
                f"💡 להחזרה: /watchadd {_ticker}"
            )
        except Exception as e:
            logger.error(f"[/watchremove] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /monthly — monthly performance summary
    if cmd in ("/monthly", "monthly", "חודש", "סיכום חודשי"):
        try:
            import database as _db
            from datetime import datetime, timezone, timedelta
            now_utc    = datetime.now(timezone.utc)
            month_ago  = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")
            trades     = _db.get_trade_history(limit=200) or []
            monthly    = [tr for tr in trades
                          if str(tr.get("exit_time",""))[:10] >= month_ago
                          and tr.get("pnl_gross") is not None]
            if not monthly:
                return "📅 <b>סיכום 30 ימים</b>\n━━━━━━━━━━━━━━━━\n😴 לא היו עסקאות החודש"
            wins        = [tr for tr in monthly if float(tr.get("pnl_gross") or 0) > 0]
            total_pnl   = sum(float(tr.get("pnl_gross") or 0) for tr in monthly)
            wr          = round(len(wins) / len(monthly) * 100, 1) if monthly else 0
            avg_pnl     = total_pnl / len(monthly) if monthly else 0
            best        = max(monthly, key=lambda x: float(x.get("pnl_gross") or 0))
            worst       = min(monthly, key=lambda x: float(x.get("pnl_gross") or 0))
            # Best/worst tickers
            ticker_pnl: dict[str, float] = {}
            for tr in monthly:
                tk = tr.get("ticker","?")
                ticker_pnl[tk] = ticker_pnl.get(tk, 0) + float(tr.get("pnl_gross") or 0)
            best_tk  = max(ticker_pnl, key=ticker_pnl.get)
            worst_tk = min(ticker_pnl, key=ticker_pnl.get)
            # Monthly equity change
            equity_now = context.get("equity", 0)
            wr_bar = "🟢" * round(wr / 10) + "⚪" * (10 - round(wr / 10))
            return (
                f"📅 <b>סיכום 30 ימים</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🔢  עסקאות:       <b>{len(monthly)}</b>  (✅{len(wins)} ❌{len(monthly)-len(wins)})\n"
                f"🎯  אחוז הצלחה:     <b>{wr}%</b>\n"
                f"    {wr_bar}\n"
                f"💰  רווח חודשי:  {_fmt_pnl(total_pnl)}\n"
                f"⚡  לעסקה:       ${avg_pnl:+.2f}\n\n"
                f"🏆  הכי טוב:   <b>{best_tk}</b>  ${ticker_pnl[best_tk]:+.2f}\n"
                f"📉  הכי גרוע: <b>{worst_tk}</b>  ${ticker_pnl[worst_tk]:+.2f}"
            )
        except Exception as e:
            logger.error(f"[/monthly] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /signals — show current buy signals from scanner
    if cmd in ("/signals", "signals", "סיגנלים", "הזדמנויות"):
        try:
            from scanner import get_watchlist
            from scoring import get_composite_score, MIN_BUY_SCORE
            from sentiment import score_sentiment
            import database as _db, random
            wl   = get_watchlist()
            held = {tr["ticker"] for tr in (_db.get_open_trades() or [])}
            wl   = [tk for tk in wl if tk not in held]
            # Limit to 3 tickers — each takes ~3-4s, 3 = max 12s within 22s timeout
            import time as _t_sig
            sample = random.sample(wl, min(3, len(wl)))
            signals = []
            _sig_start = _t_sig.time()
            for tk in sample:
                if _t_sig.time() - _sig_start > 15:  # hard stop at 15s
                    break
                try:
                    sent = score_sentiment(tk)
                    comp = get_composite_score(tk, sent.score)
                    sc   = comp["composite_score"]
                    if sc >= MIN_BUY_SCORE:
                        signals.append((tk, sc, sent.score, comp))
                except Exception:
                    continue
            signals.sort(key=lambda x: x[1], reverse=True)
            if not signals:
                return (
                    f"📡 <b>הזדמנויות קנייה</b>\n━━━━━━━━━━━━━━━━\n"
                    f"😴 אין הזדמנויות עם ציון מעל {MIN_BUY_SCORE}/100 כרגע\n"
                    f"🔍 נסרקו {len(sample)} מניות"
                )
            lines = [f"📡 <b>הזדמנויות קנייה ({len(signals)} מתוך {len(sample)})</b>\n━━━━━━━━━━━━━━━━"]
            for tk, sc, ss, comp in signals[:5]:
                tech  = comp["scores"]["technicals"]
                mkt   = comp["scores"]["market"]
                lines.append(
                    f"✅ <b>{tk}</b>  {sc:.0f}/100  <code>{_score_bar(sc, 8)}</code>\n"
                    f"   🔧{tech:.0f}  🌍{mkt:.0f}  🧠{ss}/10"
                )
            lines.append(f"\n🎯 סף קנייה: {MIN_BUY_SCORE}  |  /scan לביצוע מיידי")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/signals] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

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
                    return f"{name}:   —"
                icon = "📈" if chg >= 0 else "📉"
                return f"{name}:   {icon} <b>{chg:+.2f}%</b>"

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
                fg_label = "—"

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
                f"🌍 <b>מצב השוק</b>",
                f"━━━━━━━━━━━━━━━━",
                _idx_line("📊 SPY (S&P500)", spy_chg),
                _idx_line("💻 QQQ (נאסד\"ק)", qqq_chg),
                _idx_line("🏭 DIA (דאו)", dia_chg),
                f"━━━━━━━━━━━━━━━━",
                f"{vix_icon} VIX:            <b>{vix_val:.1f}</b>",
                f"😰 פחד/חמדנות:  <b>{fg_label}</b>",
                f"━━━━━━━━━━━━━━━━",
                f"{verdict}",
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/market] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /journal [TICKER] NOTE — add a note to trade journal
    if cmd in ("/journal", "journal", "יומן", "הערה"):
        parts = t.split(maxsplit=2)
        if len(parts) < 2:
            # Show recent entries
            try:
                entries = database.get_journal_entries(limit=7)
                if not entries:
                    return (
                        f"📓 <b>יומן עסקאות</b>\n━━━━━━━━━━━━━━━━\n"
                        f"😴 אין הערות עדיין\n\n"
                        f"💡 <code>/journal AAPL מניה עם פוטנציאל גדול</code>"
                    )
                lines = [f"📓 <b>יומן עסקאות</b>\n━━━━━━━━━━━━━━━━"]
                for e in entries:
                    tk   = f"[{e['ticker']}] " if e.get("ticker") else ""
                    date = str(e.get("created_at",""))[:10]
                    lines.append(f"📌 {tk}<i>{e['note']}</i>  <code>{date}</code>")
                return "\n".join(lines)
            except Exception as e:
                logger.error(f"[/journal] Error: {e}")
                return "❌ שגיאה פנימית — נסה שוב"
        # Check if second word is a ticker
        possible_ticker = _safe_ticker(parts[1])
        if possible_ticker and len(parts) >= 3:
            note_text = parts[2]
            ticker_arg = possible_ticker
        else:
            note_text = " ".join(parts[1:])
            ticker_arg = None
        if not note_text.strip():
            return "💡 דוגמה: <code>/journal AAPL עלייה חזקה לפני דוח</code>"
        try:
            eid = database.add_journal_entry(note_text, ticker_arg)
            tk_str = f" על <b>{ticker_arg}</b>" if ticker_arg else ""
            return (
                f"📓 <b>הערה נשמרה!</b>{tk_str}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📌 {note_text}\n"
                f"🔖 מזהה: #{eid}\n\n"
                f"💡 /journal לצפייה בכל ההערות"
            )
        except Exception as e:
            logger.error(f"[/journal] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /levels TICKER — support/resistance via pivot points
    if cmd in ("/levels", "levels", "רמות", "תמיכה", "תנגדות") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /levels AAPL"
        try:
            import yfinance as _yf
            hist = _yf.Ticker(_ticker).history(period="5d", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 2:
                return f"❌ אין מספיק נתונים עבור {_ticker}"
            # Standard pivot point (PP) calculation from yesterday
            prev   = hist.iloc[-2]
            hi, lo, cl = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
            cur    = float(hist.iloc[-1]["Close"])
            pp     = (hi + lo + cl) / 3
            r1     = 2 * pp - lo
            r2     = pp + (hi - lo)
            r3     = hi + 2 * (pp - lo)
            s1     = 2 * pp - hi
            s2     = pp - (hi - lo)
            s3     = lo - 2 * (hi - pp)
            # Mark where current price sits
            def _level_marker(price, cur_p):
                return " ◀ עכשיו" if abs(price - cur_p) / cur_p < 0.015 else ""
            lines = [
                f"📐 <b>רמות תמיכה/תנגדות — {_ticker}</b>\n━━━━━━━━━━━━━━━━",
                f"🔴 R3: {_fmt_price(r3)}{_level_marker(r3, cur)}",
                f"🔴 R2: {_fmt_price(r2)}{_level_marker(r2, cur)}",
                f"🔴 R1: {_fmt_price(r1)}{_level_marker(r1, cur)}",
                f"⚪ PP: <b>{_fmt_price(pp)}</b>{_level_marker(pp, cur)}  ← ציר",
                f"🟢 S1: {_fmt_price(s1)}{_level_marker(s1, cur)}",
                f"🟢 S2: {_fmt_price(s2)}{_level_marker(s2, cur)}",
                f"🟢 S3: {_fmt_price(s3)}{_level_marker(s3, cur)}",
                f"\n📍 מחיר עכשיו: <b>{_fmt_price(cur)}</b>",
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/levels] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /remind HH:MM TEXT — set a time-based reminder (Israel time)
    if cmd in ("/remind", "remind", "תזכורת") and len(t.split()) >= 3:
        parts = t.split(maxsplit=2)
        time_str = parts[1]
        reminder_text = parts[2] if len(parts) > 2 else "תזכורת!"
        try:
            hh, mm = map(int, time_str.split(":"))
            assert 0 <= hh <= 23 and 0 <= mm <= 59
        except Exception:
            return "❌ פורמט שגוי — דוגמה: /remind 17:30 לבדוק TSLA"
        import os as _os
        reminders = _os.getenv("USER_REMINDERS", "")
        new_entry = f"{hh:02d}:{mm:02d}|{reminder_text}"
        _os.environ["USER_REMINDERS"] = (reminders + "," + new_entry).strip(",")
        return (
            f"⏰ <b>תזכורת נוצרה!</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🕐  שעה: <b>{hh:02d}:{mm:02d}</b> (שעון ישראל)\n"
            f"📌  הודעה: {reminder_text}\n\n"
            f"✅ הבוט ישלח הודעה בשעה {hh:02d}:{mm:02d}"
        )

    # /whatsnew — last 5 bot actions
    if cmd in ("/whatsnew", "whatsnew", "מה קרה", "פעולות אחרונות"):
        try:
            trades = database.get_trade_history(limit=10) or []
            recent = trades[:5]
            if not recent:
                return "📋 <b>פעולות אחרונות</b>\n━━━━━━━━━━━━━━━━\n😴 אין פעולות עדיין"
            lines = ["📋 <b>פעולות אחרונות</b>\n━━━━━━━━━━━━━━━━"]
            for tr in recent:
                st   = tr.get("status", "open")
                tk   = tr.get("ticker", "?")
                pnl  = float(tr.get("pnl_gross") or 0)
                ep   = float(tr.get("entry_price") or 0)
                xp   = float(tr.get("exit_price") or 0)
                date = str(tr.get("exit_time") or tr.get("entry_time",""))[:10]
                if st == "open":
                    lines.append(f"🟡 <b>קנייה:</b> {tk} @ {_fmt_price(ep)}  <code>{date}</code>")
                else:
                    icon = "🟢" if pnl >= 0 else "🔴"
                    reason_map = {"take_profit":"🎯","stop_loss":"🛑","smart_sell":"🧠",
                                  "news_exit":"📰","time_exit":"⏱"}
                    r = reason_map.get(st, "📌")
                    lines.append(f"{icon} <b>מכירה {r}:</b> {tk}  ${pnl:+.2f}  <code>{date}</code>")
            # Also show open positions
            open_t = database.get_open_trades()
            if open_t:
                lines.append(f"\n📂 פתוח: {', '.join(tr['ticker'] for tr in open_t)}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/whatsnew] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /position TICKER — full single position view (stop + chart + score + news)
    if cmd in ("/position", "position", "פוזיציה") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /position AAPL"
        try:
            import database as _db
            trade = _db.get_open_trade_by_ticker(_ticker)
            if not trade:
                return f"❌ אין פוזיציה פתוחה עבור <b>{_ticker}</b>"
            # Price
            pos = broker.get_position(_ticker)
            cur    = float(pos.get("current_price", trade["entry_price"])) if pos else trade["entry_price"]
            entry  = float(trade["entry_price"])
            qty    = float(trade["qty"])
            stop   = float(trade.get("atr_stop_price") or entry * 0.95)
            wm     = float(trade.get("high_watermark") or entry)
            pct    = (cur - entry) / entry * 100
            pnl    = (cur - entry) * qty
            val    = cur * qty
            stop_pct = (cur - stop) / cur * 100
            in_prof  = stop > entry
            held_h   = 0.0
            try:
                from datetime import datetime, timezone as _tz
                ed = datetime.strptime(str(trade.get("entry_time",""))[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz.utc)
                held_h = (datetime.now(_tz.utc) - ed).total_seconds() / 3600
            except Exception:
                pass
            # Score
            sc_str = ""
            try:
                from scoring import get_composite_score
                from sentiment import score_sentiment
                _sent = score_sentiment(_ticker)
                _comp = get_composite_score(_ticker, _sent.score)
                _sc   = _comp["composite_score"]
                _buy  = _comp["should_buy"]
                sc_str = f"\n🎯  ציון עכשיו:  <b>{_sc:.0f}/100</b>  {'✅' if _buy else '⚠️ שקול מכירה'}"
            except Exception:
                pass
            # Mini chart (5 days)
            chart_str = ""
            try:
                import yfinance as _yf
                _h = _yf.Ticker(_ticker).history(period="5d", interval="1h", auto_adjust=True)
                if len(_h) >= 5:
                    _cls  = [float(v) for v in _h["Close"].dropna()][-20:]
                    _mn, _mx = min(_cls), max(_cls)
                    _bars = ["▁","▂","▃","▄","▅","▆","▇","█"]
                    _rng  = _mx - _mn or 1
                    _chart = "".join(_bars[round((v - _mn) / _rng * 7)] for v in _cls)
                    chart_str = f"\n<code>{_chart}</code>  (5 ימים)"
            except Exception:
                pass
            return (
                f"📂 <b>פוזיציה — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📍  מחיר עכשיו:  <b>{_fmt_price(cur)}</b>  ({pct:+.1f}%)"
                f"{chart_str}\n\n"
                f"📌  קנייה:        {_fmt_price(entry)}\n"
                f"🔢  כמות:          {qty} מניות\n"
                f"💼  שווי:           {_fmt_price(val)}\n"
                f"⏳  הוחזק:         {_fmt_held(held_h)}\n\n"
                f"🛑  Stop:          {_fmt_price(stop)}  ({'💚 ברווח' if in_prof else '❤️ בהפסד'}, {stop_pct:.1f}% מרחק)\n"
                f"🏆  שיא:           {_fmt_price(wm)}\n"
                f"{'💚' if pnl>=0 else '❤️'}  רווח/הפסד:          <b>{_fmt_pnl(pnl)}</b>"
                f"{sc_str}"
            )
        except Exception as e:
            logger.error(f"[/position] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /quick TICKER — instant overview: price + score + 52w + sentiment
    if cmd in ("/quick", "quick", "מהיר", "סקירה מהירה") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /quick AAPL"
        try:
            import yfinance as _yf
            fi   = _yf.Ticker(_ticker).fast_info
            cur  = float(getattr(fi, "last_price", 0) or 0)
            prev = float(getattr(fi, "previous_close", cur) or cur)
            hi52 = float(getattr(fi, "year_high",  0) or 0)
            lo52 = float(getattr(fi, "year_low",   0) or 0)
            chg  = (cur - prev) / prev * 100 if prev else 0
            w52_pos = (cur - lo52) / (hi52 - lo52) * 100 if hi52 > lo52 else 50
            if cur <= 0:
                return f"❌ לא הצלחתי לקבל נתונים"
            # Score
            try:
                from scoring import get_composite_score
                from sentiment import score_sentiment
                _sent = score_sentiment(_ticker)
                _comp = get_composite_score(_ticker, _sent.score)
                _sc   = _comp["composite_score"]
                _buy  = _comp["should_buy"]
                score_line = f"🎯  ציון: <b>{_sc:.0f}/100</b>  {'✅ קנה' if _buy else '❌ דלג'}  {_score_bar(_sc, 8)}"
                sent_line  = f"🧠  סנטימנט: {_sent.score}/10"
            except Exception:
                score_line = "🎯  ציון: חישוב..."
                sent_line  = ""
            chg_icon = "📈" if chg >= 0 else "📉"
            w52_icon = "🔴" if w52_pos >= 85 else ("🟢" if w52_pos <= 20 else "⚪")
            return (
                f"⚡ <b>סקירה מהירה — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💲  מחיר:   <b>{_fmt_price(cur)}</b>  {chg_icon} {chg:+.2f}%\n"
                f"{w52_icon}  52W:     {w52_pos:.0f}%  (שיא {_fmt_price(hi52)})\n"
                f"{score_line}\n"
                f"{sent_line}"
            )
        except Exception as e:
            logger.error(f"[/quick] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /ask QUESTION — explicit AI question (always goes to LLM)
    if cmd in ("/ask", "ask", "שאל", "שאלה") and len(t.split()) > 1:
        question = text[len(t.split()[0]):].strip()
        if not question:
            return "💬 שאל כל שאלה!\nדוגמה: <code>/ask האם כדאי לקנות AAPL עכשיו?</code>"
        client = _get_client()
        if not client:
            return "⚙️ Groq API לא מוגדר"
        try:
            ctx       = context
            ils_rate  = _get_usd_ils()
            positions = ctx.get("open_positions", [])
            pos_brief = ", ".join(f"{p['ticker']} {p['pct']:+.1f}%" for p in positions) or "אין"
            equity    = ctx.get("equity", 0)
            vix       = ctx.get("vix", "—")
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content":
                        f"אתה מנהל השקעות מקצועי. ענה בעברית בלבד, קצר וממוקד (עד 5 שורות).\n"
                        f"תיק: ${equity:,.0f} | VIX: {vix} | פוזיציות: {pos_brief}\n"
                        f"שער דולר: ₪{ils_rate:.2f}"},
                    {"role": "user", "content": question},
                ],
                max_tokens=300, temperature=0.4,
            )
            answer = resp.choices[0].message.content.strip()
            # Ensure Hebrew — translate if too much English
            _heb = sum(1 for c in answer if 'א' <= c <= 'ת')
            _lat = sum(1 for c in answer if c.isalpha() and c.isascii())
            if _lat > _heb and len(answer) > 15:
                try:
                    _tr = client.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=[{"role": "user", "content": f"תרגם לעברית בלבד:\n{answer}"}],
                        max_tokens=400, temperature=0.2,
                    )
                    answer = _tr.choices[0].message.content.strip() or answer
                except Exception:
                    pass
            return f"🤖 <b>AI</b>\n━━━━━━━━━━━━━━━━\n{answer}"
        except Exception as e:
            logger.error(f"[/ask] Error: {type(e).__name__}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /52week TICKER — 52-week context
    if cmd in ("/52week", "/52", "52week", "שיא", "שפל") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /52week AAPL"
        try:
            import yfinance as _yf
            tk   = _yf.Ticker(_ticker)
            info = tk.fast_info
            cur  = float(getattr(info, "last_price", 0) or 0)
            hi52 = float(getattr(info, "year_high",  0) or 0)
            lo52 = float(getattr(info, "year_low",   0) or 0)
            if cur <= 0 or hi52 <= 0:
                return f"❌ לא הצלחתי לקבל נתונים עבור {_ticker}"
            # Where is current price in 52w range?
            rng  = hi52 - lo52
            pos  = (cur - lo52) / rng * 100 if rng else 50
            bar_fill = max(0, min(20, round(pos / 5)))
            bar      = "▓" * bar_fill + "│" + "░" * (20 - bar_fill)
            # Signal
            if pos >= 90:   signal = "🔴 ליד שיא — שים לב לתנגדות"
            elif pos >= 70: signal = "🟡 טוב, מעל אמצע הטווח"
            elif pos <= 20: signal = "🟢 ליד שפל — הזדמנות פוטנציאלית"
            else:           signal = "⚪ אמצע הטווח"
            pct_from_high = (hi52 - cur) / hi52 * 100
            pct_from_low  = (cur - lo52) / lo52 * 100
            return (
                f"📏 <b>52 שבועות — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🏔️  שיא:    {_fmt_price(hi52)}  (-{pct_from_high:.1f}% ממנו)\n"
                f"📍  עכשיו:  <b>{_fmt_price(cur)}</b>\n"
                f"🏔️  שפל:   {_fmt_price(lo52)}  (+{pct_from_low:.1f}% ממנו)\n\n"
                f"<code>שפל{bar}שיא</code>\n"
                f"מיקום: <b>{pos:.0f}%</b> מהטווח\n\n"
                f"{signal}"
            )
        except Exception as e:
            logger.error(f"[/52week] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /explain TERM — explain financial terms in simple Hebrew
    if cmd in ("/explain", "explain", "הסבר", "מה זה") and len(t.split()) > 1:
        term = " ".join(t.split()[1:]).strip().lower()
        explanations = {
            "rsi":         ("📊 RSI — מדד כוח יחסי",
                            "מספר בין 0-100. מתחת ל-30 = מניה 'מכורה יתר' (קנייה פוטנציאלית). "
                            "מעל 70 = 'קנייה יתר' (שים לב). הבוט מחפש RSI 35-65 — לא קיצוני."),
            "atr":         ("📐 ATR — טווח אמיתי ממוצע",
                            "כמה דולר המניה זזה בממוצע ביום. TSLA עם ATR=$10 = נורמלי לזוז $10 ביום. "
                            "הבוט מגדיר Stop Loss לפי ATR — מניות תנודתיות מקבלות stop רחב יותר."),
            "stop loss":   ("🛑 Stop Loss — עצירת הפסד",
                            "מחיר שמתחתיו הבוט מוכר אוטומטית. דוגמה: קנית ב-$100, Stop ב-$95 — "
                            "אם המחיר יורד ל-$95 הבוט מוכר מיד. מגן מהפסדים גדולים."),
            "trailing stop": ("🔄 Trailing Stop — עצירה נגררת",
                              "Stop שעולה עם המחיר. קנית ב-$100, Stop ב-$95. מחיר עלה ל-$110 — "
                              "Stop עולה ל-$105. כך נועל רווחים תוך כדי ריצה."),
            "vix":         ("🌡️ VIX — מדד הפחד",
                            "מודד פחד/תנודתיות בשוק. מתחת ל-20 = שוק רגוע. 20-28 = דאגה. "
                            "מעל 28 = פחד — הבוט עוצר קניות. מעל 30 = בהלה."),
            "macd":        ("📈 MACD — המגמה הנגררת",
                            "אות מגמה: כשקו MACD חוצה מעלה את קו האות = אות קנייה. "
                            "הבוט בודק שהמניה בתנופה חיובית לפני כל קנייה."),
            "kelly":       ("📐 Kelly Criterion — גודל פוזיציה",
                            "נוסחה מתמטית לגודל פוזיציה אופטימלי. מחשבת כמה להשקיע לפי "
                            "אחוז הצלחה וממוצע רווח/הפסד. מתחיל לפעול אחרי 30 עסקאות."),
            "fear greed":  ("😨 Fear & Greed — פחד וחמדנות",
                            "מדד 0-100. מתחת ל-25 = פחד קיצוני (הזדמנות קנייה!). "
                            "מעל 75 = חמדנות קיצונית (שוק מוגזם). הבוט עדיף לקנות בפחד."),
            "sma":         ("📉 SMA — ממוצע נע פשוט",
                            "ממוצע מחיר לאורך תקופה. SMA50 = ממוצע 50 יום אחרונים. "
                            "מחיר מעל SMA50 = מגמה עולה. הבוט לא קונה כש-SPY מתחת ל-SMA50."),
            "sharpe":      ("⚖️ Sharpe Ratio — יחס שארפ",
                            "מדד לאיכות התשואה. מעל 1 = טוב. מעל 2 = מצוין. "
                            "מחשב תשואה חלקי סטיית תקן — עדיף רווח יציב על רווח גדול ותנודתי."),
        }
        # Find best match
        result = None
        for key, val in explanations.items():
            if key in term or term in key:
                result = val
                break
        if result:
            title, explanation = result
            return f"{title}\n━━━━━━━━━━━━━━━━\n{explanation}"
        else:
            available = " · ".join(explanations.keys())
            return (
                f"❓ <b>הסבר מונח</b>\n━━━━━━━━━━━━━━━━\n"
                f"לא מצאתי הסבר ל-\"{term}\"\n\n"
                f"<b>מונחים זמינים:</b>\n{available}\n\n"
                f"דוגמה: <code>/explain RSI</code>"
            )

    # /gainers — top movers from watchlist today
    if cmd in ("/gainers", "gainers", "עולים", "מנצחים היום"):
        try:
            import yfinance as _yf
            from scanner import get_watchlist
            import random
            wl     = get_watchlist()
            sample = random.sample(wl, min(30, len(wl)))
            # Quick batch download
            prices = _yf.download(sample, period="2d", progress=False, auto_adjust=True)
            movers = []
            _cols = prices.columns.get_level_values(0) if hasattr(prices.columns, "get_level_values") else prices.columns
            if not prices.empty and "Close" in _cols:
                close = prices["Close"]
                for tk in sample:
                    try:
                        if tk in close.columns and len(close[tk].dropna()) >= 2:
                            prev = float(close[tk].iloc[-2])
                            cur  = float(close[tk].iloc[-1])
                            if prev > 0:
                                movers.append((tk, (cur - prev) / prev * 100, cur))
                    except Exception:
                        pass
            movers.sort(key=lambda x: x[1], reverse=True)
            lines = [f"🚀 <b>מובילים היום (מ-{len(movers)} מניות)</b>\n━━━━━━━━━━━━━━━━"]
            if movers:
                lines.append("📈 <b>עולות:</b>")
                for tk, chg, cur in movers[:5]:
                    lines.append(f"  🟢 <b>{tk}</b>  {chg:+.2f}%  |  {_fmt_price(cur)}")
                lines.append("\n📉 <b>יורדות:</b>")
                for tk, chg, cur in movers[-3:]:
                    lines.append(f"  🔴 <b>{tk}</b>  {chg:+.2f}%  |  {_fmt_price(cur)}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/gainers] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /macro — upcoming economic events
    if cmd in ("/memory", "memory", "זיכרון", "זיכרון אירועים"):
        try:
            from event_memory import get_memory_summary
            return get_memory_summary()
        except Exception as e:
            return f"❌ שגיאה: {e}"

    if cmd in ("/macro", "macro", "אירועים", "לוח כלכלי", "אירועים כלכליים"):
        try:
            from trading_hours import is_high_impact_day
            from datetime import datetime, timezone, timedelta
            now_utc = datetime.now(timezone.utc)
            lines   = [f"📅 <b>אירועים כלכליים קרובים</b>\n━━━━━━━━━━━━━━━━"]
            # Check next 14 days
            found = []
            for d in range(14):
                check_date = now_utc + timedelta(days=d)
                # Temporarily mock the date check using trading_hours
                try:
                    import os as _os
                    import trading_hours as _th
                    orig = _th._today_str if hasattr(_th, "_today_str") else None
                    impact, event = is_high_impact_day()
                    if d == 0 and impact:
                        found.append((check_date, event, "🔴 היום!"))
                except Exception:
                    pass
            # Static upcoming events (common schedule)
            il_off = 3 if 3 <= now_utc.month <= 10 else 2
            events_static = [
                ("CPI (מדד מחירים)", "📊", 14),
                ("NFP (תעסוקה)", "💼", 7),
                ("FOMC (ריבית)", "🏦", 21),
            ]
            for name, icon, days_ahead in events_static:
                est_date = now_utc + timedelta(days=days_ahead)
                il_date  = est_date + timedelta(hours=il_off)
                lines.append(f"{icon} <b>{name}</b>\n   🗓️ בערך: {il_date.strftime('%d/%m')} (הערכה)")
            lines.append(f"\n━━━━━━━━━━━━━━━━")
            lines.append(f"⚠️ ב-3 ימים לפני אירועים — הבוט לא קונה")
            lines.append(f"💡 לתאריכים מדויקים: investing.com/economic-calendar")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/macro] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # Smart fallback when command sent without required ticker/args
    _cmd_tips = {
        "/news":        ("📰", "חדשות על מניה",      "/news AAPL"),
        "/score":       ("🎯", "ציון מניה",           "/score TSLA"),
        "/price":       ("💲", "מחיר מניה",           "/price NVDA"),
        "/earnings":    ("📅", "דוח רווחים",          "/earnings AAPL"),
        "/sector":      ("🏢", "סקטור המניה",         "/sector MSFT"),
        "/stop":        ("🛑", "סטופ לוס",             "/stop AAPL"),
        "/compare":     ("⚔️", "השוואה",              "/compare AAPL MSFT"),
        "/watchadd":    ("➕", "הוסף מניה לרשימה",    "/watchadd PLTR"),
        "/watchremove": ("➖", "הסר מניה מהרשימה",    "/watchremove PLTR"),
        "/levels":      ("📐", "רמות תמיכה/תנגדות",  "/levels AAPL"),
        "/remind":      ("⏰", "תזכורת",              "/remind 17:30 לבדוק TSLA"),
        "/target":      ("🎯", "יעד רווח ידני",       "/target AAPL 210"),
        "/volume":      ("📊", "נפח מסחר",            "/volume AAPL"),
        "/chart":       ("📊", "גרף 30 ימים",         "/chart AAPL"),
        "/dividend":    ("💰", "דיבידנד",             "/dividend AAPL"),
        "/fundamental": ("📈", "פונדמנטלס",           "/fundamental AAPL"),
        "/volatility":  ("📐", "תנודתיות",            "/volatility AAPL"),
        "/alert":       ("🔔", "התראת מחיר",          "/alert AAPL 200"),
    }
    if cmd in _cmd_tips and len(t.split()) < (3 if cmd in ("/compare", "/remind", "/target", "/alert") else 2):
        icon, label, example = _cmd_tips[cmd]
        return f"{icon} <b>{label}</b>\n\nדוגמה: <code>{example}</code>"

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
            # Translate headlines to Hebrew
            translated = []
            try:
                cli = _get_client()
                if cli and headlines:
                    _raw = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:5]))
                    _tr  = cli.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=[{"role": "user", "content":
                            f"תרגם את הכותרות הבאות לעברית קצרה (עד 12 מילים). "
                            f"החזר רק כותרות ממוספרות:\n{_raw}"}],
                        max_tokens=300, temperature=0.2,
                    )
                    for ln in _tr.choices[0].message.content.strip().split("\n"):
                        ln = ln.strip()
                        if ln and ln[0].isdigit():
                            translated.append(ln.split(". ", 1)[-1].strip())
            except Exception:
                pass
            display = translated if translated else [h[:90] for h in headlines[:5]]
            lines = [f"📰 <b>חדשות — {_ticker}</b>\n━━━━━━━━━━━━━━━━", sent_line, ""]
            for i, h in enumerate(display, 1):
                lines.append(f"{i}. {h}")
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
            vix    = result.get("vix", "—")
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
                f"{'✅ <b>המלצה: קנה</b>' if buy else '⏭️ <b>המלצה: דלג</b>'}\n\n"
                f"<i>💡 "
                + (
                    "ציון גבוה + סנטימנט חיובי + שוק תומך" if buy and sc >= 70 and sent.score >= 7 else
                    "ציון טוב אבל סנטימנט ניטרלי — כדאי לבדוק חדשות" if buy and sent.score < 6 else
                    "ציון גבולי — ממתין לאות חזק יותר" if not buy and sc >= 55 else
                    "ציון נמוך — הבוט ידלג על המניה"
                ) + "</i>"
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
                verdict  = summary.get("verdict", "")
                # Translate English blocker strings to Hebrew
                _blocker_map = {
                    "SPY below SMA50": "SPY מתחת ל-SMA50 — שוק בירידה",
                    "VIX": "VIX גבוה — שוק תנודתי",
                    "Circuit breaker": "Circuit Breaker פעיל — הפסד יומי מקסימלי",
                    "Max positions": "מספר פוזיציות מקסימלי הושג",
                    "Market closed": "השוק סגור",
                    "Not enough cash": "אין מספיק מזומן",
                    "None": "אין חסמים",
                    "Bot is paused": "הבוט מושהה",
                    "High-impact": "אירוע כלכלי משמעותי היום",
                }
                def _translate_blocker(b):
                    for eng, heb in _blocker_map.items():
                        if eng.lower() in b.lower():
                            return heb
                    return b  # return as-is if no match
                lines = [f"🔍 <b>אבחון</b>\n━━━━━━━━━━━━━━━━"]
                # Translate verdict
                if "SHOULD be buying" in verdict:
                    lines.append("✅ הבוט אמור לקנות — בדוק ציוני מניות")
                elif "BLOCKED" in verdict:
                    lines.append("❌ הבוט חסום — ראה חסמים למטה")
                else:
                    lines.append(verdict)
                if blockers:
                    lines.append("\n<b>חסמים:</b>")
                    for b in blockers:
                        lines.append(f"⛔ {_translate_blocker(b)}")
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
                f"📊 הזדמנויות שנותחו: {total}\n"
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
            from sentiment import _sentiment_cache, CACHE_TTL
            import database as _db, random, time as _t_top
            wl   = get_watchlist()
            held = {tr["ticker"] for tr in (_db.get_open_trades() or [])}
            wl   = [tk for tk in wl if tk not in held]

            # Phase 1: quick momentum pre-filter via yfinance batch (fast ~2s)
            import yfinance as _yf_top
            sample_large = random.sample(wl, min(20, len(wl)))  # reduced 40→20 for speed
            _prices_top  = _yf_top.download(sample_large, period="2d",
                                             progress=False, auto_adjust=True)
            _chg_map: dict[str, float] = {}
            _cols = _prices_top.columns.get_level_values(0) if hasattr(_prices_top.columns, "get_level_values") else _prices_top.columns
            if not _prices_top.empty and "Close" in _cols:
                _cl = _prices_top["Close"]
                import pandas as _pd_top
                _cl_df = _cl if isinstance(_cl, _pd_top.DataFrame) else _cl.to_frame()
                for _tk2 in sample_large:
                    try:
                        _s = _cl_df[_tk2].dropna() if _tk2 in _cl_df.columns else None
                        if _s is not None and len(_s) >= 2:
                            _chg_map[_tk2] = float((_s.iloc[-1] - _s.iloc[-2]) / _s.iloc[-2] * 100)
                    except Exception:
                        pass
            # Sort by momentum, take top 6 candidates (reduced 12→6 for speed)
            sample = sorted(_chg_map, key=_chg_map.get, reverse=True)[:6]
            if not sample:
                sample = random.sample(wl, min(8, len(wl)))

            # Phase 2: full score — use CACHED sentiment (avoid slow RSS fetch)
            results = []
            _top_start = _t_top.time()
            for _tk in sample:
                if _t_top.time() - _top_start > 10:  # 10s max (reduced 17→10)
                    break
                try:
                    # Use cached sentiment if fresh, else neutral (5) for speed
                    _cached_sent = _sentiment_cache.get(_tk)
                    if _cached_sent and (_t_top.time() - _cached_sent.timestamp) < CACHE_TTL:
                        _sent_score = _cached_sent.score
                    else:
                        _sent_score = 5  # neutral — avoid slow RSS for /top
                    comp = get_composite_score(_tk, _sent_score)
                    results.append((_tk, comp["composite_score"], comp["should_buy"], _sent_score))
                except Exception:
                    continue
            results.sort(key=lambda x: x[1], reverse=True)
            buys  = [(t,s,b,ss) for t,s,b,ss in results if b]
            skips = [(t,s,b,ss) for t,s,b,ss in results if not b]
            lines = [f"🏆 <b>סריקה — {len(results)} מניות</b>\n━━━━━━━━━━━━━━━━"]
            if buys:
                lines.append(f"✅ <b>מעל סף קנייה ({len(buys)}):</b>")
                for tk, sc, _, ss in buys:
                    lines.append(f"  🟢 <b>{tk}</b>  {sc:.0f}/100  {_score_bar(sc, 8)}  🧠{ss}/10")
            if skips:
                lines.append(f"\n⏭️ <b>מתחת לסף ({len(skips)}):</b>")
                for tk, sc, _, ss in skips:
                    lines.append(f"  ⚪ <b>{tk}</b>  {sc:.0f}/100  {_score_bar(sc, 8)}")
            lines.append(f"\n━━━━━━━━━━━━━━━━\n🎯 סף קנייה: <b>{MIN_BUY_SCORE}</b>  |  /scan לביצוע מיידי")
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

    # /fear — פחד וחמדנות
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
                f"😨 <b>פחד וחמדנות</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 ציון: <b>{fg}/100</b>\n"
                f"💭 מצב: {label}\n"
                f"🌡️ VIX: {vix or '—'}\n"
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
                    lines.append(f"⛔ <b>עצירה לפני דוח: {days} ימים לדוח</b>")
                else:
                    lines.append(f"✅ הדוח הבא: בעוד {days} ימים")
            beat = impact.get("beat_rate", 0)
            avg_move = impact.get("avg_move_pct", 0)
            quarters = impact.get("quarters_analyzed", 0)
            if quarters > 0:
                lines.append(f"🎯 שיעור הצלחה: <b>{beat*100:.0f}%</b> ({quarters} רבעונים)")
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
            # Use cached sentiment (fast) — avoid RSS fetch which takes 8s each
            import time as _t_cmp
            from sentiment import _sentiment_cache as _sc, CACHE_TTL as _CTL
            def _fast_sent(tk):
                c = _sc.get(tk)
                if c and (_t_cmp.time() - c.timestamp) < _CTL:
                    return c.score
                return 5  # neutral if no cache
            s1_sent = _fast_sent(t1)
            s2_sent = _fast_sent(t2)
            r1    = get_composite_score(t1, s1_sent)
            r2    = get_composite_score(t2, s2_sent)
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

            # Buffett analysis for both
            try:
                from buffett_analysis import get_buffett_analysis
                buf1 = get_buffett_analysis(t1)
                buf2 = get_buffett_analysis(t2)
                bs1, bs2 = buf1.get("score", 0), buf2.get("score", 0)
                bm1, bm2 = buf1.get("moat", "weak"), buf2.get("moat", "weak")
                moat_icon = {"strong": "💪", "medium": "🛡️", "weak": "⚠️"}
                bm1_str = f"{moat_icon.get(bm1, '?')} {bm1}"
                bm2_str = f"{moat_icon.get(bm2, '?')} {bm2}"
            except Exception:
                bs1, bs2, bm1_str, bm2_str = 0, 0, "?", "?"

            # Combined winner: 50% composite + 50% Buffett
            combined1 = (s1 + bs1) / 2
            combined2 = (s2 + bs2) / 2
            winner_combined = t1 if combined1 >= combined2 else t2
            combined_diff = abs(combined1 - combined2)

            return (
                f"⚔️ <b>{t1}  vs  {t2}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>{t1}</b>\n"
                f"  ⭐ ציון טכני: {s1:.0f}/100  {_score_bar(s1, 8)}\n"
                f"  🎩 ציון באפט: {bs1:.0f}/100  |  Moat: {bm1_str}\n"
                f"  📅 יום: {chg1:+.2f}%  |  {b1}\n\n"
                f"<b>{t2}</b>\n"
                f"  ⭐ ציון טכני: {s2:.0f}/100  {_score_bar(s2, 8)}\n"
                f"  🎩 ציון באפט: {bs2:.0f}/100  |  Moat: {bm2_str}\n"
                f"  📅 יום: {chg2:+.2f}%  |  {b2}\n\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🏆 <b>עדיף: {winner_combined}</b> (פער משוקלל {combined_diff:.0f}/100)\n"
                f"💡 השווייה משוקללת: 50% טכני + 50% באפט"
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
            chart_line = f"\n<code>רווח/הפסד: {chart}</code>  ({len(days_sorted)} ימים)"

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
                f"🎯  אחוז הצלחה: <b>{wr}%</b>\n"
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

    # /health — combined position health: score + news + stop in one view
    if cmd in ("/health", "health", "בריאות", "מצב פוזיציות"):
        try:
            import database as _db
            open_trades = _db.get_open_trades() or []
            if not open_trades:
                return "📭 אין פוזיציות פתוחות"
            from scoring import get_composite_score
            from sentiment import _sentiment_cache as _hsc, CACHE_TTL as _HCTL
            import time as _t_health
            lines = [f"🩺 <b>בריאות תיק</b>\n━━━━━━━━━━━━━━━━"]
            _health_start = _t_health.time()
            for tr in open_trades:
                if _t_health.time() - _health_start > 16:  # hard-stop
                    lines.append("⏳ עוד פוזיציות — לא הספקתי לבדוק הכל")
                    break
                tk    = tr.get("ticker", "?")
                entry = float(tr.get("entry_price") or 0)
                stop  = float(tr.get("atr_stop_price") or 0)
                try:
                    pos = broker.get_position(tk)
                    cur = float(pos.get("current_price", entry)) if pos else entry
                    pct = (cur - entry) / entry * 100 if entry else 0
                except Exception:
                    cur, pct = entry, 0
                # Score — use cached sentiment to avoid RSS slowness
                try:
                    _hc = _hsc.get(tk)
                    _hs = _hc.score if _hc and (_t_health.time()-_hc.timestamp)<_HCTL else 5
                    comp  = get_composite_score(tk, _hs)
                    sc    = comp["composite_score"]
                    buy   = comp["should_buy"]
                    # Create a minimal sent-like object for display
                    class _S: score = _hs
                    sent = _S() if _hc else None
                except Exception:
                    sc, buy, sent = 0, False, None
                # Stop distance
                stop_dist = (cur - stop) / cur * 100 if stop and cur else 0
                # Health icons
                score_icon = "🟢" if sc >= 65 else ("🟡" if sc >= 50 else "🔴")
                pnl_icon   = "📈" if pct >= 0 else "📉"
                news_icon  = "📰"
                sent_icon  = ("🟢" if sent and sent.score >= 7 else
                              "🟡" if sent and sent.score >= 5 else "🔴") if sent else "⚪"
                lines.append(
                    f"\n{'🟢' if pct >= 0 else '🔴'} <b>{tk}</b>  {pct:+.1f}%\n"
                    f"  {score_icon} ציון: <b>{sc:.0f}/100</b>  {'✅' if buy else '⚠️ שקול מכירה'}\n"
                    f"  {sent_icon} סנטימנט: {sent.score if sent else '?'}/10\n"
                    f"  🛑 סטופ מרחק: <b>{stop_dist:.1f}%</b>  "
                    f"({'💚 ברווח' if stop > entry else '❤️ בהפסד'} אם מופעל)"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/health] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /volume TICKER — volume analysis
    if cmd in ("/volume", "volume", "נפח", "נפח מסחר") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /volume AAPL"
        try:
            from volume_confirm import check as _vol_check, get_current_ratio as _vol_ratio
            passed, reason, details = _vol_check(_ticker)
            ratio = details.get("ratio") or _vol_ratio(_ticker) or 0
            cur_vol = details.get("current_volume", 0)
            ma_vol  = details.get("ma_volume", 0)
            # Visual bar
            fill = min(20, round(ratio * 10)) if ratio else 0
            bar  = "█" * fill + "░" * (20 - fill)
            status_icon = "🟢" if passed else "🔴"
            status_str  = "✅ נפח מאשר אות" if passed else "⚠️ נפח נמוך — אות חלש"
            return (
                f"📊 <b>נפח מסחר — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"{status_icon} {status_str}\n\n"
                f"📈  נפח עכשיו:   <b>{cur_vol:,.0f}</b>\n"
                f"📉  ממוצע 20 יום: {ma_vol:,.0f}\n"
                f"⚡  יחס:           <b>{ratio:.2f}×</b>  (נדרש ≥ 1.0)\n"
                f"<code>{bar}</code>"
            )
        except Exception as e:
            logger.error(f"[/volume] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /target TICKER PRICE — set custom take profit
    if cmd in ("/target", "target", "יעד") and len(t.split()) >= 3:
        parts   = t.split()
        _ticker = _safe_ticker(parts[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /target AAPL 210"
        try:
            _tp_price = float(parts[2])
        except ValueError:
            return "❌ מחיר לא חוקי — דוגמה: /target AAPL 210"
        try:
            import database as _db
            trade = _db.get_open_trade_by_ticker(_ticker)
            if not trade:
                return f"❌ אין פוזיציה פתוחה עבור <b>{_ticker}</b>"
            entry  = float(trade.get("entry_price") or 0)
            upside = (_tp_price - entry) / entry * 100 if entry else 0
            # Store as env var (simple persistence)
            import os as _os
            key   = f"CUSTOM_TP_{_ticker}"
            _os.environ[key] = str(_tp_price)
            return (
                f"🎯 <b>יעד מותאם אישית — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📌  כניסה:  {_fmt_price(entry)}\n"
                f"🎯  יעד חדש: <b>{_fmt_price(_tp_price)}</b>\n"
                f"📈  פוטנציאל: <b>{upside:+.1f}%</b>\n\n"
                f"✅ הבוט ישמור על עין ויתריע כשמגיעים ליעד"
            )
        except Exception as e:
            logger.error(f"[/target] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /pnl — quick profit/loss number
    if cmd in ("/pnl", "pnl", "כמה עשיתי"):
        positions = context.get("open_positions", [])
        open_pnl  = context.get("open_pnl", 0)
        realized  = context.get("realized_pnl_net", 0)
        total     = open_pnl + realized
        equity    = context.get("equity", 0)
        cash      = context.get("cash", 0)
        invested  = equity - cash

        lines = [f"💰 <b>רווח/הפסד — תיק</b>\n━━━━━━━━━━━━━━━━"]

        if positions:
            for p in positions:
                e = "🟢" if p["pnl"] >= 0 else "🔴"
                qty_str = f"{p['qty']:.4f}".rstrip('0').rstrip('.')
                val = p.get("value") or round(p["current"] * p["qty"], 2)
                held = _fmt_held(p.get("held_hours", 0))
                lines.append(f"\n{e} <b>{p['ticker']}</b>")
                lines.append(f"   🔢 כמות:          {qty_str} מניות")
                lines.append(f"   💵 כניסה:         {_fmt_price(p['entry'])}")
                lines.append(f"   📍 עכשיו:         <b>{_fmt_price(p['current'])}</b>  ({p['pct']:+.2f}%)")
                lines.append(f"   💰 רווח/הפסד:   {_fmt_pnl(p['pnl'], False)}")
                lines.append(f"   💼 שווי:           {_fmt_price(val)}")
                lines.append(f"   ⏱️ הוחזק:         {held}")
        else:
            lines.append("אין פוזיציות פתוחות כרגע")

        lines.append(f"\n━━━━━━━━━━━━━━━━")
        lines.append(f"💼  מושקע:    {_fmt_price(invested)}")
        lines.append(f"📂  רווח פתוח:  {_fmt_pnl(open_pnl)}")
        if realized:
            lines.append(f"💳  ממומש:     {_fmt_pnl(realized)}")
        icon = "📈" if total >= 0 else "📉"
        lines.append(f"{icon}  <b>סה״כ: {_fmt_pnl(total)}</b>")
        return "\n".join(lines)

    # /chart TICKER — ASCII price chart (30 days)
    if cmd in ("/chart", "chart", "גרף") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /chart AAPL"
        try:
            import yfinance as _yf
            hist = _yf.Ticker(_ticker).history(period="30d", auto_adjust=True)
            if hist.empty or len(hist) < 5:
                return f"❌ אין מספיק נתונים עבור {_ticker}"
            closes = [float(v) for v in hist["Close"].dropna()]
            hi, lo = max(closes), min(closes)
            rng = hi - lo or 1
            # 8-level bar chart (one bar per day, max 30)
            bars   = ["▁","▂","▃","▄","▅","▆","▇","█"]
            chart  = "".join(bars[round((v - lo) / rng * 7)] for v in closes[-20:])
            cur    = closes[-1]
            prev   = closes[-2] if len(closes) >= 2 else cur
            chg    = (cur - prev) / prev * 100 if prev else 0
            chg30  = (cur - closes[0]) / closes[0] * 100 if closes[0] else 0
            trend  = "📈" if chg30 >= 0 else "📉"
            return (
                f"📊 <b>גרף — {_ticker}  (30 ימים)</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<code>{chart}</code>\n\n"
                f"📍  עכשיו:  <b>{_fmt_price(cur)}</b>  ({chg:+.2f}% היום)\n"
                f"🏔️  שיא 30י:  {_fmt_price(hi)}\n"
                f"🏔️  שפל 30י: {_fmt_price(lo)}\n"
                f"{trend}  שינוי 30 יום: <b>{chg30:+.1f}%</b>"
            )
        except Exception as e:
            logger.error(f"[/chart] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /dividend TICKER — dividend info
    if cmd in ("/dividend", "dividend", "דיבידנד") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /dividend AAPL"
        try:
            import yfinance as _yf
            info = _yf.Ticker(_ticker).info
            div_rate   = float(info.get("dividendRate") or 0)
            div_yield  = float(info.get("dividendYield") or 0) * 100
            ex_date    = info.get("exDividendDate")
            pay_date   = info.get("dividendDate")
            freq       = info.get("dividendFrequency")
            freq_map   = {1:"שנתי", 2:"חצי-שנתי", 4:"רבעוני", 12:"חודשי"}
            freq_str   = freq_map.get(freq, "לא ידוע") if freq else "לא ידוע"
            if div_rate == 0 and div_yield == 0:
                return (
                    f"💰 <b>דיבידנד — {_ticker}</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"❌ מניה זו לא משלמת דיבידנד"
                )
            from datetime import datetime, timezone
            def _fmt_ts(ts):
                if not ts: return "לא ידוע"
                try:
                    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d/%m/%Y")
                except Exception: return str(ts)
            return (
                f"💰 <b>דיבידנד — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💵  דיבידנד שנתי:  <b>${div_rate:.2f}</b>\n"
                f"📊  תשואת דיבידנד: <b>{div_yield:.2f}%</b>\n"
                f"📅  תאריך פקיעה:   {_fmt_ts(ex_date)}\n"
                f"💳  תאריך תשלום:   {_fmt_ts(pay_date)}\n"
                f"🔄  תדירות:          {freq_str}"
            )
        except Exception as e:
            logger.error(f"[/dividend] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /buffett TICKER — Warren Buffett style deep analysis
    if cmd in ("/buffett", "buffett", "באפט") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /buffett AAPL"
        try:
            from buffett_analysis import get_buffett_analysis, format_buffett_report
            analysis = get_buffett_analysis(_ticker)
            return format_buffett_report(analysis)
        except Exception as e:
            logger.error(f"[/buffett] Error: {e}")
            return f"❌ לא הצלחתי לנתח את {_ticker} — נסה שוב מאוחר יותר"

    # /fundamental TICKER — P/E, revenue, margins
    if cmd in ("/fundamental", "fundamental", "פונדמנטלס", "יסודות") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /fundamental AAPL"
        try:
            import yfinance as _yf
            info = _yf.Ticker(_ticker).info
            pe         = info.get("trailingPE") or info.get("forwardPE")
            eps        = info.get("trailingEps")
            revenue    = info.get("totalRevenue")
            margin     = info.get("profitMargins")
            growth     = info.get("revenueGrowth")
            mktcap     = info.get("marketCap")
            target     = info.get("targetMeanPrice")
            cur        = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
            def _bil(v): return f"${v/1e9:.1f}B" if v and v >= 1e9 else (f"${v/1e6:.0f}M" if v else "—")
            upside     = (target - cur) / cur * 100 if target and cur else None
            analyst    = info.get("recommendationMean")
            rec_map    = {1:"🟢 קנה חזק", 1.5:"🟢 קנה", 2:"🟡 קנה", 2.5:"🟡 קנה מתון",
                          3:"⚪ החזק", 3.5:"🟡 מכור מתון", 4:"🔴 מכור", 4.5:"🔴 מכור חזק"}
            rec_str = next((v for k,v in rec_map.items() if analyst and abs(analyst-k)<=0.25), "—") if analyst else "—"
            lines = [f"📈 <b>פונדמנטלס — {_ticker}</b>\n━━━━━━━━━━━━━━━━"]
            if pe:    lines.append(f"💹  P/E Ratio:      <b>{pe:.1f}</b>")
            if eps:   lines.append(f"💰  EPS:              <b>${eps:.2f}</b>")
            if revenue: lines.append(f"📊  הכנסות:          <b>{_bil(revenue)}</b>")
            if margin:  lines.append(f"📏  מרווח נקי:      <b>{margin*100:.1f}%</b>")
            if growth:  lines.append(f"📈  צמיחת הכנסות: <b>{growth*100:+.1f}%</b>")
            if mktcap:  lines.append(f"🏦  שווי שוק:        <b>{_bil(mktcap)}</b>")
            if target:  lines.append(f"🎯  יעד אנליסטים: <b>{_fmt_price(target)}</b>"
                                     + (f"  ({upside:+.1f}%)" if upside else ""))
            lines.append(f"📋  המלצה:          {rec_str}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/fundamental] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /review — AI reviews all positions and gives recommendations
    if cmd in ("/review", "review", "סקירה", "בדוק פוזיציות"):
        client = _get_client()
        if not client:
            return "⚙️ Groq API לא מוגדר"
        try:
            positions = context.get("open_positions", [])
            if not positions:
                return "📭 אין פוזיציות לסקירה"
            vix     = context.get("vix", "—")
            equity  = context.get("equity", 0)
            ils_r   = _get_usd_ils()
            pos_lines = "\n".join(
                f"- {p['ticker']}: קנייה ${p['entry']:.2f}, עכשיו ${p['current']:.2f} "
                f"({p['pct']:+.1f}%), P&L ${p['pnl']:+.2f}, "
                f"Stop ${p.get('atr_stop',0):.2f}, הוחזק {_fmt_held(p.get('held_hours',0))}"
                for p in positions
            )
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content":
                        "אתה יועץ השקעות מקצועי. ענה בעברית בלבד. "
                        f"VIX={vix}, שווי תיק=${equity:.0f}"},
                    {"role": "user", "content":
                        f"סקור את הפוזיציות הבאות ותן המלצה קצרה לכל אחת (החזק/מכור/הוסף):\n{pos_lines}"},
                ],
                max_tokens=400, temperature=0.3,
            )
            review = resp.choices[0].message.content.strip()
            return f"🤖 <b>סקירת פוזיציות AI</b>\n━━━━━━━━━━━━━━━━\n{review}"
        except Exception as e:
            logger.error(f"[/review] Error: {type(e).__name__}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /quiet — reduce notifications (only critical alerts)
    if cmd in ("/quiet", "quiet", "שקט", "פחות התראות"):
        import os as _os
        _os.environ["QUIET_MODE"] = "true"
        return (
            "🔕 <b>מצב שקט הופעל</b>\n"
            "━━━━━━━━━━━━━━━━\n"
            "✅ קניות ומכירות — עדיין נשלחות\n"
            "🔇 סטופ לוס, חדשות, סריקות — מושתקות\n\n"
            "לחזרה: /loud"
        )

    # /loud — restore all notifications
    if cmd in ("/loud", "loud", "רועש", "כל ההתראות"):
        import os as _os
        _os.environ.pop("QUIET_MODE", None)
        return "🔔 <b>כל ההתראות הופעלו מחדש</b> ✅"

    # /exposure — sector exposure of open portfolio
    if cmd in ("/exposure", "exposure", "חשיפה", "סקטורים בתיק"):
        try:
            positions = context.get("open_positions", [])
            equity    = context.get("equity", 1) or 1
            if not positions:
                return "📭 אין פוזיציות פתוחות"
            from sector_rotation import get_sector_for_ticker, SECTOR_ETFS
            sector_vals: dict[str, float] = {}
            no_sector = []
            for p in positions:
                tk  = p["ticker"]
                val = p.get("value", p["qty"] * p["current"])
                etf = get_sector_for_ticker(tk)
                if etf:
                    name = SECTOR_ETFS.get(etf, etf)
                    sector_vals[name] = sector_vals.get(name, 0) + val
                else:
                    sector_vals["אחר"] = sector_vals.get("אחר", 0) + val
            lines = [f"🏢 <b>חשיפת תיק לסקטורים</b>\n━━━━━━━━━━━━━━━━"]
            total_invested = sum(sector_vals.values())
            for sec, val in sorted(sector_vals.items(), key=lambda x: -x[1]):
                pct  = val / equity * 100
                bar  = "█" * max(1, round(pct / 5)) + "░" * max(0, 20 - round(pct / 5))
                lines.append(f"🏢 <b>{sec}</b>  {pct:.1f}%\n   <code>{bar}</code>  {_fmt_price(val)}")
            cash     = context.get("cash", 0)
            cash_pct = cash / equity * 100
            lines.append(f"\n💵 <b>מזומן</b>  {cash_pct:.1f}%  |  <b>סה\"כ: {_fmt_price(equity)}</b>")
            # Diversification tip
            if len(sector_vals) == 1:
                lines.append(f"\n⚠️ כל הכסף בסקטור אחד — שקול פיזור")
            elif len(sector_vals) >= 3:
                lines.append(f"\n✅ מפוזר טוב — {len(sector_vals)} סקטורים")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/exposure] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /volatility TICKER — ATR-based volatility vs market
    if cmd in ("/volatility", "volatility", "תנודתיות") and len(t.split()) > 1:
        _ticker = _safe_ticker(t.split()[1])
        if not _ticker:
            return "❌ טיקר לא חוקי — דוגמה: /volatility AAPL"
        try:
            import yfinance as _yf
            hist = _yf.Ticker(_ticker).history(period="30d", auto_adjust=True)
            spy  = _yf.Ticker("SPY").history(period="30d", auto_adjust=True)
            if hist.empty or len(hist) < 10:
                return f"❌ אין מספיק נתונים עבור {_ticker}"
            # ATR%
            def _atr_pct(h):
                hi, lo, cl = h["High"], h["Low"], h["Close"]
                tr = (hi - lo).combine((hi - cl.shift(1)).abs(), max).combine((lo - cl.shift(1)).abs(), max)
                return float(tr.ewm(span=14, adjust=False).mean().iloc[-1] / cl.iloc[-1] * 100)
            atr_stock = _atr_pct(hist)
            atr_spy   = _atr_pct(spy)
            ratio     = atr_stock / atr_spy if atr_spy else 1
            # 30-day std dev %
            daily_ret = hist["Close"].pct_change().dropna()
            std_pct   = float(daily_ret.std() * 100)
            # Beta approx
            spy_ret   = spy["Close"].pct_change().dropna()
            if len(daily_ret) == len(spy_ret):
                cov   = float(daily_ret.cov(spy_ret))
                var   = float(spy_ret.var())
                beta  = cov / var if var else 1.0
            else:
                beta = ratio
            # Labels
            if ratio < 0.7:   vol_label = "🟢 נמוכה מהשוק"
            elif ratio < 1.3: vol_label = "⚪ דומה לשוק"
            elif ratio < 2.0: vol_label = "🟡 גבוהה מהשוק"
            else:             vol_label = "🔴 תנודתית מאוד"
            cur = float(hist["Close"].iloc[-1])
            return (
                f"📐 <b>תנודתיות — {_ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊  ATR%:       <b>{atr_stock:.2f}%</b>  (SPY: {atr_spy:.2f}%)\n"
                f"📈  סטיית תקן: <b>{std_pct:.2f}%</b> ביום\n"
                f"⚡  Beta:        <b>{beta:.2f}</b>\n"
                f"🏷️  {vol_label}\n\n"
                f"📍  מחיר:  {_fmt_price(cur)}"
            )
        except Exception as e:
            logger.error(f"[/volatility] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /morning — manually trigger morning briefing
    if cmd in ("/morning", "morning", "תדרוך בוקר", "בריפינג"):
        try:
            import requests as _req, os as _os
            base   = _os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000").rstrip("/")
            secret = settings.WEBHOOK_SECRET
            if not secret:
                return "⚙️ WEBHOOK_SECRET לא מוגדר"
            # timeout=8 — just kick off the briefing, don't wait for it to finish
            # The briefing sends its own message when done (async server-side)
            r = _req.post(f"{base}/telegram/briefing",
                          headers={"X-Webhook-Secret": secret}, timeout=8)
            if r.status_code in (200, 202):
                return "☀️ <b>תדרוך בוקר בדרך!</b>\n📨 יגיע תוך 30-60 שניות..."
            return f"⚠️ לא הצלחתי לשלוח (קוד {r.status_code})"
        except Exception as e:
            logger.error(f"[/morning] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # /trending — top momentum stocks from watchlist (price-based, fast)
    if cmd in ("/trending", "trending", "טרנד", "מומנטום"):
        try:
            import yfinance as _yf
            from scanner import get_watchlist
            import database as _db
            import random
            wl   = get_watchlist()
            held = {tr["ticker"] for tr in (_db.get_open_trades() or [])}
            wl   = [tk for tk in wl if tk not in held]
            sample = random.sample(wl, min(40, len(wl)))
            prices = _yf.download(sample, period="5d", progress=False, auto_adjust=True)
            trending = []
            _cols = prices.columns.get_level_values(0) if hasattr(prices.columns, "get_level_values") else prices.columns
            if not prices.empty and "Close" in _cols:
                close = prices["Close"]
                for tk in sample:
                    try:
                        s = close[tk].dropna()
                        if len(s) >= 5:
                            chg5d = (float(s.iloc[-1]) - float(s.iloc[0])) / float(s.iloc[0]) * 100
                            chg1d = (float(s.iloc[-1]) - float(s.iloc[-2])) / float(s.iloc[-2]) * 100
                            # Strong momentum: up 5 days in a row OR 3%+ in 5 days
                            if chg5d >= 3.0 or (chg5d >= 1.5 and chg1d > 0):
                                trending.append((tk, chg5d, chg1d, float(s.iloc[-1])))
                    except Exception:
                        pass
            trending.sort(key=lambda x: x[1], reverse=True)
            if not trending:
                return "📉 <b>טרנדינג</b>\n━━━━━━━━━━━━━━━━\n😴 אין מניות בתנופה עכשיו"
            lines = [f"🚀 <b>מניות בתנופה ({len(trending[:8])})</b>\n━━━━━━━━━━━━━━━━"]
            for tk, chg5, chg1, cur in trending[:8]:
                icon = "🔥" if chg5 >= 5 else "📈"
                lines.append(
                    f"{icon} <b>{tk}</b>  5י: <b>{chg5:+.1f}%</b>  |  היום: {chg1:+.1f}%  |  {_fmt_price(cur)}"
                )
            lines.append(f"\n💡 /score TICKER לניתוח מלא")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[/trending] Error: {e}")
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
                from budget import kelly_fraction
                kf = kelly_fraction()
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
                f"🛑  סטופ לוס:          {settings.STOP_LOSS_PCT}%\n"
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
            import time as _t_nc
            lines = [f"📰 <b>בדיקת חדשות בזמן אמת</b>\n━━━━━━━━━━━━━━━━"]
            _nc_start = _t_nc.time()
            for tr in open_trades:
                tk = tr.get("ticker", "")
                if not tk:
                    continue
                if _t_nc.time() - _nc_start > 7:   # hard-stop (7s max for Telegram)
                    lines.append("⏳ בדיקה חלקית — הוגבלה בזמן")
                    break
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
            lines = [f"📅 <b>סיכום היום</b>", "━━━━━━━━━━━━━━━━"]
            lines.append(f"📆 תאריך:         {today_str}")
            lines.append(f"🛒 קניות:          <b>{len(opened)}</b>")
            lines.append(f"💸 מכירות:        <b>{len(closed)}</b>")
            if closed:
                lines.append(f"💰 רווח/הפסד:   {_fmt_pnl(total_pnl)}")
                lines.append("━━━━━━━━━━━━━━━━")
                for _ct in closed:
                    _sym  = _ct.get("ticker","?")
                    _pnl  = float(_ct.get("pnl_gross") or 0)
                    _ep   = float(_ct.get("entry_price") or 0)
                    _xp   = float(_ct.get("exit_price") or 0)
                    _pct  = (_xp-_ep)/_ep*100 if _ep else 0
                    _icon = "🟢" if _pnl >= 0 else "🔴"
                    lines.append(f"{_icon} <b>{_sym}</b>")
                    lines.append(f"   📍 שינוי:      {_pct:+.1f}%")
                    lines.append(f"   💰 רווח:       {_fmt_pnl(_pnl, False)}")
            if opened:
                lines.append("━━━━━━━━━━━━━━━━")
                lines.append(f"📂 נקנו היום:")
                for _ot in opened:
                    lines.append(f"   📌 <b>{_ot.get('ticker','?')}</b>  @ {_fmt_price(_ot.get('entry_price',0))}")
            if not opened and not closed:
                lines.append("━━━━━━━━━━━━━━━━")
                lines.append("😴 לא היו עסקאות היום")
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
        # Return immediately — don't wait 60s for scan to complete (Telegram webhook timeout)
        import os as _os
        base   = _os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000").rstrip("/")
        secret = settings.WEBHOOK_SECRET
        if not secret:
            return "⚙️ WEBHOOK_SECRET לא מוגדר"
        # Fire-and-forget: run scan in background, send result when done
        async def _run_scan_and_notify():
            import requests as _rq
            try:
                # MUST use asyncio.to_thread — requests.post blocks the event loop!
                r = await asyncio.to_thread(
                    lambda: _rq.post(f"{base}/scan/now",
                                     headers={"X-Webhook-Secret": secret}, timeout=90)
                )
                if r.status_code == 200:
                    data     = r.json()
                    bought_l = data.get("bought", [])   # list of dicts
                    skipped_l= data.get("skipped", [])
                    bought   = len(bought_l) if isinstance(bought_l, list) else int(bought_l or 0)
                    scanned  = bought + (len(skipped_l) if isinstance(skipped_l, list) else 0)
                    cash     = data.get("cash_remaining", data.get("remaining_cash", 0))
                    if bought > 0:
                        tickers = ", ".join(b.get("ticker","?") for b in bought_l[:5]) if isinstance(bought_l, list) else ""
                        await send_message(
                            f"🔍 <b>סריקה הושלמה</b>\n━━━━━━━━━━━━━━━━\n"
                            f"✅  נקנו: <b>{bought} מניות</b>"
                            + (f"  ({tickers})" if tickers else "") + "\n"
                            f"🔎  נסרקו: {scanned}\n"
                            f"💵  מזומן נותר: ${cash:,.2f}"
                        )
                    else:
                        reason = data.get("reason", "לא נמצאו הזדמנויות")
                        await send_message(
                            f"🔍 <b>סריקה הושלמה</b>\n━━━━━━━━━━━━━━━━\n"
                            f"⏭️  {reason}\n"
                            f"🔎  נסרקו: {scanned} מניות"
                        )
                else:
                    await send_message(f"⚠️ סריקה נכשלה (קוד {r.status_code})")
            except Exception as _se:
                logger.error(f"[/scan background] Error: {_se}")
                await send_message("❌ סריקה נכשלה — נסה שוב")
        # _handle_command runs in asyncio.to_thread — can't use create_task directly.
        # Use run_coroutine_threadsafe with the stored main event loop.
        import asyncio as _aio
        try:
            from discord_bot import get_event_loop as _get_loop
            _loop = _get_loop()   # loop stored at startup in main.py
        except Exception:
            _loop = None
        if _loop and _loop.is_running():
            _aio.run_coroutine_threadsafe(_run_scan_and_notify(), _loop)
        else:
            # Fallback: synchronous scan (blocks but won't crash)
            import requests as _rq2
            try:
                _r = _rq2.post(f"{base}/scan/now",
                               headers={"X-Webhook-Secret": secret}, timeout=60)
                if _r.status_code == 200:
                    _d = _r.json()
                    _bl = _d.get("bought", [])
                    _bought = len(_bl) if isinstance(_bl, list) else int(_bl or 0)
                    return (f"🔍 <b>סריקה הושלמה</b>\n✅ נקנו: {_bought} מניות"
                            if _bought else f"🔍 <b>סריקה הושלמה</b>\n⏭️ {_d.get('reason','לא נמצאו')}")
            except Exception:
                pass
        return "🔍 <b>סריקה התחילה!</b>\nתקבל תוצאות בעוד כ-30-60 שניות... ⏳"

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
                f"🎯 אחוז הצלחה: <b>{wr}%</b>  ({wins}/{total})"
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
            vix       = context.get("vix", "—")
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

    # ── Explicit command aliases → keyword scan ────────────────────────────
    _alias_kw = {"/manioth":"manioth","/revach":"revach",
                 "/shovi":"shovi","/mazon":"mazon","/biztsuim":"biztsuim"}
    if cmd in _alias_kw:
        t = _alias_kw[cmd]   # safe: replaces only for keyword scan below

    # ── שאלות מניות/פוזיציות ───────────────────────────────────────────────
    stocks_keywords = ["מניות", "מניה", "פוזיציות", "מה יש", "מה קניתי", "מחזיק", "תיק שלי", "איזה", "manioth", "/manioth"]
    if any(k in t for k in stocks_keywords):
        # Always fetch live from DB — bypass cache so new positions show immediately
        try:
            import database as _dbm
            import yfinance as _yf_m
            _live_trades = _dbm.get_open_trades()
            if _live_trades:
                _tickers = [tr["ticker"] for tr in _live_trades]
                _hist = _yf_m.download(_tickers, period="2d", progress=False, auto_adjust=True)
                positions = []
                for tr in _live_trades:
                    tk = tr["ticker"]
                    try:
                        if len(_tickers) == 1:
                            cur = float(_hist["Close"].dropna().iloc[-1])
                        else:
                            cur = float(_hist["Close"][tk].dropna().iloc[-1])
                    except Exception:
                        cur = float(tr.get("entry_price", 0))
                    entry = float(tr.get("entry_price", 0))
                    qty   = float(tr.get("qty", 0))
                    pnl   = (cur - entry) * qty
                    pct   = (cur - entry) / entry * 100 if entry > 0 else 0
                    positions.append({
                        "ticker":  tk,
                        "qty":     qty,
                        "entry":   entry,
                        "current": cur,
                        "pnl":     pnl,
                        "pct":     pct,
                        "value":   cur * qty,
                        "atr_stop": tr.get("atr_stop_price") or 0,
                        "held_hours": 0,
                    })
            else:
                positions = []
        except Exception:
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
                tp_pct = max(3.0, ((p["entry"] - stop) / p["entry"] * 100) * 1.5) if stop and p["entry"] else 5.0
                tp_price = round(p["entry"] * (1 + tp_pct / 100), 2)
            except Exception:
                tp_price = 0
            total_pnl += p["pnl"]
            lines.append(
                f"\n{'🟢' if profit else '🔴'} <b>{p['ticker']}</b>\n"
                f"   🔢 כמות:                    {p['qty']} מניות\n"
                f"   📌 מחיר קנייה:         {_fmt_price(p['entry'])}\n"
                f"   📍 מחיר עכשיו:          {_fmt_price(p['current'])} ({p['pct']:+.1f}%)\n"
                f"   📈 יעד רווח:        {_fmt_price(tp_price) if tp_price else '—'}\n"
                f"   📉 סטופ לוס:      {_fmt_price(stop) if stop else '—'}\n"
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
                f"🎯  אחוז הצלחה:       <b>{wr}%</b>\n"
                f"    {wr_bar}\n\n"
                f"💰  רווח כולל:       {_fmt_pnl(total_pnl)}\n"
                f"📈  ממוצע ניצחון:  <b>${avg_win:+.2f}</b>\n"
                f"📉  ממוצע הפסד:   <b>${avg_loss:+.2f}</b>\n"
                f"⚖️   יחס סיכון/רווח:      <b>{rr_ratio:.2f}</b>\n"
                f"⏱️   זמן ממוצע:     <b>{hold_str}</b>\n\n"
                f"🏆  הכי טוב:  <b>{best_trade['ticker']}</b>  ${float(best_trade.get('pnl_gross',0)):+.2f}\n"
                f"📉  הכי גרוע: <b>{worst_trade['ticker']}</b>  ${float(worst_trade.get('pnl_gross',0)):+.2f}"
            )
        except Exception as e:
            logger.error(f"[/biztsuim] Error: {e}")
            return "❌ שגיאה פנימית — נסה שוב"

    # ── Auto-detect "קנה TICKER" / "buy TICKER" ────────────────────────────────
    _buy_trigger = ["קנה", "תקנה", "buy", "רכוש"]
    _sell_trigger = ["מכור", "תמכור", "sell"]
    _parts = t.split()
    if len(_parts) >= 2 and _parts[0] in _buy_trigger:
        _auto_buy_tk = _safe_ticker(_parts[1])
        if _auto_buy_tk:
            # Check if already held
            import database as _db2
            if _db2.get_open_trade_by_ticker(_auto_buy_tk):
                return f"⚠️ כבר מחזיקים <b>{_auto_buy_tk}</b>!\nשלח /stop {_auto_buy_tk} לראות פרטים"
            return (
                f"🛒 <b>קנייה ידנית — {_auto_buy_tk}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"הבוט קונה רק דרך הסריקה האוטומטית.\n\n"
                f"💡 <b>מה לעשות:</b>\n"
                f"1️⃣  /score {_auto_buy_tk} — בדוק ציון\n"
                f"2️⃣  /watchadd {_auto_buy_tk} — הוסף לרשימה\n"
                f"3️⃣  /scan — הפעל סריקה מיידית\n\n"
                f"אם הציון מעל הסף — הבוט יקנה אוטומטית!"
            )
    if len(_parts) >= 2 and _parts[0] in _sell_trigger:
        _auto_sell_tk = _safe_ticker(_parts[1])
        if _auto_sell_tk:
            return _handle_command(f"/sell {_auto_sell_tk}", context)

    # ── Auto-detect ticker in free Hebrew text ─────────────────────────────────
    _auto_ticker = _extract_ticker_from_text(text)
    if _auto_ticker:
        _score_kws = ["ציון", "score", "כמה", "כדאי", "לקנות", "לא לקנות", "מה דעתך", "להמליץ"]
        _news_kws  = ["חדשות", "news", "כותרות", "מה קורה", "מה היה", "עדכון"]
        _price_kws = ["מחיר", "price", "שווה", "עולה", "יורד", "כמה עולה"]
        _vol_kws   = ["נפח", "volume", "כמה נסחר"]
        _52w_kws   = ["שיא", "שפל", "52", "גבוה", "נמוך"]
        if any(k in t for k in _score_kws):
            return _handle_command(f"/score {_auto_ticker}", context)
        if any(k in t for k in _news_kws):
            return _handle_command(f"/news {_auto_ticker}", context)
        if any(k in t for k in _price_kws):
            return _handle_command(f"/price {_auto_ticker}", context)
        if any(k in t for k in _vol_kws):
            return _handle_command(f"/volume {_auto_ticker}", context)
        if any(k in t for k in _52w_kws):
            return _handle_command(f"/52week {_auto_ticker}", context)
        # If user mentions a ticker they OWN → show their position details
        owned_tickers = {p["ticker"] for p in context.get("open_positions", [])}
        if _auto_ticker in owned_tickers:
            return _handle_command(f"/stop {_auto_ticker}", context)

        # If message is ONLY a ticker (e.g. user answers "V" after /news asked "which ticker?")
        # → show quick overview + offer options
        _clean = text.strip().upper()
        if _TICKER_RE.match(_clean) and len(_clean) <= 5:
            return (
                f"📊 <b>{_clean}</b> — מה תרצה לדעת?\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📰  /news {_clean} — חדשות\n"
                f"🎯  /score {_clean} — ציון\n"
                f"💲  /price {_clean} — מחיר\n"
                f"📏  /52week {_clean} — מיקום בשנה\n"
                f"📈  /fundamental {_clean} — פונדמנטלס"
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
    from_user = message.get("from", {}) or {}
    from_user_id = str(from_user.get("id", ""))
    is_bot      = bool(from_user.get("is_bot", False))
    text = (message.get("text") or "").strip()

    # Security: only respond to the configured chat AND the user
    if not settings.TELEGRAM_CHAT_ID:
        return {"status": "ignored", "reason": "TELEGRAM_CHAT_ID not configured"}
    if chat_id != str(settings.TELEGRAM_CHAT_ID):
        logger.warning(f"[CHAT] Ignoring message from unauthorized chat {chat_id}")
        return {"status": "ignored", "reason": "unauthorized chat"}
    # In a private chat, chat.id == from.id. Reject if from.id mismatches (anti-spoof).
    if from_user_id and from_user_id != str(settings.TELEGRAM_CHAT_ID):
        logger.warning(f"[CHAT] Ignoring message from unauthorized user {from_user_id} (chat={chat_id})")
        return {"status": "ignored", "reason": "unauthorized user"}
    # Reject messages from other bots
    if is_bot:
        logger.warning(f"[CHAT] Ignoring message from bot user {from_user_id}")
        return {"status": "ignored", "reason": "bot user"}

    if not text:
        return {"status": "ignored", "reason": "empty message"}
    # Limit text length to prevent LLM prompt injection / resource exhaustion
    if len(text) > 1000:
        text = text[:1000]

    # Map Hebrew button labels to commands
    _BUTTON_MAP = {
        "💰 רווח/הפסד":      "/pnl",
        "📊 מצב התיק":       "/status",
        "📋 תקציר יומי":     "/digest",
        "📈 מניות שלי":      "/manioth",
        "🔢 כמה עסקאות":     "/count",
        "🤖 מה אתה עושה":    "/activity_now",
        "🌟 הכי טובות":      "/best",
        "🌍 מצב השוק":       "/market",
        "🏆 מובילים היום":   "/gainers",
        "📰 חדשות":          "/newscheck",
        "💡 ייעוץ AI":       "/advice",
        "⚠️ ניתוח סיכון":    "/risk_score",
        "📅 מה היה היום":    "/today",
        "📋 כל הפקודות":     "/help",
    }
    if text in _BUTTON_MAP:
        text = _BUTTON_MAP[text]

    logger.info(f"[CHAT] Incoming: {text[:100]}")

    # Send typing indicator immediately so user knows bot is working
    await _send_typing(chat_id)

    # Smart ticker detection: if user typed only a ticker (e.g. "NVDA"),
    # offer quick analysis options instead of guessing.
    _detected_ticker = _detect_ticker(text)

    # Generate reply — total timeout 25s (Telegram drops webhook at ~30s)
    try:
        async def _generate_reply():
            ctx = await asyncio.to_thread(_build_context)

            # Auto-respond to bare ticker with quick analysis menu
            if _detected_ticker and len(text.split()) == 1:
                return (
                    f"📊 <b>{_detected_ticker}</b> — מה תרצה לדעת?\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"💲 /price {_detected_ticker} — מחיר נוכחי\n"
                    f"🎯 /score {_detected_ticker} — ציון טכני (0-100)\n"
                    f"🎩 /buffett {_detected_ticker} — ניתוח באפט מלא\n"
                    f"📰 /news {_detected_ticker} — חדשות + סנטימנט AI\n"
                    f"📑 /earnings {_detected_ticker} — דוח רווחים\n"
                    f"📐 /volatility {_detected_ticker} — תנודתיות + Beta\n"
                    f"💡 או שלח שאלה: <i>\"האם כדאי לקנות {_detected_ticker}?\"</i>"
                )

            rep = await asyncio.to_thread(_handle_command, text, ctx)
            if rep is None:
                client = _get_client()
                if client:
                    # Pass conversation history for follow-up questions
                    hist = _get_history(chat_id)
                    rep = await asyncio.to_thread(_llm_reply, text, ctx, hist)
                else:
                    # LLM not configured — suggest relevant commands instead of just portfolio
                    rep = (
                        f"💡 <b>לא מובנת הבקשה — נסה פקודה:</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"📊 /status — מצב התיק\n"
                        f"🏆 /top — מניות מובילות\n"
                        f"🎯 /score AAPL — ציון מניה\n"
                        f"📰 /news AAPL — חדשות\n"
                        f"💲 /price AAPL — מחיר\n"
                        f"❓ /help — כל הפקודות"
                    )
            return rep

        reply = await asyncio.wait_for(_generate_reply(), timeout=25)

    except asyncio.TimeoutError:
        logger.warning(f"[CHAT] Reply timed out for: {text[:50]}")
        reply = "⏳ הבוט עסוק כרגע — נסה שוב בעוד רגע"
    except Exception as exc:
        logger.error(f"[CHAT] Reply generation failed: {exc}")
        reply = "מצטער, נתקלתי בשגיאה. נסה שוב."

    # Send reply
    try:
        ok = await send_message(reply)
        # Save to conversation history for follow-up questions
        if ok:
            _remember(chat_id, "user", text)
            _remember(chat_id, "assistant", reply)
        return {
            "status": "replied" if ok else "send_failed",
            "incoming": text[:200],
            "reply": reply[:200],
        }
    except Exception as exc:
        logger.error(f"[CHAT] Failed to send reply: {exc}")
        return {"status": "error", "reason": str(exc)}
