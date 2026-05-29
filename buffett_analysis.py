"""
ניתוח מניה ברמת וורן באפט — בודק את העקרונות שלו:

1. Return on Equity (ROE) — האם החברה מייצרת ערך?
2. Profit margins — האם יש moat (יתרון תחרותי)?
3. Debt levels — האם החברה יציבה פיננסית?
4. Earnings growth — האם הרווחים גדלים בעקביות?
5. Valuation (P/E, P/B) — האם המחיר הוגן?
6. Free Cash Flow — האם מייצרת מזומן אמיתי?
7. Dividend — האם מחלקת לבעלי מניות?

Returns: dict with detailed analysis + verdict
"""
import logging
import math
import threading

logger = logging.getLogger(__name__)

# Cache: ticker → (timestamp, analysis_dict)
_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 24 * 3600   # 24 hours


def _safe_float(val):
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def get_buffett_analysis(ticker: str) -> dict:
    """
    מנתח מניה ברמת וורן באפט. מחזיר:
    {
      "ticker": "AAPL",
      "verdict": "QUALITY_BUY" | "FAIR" | "AVOID" | "UNCLEAR",
      "score": 0-100,
      "criteria": {...},
      "moat": "strong" | "medium" | "weak",
      "summary_he": "..."
    }
    """
    import time
    now = time.time()

    # Check cache
    with _cache_lock:
        cached = _cache.get(ticker)
        if cached and now - cached[0] < _CACHE_TTL:
            return cached[1]

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        # yfinance can return None or empty dict when Yahoo rate-limits us
        # or rejects the crumb token. Treat both as "no data".
        if not info or not isinstance(info, dict):
            raise ValueError("yfinance returned no data (rate-limited or crumb expired)")
    except Exception as e:
        logger.debug(f"[BUFFETT] yfinance failed for {ticker}: {e}")
        # Cache the "unclear" verdict briefly so we don't hammer yfinance
        result = {
            "ticker": ticker,
            "verdict": "UNCLEAR",
            "score": 50,
            "summary_he": f"נתונים זמניים לא זמינים — {ticker}",
            "criteria": {},
            "moat": "unknown",
        }
        # Cache it for 5 minutes to avoid repeated yfinance calls during rate-limit
        with _cache_lock:
            _cache[ticker] = (now - _CACHE_TTL + 300, result)  # expires in 5 min
        return result

    # ── Extract metrics ──────────────────────────────────────────────────
    roe         = _safe_float(info.get("returnOnEquity"))
    margin      = _safe_float(info.get("profitMargins"))
    op_margin   = _safe_float(info.get("operatingMargins"))
    debt_eq     = _safe_float(info.get("debtToEquity"))
    if debt_eq is not None:
        debt_eq = debt_eq / 100.0  # yfinance returns percentage
    eps_growth  = _safe_float(info.get("earningsGrowth"))
    rev_growth  = _safe_float(info.get("revenueGrowth"))
    pe          = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
    pb          = _safe_float(info.get("priceToBook"))
    fcf         = _safe_float(info.get("freeCashflow"))
    div_yield   = _safe_float(info.get("dividendYield"))
    market_cap  = _safe_float(info.get("marketCap"))
    name        = info.get("shortName") or info.get("longName") or ticker

    # ── Buffett Criteria Scoring (0-100) ───────────────────────────────
    score = 0
    criteria = {}

    # 1. ROE — Quality (max 20 points)
    if roe is not None:
        if roe > 0.20:
            score += 20; criteria["roe"] = f"מעולה — {roe*100:.1f}%"
        elif roe > 0.15:
            score += 15; criteria["roe"] = f"טוב — {roe*100:.1f}%"
        elif roe > 0.10:
            score += 8;  criteria["roe"] = f"סביר — {roe*100:.1f}%"
        elif roe > 0:
            score += 2;  criteria["roe"] = f"חלש — {roe*100:.1f}%"
        else:
            criteria["roe"] = f"שלילי — {roe*100:.1f}% (מפסיד כסף)"

    # 2. Profit Margin — Moat (max 20 points)
    if margin is not None:
        if margin > 0.25:
            score += 20; criteria["margin"] = f"מעולה — {margin*100:.1f}% (moat חזק)"
        elif margin > 0.15:
            score += 15; criteria["margin"] = f"טוב — {margin*100:.1f}%"
        elif margin > 0.08:
            score += 8;  criteria["margin"] = f"סביר — {margin*100:.1f}%"
        elif margin > 0:
            score += 2;  criteria["margin"] = f"דק — {margin*100:.1f}%"
        else:
            criteria["margin"] = f"שלילי — {margin*100:.1f}%"

    # 3. Debt — Stability (max 15 points)
    if debt_eq is not None:
        if debt_eq < 0.3:
            score += 15; criteria["debt"] = f"מעולה — D/E={debt_eq:.2f}"
        elif debt_eq < 0.7:
            score += 10; criteria["debt"] = f"טוב — D/E={debt_eq:.2f}"
        elif debt_eq < 1.5:
            score += 4;  criteria["debt"] = f"סביר — D/E={debt_eq:.2f}"
        else:
            criteria["debt"] = f"גבוה מדי — D/E={debt_eq:.2f}"

    # 4. Earnings Growth — Predictability (max 15 points)
    if eps_growth is not None:
        if eps_growth > 0.15:
            score += 15; criteria["growth"] = f"חזק — EPS גדל {eps_growth*100:.1f}%"
        elif eps_growth > 0.05:
            score += 10; criteria["growth"] = f"טוב — EPS גדל {eps_growth*100:.1f}%"
        elif eps_growth > 0:
            score += 5;  criteria["growth"] = f"חלש — EPS גדל {eps_growth*100:.1f}%"
        else:
            criteria["growth"] = f"יורד — EPS ירד {eps_growth*100:.1f}%"

    # 5. Valuation P/E (max 15 points)
    if pe is not None:
        if 10 <= pe <= 20:
            score += 15; criteria["pe"] = f"הוגן — P/E={pe:.1f}"
        elif 20 < pe <= 30:
            score += 10; criteria["pe"] = f"קצת יקר — P/E={pe:.1f}"
        elif 5 <= pe < 10:
            score += 8;  criteria["pe"] = f"זול — P/E={pe:.1f} (בדוק למה!)"
        elif 30 < pe <= 40:
            score += 3;  criteria["pe"] = f"יקר — P/E={pe:.1f}"
        else:
            criteria["pe"] = f"יקר מאוד — P/E={pe:.1f}" if pe > 40 else f"חשוד — P/E={pe:.1f}"

    # 6. Free Cash Flow (max 10 points)
    if fcf is not None and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap
        if fcf_yield > 0.06:
            score += 10; criteria["fcf"] = f"מעולה — תשואת FCF {fcf_yield*100:.1f}%"
        elif fcf_yield > 0.03:
            score += 6;  criteria["fcf"] = f"טוב — תשואת FCF {fcf_yield*100:.1f}%"
        elif fcf > 0:
            score += 2;  criteria["fcf"] = f"חיובי — תשואת FCF {fcf_yield*100:.1f}%"
        else:
            criteria["fcf"] = "שלילי — לא מייצר מזומן"

    # 7. Dividend bonus (max 5 points)
    if div_yield is not None and div_yield > 0:
        if div_yield > 0.03:
            score += 5; criteria["dividend"] = f"דיבידנד טוב — {div_yield*100:.2f}%"
        else:
            score += 2; criteria["dividend"] = f"דיבידנד — {div_yield*100:.2f}%"

    # ── Determine Moat ─────────────────────────────────────────────────
    moat = "weak"
    if margin and margin > 0.20 and roe and roe > 0.15:
        moat = "strong"
    elif (margin and margin > 0.10) or (roe and roe > 0.12):
        moat = "medium"

    # ── Verdict ────────────────────────────────────────────────────────
    if score >= 75:
        verdict = "QUALITY_BUY"
        verdict_he = "🟢 איכותית מאוד — כדאי לקנייה לטווח ארוך"
    elif score >= 55:
        verdict = "FAIR"
        verdict_he = "🟡 סבירה — אבל לא הצעקה האחרונה"
    elif score >= 35:
        verdict = "WEAK"
        verdict_he = "🟠 חלשה — לא היה בוחר בה באפט"
    else:
        verdict = "AVOID"
        verdict_he = "🔴 הימנע — סיכון פיננסי / מתומחרת יתר"

    # ── Build summary ──────────────────────────────────────────────────
    cap_str = ""
    if market_cap:
        if market_cap >= 1e12:
            cap_str = f"${market_cap/1e12:.1f}T"
        elif market_cap >= 1e9:
            cap_str = f"${market_cap/1e9:.1f}B"
        else:
            cap_str = f"${market_cap/1e6:.0f}M"

    result = {
        "ticker": ticker,
        "name": name,
        "verdict": verdict,
        "verdict_he": verdict_he,
        "score": round(score, 1),
        "moat": moat,
        "market_cap": cap_str,
        "criteria": criteria,
        "metrics": {
            "roe": roe, "margin": margin, "debt_eq": debt_eq,
            "eps_growth": eps_growth, "pe": pe, "pb": pb,
            "fcf": fcf, "div_yield": div_yield,
        }
    }

    with _cache_lock:
        _cache[ticker] = (now, result)

    logger.info(f"[BUFFETT] {ticker}: score={score:.0f}/100 | verdict={verdict} | moat={moat}")
    return result


def format_buffett_report(analysis: dict) -> str:
    """Format Buffett analysis as a Telegram-ready Hebrew report."""
    ticker = analysis["ticker"]
    name   = analysis.get("name", ticker)
    score  = analysis["score"]
    verdict_he = analysis["verdict_he"]
    moat   = analysis["moat"]
    cap    = analysis.get("market_cap", "")
    crit   = analysis.get("criteria", {})

    moat_str = {"strong": "💪 חזק", "medium": "🛡️ בינוני", "weak": "⚠️ חלש", "unknown": "❓ לא ידוע"}[moat]

    # Score bar
    bar_filled = round(score / 10)
    score_bar  = "🟩" * bar_filled + "⬜" * (10 - bar_filled)

    lines = [
        f"📊 <b>ניתוח באפט — {ticker}</b>",
        f"━━━━━━━━━━━━━━━━",
        f"🏢 {name}" + (f" ({cap})" if cap else ""),
        f"",
        f"{verdict_he}",
        f"🎯 ציון איכות: <b>{score:.0f}/100</b>",
        f"{score_bar}",
        f"🛡️ יתרון תחרותי (moat): {moat_str}",
        f"━━━━━━━━━━━━━━━━",
        f"<b>📋 ניתוח לפי קריטריונים:</b>",
    ]

    labels = {
        "roe":      "📈 ROE (תשואה על הון)",
        "margin":   "💰 שולי רווח",
        "debt":     "🏦 חוב",
        "growth":   "📊 צמיחת רווחים",
        "pe":       "💲 תמחור",
        "fcf":      "💵 תזרים מזומנים",
        "dividend": "🎁 דיבידנד",
    }
    for key, label in labels.items():
        if key in crit:
            lines.append(f"   {label}: {crit[key]}")

    # Verdict explanation
    lines.append("━━━━━━━━━━━━━━━━")
    if analysis["verdict"] == "QUALITY_BUY":
        lines.append("💡 <b>סיכום:</b> חברה איכותית עם moat חזק. מתאימה להחזקה ארוכת טווח.")
    elif analysis["verdict"] == "FAIR":
        lines.append("💡 <b>סיכום:</b> איכות סבירה. כדאי להמתין למחיר טוב יותר.")
    elif analysis["verdict"] == "WEAK":
        lines.append("💡 <b>סיכום:</b> חברה חלשה. וורן באפט היה מעדיף איכות גבוהה יותר.")
    else:
        lines.append("💡 <b>סיכום:</b> סיכון פיננסי או overvaluation. הימנע.")

    return "\n".join(lines)
