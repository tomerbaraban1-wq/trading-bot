"""
Stock Scanner — finds the best buy opportunity right now.
Scans top US stocks using: technical indicators + AI news sentiment.
"""

import time
import logging
import yfinance as yf
from indicators import get_current_indicators
from sentiment import score_sentiment

logger = logging.getLogger(__name__)

WATCHLIST = [
    # ── מניות ──
    # טכנולוגיה — גדולות
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "AMD", "INTC", "QCOM", "MU", "NFLX", "CRM", "ORCL", "UBER", "PLTR",
    # טכנולוגיה — בינוניות וצמיחה
    "SNOW", "NET", "DDOG", "ZS", "CRWD", "PANW", "FTNT", "NOW",
    "SHOP", "SQ", "PYPL", "AFRM", "HOOD", "SOFI", "BILL",
    "ARM", "SMCI", "DELL", "HPQ", "IBM", "CSCO", "TXN",
    "MRVL", "KLAC", "AMAT", "LRCX", "ASML", "TSM",
    # AI / ענן
    "AI", "BBAI", "SOUN", "IONQ", "RKLB", "LUNR",
    # פיננסים
    "JPM", "BAC", "GS", "V", "MA", "AXP", "C", "WFC", "MS", "BLK",
    "SCHW", "COF", "ALLY", "NU",
    # בריאות / ביוטק
    "UNH", "LLY", "PFE", "MRNA", "BNTX", "ABBV", "JNJ", "MRK",
    "AMGN", "GILD", "BIIB", "REGN", "VRTX", "ISRG",
    # אנרגיה
    "XOM", "CVX", "COP", "SLB", "OXY", "BP",
    # צרכנות / קמעונאות
    "WMT", "COST", "HD", "MCD", "SBUX", "TGT", "AMZN", "BABA",
    "NKE", "LULu", "DIS", "CMCSA",
    # רכב / EV
    "TSLA", "RIVN", "LCID", "GM", "F", "NIO", "LI",
    # קריפטו סטוקים
    "COIN", "MSTR", "MARA", "RIOT", "CLSK", "BITF", "HUT",

    # ── קרנות סל (ETFs) ──
    # שוק רחב
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    # סקטוריאליים
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLRE", "XLY", "XLP", "XLB",
    # ממונפים (x2/x3) — תנועות חזקות
    "TQQQ", "SOXL", "UPRO", "TECL", "FNGU", "LABU", "WEBL",
    "QLD", "SSO", "UDOW",
    # הפוכים — לירידות
    "SQQQ", "SDOW", "SPXU", "SOXS",
    # תנודתיות
    "UVXY", "VXX",

    # ── סחורות (Commodities ETFs) ──
    "GLD",   # זהב
    "SLV",   # כסף
    "GDX",   # מניות זהב
    "GDXJ",  # מניות זהב קטנות
    "USO",   # נפט גולמי
    "UNG",   # גז טבעי
    "CORN",  # תירס
    "WEAT",  # חיטה
    "PDBC",  # סחורות מגוונות
    "DBO",   # נפט

    # ── נגזרים ושווקים אלטרנטיביים ──
    "BITO",  # ביטקוין ETF
    "ETHE",  # אית'ריום ETF
    "IBIT",  # ביטקוין ETF של BlackRock
]

# קטגוריות לכל סימבול
ASSET_CATEGORY = {
    # מניות טק
    "AAPL":"מניה","MSFT":"מניה","NVDA":"מניה","GOOGL":"מניה","AMZN":"מניה",
    "META":"מניה","TSLA":"מניה","AVGO":"מניה","AMD":"מניה","INTC":"מניה",
    "QCOM":"מניה","MU":"מניה","NFLX":"מניה","CRM":"מניה","ORCL":"מניה",
    "UBER":"מניה","PLTR":"מניה","COIN":"מניה","MSTR":"מניה","MARA":"מניה","RIOT":"מניה",
    # מניות אחרות
    "JPM":"מניה","BAC":"מניה","GS":"מניה","V":"מניה","MA":"מניה",
    "UNH":"מניה","LLY":"מניה","PFE":"מניה","XOM":"מניה","CVX":"מניה",
    "WMT":"מניה","COST":"מניה","HD":"מניה","MCD":"מניה",
    # ETFs שוק רחב
    "SPY":"קרן סל","QQQ":"קרן סל","IWM":"קרן סל","DIA":"קרן סל","VTI":"קרן סל","VOO":"קרן סל",
    # ETFs סקטוריאליים
    "XLK":"קרן סל","XLF":"קרן סל","XLE":"קרן סל","XLV":"קרן סל","XLI":"קרן סל","XLRE":"קרן סל",
    # ממונפים
    "TQQQ":"ממונף x3","SOXL":"ממונף x3","UPRO":"ממונף x3","TECL":"ממונף x3",
    "SQQQ":"הפוך x3","SDOW":"הפוך x3",
    # תנודתיות
    "UVXY":"נגזר VIX","VXX":"נגזר VIX",
    # סחורות
    "GLD":"סחורה - זהב","SLV":"סחורה - כסף","GDX":"סחורה - זהב",
    "USO":"סחורה - נפט","UNG":"סחורה - גז","DBO":"סחורה - נפט",
    "CORN":"סחורה - חקלאות","WEAT":"סחורה - חקלאות","PDBC":"סחורות",
    # קריפטו ETF
    "BITO":"קריפטו ETF","ETHE":"קריפטו ETF",
}

_cache: dict = {"result": None, "time": 0}
CACHE_TTL = 300  # 5 minutes


def _score_stock(ticker: str, ind: dict) -> tuple[float, str]:
    """
    Score a stock based on technical indicators.
    Returns (score, reason_string).
    """
    score = 0.0
    reasons = []

    # --- RSI scoring ---
    rsi = ind.get("rsi")
    if rsi is not None:
        if 40 <= rsi <= 55:
            score += 3
            reasons.append(f"RSI={rsi:.0f}✅")
        elif 30 <= rsi < 40:
            score += 2
            reasons.append(f"RSI={rsi:.0f}⚠️")
        elif rsi < 30:
            score += 1
            reasons.append(f"RSI={rsi:.0f}🔻")
        elif rsi > 65:
            score -= 2
            reasons.append(f"RSI={rsi:.0f}❌")
        else:
            # 55 < rsi <= 65 — neutral, no points
            reasons.append(f"RSI={rsi:.0f}")

    # --- MACD scoring ---
    macd = ind.get("macd")
    macd_signal = ind.get("macd_signal")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            score += 3
            reasons.append("MACD חיובי✅")
        elif macd > 0:
            score += 1
            reasons.append("MACD חלש⚠️")
        else:
            reasons.append("MACD שלילי")

    # --- Bollinger Bands scoring ---
    # get_current_indicators returns bb_position as "lower"/"middle"/"upper"/"unknown"
    bb_pos_str = ind.get("bb_position", "unknown")
    if bb_pos_str == "lower":
        score += 2
        reasons.append("BB תחתון✅")
    elif bb_pos_str == "middle":
        score += 1
        reasons.append("BB אמצע")
    elif bb_pos_str == "upper":
        score -= 1
        reasons.append("BB עליון❌")

    # --- Volume ratio scoring ---
    volume_ratio = ind.get("volume_ratio")
    if volume_ratio is not None:
        if volume_ratio > 1.5:
            score += 2
            reasons.append("נפח גבוה✅")
        elif volume_ratio > 1.2:
            score += 1
            reasons.append("נפח בינוני")

    reason = " | ".join(reasons) if reasons else "אין נתונים"
    return score, reason


def _get_price_change(ticker: str) -> tuple[float, float]:
    """
    Fetch current price and daily change % using yfinance fast_info.
    Returns (price, change_pct). Falls back to (0.0, 0.0) on error.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(info.last_price or 0)
        prev_close = float(info.previous_close or 0)
        if prev_close > 0:
            change_pct = ((price - prev_close) / prev_close) * 100
        else:
            change_pct = 0.0
        return round(price, 2), round(change_pct, 2)
    except Exception:
        return 0.0, 0.0


# פילטר שווי שוק מינימלי (ניתן לשינוי)
MIN_MARKET_CAP = 100_000_000  # $100 מיליון


def _get_market_cap(ticker: str) -> float:
    """מחזיר שווי שוק / AUM בדולרים. 0 אם לא ידוע."""
    try:
        t = yf.Ticker(ticker)
        # מניות — market cap
        mc = float(getattr(t.fast_info, "market_cap", 0) or 0)
        if mc > 0:
            return mc
        # קרנות סל — totalAssets (AUM)
        info = t.info
        return float(info.get("totalAssets") or 0)
    except Exception:
        return 0.0


def scan_stocks(max_results: int = 3) -> list[dict]:
    """
    Scan the watchlist with technical indicators and return the top stocks.
    מסנן חברות עם שווי שוק מתחת ל-MIN_MARKET_CAP.

    Returns a list of dicts sorted by score (highest first), each with:
    ticker, score, rsi, macd, signal, bb_position, volume_ratio,
    price, change_pct, market_cap, reason
    """
    results = []

    for ticker in WATCHLIST:
        try:
            # ── פילטר שווי שוק / AUM ──
            market_cap = _get_market_cap(ticker)
            category = ASSET_CATEGORY.get(ticker, "מניה")
            # חל על מניות וקרנות סל — סחורות/נגזרים עוברים בחופשי
            needs_filter = category in ("מניה", "קרן סל")
            if needs_filter and market_cap > 0 and market_cap < MIN_MARKET_CAP:
                logger.debug(
                    f"Scanner: {ticker} filtered out — "
                    f"market cap/AUM ${market_cap/1e6:.0f}M < ${MIN_MARKET_CAP/1e6:.0f}M"
                )
                continue

            ind = get_current_indicators(ticker)
            if ind is None:
                logger.debug(f"Scanner: no indicator data for {ticker}, skipping")
                continue

            tech_score, reason = _score_stock(ticker, ind)
            price, change_pct = _get_price_change(ticker)

            results.append({
                "ticker": ticker,
                "score": tech_score,
                "category": category,
                "market_cap": market_cap,
                "rsi": ind.get("rsi"),
                "macd": ind.get("macd"),
                "signal": ind.get("macd_signal"),
                "bb_position": ind.get("bb_position"),
                "volume_ratio": ind.get("volume_ratio"),
                "price": price if price else ind.get("close", 0),
                "change_pct": change_pct,
                "reason": reason,
            })
        except Exception as e:
            logger.warning(f"Scanner: failed to process {ticker}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def get_top_pick() -> dict:
    """
    Return the single best stock to buy right now.

    Pipeline:
    1. Check 5-minute cache — return cached result if fresh.
    2. scan_stocks(max_results=5) for top technical candidates.
    3. Score sentiment for the top 3 candidates via Groq AI.
    4. final_score = technical_score + (sentiment.score / 2)
    5. Return the winner with all data + scan_time.
    """
    # Cache check
    now = time.time()
    if _cache["result"] is not None and (now - _cache["time"]) < CACHE_TTL:
        logger.info("Scanner: returning cached result")
        return _cache["result"]

    logger.info("Scanner: starting full scan of watchlist...")
    candidates = scan_stocks(max_results=5)

    if not candidates:
        return {
            "ticker": None,
            "score": 0,
            "error": "No stocks passed technical screening",
            "scan_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # Enrich top 3 with sentiment
    enriched = []
    for stock in candidates[:3]:
        ticker = stock["ticker"]
        try:
            sentiment = score_sentiment(ticker)
            sent_score = sentiment.score
            sent_reasoning = sentiment.reasoning
        except Exception as e:
            logger.warning(f"Scanner: sentiment failed for {ticker}: {e}")
            sent_score = 5  # neutral fallback
            sent_reasoning = "Sentiment check failed — defaulting to neutral"

        final_score = stock["score"] + (sent_score / 2)
        enriched.append({
            **stock,
            "sentiment_score": sent_score,
            "sentiment_reasoning": sent_reasoning,
            "final_score": final_score,
        })

    # Pick the winner
    enriched.sort(key=lambda x: x["final_score"], reverse=True)
    winner = enriched[0]

    result = {
        "ticker": winner["ticker"],
        "score": round(winner["final_score"], 2),
        "rsi": winner.get("rsi"),
        "macd_signal": winner.get("signal"),
        "bb_position": winner.get("bb_position"),
        "volume_ratio": winner.get("volume_ratio"),
        "sentiment_score": winner.get("sentiment_score"),
        "sentiment_reasoning": winner.get("sentiment_reasoning"),
        "price": winner.get("price"),
        "change_pct": winner.get("change_pct"),
        "reason": winner.get("reason"),
        "scan_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Store in cache
    _cache["result"] = result
    _cache["time"] = now

    logger.info(
        f"Scanner: top pick is {result['ticker']} "
        f"(score={result['score']}, sentiment={result['sentiment_score']}/10)"
    )
    return result
