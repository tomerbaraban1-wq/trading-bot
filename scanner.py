"""
Stock Scanner — finds the best buy opportunity right now.
Scans top US stocks using: technical indicators + AI news sentiment.
Dynamic watchlist: fetches S&P500 + Nasdaq100 + Russell1000 from Wikipedia
daily and filters to only companies with market cap > MIN_MARKET_CAP.
"""

import os
import time
import threading
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
from indicators import get_current_indicators
from sentiment import score_sentiment

logger = logging.getLogger(__name__)

# ── Dynamic watchlist state ────────────────────────────────────────────────────
_dynamic_list: list[str] = []
_dynamic_list_lock = threading.Lock()
_dynamic_list_date: str = ""


def _fetch_index_tickers() -> list[str]:
    """Fetch tickers from S&P 500 + Nasdaq 100 via Wikipedia.
    Uses requests for the HTTP timeout (pandas.read_html doesn't support it),
    then parses the HTML with pandas.
    """
    import pandas as pd
    import requests
    tickers = set()
    sources = [
        ("S&P 500",    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, "Symbol"),
        ("Nasdaq 100", "https://en.wikipedia.org/wiki/Nasdaq-100",                  5, "Ticker"),
    ]
    headers = {"User-Agent": "Mozilla/5.0 TradingBot/1.0"}
    for name, url, table_idx, col in sources:
        try:
            resp = requests.get(url, timeout=15, headers=headers,
                                 verify=os.getenv("REQUESTS_CA_BUNDLE", True))
            resp.raise_for_status()
            import io
            df = pd.read_html(io.StringIO(resp.text))[table_idx]
            raw = df[col].dropna().tolist()
            cleaned = [str(t).replace(".", "-").strip() for t in raw]
            tickers.update(cleaned)
            logger.info(f"Dynamic watchlist: fetched {len(cleaned)} tickers from {name}")
        except Exception as e:
            logger.warning(f"Dynamic watchlist: failed to fetch {name}: {e}")
    return list(tickers)


def refresh_large_cap_list() -> None:
    """
    Background: fetch all S&P500+Nasdaq100 tickers, filter by MIN_MARKET_CAP,
    update the global dynamic watchlist. Skips if already done today.
    """
    global _dynamic_list_date
    today = datetime.now().strftime("%Y-%m-%d")

    with _dynamic_list_lock:
        if _dynamic_list_date == today and _dynamic_list:
            return  # already fresh today

    logger.info("Dynamic watchlist: starting refresh...")
    index_tickers = _fetch_index_tickers()

    # Merge with static WATCHLIST so we never lose known large caps
    all_tickers = list(set(WATCHLIST + index_tickers))
    logger.info(f"Dynamic watchlist: checking market cap for {len(all_tickers)} tickers...")

    result: list[str] = []
    lock = threading.Lock()

    def _check(ticker: str):
        try:
            mc = _get_market_cap(ticker)
            if mc >= MIN_MARKET_CAP:
                with lock:
                    result.append(ticker)
        except Exception:
            pass

    # Parallel market-cap checks with hard timeout — prevents hanging the bot
    import concurrent.futures as _cf
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_check, t): t for t in all_tickers}
        try:
            _cf.wait(futures, timeout=60)   # max 60 seconds total
        except Exception:
            pass
        # Cancel any still-running futures
        for f in futures:
            f.cancel()

    with _dynamic_list_lock:
        _dynamic_list.clear()
        _dynamic_list.extend(result)
        _dynamic_list_date = today

    logger.info(f"Dynamic watchlist: {len(result)} stocks above ${MIN_MARKET_CAP/1e9:.0f}B market cap")


def get_watchlist() -> list[str]:
    """
    Returns the dynamic large-cap watchlist if ready, else static WATCHLIST.
    Merges USER_WATCHLIST env var (custom tickers added via /watchadd).
    Respects USER_WATCHLIST_REMOVE env var (tickers removed via /watchremove).
    """
    import os as _os
    with _dynamic_list_lock:
        base = list(_dynamic_list) if _dynamic_list else list(WATCHLIST)

    # User-added tickers (via /watchadd)
    user_add = [t.strip().upper() for t in _os.getenv("USER_WATCHLIST", "").split(",") if t.strip()]
    # User-removed tickers (via /watchremove)
    user_remove = {t.strip().upper() for t in _os.getenv("USER_WATCHLIST_REMOVE", "").split(",") if t.strip()}

    # Merge: user additions first (higher priority), then filter removes
    merged = user_add + [t for t in base if t not in set(user_add) and t not in user_remove]
    return merged


WATCHLIST = [
    # ══════════════════════════════════════════════════
    # מניות בלבד — ממוינות לפי עדיפות: טכנולוגיה + מומנטום ראשונות
    # (הפילטר MIN_MARKET_CAP מסנן אוטומטית אם ירדו)
    # הוסרו: ETFs, כלי שירות, נדל"ן, ADRs זרים, מניות סיניות
    # ══════════════════════════════════════════════════

    # ── 1. טכנולוגיה + AI + שבבים (הכי תנועתיים) ──
    "NVDA", "MSFT", "AAPL", "META", "GOOGL", "AMZN", "TSLA",
    "AMD", "AVGO", "ORCL", "QCOM", "ARM", "PLTR", "SMCI",
    "AMAT", "LRCX", "KLAC", "MU", "TXN", "ADI", "NXPI",
    "ASML", "TSM",  # שבבים בינלאומיים עם נפח גבוה

    # ── 2. תוכנה / ענן / SaaS ──
    "NOW", "CRM", "ADBE", "INTU", "PANW", "CRWD", "FTNT",
    "SNPS", "CDNS", "SHOP", "MELI", "UBER", "NFLX",

    # ── 3. פיננסים — רק הגדולים עם מומנטום ──
    "JPM", "GS", "MS", "V", "MA", "AXP",
    "BLK", "SPGI", "CME", "SCHW", "COF",

    # ── 4. בריאות — רק מניות תנועתיות ──
    "LLY", "UNH", "ABBV", "ISRG", "TMO", "AMGN",
    "REGN", "SYK", "DHR", "ABT",

    # ── 5. צרכנות חזקה ──
    "COST", "HD", "MCD", "WMT", "BKNG", "ABNB",
    "LOW", "CMG", "NKE",

    # ── 6. תעשייה / ביטחון ──
    "RTX", "LMT", "GE", "CAT", "HON", "DE",
    "NOC", "GD", "ETN",

    # ── 7. אנרגיה — רק הגדולים ──
    "XOM", "CVX", "COP", "EOG",

    # ── 8. קריפטו / פינטק ──
    "COIN", "PYPL", "MSTR",

    # ── 9. תקשורת ──
    "TMUS", "DIS", "CMCSA",

    # ── 10. קרנות סל (ETFs) — שוק רחב + סקטוריאליות תנועתיות ──
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    # סקטוריאליות עם נפח ותנודתיות
    "XLK",   # טכנולוגיה — תנועתי
    "XLF",   # פיננסים — תנועתי
    "XLE",   # אנרגיה — תנועתי
    "XLV",   # בריאות
    "XLY",   # צריכה מותרת
    "XLI",   # תעשייה
    "SMH",   # שבבים — נפח גבוה מאוד
    "ARKK",  # ARK Innovation — חדשנות + תנודתיות
    # ממונפים (סיכון גבוה — אבל תנועה גדולה)
    "TQQQ",  # 3x QQQ
    "SOXL",  # 3x semis
    "UPRO",  # 3x SPY
    # סחורות
    "GLD",   # זהב
    "SLV",   # כסף
    # קריפטו
    "IBIT",  # Bitcoin ETF
    "ETHE",  # Ethereum ETF
    # אג"ח (תגובה לריבית)
    "TLT",   # 20Y Treasury
]

# הסר כפילויות תוך שמירת הסדר המקורי
WATCHLIST = list(dict.fromkeys(WATCHLIST))

# קטגוריות לכל סימבול — הגדרת ברירת מחדל: "מניה"
# רק ETFs וסחורות צריכים הגדרה מפורשת (הפילטר חל עליהם אחרת)
ASSET_CATEGORY = {
    # ── ETFs שוק רחב ──
    "SPY":"קרן סל","QQQ":"קרן סל","IWM":"קרן סל","DIA":"קרן סל",
    "VTI":"קרן סל","VOO":"קרן סל",
    # ── ETFs סקטוריאליים ──
    "XLK":"קרן סל","XLF":"קרן סל","XLE":"קרן סל","XLV":"קרן סל",
    "XLI":"קרן סל","XLRE":"קרן סל","XLY":"קרן סל","XLP":"קרן סל","XLB":"קרן סל",
    # ── ממונפים ──
    "TQQQ":"ממונף x3","SOXL":"ממונף x3","UPRO":"ממונף x3","TECL":"ממונף x3",
    "QLD":"ממונף x2","SSO":"ממונף x2","FNGU":"ממונף x3","LABU":"ממונף x3","WEBL":"ממונף x3",
    "UDOW":"ממונף x3",
    # ── הפוכים ──
    "SQQQ":"הפוך x3","SDOW":"הפוך x3","SPXU":"הפוך x3","SOXS":"הפוך x3",
    # ── תנודתיות ──
    "UVXY":"נגזר VIX","VXX":"נגזר VIX",
    # ── סחורות ──
    "GLD":"סחורה - זהב","SLV":"סחורה - כסף","GDX":"סחורה - זהב","GDXJ":"סחורה - זהב",
    "USO":"סחורה - נפט","UNG":"סחורה - גז","DBO":"סחורה - נפט",
    "CORN":"סחורה - חקלאות","WEAT":"סחורה - חקלאות","PDBC":"סחורות",
    # ── קריפטו ETF ──
    "BITO":"קריפטו ETF","ETHE":"קריפטו ETF","IBIT":"קריפטו ETF",
    # כל שאר הסימבולים = "מניה" (ברירת מחדל ב-ASSET_CATEGORY.get(ticker, "מניה"))
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
    # get_current_indicators returns bb_position as float 0.0 (lower) → 1.0 (upper)
    bb_pos = ind.get("bb_position")
    if bb_pos is not None:
        try:
            bb_pos = float(bb_pos)
            if bb_pos < 0.35:
                score += 2
                reasons.append("BB תחתון✅")
            elif bb_pos < 0.65:
                score += 1
                reasons.append("BB אמצע")
            else:
                score -= 1
                reasons.append("BB עליון❌")
        except (TypeError, ValueError):
            pass

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
MIN_MARKET_CAP = int(os.getenv("MIN_MARKET_CAP_USD", "20000000000"))  # $20B default (was $100B — far too narrow: only ~80 mega-caps, the most efficient/hardest-to-beat names). $20B ≈ ~300 quality large-caps across all sectors = a much broader hunting ground, still safe & liquid (no small-caps). Tune via MIN_MARKET_CAP_USD env.

# Start background refresh AFTER all module-level names are defined
# (WATCHLIST + MIN_MARKET_CAP must exist before the thread can reference them)
threading.Thread(target=refresh_large_cap_list, daemon=True, name="watchlist-refresh").start()


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
    Uses get_watchlist() (already pre-filtered by market cap) to avoid
    249 extra yfinance market-cap calls.
    """
    results = []

    # Use dynamic watchlist (pre-filtered large-caps) — skip per-ticker market cap checks
    watchlist = get_watchlist()
    # Shuffle for diversity — different stocks scanned each cycle
    import random as _random
    shuffled = list(watchlist)
    _random.shuffle(shuffled)

    for ticker in shuffled:
        try:
            category = ASSET_CATEGORY.get(ticker, "מניה")

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
                "market_cap": 0,   # pre-filtered, no need to re-fetch
                "rsi": ind.get("rsi"),
                "macd": ind.get("macd"),
                "signal": ind.get("macd_signal"),
                "bb_position": ind.get("bb_position"),
                "volume_ratio": ind.get("volume_ratio"),
                "price": price if price else ind.get("close", 0),
                "change_pct": change_pct,
                "reason": reason,
            })

            # Early exit: stop scanning once we have enough strong candidates
            if len(results) >= max_results * 5:
                break

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
