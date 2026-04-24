"""
Stock Scanner — finds the best buy opportunity right now.
Scans top US stocks using: technical indicators + AI news sentiment.
Dynamic watchlist: fetches S&P500 + Nasdaq100 + Russell1000 from Wikipedia
daily and filters to only companies with market cap > MIN_MARKET_CAP.
"""

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
    """Fetch tickers from S&P 500 + Nasdaq 100 via Wikipedia."""
    import pandas as pd
    tickers = set()
    sources = [
        ("S&P 500",    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, "Symbol"),
        ("Nasdaq 100", "https://en.wikipedia.org/wiki/Nasdaq-100",                  4, "Ticker"),
    ]
    for name, url, table_idx, col in sources:
        try:
            df = pd.read_html(url, timeout=15)[table_idx]
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

    # Parallel market-cap checks — 30 workers, should finish in ~60s
    with ThreadPoolExecutor(max_workers=30) as ex:
        list(ex.map(_check, all_tickers))

    with _dynamic_list_lock:
        _dynamic_list.clear()
        _dynamic_list.extend(result)
        _dynamic_list_date = today

    logger.info(f"Dynamic watchlist: {len(result)} stocks above ${MIN_MARKET_CAP/1e9:.0f}B market cap")


def get_watchlist() -> list[str]:
    """
    Returns the dynamic large-cap watchlist if ready, else static WATCHLIST.
    Call refresh_large_cap_list() once at startup (in a background thread).
    """
    with _dynamic_list_lock:
        if _dynamic_list:
            return list(_dynamic_list)
    return list(WATCHLIST)


# Start background refresh immediately on import (non-blocking)
threading.Thread(target=refresh_large_cap_list, daemon=True, name="watchlist-refresh").start()

WATCHLIST = [
    # ══════════════════════════════════════════════════
    # מניות — כל החברות מעל $100 מיליארד שווי שוק
    # (הפילטר MIN_MARKET_CAP מסנן אוטומטית אם ירדו)
    # ══════════════════════════════════════════════════

    # טכנולוגיה ענקים
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "AVGO", "ORCL", "AMD", "INTC", "CSCO", "TXN", "QCOM", "IBM",
    "AMAT", "LRCX", "KLAC", "MU", "ADI", "MCHP", "NXPI",

    # תוכנה / ענן / SaaS
    "NOW", "CRM", "ADBE", "INTU", "PANW", "CRWD", "FTNT", "SNPS",
    "CDNS", "ANSS", "PTC", "PLTR", "SAP", "ASML",

    # אינטרנט / מדיה / מסחר
    "NFLX", "UBER", "BKNG", "ABNB", "EBAY", "PYPL", "SHOP", "MELI",

    # שבבים / חומרה
    "ARM", "SMCI", "DELL", "HPQ", "HPE", "STX", "WDC",

    # AI / ענן
    "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # כבר למעלה — הדגשה

    # קריפטו / פינטק גדולים
    "COIN", "MSTR", "SQ", "PYPL",

    # ── פיננסים ──
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA",
    "BLK", "SCHW", "CB", "PGR", "MMC", "AON", "SPGI", "MCO",
    "ICE", "CME", "COF", "USB", "TFC", "PNC", "BK", "STT",
    "BRK-B",  # Berkshire Hathaway

    # ── בריאות / תרופות / ביוטק ──
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "DHR", "AMGN", "ISRG", "VRTX", "REGN", "BSX", "ELV", "CVS",
    "SYK", "ZTS", "GILD", "MDT", "CI", "HUM", "BIIB", "ILMN",
    "IDXX", "MTD", "WAT", "A",

    # ── צרכנות / קמעונאות ──
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW",
    "TJX", "ROST", "DG", "DLTR", "YUM", "CMG", "DPZ",
    "BABA", "JD", "PDD",

    # ── מוצרי צריכה / מזון ──
    "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "K", "SJM", "HRL", "MKC",

    # ── מדיה ובידור ──
    "DIS", "CMCSA", "WBD", "PARA", "NFLX",

    # ── תקשורת ──
    "T", "VZ", "TMUS",

    # ── אנרגיה ──
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "VLO",
    "MPC", "HES", "DVN", "FANG",

    # ── תעשייה / ביטחון ──
    "BA", "CAT", "HON", "RTX", "LMT", "GE", "MMM", "DE",
    "UPS", "FDX", "ETN", "EMR", "ROK", "PH", "ITW",
    "NOC", "GD", "L3H", "HII", "TDG",

    # ── נדל"ן / תשתיות ──
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "DLR",

    # ── חומרים ──
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE",

    # ── רכב ──
    "TSLA", "TM", "GM", "F",

    # ── ניהול נכסים / השקעות אלטרנטיביות ──
    "BX",   # Blackstone
    "KKR",  # KKR
    "APO",  # Apollo Global
    "CG",   # Carlyle
    "BAM",  # Brookfield

    # ── ייעוץ / IT שירותים ──
    "ACN",  # Accenture
    "FI",   # Fiserv
    "FIS",  # Fidelity National

    # ── מניות בינלאומיות הנסחרות בארה"ב ──
    "TSM",   # Taiwan Semiconductor (~$800B)
    "NVO",   # Novo Nordisk (~$350B) — תרופות סוכרת
    "AZN",   # AstraZeneca (~$250B)
    "SHEL",  # Shell (~$200B)
    "ACN",   # Accenture (~$200B)
    "SNY",   # Sanofi (~$130B)
    "UL",    # Unilever (~$120B)
    "BHP",   # BHP Group (~$130B)
    "RY",    # Royal Bank of Canada (~$170B)
    "TD",    # Toronto-Dominion Bank (~$100B)
    "HSBC",  # HSBC (~$160B)
    "NVS",   # Novartis (~$200B)
    "RHHBY", # Roche (~$200B)
    "TTE",   # TotalEnergies (~$140B)
    "RIO",   # Rio Tinto (~$100B)
    "SONY",  # Sony (~$100B)
    "MUFG",  # Mitsubishi UFJ Financial (~$150B)
    "HDB",   # HDFC Bank India (~$150B)
    "SIEGY", # Siemens (~$150B)
    "LVMUY", # LVMH — יוקרה (~$350B)
    "LRLCY", # L'Oreal (~$200B)
    "TCEHY", # Tencent (~$400B)
    "SAP",   # SAP (~$250B)

    # ── שירותים / תשתיות ──
    "NEE",   # NextEra Energy (~$120B) — אנרגיה ירוקה
    "ADP",   # Automatic Data Processing (~$100B)
    "EQIX",  # Equinix (~$80B — data centers)
    "WM",    # Waste Management (~$80B)

    # ── קרנות סל (ETFs) — שוק רחב ──
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    # ממונפים
    "TQQQ", "SOXL", "UPRO", "QLD", "SSO",
    # סקטוריאליים
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
    # סחורות
    "GLD", "SLV", "USO", "GDX",
    # קריפטו ETF
    "IBIT", "BITO",
]

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
MIN_MARKET_CAP = 100_000_000_000  # $100 מיליארד


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
