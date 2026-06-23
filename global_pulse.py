"""
Global Market Pulse — overnight / overseas leading signal for the US open.
=========================================================================

When the US market is CLOSED, the best available predictor of how the US will
open is what the OPEN markets are doing: S&P 500 futures (trade ~23h) plus the
major overseas indices — Tokyo Nikkei, London FTSE, Frankfurt DAX, Hong Kong
Hang Seng. A coordinated overseas sell-off (especially on news) almost always
drags US futures down → the bot should open the US session CAUTIOUSLY.

This module turns those moves into ONE risk verdict + a caution multiplier:

    caution_mult ∈ {0.5, 0.75, 1.0}   (≤ 1.0 — only REDUCES US risk, never raises it)

So the worst case of acting on it is being slightly too conservative — never
more aggressive. The trading code reads get_cached_caution_mult().
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# yfinance symbols for markets open while the US is closed.
_PULSE_SYMBOLS = {
    "S&P futures": "ES=F",    # ~23h trading — the most direct US-open predictor
    "Nikkei":      "^N225",   # 🇯🇵 Tokyo
    "FTSE":        "^FTSE",    # 🇬🇧 London
    "DAX":         "^GDAXI",   # 🇩🇪 Frankfurt
    "Hang Seng":   "^HSI",     # 🇭🇰 Hong Kong
}

# S&P futures dominate (they ARE the US open); overseas indices confirm the regime.
_WEIGHTS = {"S&P futures": 3.0, "Nikkei": 1.0, "FTSE": 1.0, "DAX": 1.0, "Hang Seng": 0.7}

_cache: dict = {"verdict": None, "ts": 0.0, "detail": {}, "score": 0.0, "caution_mult": 1.0}
_lock = threading.Lock()
_TTL = 600                # re-poll at most every 10 min
_MAX_AGE_FOR_USE = 6 * 3600  # only let a pulse < 6h old influence trading


def _pct_change(symbol: str) -> float | None:
    """Latest session % change for an index/future via yfinance. None on failure."""
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="2d")
        if h is None or len(h) < 2:
            return None
        prev = float(h["Close"].iloc[-2])
        cur = float(h["Close"].iloc[-1])
        if prev <= 0:
            return None
        return (cur - prev) / prev * 100.0
    except Exception:
        return None


def _vix_level() -> float | None:
    """Current VIX (fear index) LEVEL. None on failure. A high/spiking VIX overnight
    is risk-off even when the equity indices look calm."""
    try:
        import yfinance as yf
        h = yf.Ticker("^VIX").history(period="2d")
        if h is None or len(h) < 1:
            return None
        return float(h["Close"].iloc[-1])
    except Exception:
        return None


def get_global_pulse(force: bool = False) -> dict:
    """
    Compute the global risk verdict from the open overseas markets + S&P futures.
    Returns {verdict, score, caution_mult, detail{name:pct}, ts}. Cached for 10 min.
    """
    now = time.time()
    if not force:
        with _lock:
            if _cache["verdict"] and now - _cache["ts"] < _TTL:
                return dict(_cache)

    detail: dict[str, float] = {}
    for name, sym in _PULSE_SYMBOLS.items():
        p = _pct_change(sym)
        if p is not None:
            detail[name] = round(p, 2)

    if not detail:
        return {"verdict": "unknown", "score": 0.0, "caution_mult": 1.0,
                "detail": {}, "ts": now}

    wsum = sum(_WEIGHTS.get(k, 1.0) for k in detail) or 1.0
    score = sum(detail[k] * _WEIGHTS.get(k, 1.0) for k in detail) / wsum

    # Equity-index caution
    if score <= -1.0:
        caution = 0.5     # overseas selling hard → halve US risk
    elif score <= -0.4:
        caution = 0.75
    else:
        caution = 1.0

    # VIX (fear index) — a LEVEL-based caution that catches stress even when the
    # equity indices look calm. Take the MORE cautious of the two signals.
    vix = _vix_level()
    if vix is not None:
        detail["VIX"] = round(vix, 1)
        if   vix >= 32: caution = min(caution, 0.5)
        elif vix >= 26: caution = min(caution, 0.65)
        elif vix >= 21: caution = min(caution, 0.85)

    # Derive the verdict from the FINAL caution so the label matches what we act on.
    if   caution <= 0.5:  verdict = "risk_off"
    elif caution <= 0.75: verdict = "cautious"
    elif score >= 0.8:    verdict = "risk_on"   # positive abroad + calm VIX
    else:                 verdict = "neutral"

    result = {"verdict": verdict, "score": round(score, 2), "caution_mult": caution,
              "vix": detail.get("VIX"), "detail": detail, "ts": now}
    with _lock:
        _cache.update(result)
    logger.info(f"[GLOBAL PULSE] {verdict} (score={score:+.2f}%, caution×{caution}) | {detail}")
    return result


def get_cached_caution_mult() -> float:
    """
    The latest risk multiplier for the trading code (1.0 = normal, <1 = be cautious).
    Safe default 1.0; ignores a stale (>6h) pulse so it never over-restricts.
    """
    with _lock:
        if _cache.get("caution_mult") is not None and time.time() - _cache.get("ts", 0) < _MAX_AGE_FOR_USE:
            return float(_cache["caution_mult"])
    return 1.0


def format_telegram() -> str:
    """Human-readable Hebrew summary for a Telegram report."""
    p = get_global_pulse()
    if p["verdict"] == "unknown":
        return "🌍 דופק גלובלי: אין נתונים זמינים כרגע"
    icons = {"risk_off": "🔴", "cautious": "🟠", "neutral": "⚪", "risk_on": "🟢"}
    names = {"risk_off": "סיכון גבוה — שווקים נופלים",
             "cautious": "זהירות — חולשה בחו\"ל",
             "neutral": "ניטרלי",
             "risk_on": "חיובי — שווקים עולים"}
    lines = [f"🌍 <b>דופק גלובלי: {icons.get(p['verdict'],'')} {names.get(p['verdict'], p['verdict'])}</b>",
             "━━━━━━━━━━━━━━━━"]
    for k, v in p["detail"].items():
        em = "🟢" if v >= 0 else "🔴"
        lines.append(f"  {em} {k}: {v:+.2f}%")
    if p["caution_mult"] < 1.0:
        lines.append(f"\n⚠️ פתיחת ארה\"ב בזהירות — גודל פוזיציה ×{p['caution_mult']}")
    return "\n".join(lines)
