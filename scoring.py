"""
Composite Scoring Engine
Combines ALL signals into a single score 0-100.
Bot only buys when score >= MIN_SCORE (default 60).
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from indicators import get_current_indicators, get_market_conditions, get_stock_data, add_all_indicators

logger = logging.getLogger(__name__)

MIN_BUY_SCORE = 50  # minimum composite score to allow a buy


def _safe(val, default=None):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default


def score_technicals(ticker: str) -> tuple[float, dict]:
    """
    Score all technical indicators for a ticker.
    Returns (score 0-100, breakdown dict).
    """
    indicators = get_current_indicators(ticker)
    if not indicators:
        return 0.0, {"error": "Could not fetch indicators"}

    score = 0
    max_score = 0
    breakdown = {}

    # ── RSI (0-15 points) ──────────────────────────────────────────────
    rsi = _safe(indicators.get("rsi"))
    max_score += 15
    if rsi is not None:
        if 40 <= rsi <= 60:      score += 15; breakdown["rsi"] = f"✅ Neutral zone ({rsi:.1f})"
        elif 30 <= rsi < 40:     score += 12; breakdown["rsi"] = f"✅ Slight oversold ({rsi:.1f})"
        elif 60 < rsi <= 70:     score += 8;  breakdown["rsi"] = f"⚠️ Slight overbought ({rsi:.1f})"
        elif rsi < 30:           score += 5;  breakdown["rsi"] = f"⚠️ Very oversold ({rsi:.1f})"
        else:                    score += 0;  breakdown["rsi"] = f"❌ Overbought ({rsi:.1f})"
    else:
        breakdown["rsi"] = "⚪ N/A"

    # ── MACD (0-15 points) ─────────────────────────────────────────────
    max_score += 15
    macd = _safe(indicators.get("macd"))
    macd_sig = _safe(indicators.get("macd_signal"))
    macd_hist = _safe(indicators.get("macd_hist"))
    macd_bullish = indicators.get("macd_bullish")
    if macd is not None and macd_sig is not None:
        if macd_bullish and macd_hist and macd_hist > 0:
            score += 15; breakdown["macd"] = f"✅ Bullish crossover (hist={macd_hist:.4f})"
        elif macd_bullish:
            score += 10; breakdown["macd"] = "✅ MACD above signal"
        elif macd_hist and macd_hist > -0.05:
            score += 5;  breakdown["macd"] = "⚠️ Weakly bearish"
        else:
            score += 0;  breakdown["macd"] = "❌ Bearish MACD"
    else:
        breakdown["macd"] = "⚪ N/A"

    # ── Bollinger Bands (0-10 points) ──────────────────────────────────
    max_score += 10
    bb_pos = _safe(indicators.get("bb_position"))
    bb_width = _safe(indicators.get("bb_width"))
    if bb_pos is not None:
        if bb_pos < 0.3:          score += 10; breakdown["bb"] = f"✅ Near lower band ({bb_pos:.2f}) - good entry"
        elif bb_pos < 0.5:        score += 8;  breakdown["bb"] = f"✅ Below midline ({bb_pos:.2f})"
        elif bb_pos < 0.7:        score += 5;  breakdown["bb"] = f"⚠️ Above midline ({bb_pos:.2f})"
        elif bb_pos < 0.85:       score += 2;  breakdown["bb"] = f"⚠️ Near upper band ({bb_pos:.2f})"
        else:                     score += 0;  breakdown["bb"] = f"❌ At upper band ({bb_pos:.2f})"
    else:
        breakdown["bb"] = "⚪ N/A"

    # ── Moving Averages trend (0-15 points) ────────────────────────────
    max_score += 15
    ma_score = 0
    above_20 = indicators.get("above_sma20")
    above_50 = indicators.get("above_sma50")
    above_200 = indicators.get("above_sma200")
    if above_20: ma_score += 5
    if above_50: ma_score += 5
    if above_200: ma_score += 5
    score += ma_score
    trend_str = f"SMA20={'✅' if above_20 else '❌'} SMA50={'✅' if above_50 else '❌'} SMA200={'✅' if above_200 else '❌'}"
    breakdown["moving_averages"] = trend_str

    # ── Stochastic (0-10 points) ────────────────────────────────────────
    max_score += 10
    stoch_k = _safe(indicators.get("stoch_k"))
    if stoch_k is not None:
        if stoch_k < 20:          score += 10; breakdown["stochastic"] = f"✅ Oversold ({stoch_k:.1f}) - bounce expected"
        elif stoch_k < 40:        score += 8;  breakdown["stochastic"] = f"✅ Low zone ({stoch_k:.1f})"
        elif stoch_k < 60:        score += 5;  breakdown["stochastic"] = f"⚠️ Neutral ({stoch_k:.1f})"
        elif stoch_k < 80:        score += 2;  breakdown["stochastic"] = f"⚠️ High zone ({stoch_k:.1f})"
        else:                     score += 0;  breakdown["stochastic"] = f"❌ Overbought ({stoch_k:.1f})"
    else:
        breakdown["stochastic"] = "⚪ N/A"

    # ── CCI (0-5 points) ───────────────────────────────────────────────
    max_score += 5
    cci = _safe(indicators.get("cci"))
    if cci is not None:
        if -100 <= cci <= 0:      score += 5;  breakdown["cci"] = f"✅ Good CCI ({cci:.1f})"
        elif 0 < cci <= 100:      score += 3;  breakdown["cci"] = f"⚠️ Neutral CCI ({cci:.1f})"
        else:                     score += 0;  breakdown["cci"] = f"❌ Extreme CCI ({cci:.1f})"
    else:
        breakdown["cci"] = "⚪ N/A"

    # ── Williams %R (0-5 points) ────────────────────────────────────────
    max_score += 5
    wr = _safe(indicators.get("williams_r"))
    if wr is not None:
        if wr < -80:              score += 5;  breakdown["williams_r"] = f"✅ Oversold ({wr:.1f})"
        elif wr < -50:            score += 3;  breakdown["williams_r"] = f"⚠️ Neutral ({wr:.1f})"
        else:                     score += 0;  breakdown["williams_r"] = f"❌ Overbought ({wr:.1f})"
    else:
        breakdown["williams_r"] = "⚪ N/A"

    # ── Volume (0-10 points) ────────────────────────────────────────────
    max_score += 10
    vol_ratio = _safe(indicators.get("volume_ratio"))
    if vol_ratio is not None:
        if vol_ratio >= 1.5:      score += 10; breakdown["volume"] = f"✅ High volume ({vol_ratio:.2f}x)"
        elif vol_ratio >= 1.0:    score += 7;  breakdown["volume"] = f"✅ Normal volume ({vol_ratio:.2f}x)"
        elif vol_ratio >= 0.7:    score += 4;  breakdown["volume"] = f"⚠️ Low volume ({vol_ratio:.2f}x)"
        else:                     score += 0;  breakdown["volume"] = f"❌ Very low volume ({vol_ratio:.2f}x)"
    else:
        breakdown["volume"] = "⚪ N/A"

    # ── OBV trend (0-10 points) ────────────────────────────────────────
    # OBV rising with price = confirmed move. OBV falling with price = fakeout warning.
    max_score += 10
    try:
        import yfinance as _yf
        import numpy as _np
        _hist = _yf.Ticker(ticker).history(period="20d", interval="1d")
        if len(_hist) >= 10:
            from indicators import _obv as _calc_obv
            _obv_series = _calc_obv(_hist)
            _price_chg = float(_hist["Close"].iloc[-1]) - float(_hist["Close"].iloc[-5])
            _obv_chg   = float(_obv_series.iloc[-1])   - float(_obv_series.iloc[-5])
            _obv_trend_up = _obv_chg > 0
            _price_trend_up = _price_chg > 0
            if _obv_trend_up and _price_trend_up:
                score += 10; breakdown["obv"] = "✅ OBV מאשר מגמה עולה"
            elif _obv_trend_up and not _price_trend_up:
                score += 7;  breakdown["obv"] = "✅ OBV חיובי (צבירה)"
            elif not _obv_trend_up and not _price_trend_up:
                score += 3;  breakdown["obv"] = "⚠️ OBV יורד עם מחיר"
            else:
                score += 0;  breakdown["obv"] = "❌ divergence: מחיר עולה, OBV יורד — אזהרה"
        else:
            breakdown["obv"] = "⚪ N/A"
    except Exception:
        breakdown["obv"] = "⚪ N/A"

    # ── Momentum (0-5 points) ───────────────────────────────────────────
    max_score += 5
    momentum = _safe(indicators.get("momentum_10"))
    if momentum is not None:
        if momentum > 0:          score += 5;  breakdown["momentum"] = f"✅ Positive ({momentum:.2f})"
        elif momentum > -2:       score += 2;  breakdown["momentum"] = f"⚠️ Slightly negative ({momentum:.2f})"
        else:                     score += 0;  breakdown["momentum"] = f"❌ Negative ({momentum:.2f})"
    else:
        breakdown["momentum"] = "⚪ N/A"

    # ── Volatility (0-5 points) ─────────────────────────────────────────
    max_score += 5
    vol20 = _safe(indicators.get("volatility_20"))
    if vol20 is not None:
        if vol20 < 1.5:           score += 5;  breakdown["volatility"] = f"✅ Low volatility ({vol20:.2f}%)"
        elif vol20 < 3.0:         score += 3;  breakdown["volatility"] = f"⚠️ Medium volatility ({vol20:.2f}%)"
        else:                     score += 0;  breakdown["volatility"] = f"❌ High volatility ({vol20:.2f}%)"
    else:
        breakdown["volatility"] = "⚪ N/A"

    # Normalize to 0-100
    final_score = round((score / max_score) * 100, 1) if max_score > 0 else 0
    return final_score, breakdown


def score_market(market: dict) -> tuple[float, dict]:
    """Score overall market conditions. Returns (0-100, breakdown)."""
    score = 50  # neutral default
    breakdown = {}

    vix = market.get("vix")
    if vix is not None:
        if vix < 15:      score += 20; breakdown["vix"] = f"✅ Very calm ({vix})"
        elif vix < 20:    score += 10; breakdown["vix"] = f"✅ Calm ({vix})"
        elif vix < 25:    score += 0;  breakdown["vix"] = f"⚠️ Elevated ({vix})"
        elif vix < 30:    score -= 10; breakdown["vix"] = f"⚠️ High fear ({vix})"
        else:             score -= 30; breakdown["vix"] = f"❌ Extreme fear ({vix})"

    spy_up = market.get("spy_above_sma50")
    if spy_up is True:    score += 15; breakdown["spy"] = "✅ SPY above SMA50 (uptrend)"
    elif spy_up is False: score -= 15; breakdown["spy"] = "❌ SPY below SMA50 (downtrend)"

    spy_rsi = market.get("spy_rsi")
    if spy_rsi is not None:
        if spy_rsi < 70:  score += 5;  breakdown["spy_rsi"] = f"✅ SPY RSI ok ({spy_rsi:.1f})"
        else:             score -= 10; breakdown["spy_rsi"] = f"❌ SPY overbought ({spy_rsi:.1f})"

    return max(0, min(100, score)), breakdown


def get_composite_score(ticker: str, sentiment_score: int = 5) -> dict:
    """
    Full composite score combining technicals + market + sentiment.
    Returns dict with final score and full breakdown.
    """
    # Technical score (60% weight)
    tech_score, tech_breakdown = score_technicals(ticker)

    # Market conditions score (20% weight)
    market = get_market_conditions()
    mkt_score, mkt_breakdown = score_market(market)

    # Sentiment score — convert 1-10 to 0-100 (20% weight)
    sent_score = max(0, min(100, (max(1, sentiment_score) - 1) / 9 * 100))

    # Weighted composite
    composite = round(
        tech_score * 0.60 +
        mkt_score  * 0.20 +
        sent_score * 0.20,
        1
    )

    decision = "BUY ✅" if composite >= MIN_BUY_SCORE else "SKIP ❌"

    logger.info(
        f"[SCORE] {ticker}: composite={composite}/100 "
        f"(tech={tech_score}, market={mkt_score}, sentiment={sent_score:.0f}) → {decision}"
    )

    return {
        "ticker": ticker,
        "composite_score": composite,
        "min_score": MIN_BUY_SCORE,
        "decision": decision,
        "should_buy": composite >= MIN_BUY_SCORE,
        "weights": {"technicals": "60%", "market": "20%", "sentiment": "20%"},
        "scores": {
            "technicals": tech_score,
            "market": mkt_score,
            "sentiment": round(sent_score, 1),
        },
        "breakdown": {
            "technicals": tech_breakdown,
            "market": mkt_breakdown,
        },
        "vix": market.get("vix"),
    }
