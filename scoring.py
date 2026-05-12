"""
Composite Scoring Engine
Combines ALL signals into a single score 0-100.
Bot only buys when score >= MIN_SCORE (default 60).
"""
import logging
import time as _time_module
import numpy as np
import pandas as pd
import yfinance as yf
from indicators import get_current_indicators, get_market_conditions, get_stock_data, add_all_indicators

logger = logging.getLogger(__name__)

# ── Higher-Timeframe (Weekly) Cache ──────────────────────────────────────────
# Stores (timestamp, is_bearish: bool) per ticker; TTL = 4 hours
_htf_cache: dict[str, tuple[float, bool]] = {}
_HTF_CACHE_TTL = 4 * 3600  # seconds


def _get_weekly_bearish(ticker: str) -> bool:
    """
    Return True if the weekly timeframe is bearish for *ticker*.
    Bearish = weekly close < weekly SMA(20) AND weekly MACD < weekly MACD signal.
    Result is cached for 4 hours per ticker.
    """
    now_ts = _time_module.time()
    cached = _htf_cache.get(ticker)
    if cached is not None:
        ts, result = cached
        if now_ts - ts < _HTF_CACHE_TTL:
            return result

    try:
        import yfinance as _yf
        _hist = _yf.Ticker(ticker).history(period="2y", interval="1wk", timeout=10)
        if _hist is None or len(_hist) < 26:
            _htf_cache[ticker] = (now_ts, False)
            return False

        close = _hist["Close"].dropna()

        # Weekly SMA(20)
        sma20 = close.rolling(20).mean()
        price_below_sma20 = float(close.iloc[-1]) < float(sma20.iloc[-1])

        # Weekly MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_below_signal = float(macd_line.iloc[-1]) < float(signal_line.iloc[-1])

        is_bearish = price_below_sma20 and macd_below_signal
        _htf_cache[ticker] = (now_ts, is_bearish)
        logger.debug(
            f"[HTF] {ticker}: weekly bearish={is_bearish} "
            f"(close<SMA20={price_below_sma20}, MACD<signal={macd_below_signal})"
        )
        return is_bearish

    except Exception as e:
        logger.warning(f"[HTF] Weekly trend check failed for {ticker}: {e}")
        _htf_cache[ticker] = (now_ts, False)  # fail-open: don't penalise on error
        return False

import os as _os
MIN_BUY_SCORE: int = int(_os.getenv("MIN_BUY_SCORE", "58"))  # balanced: was 65 (too strict), now 58


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
        if 35 <= rsi <= 55:      score += 15; breakdown["rsi"] = f"✅ Ideal zone ({rsi:.1f})"
        elif 55 < rsi <= 65:     score += 12; breakdown["rsi"] = f"✅ Healthy uptrend ({rsi:.1f})"
        elif 25 <= rsi < 35:     score += 10; breakdown["rsi"] = f"✅ Oversold ({rsi:.1f})"
        elif 65 < rsi <= 72:     score += 7;  breakdown["rsi"] = f"⚠️ Extended ({rsi:.1f})"
        elif rsi < 25:           score += 4;  breakdown["rsi"] = f"⚠️ Very oversold ({rsi:.1f})"
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
    sma_50  = indicators.get("sma_50")
    sma_200 = indicators.get("sma_200")
    # Strong trend bonus: all 3 MAs aligned = momentum
    if above_20 and above_50 and above_200:
        ma_score += 18  # all aligned = strong uptrend
    elif above_20 and above_50:
        ma_score += 12
    elif above_50 and above_200:
        ma_score += 10
    elif above_20:
        ma_score += 5
    elif above_50:
        ma_score += 3
    # Penalty for below all MAs
    if not above_20 and not above_50:
        ma_score -= 5
    score += ma_score

    # Golden Cross bonus (+8): SMA50 > SMA200 = long-term bull signal
    # Death Cross penalty (-8): SMA50 < SMA200 = long-term bear signal
    max_score += 8
    if sma_50 and sma_200:
        if sma_50 > sma_200:
            score += 8;  breakdown["golden_cross"] = f"✅ Golden Cross (SMA50 > SMA200)"
        else:
            score -= 4;  breakdown["golden_cross"] = f"❌ Death Cross (SMA50 < SMA200)"
    else:
        breakdown["golden_cross"] = "⚪ N/A"

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

    # ── Momentum (0-8 points, raised from 5) ────────────────────────────
    max_score += 8
    momentum = _safe(indicators.get("momentum_10"))
    if momentum is not None:
        if momentum > 3:          score += 8;  breakdown["momentum"] = f"✅ Strong upward ({momentum:.2f})"
        elif momentum > 0:        score += 5;  breakdown["momentum"] = f"✅ Positive ({momentum:.2f})"
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

    # ── VWAP distance (0-8 points) ─────────────────────────────────────
    max_score += 8
    vwap_dist = _safe(indicators.get("vwap_distance_pct"))
    if vwap_dist is not None:
        if -3 <= vwap_dist <= -0.3:  score += 8; breakdown["vwap"] = f"✅ Below VWAP ({vwap_dist:.1f}%) — institutional discount"
        elif -0.3 < vwap_dist <= 1:  score += 5; breakdown["vwap"] = f"✅ Near VWAP ({vwap_dist:.1f}%)"
        elif 1 < vwap_dist <= 3:     score += 2; breakdown["vwap"] = f"⚠️ Above VWAP ({vwap_dist:.1f}%)"
        else:                         score += 0; breakdown["vwap"] = f"❌ Far above VWAP ({vwap_dist:.1f}%)"
    else:
        breakdown["vwap"] = "⚪ N/A"

    # ── Candlestick patterns (0-7 points) ──────────────────────────────
    max_score += 7
    bull_engulf = indicators.get("pattern_bull_engulf", False)
    hammer      = indicators.get("pattern_hammer", False)
    bear_engulf = indicators.get("pattern_bear_engulf", False)
    if bull_engulf:   score += 7; breakdown["candle"] = "✅ Bullish Engulfing — strong reversal"
    elif hammer:      score += 5; breakdown["candle"] = "✅ Hammer — bounce signal"
    elif bear_engulf: score -= 4; breakdown["candle"] = "❌ Bearish Engulfing — avoid entry"
    else:             breakdown["candle"] = "⚪ No pattern"

    # Normalize to 0-100 (clamped both ends — MA bonus can push above max_score,
    # and the MA penalty can push below zero in extreme bearish conditions)
    final_score = round(min(100.0, max(0.0, (score / max_score) * 100)), 1) if max_score > 0 else 0
    return final_score, breakdown


def score_market(market: dict) -> tuple[float, dict]:
    """Score overall market conditions. Returns (0-100, breakdown).
    Stricter than before — penalises bad conditions more heavily.
    """
    score = 50  # neutral default
    breakdown = {}

    vix = market.get("vix")
    if vix is not None:
        if vix < 15:      score += 25; breakdown["vix"] = f"✅ Very calm ({vix})"
        elif vix < 18:    score += 15; breakdown["vix"] = f"✅ Calm ({vix})"
        elif vix < 22:    score += 5;  breakdown["vix"] = f"⚠️ Slightly elevated ({vix})"
        elif vix < 27:    score -= 15; breakdown["vix"] = f"⚠️ High fear ({vix})"
        elif vix < 32:    score -= 25; breakdown["vix"] = f"❌ Very high fear ({vix})"
        else:             score -= 40; breakdown["vix"] = f"❌ Extreme panic ({vix})"

    spy_up = market.get("spy_above_sma50")
    if spy_up is True:    score += 20; breakdown["spy"] = "✅ SPY above SMA50 (uptrend)"
    elif spy_up is False: score -= 25; breakdown["spy"] = "❌ SPY below SMA50 (downtrend — avoid buys)"

    spy_rsi = market.get("spy_rsi")
    if spy_rsi is not None:
        if spy_rsi < 40:  score += 10; breakdown["spy_rsi"] = f"✅ SPY oversold — potential reversal ({spy_rsi:.1f})"
        elif spy_rsi < 65: score += 5; breakdown["spy_rsi"] = f"✅ SPY RSI healthy ({spy_rsi:.1f})"
        elif spy_rsi < 75: score -= 5; breakdown["spy_rsi"] = f"⚠️ SPY slightly overbought ({spy_rsi:.1f})"
        else:             score -= 15; breakdown["spy_rsi"] = f"❌ SPY overbought ({spy_rsi:.1f})"

    # Fear & Greed Index — CNN market sentiment
    fg = market.get("fear_greed")
    if fg is not None:
        if fg <= 25:      score += 15; breakdown["fear_greed"] = f"✅ Extreme Fear ({fg}) — contrarian BUY signal"
        elif fg <= 45:    score += 8;  breakdown["fear_greed"] = f"✅ Fear ({fg}) — good entry"
        elif fg <= 55:    score += 3;  breakdown["fear_greed"] = f"⚪ Neutral ({fg})"
        elif fg <= 75:    score -= 5;  breakdown["fear_greed"] = f"⚠️ Greed ({fg}) — caution"
        else:             score -= 15; breakdown["fear_greed"] = f"❌ Extreme Greed ({fg}) — avoid new buys"

    return max(0, min(100, score)), breakdown


def get_composite_score(ticker: str, sentiment_score: int = 5) -> dict:
    """
    Full composite score combining technicals + market + sentiment.
    Returns dict with final score and full breakdown.
    """
    # Technical score (60% weight)
    tech_score, tech_breakdown = score_technicals(ticker)

    # ── Multi-Timeframe Confirmation: weekly trend filter ─────────────────
    # If the weekly chart is bearish (price < SMA20 AND MACD < signal),
    # discount the daily tech score by 30% to avoid fighting the macro trend.
    if _get_weekly_bearish(ticker):
        original_tech = tech_score
        tech_score = round(tech_score * 0.70, 1)
        tech_breakdown["htf_weekly"] = (
            f"⚠️ Weekly bearish — tech score discounted 30% "
            f"({original_tech:.1f} → {tech_score:.1f})"
        )
        logger.info(f"[HTF] {ticker}: weekly bearish — tech score {original_tech} → {tech_score}")
    else:
        tech_breakdown["htf_weekly"] = "✅ Weekly trend OK (no discount)"

    # Market conditions score (25% weight)
    market = get_market_conditions()
    mkt_score, mkt_breakdown = score_market(market)

    # Sentiment score — convert 1-10 to 0-100 (15% weight)
    sent_score = max(0, min(100, (max(1, sentiment_score) - 1) / 9 * 100))

    # VIX-adaptive weights — market filter gets heavier in fear environments
    vix = market.get("vix") or 20
    if vix < 16:
        w_tech, w_mkt, w_sent = 0.65, 0.20, 0.15   # calm: trust technicals
    elif vix < 22:
        w_tech, w_mkt, w_sent = 0.60, 0.25, 0.15   # normal
    elif vix < 28:
        w_tech, w_mkt, w_sent = 0.50, 0.35, 0.15   # elevated fear
    else:
        w_tech, w_mkt, w_sent = 0.40, 0.45, 0.15   # high fear: macro dominates

    composite = round(
        tech_score * w_tech +
        mkt_score  * w_mkt +
        sent_score * w_sent,
        1
    )

    # ── Relative Strength vs SPY ──────────────────────────────────────────
    # Stocks outperforming the market get a bonus — buy market leaders, not laggards
    rs_bonus = 0
    try:
        import yfinance as _yf
        _tickers = _yf.download([ticker, "SPY"], period="3mo", progress=False, auto_adjust=True)["Close"]
        if ticker in _tickers.columns and "SPY" in _tickers.columns:
            _sr = float(_tickers[ticker].iloc[-1] / _tickers[ticker].iloc[0])
            _sb = float(_tickers["SPY"].iloc[-1] / _tickers["SPY"].iloc[0])
            _rs = _sr / _sb if _sb > 0 else 1.0
            if _rs >= 1.15:   rs_bonus = 8   # strong leader: +15% vs SPY
            elif _rs >= 1.05: rs_bonus = 4   # mild outperformance
            elif _rs <= 0.90: rs_bonus = -5  # laggard: penalize
    except Exception:
        pass
    composite = round(min(100, max(0, composite + rs_bonus)), 1)

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
        "weights": {"technicals": "60%", "market": "25%", "sentiment": "15%"},
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
