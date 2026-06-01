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
_htf_cache: dict[str, tuple[float, bool]] = {}
_HTF_CACHE_TTL = 4 * 3600
_MAX_CACHE_SIZE = 300  # prevent unbounded memory growth

# ── Relative Strength vs SPY Cache ───────────────────────────────────────────
_rs_cache: dict[str, tuple[float, float]] = {}
_RS_CACHE_TTL = 2 * 3600

# ── Pre-Market Gap Cache ──────────────────────────────────────────────────────
_premarket_gap_cache: dict[str, tuple[float, dict]] = {}
_PREMARKET_GAP_TTL = 30 * 60

def _evict_cache(cache: dict, max_size: int = _MAX_CACHE_SIZE) -> None:
    """Remove oldest entries when cache exceeds max_size."""
    if len(cache) > max_size:
        # Remove oldest 20% of entries
        to_remove = sorted(cache.items(), key=lambda x: x[1][0])[:max_size // 5]
        for k, _ in to_remove:
            cache.pop(k, None)


def get_premarket_gap(ticker: str) -> dict | None:
    """
    Calculate pre-market gap for *ticker* relative to previous close.
    Uses 1-minute bars with prepost=True to get current pre-market price.

    Returns:
        {"gap_pct": float, "gap_up": bool, "gap_down": bool}
        or None on any error (fail-open).
    """
    now_ts = _time_module.time()
    cached = _premarket_gap_cache.get(ticker)
    if cached is not None:
        ts, result = cached
        if now_ts - ts < _PREMARKET_GAP_TTL:
            return result

    try:
        import yfinance as _yf
        # Fetch 1-minute intraday bars including pre/post market
        hist = _yf.Ticker(ticker).history(period="2d", interval="1m", prepost=True, timeout=10)
        if hist is None or hist.empty:
            return None

        # Previous regular-session close: last bar where Time is during market hours
        # Use the last Close of the prior calendar day as prev_close
        hist.index = hist.index.tz_convert("America/New_York") if hist.index.tzinfo else hist.index
        import pandas as _pd
        today = _pd.Timestamp.now(tz="America/New_York").date()
        prior_bars = hist[hist.index.date < today]
        if prior_bars.empty:
            return None
        prev_close = float(prior_bars["Close"].iloc[-1])

        # Current premarket price: latest bar from today
        today_bars = hist[hist.index.date == today]
        if today_bars.empty:
            return None
        current_price = float(today_bars["Close"].iloc[-1])

        if prev_close <= 0:
            return None

        gap_pct = (current_price - prev_close) / prev_close * 100
        result = {
            "gap_pct": round(gap_pct, 2),
            "gap_up":   gap_pct > 0,
            "gap_down": gap_pct < 0,
        }
        _evict_cache(_premarket_gap_cache)
        _premarket_gap_cache[ticker] = (now_ts, result)
        logger.debug(f"[PREMARKET] {ticker}: prev_close={prev_close:.2f} current={current_price:.2f} gap={gap_pct:.2f}%")
        return result
    except Exception as e:
        logger.debug(f"[PREMARKET] {ticker}: gap fetch failed — {e}")
        return None


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
MIN_BUY_SCORE: int = int(_os.getenv("MIN_BUY_SCORE", "51"))  # מינימום 51 — קונה כל מניה מעל 50


def get_min_buy_score() -> int:
    """Read MIN_BUY_SCORE freshly from env — apply_insights() may have changed it.
    Use this in scoring decisions to pick up training updates without restart.
    """
    return int(_os.getenv("MIN_BUY_SCORE", "51"))

# ── Fundamental Quality Cache ─────────────────────────────────────────────────
# Stores (timestamp, score: float) per ticker; TTL = 24 hours
_fundamental_cache: dict[str, tuple[float, float]] = {}
_FUNDAMENTAL_CACHE_TTL = 7 * 24 * 3600  # 7 days — fundamentals don't change daily
# Track rate-limit cooldown: if yfinance returns 429, back off for 10 min
_fundamental_rate_limit_until: float = 0.0


def get_fundamental_score(ticker: str) -> float:
    """
    Return a fundamental quality score 0-10 for *ticker* — Buffett-style analysis.

    הקריטריונים של וורן באפט:
      ROE > 15%            → +2.5  (חברה מייצרת ערך מבעלי המניות)
      ROE > 20%            → +1    (איכות מצוינת — נוסף)
      Profit margin > 15%  → +2    (יתרון תחרותי)
      Profit margin > 25%  → +1    (moat — נוסף)
      Debt/Equity < 0.5    → +2    (אין סיכון פיננסי)
      Debt/Equity 0.5-1.0  → +1    (סביר)
      EPS growth > 10%     → +1.5  (צמיחה יציבה)
      P/E 10-25            → +1    (תמחור הוגן)
      P/E > 40             → -2    (overvalued — בורח מ-bubble)
      Free Cash Flow > 0   → +1    (מייצר מזומן אמיתי)

    Result is cached for 24 hours per ticker.
    Returns 5.0 (neutral) if data is unavailable (fail-open).
    """
    now_ts = _time_module.time()
    cached = _fundamental_cache.get(ticker)
    if cached is not None:
        ts, result = cached
        if now_ts - ts < _FUNDAMENTAL_CACHE_TTL:
            logger.debug(f"[FUND] {ticker}: using cached score={result}")
            return result

    # Back off if yfinance is rate-limiting us
    global _fundamental_rate_limit_until
    if _time_module.time() < _fundamental_rate_limit_until:
        logger.debug(f"[FUND] {ticker}: rate-limit backoff — returning neutral 5.0")
        return 5.0

    try:
        # Wrap .info separately — yfinance can raise TypeError internally
        try:
            raw = yf.Ticker(ticker).info
            info = raw if isinstance(raw, dict) else {}
        except Exception as _fetch_err:
            logger.debug(f"[FUND] {ticker}: info fetch error — {_fetch_err}")
            info = {}

        if len(info) < 5:
            raise ValueError("Empty or invalid info response")

        # Remove None values to prevent 'NoneType is not iterable' TypeError
        info = {k: v for k, v in info.items() if v is not None}

        score = 0.0

        # ── Return on Equity (ROE) — מטריקה מס' 1 של באפט ──────────────────
        roe = info.get("returnOnEquity")
        if roe is not None:
            try:
                roe = float(roe)
                if not np.isnan(roe):
                    if roe > 0.20:    score += 3.5   # excellent ROE
                    elif roe > 0.15:  score += 2.5   # good ROE
                    elif roe > 0.10:  score += 1     # acceptable
                    elif roe < 0:     score -= 2     # destroying value
            except (TypeError, ValueError):
                pass

        # ── Profit Margin — היתרון התחרותי (moat) ─────────────────────────
        profit_margin = info.get("profitMargins")
        if profit_margin is not None:
            try:
                profit_margin = float(profit_margin)
                if not np.isnan(profit_margin):
                    if profit_margin > 0.25:    score += 3     # exceptional margin
                    elif profit_margin > 0.15:  score += 2     # strong margin
                    elif profit_margin > 0.05:  score += 0.5
                    elif profit_margin < 0:     score -= 2     # losing money
            except (TypeError, ValueError):
                pass

        # ── Debt / Equity — יציבות פיננסית ─────────────────────────────────
        debt_equity = info.get("debtToEquity")
        if debt_equity is not None:
            try:
                de_ratio = float(debt_equity) / 100.0
                if not np.isnan(de_ratio):
                    if de_ratio < 0.5:     score += 2      # very safe
                    elif de_ratio < 1.0:   score += 1      # acceptable
                    elif de_ratio > 2.0:   score -= 2      # too much debt
            except (TypeError, ValueError):
                pass

        # ── EPS Growth — צמיחה ───────────────────────────────────────────
        eps_growth = info.get("earningsGrowth")
        if eps_growth is not None:
            try:
                eps_growth = float(eps_growth)
                if not np.isnan(eps_growth):
                    if eps_growth > 0.10:     score += 1.5
                    elif eps_growth > 0:      score += 0.5
                    elif eps_growth < -0.10:  score -= 1
            except (TypeError, ValueError):
                pass

        # ── P/E — תמחור הוגן ─────────────────────────────────────────────
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is not None:
            try:
                pe = float(pe)
                if not np.isnan(pe):
                    if 10 <= pe <= 25:    score += 1.5     # value zone
                    elif 25 < pe <= 35:   score += 0.5
                    elif pe > 40:         score -= 2       # bubble zone
                    elif pe < 5:          score -= 1       # too cheap = problem
            except (TypeError, ValueError):
                pass

        # ── Free Cash Flow — מייצר מזומן אמיתי ────────────────────────────
        fcf = info.get("freeCashflow")
        if fcf is not None:
            try:
                fcf = float(fcf)
                if not np.isnan(fcf) and fcf > 0:
                    score += 1
            except (TypeError, ValueError):
                pass

        # Clamp 0-10
        final = round(max(0.0, min(10.0, score)), 2)
        _fundamental_cache[ticker] = (now_ts, final)
        logger.info(
            f"[FUND] {ticker}: roe={roe} margin={profit_margin} d/e={debt_equity} "
            f"eps_growth={eps_growth} pe={pe} → score={final}"
        )
        return final

    except Exception as e:
        err_str = str(e)
        if "Too Many Requests" in err_str or "Rate limited" in err_str or "429" in err_str:
            # Back off for 10 minutes so we stop hammering yfinance
            _fundamental_rate_limit_until = _time_module.time() + 600
            logger.warning(f"[FUND] yfinance rate-limited — backing off 10 min")
        else:
            logger.warning(f"[FUND] Fundamental score failed for {ticker}: {e} — returning neutral 5.0")
        _fundamental_cache[ticker] = (now_ts, 5.0)
        return 5.0


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
        # Re-calibrated based on live trade data:
        # RSI 30-40 = 100% WR in our trades | RSI 40-50 = 0% WR!
        if 30 <= rsi <= 42:      score += 15; breakdown["rsi"] = f"✅ Oversold reversal zone ({rsi:.1f}) — best WR"
        elif 55 < rsi <= 65:     score += 13; breakdown["rsi"] = f"✅ Momentum zone ({rsi:.1f})"
        elif 42 < rsi <= 55:     score += 6;  breakdown["rsi"] = f"⚠️ Neutral zone ({rsi:.1f}) — low WR historically"
        elif 65 < rsi <= 72:     score += 7;  breakdown["rsi"] = f"⚠️ Extended ({rsi:.1f})"
        elif 20 <= rsi < 30:     score += 12; breakdown["rsi"] = f"✅ Deep oversold ({rsi:.1f})"
        elif rsi < 20:           score += 4;  breakdown["rsi"] = f"⚠️ Extreme oversold ({rsi:.1f})"
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

    # ── Moving Averages trend (0-18 points) ────────────────────────────
    max_score += 18
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
        from yfinance_cache import get_ohlcv as _get_ohlcv
        _hist = _get_ohlcv(ticker, period="3mo")  # PERF: reuse the already-prefetched 3mo daily cache (cache hit, 0 network) instead of a separate 20d fetch. OBV uses only the last-5-bar change → identical.
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

    # ── 52-Week High / Low (0-8 points) ────────────────────────────────
    # Near 52w high = breakout momentum.  Near 52w low = potential bounce.
    max_score += 8
    near_52w_high   = indicators.get("near_52w_high", False)
    near_52w_low    = indicators.get("near_52w_low", False)
    pct_from_52w_hi = _safe(indicators.get("pct_from_52w_high"))   # negative = below high
    if near_52w_high:
        # Within 5% of 52w high — breakout momentum signal
        score += 8
        breakdown["week52"] = f"✅ Near 52w high ({pct_from_52w_hi:.1f}%) — breakout zone"
    elif pct_from_52w_hi is not None and pct_from_52w_hi >= -20:
        # Within 20% below 52w high — still a strong chart
        score += 5
        breakdown["week52"] = f"✅ Strong chart ({pct_from_52w_hi:.1f}% from 52w high)"
    elif near_52w_low:
        # Within 10% above 52w low — support bounce zone
        score += 3
        breakdown["week52"] = "⚠️ Near 52w low — support bounce zone"
    elif pct_from_52w_hi is not None and pct_from_52w_hi <= -40:
        # Very far below 52w high — weak chart
        score += 0
        breakdown["week52"] = f"❌ Far from 52w high ({pct_from_52w_hi:.1f}%) — weak chart"
    elif pct_from_52w_hi is not None:
        # Between -20% and -40% — neutral
        score += 2
        breakdown["week52"] = f"⚪ {pct_from_52w_hi:.1f}% from 52w high"
    else:
        breakdown["week52"] = "⚪ 52w data N/A"

    # Normalize to 0-100 (clamped both ends — MA bonus can push above max_score,
    # and the MA penalty can push below zero in extreme bearish conditions)
    final_score = round(min(100.0, max(0.0, (score / max_score) * 100)), 1) if max_score > 0 else 0
    return final_score, breakdown


def score_market(market: dict) -> tuple[float, dict]:
    """Score overall market conditions. Returns (0-100, breakdown).
    Stricter than before — penalises bad conditions more heavily.
    """
    score = 50  # neutral baseline — penalties/bonuses calibrated around this midpoint
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

    # Put/Call Ratio — options market sentiment (contrarian indicator)
    pcr = market.get("put_call_ratio")
    if pcr is not None:
        if pcr >= 1.2:    score += 12; breakdown["pcr"] = f"✅ Extreme puts ({pcr:.2f}) — contrarian BUY"
        elif pcr >= 1.0:  score += 6;  breakdown["pcr"] = f"✅ Bearish sentiment ({pcr:.2f}) — good entry"
        elif pcr <= 0.7:  score -= 8;  breakdown["pcr"] = f"❌ Complacency ({pcr:.2f}) — market too bullish"
        elif pcr <= 0.85: score -= 3;  breakdown["pcr"] = f"⚠️ Low fear ({pcr:.2f})"

    # Market Breadth — % of last 20 days SPY+QQQ both closed up
    breadth = market.get("breadth_score")
    if breadth is not None:
        if breadth > 60:   score += 5;  breakdown["breadth"] = f"✅ Healthy breadth ({breadth:.0f}% days up)"
        elif breadth < 40: score -= 5;  breakdown["breadth"] = f"❌ Weak breadth ({breadth:.0f}% days up)"
        else:              breakdown["breadth"] = f"⚪ Neutral breadth ({breadth:.0f}% days up)"

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

    # ── Sector Rotation Bonus (data-driven: tech +16.7% leading) ─────────
    # Top 3 sectors: +8 pts, Lagging sectors: -6 pts
    # Evidence: tech stocks dominate winning trades in our history
    try:
        from sector_rotation import get_sector_multiplier
        _smult = get_sector_multiplier(ticker)
        if _smult >= 1.20:
            composite = min(100, composite + 8)
            logger.debug(f"[SCORE] {ticker}: leading sector boost +8 → {composite}")
        elif _smult <= 0.85:
            composite = max(0,   composite - 6)
            logger.debug(f"[SCORE] {ticker}: lagging sector penalty -6 → {composite}")
    except Exception:
        pass

    # ── Relative Strength vs SPY (cached 2h) ─────────────────────────────────
    # Stocks outperforming the market get a bonus — buy market leaders, not laggards
    rs_bonus = 0
    try:
        import yfinance as _yf
        import threading as _thr
        _rs_now = _time_module.time()
        _cached_rs = _rs_cache.get(ticker)
        if _cached_rs and _rs_now - _cached_rs[1] < _RS_CACHE_TTL:
            _rs = _cached_rs[0]
        else:
            _raw_rs = _yf.download([ticker, "SPY"], period="3mo", progress=False, auto_adjust=True)["Close"]
            # Handle both MultiIndex (yfinance 0.2+) and flat column structures
            import pandas as _pd_rs
            _tickers_dl = _raw_rs if isinstance(_raw_rs, _pd_rs.DataFrame) else _raw_rs.to_frame()
            # Use get_level_values for MultiIndex safety
            _safe_cols = _tickers_dl.columns.get_level_values(-1) if hasattr(_tickers_dl.columns, "get_level_values") else _tickers_dl.columns
            if ticker in _safe_cols and "SPY" in _safe_cols:
                _sr = float(_tickers_dl[ticker].dropna().iloc[-1] / _tickers_dl[ticker].dropna().iloc[0])
                _sb = float(_tickers_dl["SPY"].dropna().iloc[-1] / _tickers_dl["SPY"].dropna().iloc[0])
                _rs = _sr / _sb if _sb > 0 else 1.0
                _evict_cache(_rs_cache)
                _rs_cache[ticker] = (_rs, _rs_now)
            else:
                _rs = 1.0
        if _rs >= 1.15:   rs_bonus = 8   # strong leader: +15% vs SPY
        elif _rs >= 1.05: rs_bonus = 4   # mild outperformance
        elif _rs <= 0.90: rs_bonus = -5  # laggard: penalize
    except Exception:
        pass
    composite = round(min(100, max(0, composite + rs_bonus)), 1)

    # ── Fundamental Quality Bonus ─────────────────────────────────────────────
    fund_score = get_fundamental_score(ticker)
    composite = round(min(100.0, max(0.0, composite + (fund_score - 5) * 1.2)), 1)

    # ── Post-Earnings Momentum ────────────────────────────────────────────────
    # Stock that just BEAT earnings = strong short-term momentum signal
    # Stock that just MISSED = avoid entry
    try:
        from earnings import check_post_earnings_momentum
        earn_momentum = check_post_earnings_momentum(ticker)
        if earn_momentum.get("post_earnings"):
            earn_bonus = earn_momentum["momentum_score"] - 5  # -5 to +5 swing
            composite = round(min(100, max(0, composite + earn_bonus * 1.5)), 1)
            if earn_momentum["beat"] is True:
                logger.info(f"[EARNINGS] {ticker}: post-earnings beat bonus +{earn_bonus * 1.5:.1f}")
            elif earn_momentum["beat"] is False:
                logger.info(f"[EARNINGS] {ticker}: post-earnings miss penalty {earn_bonus * 1.5:.1f}")
    except Exception:
        pass

    # ── Analyst Consensus + Short Squeeze ─────────────────────────────────────
    try:
        _info = yf.Ticker(ticker).info
        # Analyst target upside: if analysts see >15% upside → +5 bonus
        _target = float(_info.get("targetMeanPrice") or 0)
        _cur    = float(_info.get("currentPrice") or _info.get("regularMarketPrice") or 0)
        if _target > 0 and _cur > 0:
            _upside = (_target - _cur) / _cur * 100
            if _upside >= 20:    composite = min(100, composite + 6)
            elif _upside >= 10:  composite = min(100, composite + 3)
            elif _upside < -10:  composite = max(0,   composite - 4)

        # Short squeeze potential: high short interest = contrarian signal
        _short_pct = float(_info.get("shortPercentOfFloat") or 0) * 100
        if _short_pct >= 20:   composite = min(100, composite + 5)  # squeeze fuel
        elif _short_pct >= 10: composite = min(100, composite + 2)
    except Exception:
        pass

    composite = round(composite, 1)

    # ── PROFIT BOOST: Multi-timeframe momentum scoring (BONUS ONLY) ─────────
    # FIX V2: Removed the -4 penalty because it dropped ARM/AMD/NOW/ASML by 9-14
    # points → blocked them from MIN_BUY_SCORE=60. Now: bonus only, no penalty.
    # This catches momentum WITHOUT punishing recent dips (which are often
    # buying opportunities for mean-reversion).
    try:
        import yfinance as _yf_mom
        from yfinance_cache import get_ohlcv as _get_ohlcv
        _hist_mom = _get_ohlcv(ticker, period="3mo")  # PERF: reuse the already-prefetched 3mo daily cache (cache hit, 0 network) instead of a separate 1mo fetch. Uses last 21 closes → verified identical values.
        if len(_hist_mom) >= 20:
            _closes = _hist_mom["Close"].values
            _cur = float(_closes[-1])
            _d5 = float(_closes[-6]) if len(_closes) >= 6 else _cur
            _d10 = float(_closes[-11]) if len(_closes) >= 11 else _cur
            _d20 = float(_closes[-21]) if len(_closes) >= 21 else _cur

            _ret_5d = (_cur - _d5) / _d5 * 100 if _d5 > 0 else 0
            _ret_10d = (_cur - _d10) / _d10 * 100 if _d10 > 0 else 0
            _ret_20d = (_cur - _d20) / _d20 * 100 if _d20 > 0 else 0

            _positive_count = sum(1 for r in [_ret_5d, _ret_10d, _ret_20d] if r > 0)
            # BONUS ONLY — no penalties (penalties were too aggressive and blocked all entries)
            if _positive_count == 3 and _ret_5d > 2:
                composite = round(min(100, composite + 5), 1)
                logger.debug(f"[MOMENTUM] {ticker}: 3-TF uptrend → +5")
            elif _positive_count == 3:
                composite = round(min(100, composite + 3), 1)
                logger.debug(f"[MOMENTUM] {ticker}: 3-TF positive → +3")
            elif _positive_count == 2:
                # Mixed but trending positive — small bonus
                composite = round(min(100, composite + 1), 1)
            # No penalty for negative momentum — could be a dip-buy opportunity
    except Exception:
        pass

    # ── Pre-Market Gap Adjustment ─────────────────────────────────────────────
    # Only relevant during pre-market / early market (6:00–10:00 AM ET)
    try:
        from trading_hours import _now_et
        _now = _now_et()
        _mins = _now.hour * 60 + _now.minute
        if 6 * 60 <= _mins < 10 * 60:
            _gap = get_premarket_gap(ticker)
            if _gap is not None:
                _gap_pct = _gap["gap_pct"]
                # volume already captured in tech_breakdown pass
                if _gap["gap_up"] and _gap_pct >= 2:
                    composite = round(min(100, composite + 6), 1)
                    logger.info(f"[GAP] {ticker}: gap up {_gap_pct:.2f}% — momentum entering +6")
                elif _gap["gap_down"] and abs(_gap_pct) >= 2:
                    composite = round(max(0, composite - 8), 1)
                    logger.info(f"[GAP] {ticker}: gap down {_gap_pct:.2f}% — avoid gap down -8")
    except Exception:
        pass

    # ── Time-of-Day Bias ─────────────────────────────────────────────────────
    # Avoid buying during chaotic market open (9:30-10:00 ET) and MOC noise (3:30-4:00)
    try:
        from trading_hours import _now_et
        _now = _now_et()
        _h, _m = _now.hour, _now.minute
        _mins = _h * 60 + _m
        if 9 * 60 + 30 <= _mins < 10 * 60:        # 9:30-10:00 → wide spreads
            composite = max(0, composite - 5)   # reduced 12→5 (was too aggressive)
        elif 15 * 60 + 30 <= _mins < 16 * 60:     # 3:30-4:00 → MOC noise
            composite = max(0, composite - 4)   # reduced 8→4
    except Exception:
        pass

    composite = round(composite, 1)
    _min_score = get_min_buy_score()  # read fresh — apply_insights may have updated env
    should_buy_score = composite >= _min_score

    # ── Earnings Blackout — חסום לפני דוח (נשלט ע"י EARNINGS_BLACKOUT_DAYS) ──
    # Pre-earnings = binary event risk — never buy into unknown catalyst
    _earnings_block_reason = None
    _blackout_days = int(_os.getenv("EARNINGS_BLACKOUT_DAYS", "5"))  # default 5d
    try:
        import yfinance as _yf_eb
        from datetime import datetime as _dt_eb, timezone as _tz_eb, timedelta as _td_eb
        _cal = _yf_eb.Ticker(ticker).calendar
        if _cal is not None and not _cal.empty:
            _earn_col = None
            for _ec in ["Earnings Date", "Earnings Date 1"]:
                if _ec in _cal.columns:
                    _earn_col = _ec
                    break
            if _earn_col:
                _earn_dates = _cal[_earn_col].dropna()
                if not _earn_dates.empty:
                    _now_utc = _dt_eb.now(_tz_eb.utc)
                    for _ed in _earn_dates:
                        try:
                            _ed_dt = _ed if hasattr(_ed, "tzinfo") and _ed.tzinfo else _dt_eb.combine(_ed, _dt_eb.min.time(), tzinfo=_tz_eb.utc)
                            _days_to = (_ed_dt - _now_utc).days
                            if 0 <= _days_to <= _blackout_days:
                                # Score penalty scales with proximity
                                _penalty = 30 if _days_to <= 1 else 20 if _days_to <= 3 else 10
                                composite = max(0, composite - _penalty)
                                logger.info(f"[EARNINGS BLACKOUT] {ticker}: earnings in {_days_to}d — score -{_penalty} → {composite}")
                                if _days_to <= 2:
                                    _earnings_block_reason = f"Earnings in {_days_to}d — binary risk, skip"
                                break
                        except Exception:
                            pass
    except Exception:
        pass

    # ── Recent Analyst Upgrades ────────────────────────────────────────────────
    # Buy upgrade in last 7 days = strong institutional signal
    try:
        from datetime import datetime as _dt_an, timezone as _tz_an, timedelta as _td_an
        import yfinance as _yf_an
        _upgrades = _yf_an.Ticker(ticker).upgrades_downgrades
        if _upgrades is not None and not _upgrades.empty:
            _week_ago = _dt_an.now(_tz_an.utc) - _td_an(days=7)
            _rec_upgrades = _upgrades[
                (_upgrades.index > _week_ago) &
                (_upgrades["ToGrade"].str.lower().str.contains("buy|overweight|outperform", na=False))
            ]
            _rec_downgrades = _upgrades[
                (_upgrades.index > _week_ago) &
                (_upgrades["ToGrade"].str.lower().str.contains("sell|underperform|underweight", na=False))
            ]
            if len(_rec_upgrades) >= 2:
                composite = min(100, composite + 8)
                logger.info(f"[ANALYST] {ticker}: {len(_rec_upgrades)} upgrades in 7d → +8")
            elif len(_rec_upgrades) == 1:
                composite = min(100, composite + 4)
                logger.info(f"[ANALYST] {ticker}: 1 upgrade in 7d → +4")
            if len(_rec_downgrades) >= 1:
                composite = max(0, composite - 6)
                logger.info(f"[ANALYST] {ticker}: {len(_rec_downgrades)} downgrades in 7d → -6")
    except Exception:
        pass

    # ── Hard filters (env-controlled) ─────────────────────────────────────
    # These override the score and block buying regardless of composite_score
    hard_block_reason = _earnings_block_reason  # may already be set from earnings blackout

    # 1. SMA50 + Golden Cross hard filters (data-driven from own trade history)
    # Evidence: 60x below SMA50 & 36x SMA50<SMA200 in losing trades
    # FIX V2: Allow high-score exception (>=70) — like death cross does
    # This catches quality dips where the price temporarily fell below SMA50
    # (e.g., CRM at score 55.4 — was getting blocked despite reasonable score)
    if _os.getenv("REQUIRE_ABOVE_SMA50", "true").lower() == "true":
        indicators_for_filter = get_current_indicators(ticker) or {}
        above_sma50  = indicators_for_filter.get("above_sma50")
        above_sma200 = indicators_for_filter.get("above_sma200")

        if above_sma50 is False:
            # NEW: Allow exception for high-conviction trades (composite >= 70)
            # Quality companies often dip below SMA50 — these are buying opportunities
            if composite < 70:
                hard_block_reason = f"מתחת SMA50 — מגמה יורדת (ציון {composite:.0f} < 70)"
            else:
                logger.info(f"[SCORE] {ticker}: below SMA50 BUT score={composite:.0f} ≥70 — allowing dip-buy exception")
        elif _os.getenv("REQUIRE_GOLDEN_CROSS", "true").lower() == "true" and above_sma200 is False:
            # Death Cross: SMA50 < SMA200 appeared in 36/36 losing trades
            # Still allow if score is exceptionally high (≥80) — QCOM won despite death cross
            if composite < 80:
                hard_block_reason = f"דת קרוס SMA50<SMA200 — מגמה ארוכת-טווח שלילית (ציון {composite:.0f} < 80)"
            else:
                logger.info(f"[SCORE] {ticker}: death cross BUT score={composite:.0f} ≥80 — allowing exception")

    # 1b. Bollinger Band ceiling filter — 24x losses had bb_position > 82%
    # "קרוב לגג" = price near top of Bollinger Band = stretched, likely to pull back
    _max_bb = float(_os.getenv("MAX_BB_POSITION", "0.97"))  # raised 0.92→0.97 (market BB at 89-107%, only block extremes)
    try:
        _bb_pos = (get_current_indicators(ticker) or {}).get("bb_position")
        if _bb_pos is not None and float(_bb_pos) > _max_bb:
            hard_block_reason = hard_block_reason or (
                f"BB {float(_bb_pos)*100:.0f}% — קרוב לגג בולינגר (24x הפסדים)"
            )
    except Exception:
        pass

    # 1c. Negative Momentum Filter — 30x losses had negative momentum
    # momentum = difference between current price and SMA(5). Negative = trending down short-term
    try:
        if not hard_block_reason:
            _ind_mom = (get_current_indicators(ticker) or {})
            _macd = _ind_mom.get("macd")
            _macd_sig = _ind_mom.get("macd_signal")
            # If both MACD and signal are negative — short-term momentum bearish
            if _macd is not None and _macd_sig is not None:
                if float(_macd) < -1.0 and float(_macd_sig) < 0:
                    composite = max(0, composite - 10)
                    logger.info(f"[SCORE] {ticker}: negative MACD momentum → discount -10 pts to {composite}")
    except Exception:
        pass

    # 2. Volume hard filter — low volume = weak signal
    _min_vol = float(_os.getenv("MIN_VOLUME_RATIO", "0.5"))
    _vol_ratio = None
    try:
        _ind2 = tech_breakdown.get("volume_ratio")
        if _ind2:
            import re as _re
            _m = _re.search(r"×([\d.]+)", str(_ind2))
            if _m:
                _vol_ratio = float(_m.group(1))
    except Exception:
        pass
    if _vol_ratio is not None and _vol_ratio < _min_vol:
        hard_block_reason = hard_block_reason or f"Volume too low ({_vol_ratio:.2f}x < {_min_vol}x)"

    # Re-evaluate should_buy with updated composite and hard blocks
    should_buy_score = composite >= _min_score and hard_block_reason is None
    decision = "BUY ✅" if should_buy_score else f"SKIP ❌{(' — ' + hard_block_reason) if hard_block_reason else ''}"

    logger.info(
        f"[SCORE] {ticker}: composite={composite}/100 "
        f"(tech={tech_score}, market={mkt_score}, sentiment={sent_score:.0f}) → {decision}"
    )

    return {
        "ticker": ticker,
        "composite_score": composite,
        "min_score": _min_score,
        "decision": decision,
        "should_buy": should_buy_score,
        "hard_block_reason": hard_block_reason,
        "weights": {"technicals": f"{round(w_tech*100)}%", "market": f"{round(w_mkt*100)}%", "sentiment": f"{round(w_sent*100)}%"},
        "scores": {
            "technicals": tech_score,
            "market": mkt_score,
            "sentiment": round(sent_score, 1),
            "fundamental": fund_score,
        },
        "breakdown": {
            "technicals": tech_breakdown,
            "market": mkt_breakdown,
        },
        "vix": market.get("vix"),
    }
