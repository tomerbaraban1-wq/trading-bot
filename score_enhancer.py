"""
Score Enhancer
==============

Wraps the existing composite_score and adds bonus/penalty points using:
1. ML win probability (ml_predictor)
2. Chart pattern signals (pattern_recognition)
3. Multi-timeframe alignment (multi_timeframe)
4. News sentiment (news_intelligence)
5. Anomaly detection (anomaly_detector)
6. AI decision engine (ai_decision_engine)

This module enhances the existing scoring without breaking it.
The composite_score stays the primary driver; AI signals add ±15 points.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def enhance_score(
    ticker: str,
    base_score: float,
    rsi: float = 50.0,
    macd: float = 0.0,
    volume_ratio: float = 1.0,
    sentiment_score: float = 5.0,
) -> dict:
    """
    Takes the existing composite_score and enhances it with AI/ML signals.

    Returns:
        {
          "original_score": float,
          "enhanced_score": float,
          "adjustment": float,      # total ± points added
          "signals": {...},         # breakdown of each signal
          "confidence": float,      # 0-1, how confident the enhancement is
          "skip_trade": bool,       # True if news/anomaly says skip
          "skip_reason": str,
        }
    """
    signals = {}
    adjustment = 0.0
    skip_trade = False
    skip_reason = ""
    confidence_parts = []

    # ── 1. ML Win Probability ──────────────────────────────────────────────
    try:
        from ml_predictor import predict_trade_outcome
        ml_pred = await asyncio.wait_for(
            predict_trade_outcome(
                ticker=ticker,
                rsi=rsi,
                macd=macd,
                volume_ratio=volume_ratio,
                sentiment_score=sentiment_score,
            ),
            timeout=15,
        )

        win_prob = ml_pred.win_probability
        # ML: ±8 points
        if win_prob >= 0.70:
            ml_adj = +8.0
        elif win_prob >= 0.60:
            ml_adj = +4.0
        elif win_prob >= 0.50:
            ml_adj = 0.0
        elif win_prob >= 0.40:
            ml_adj = -4.0
        else:
            ml_adj = -8.0

        adjustment += ml_adj
        signals["ml"] = {
            "win_probability": win_prob,
            "adjustment": ml_adj,
            "confidence": ml_pred.confidence,
        }
        confidence_parts.append(ml_pred.confidence)

    except Exception as e:
        logger.debug(f"ML enhancement failed for {ticker}: {e}")
        signals["ml"] = {"error": str(e)}

    # ── 2. Chart Patterns ──────────────────────────────────────────────────
    try:
        from pattern_recognition import analyze_chart_patterns
        patterns = await asyncio.wait_for(
            analyze_chart_patterns(ticker, period_days=60),
            timeout=20,
        )

        if "error" not in patterns:
            overall = patterns.get("overall_signal", "")
            breakout = patterns.get("breakout")

            pat_adj = 0.0
            if "🟢🟢" in overall:   # STRONG BULLISH
                pat_adj = +7.0
            elif "🟢" in overall:   # BULLISH
                pat_adj = +4.0
            elif "🔴🔴" in overall:  # STRONG BEARISH — skip!
                pat_adj = -8.0
                skip_trade = True
                skip_reason = f"Pattern: {overall}"
            elif "🔴" in overall:   # BEARISH
                pat_adj = -4.0

            # Bonus for confirmed breakout with volume
            if breakout and breakout.get("type") == "bullish_breakout":
                if breakout.get("volume_confirmed"):
                    pat_adj += 5.0

            adjustment += pat_adj
            signals["patterns"] = {
                "overall": overall,
                "adjustment": pat_adj,
                "breakout": breakout,
                "pattern_count": len(patterns.get("patterns", [])),
            }
            confidence_parts.append(0.6)

    except Exception as e:
        logger.debug(f"Pattern enhancement failed for {ticker}: {e}")
        signals["patterns"] = {"error": str(e)}

    # ── 3. Multi-Timeframe Alignment ───────────────────────────────────────
    try:
        from multi_timeframe import analyze_multi_timeframe
        mtf = await asyncio.wait_for(
            analyze_multi_timeframe(ticker),
            timeout=30,
        )

        mtf_adj = 0.0
        if mtf.high_confidence:
            if mtf.overall_trend == "BULLISH":
                mtf_adj = +6.0
            elif mtf.overall_trend == "BEARISH":
                mtf_adj = -7.0
                skip_trade = True
                skip_reason = f"MTF: All timeframes bearish (alignment={mtf.alignment_score:.0%})"
        elif mtf.alignment_score > 0.6:
            if mtf.overall_trend == "BULLISH":
                mtf_adj = +3.0
            elif mtf.overall_trend == "BEARISH":
                mtf_adj = -3.0

        adjustment += mtf_adj
        signals["multi_timeframe"] = {
            "trend": mtf.overall_trend,
            "alignment": mtf.alignment_score,
            "high_confidence": mtf.high_confidence,
            "setup": mtf.actionable_setup,
            "adjustment": mtf_adj,
        }
        confidence_parts.append(mtf.alignment_score)

    except Exception as e:
        logger.debug(f"MTF enhancement failed for {ticker}: {e}")
        signals["multi_timeframe"] = {"error": str(e)}

    # ── 4. News Sentiment Check ────────────────────────────────────────────
    try:
        from news_intelligence import detect_catalysts
        catalyst_data = await asyncio.wait_for(
            detect_catalysts(ticker),
            timeout=12,
        )

        if "error" not in catalyst_data:
            news_sentiment = catalyst_data.get("avg_sentiment_24h", 0)
            has_breaking = catalyst_data.get("has_breaking_news", False)
            has_catalyst = catalyst_data.get("has_catalysts", False)

            news_adj = 0.0

            # Strong positive news
            if news_sentiment > 0.5:
                news_adj += 4.0
            elif news_sentiment > 0.3:
                news_adj += 2.0
            # Strong negative news — consider skipping
            elif news_sentiment < -0.5:
                news_adj -= 5.0
                if has_breaking:
                    skip_trade = True
                    skip_reason = f"Breaking bearish news (sentiment={news_sentiment:.2f})"
            elif news_sentiment < -0.3:
                news_adj -= 2.0

            adjustment += news_adj
            signals["news"] = {
                "sentiment_24h": news_sentiment,
                "has_breaking": has_breaking,
                "has_catalyst": has_catalyst,
                "adjustment": news_adj,
            }
            confidence_parts.append(0.5)

    except Exception as e:
        logger.debug(f"News enhancement failed for {ticker}: {e}")
        signals["news"] = {"error": str(e)}

    # ── 5. Anomaly Check ───────────────────────────────────────────────────
    try:
        from anomaly_detector import detect_price_anomaly, detect_volume_anomaly

        price_anomaly = await asyncio.wait_for(
            detect_price_anomaly(ticker, threshold=3.5),
            timeout=15,
        )

        volume_anomaly = await asyncio.wait_for(
            detect_volume_anomaly(ticker, threshold=3.0),
            timeout=15,
        )

        anom_adj = 0.0

        # Unusual downward price movement during scan = risk
        if price_anomaly:
            if price_anomaly.z_score < -3.5:
                anom_adj -= 6.0  # Extreme drop — be cautious
            elif price_anomaly.z_score > 3.5:
                anom_adj -= 2.0  # Extreme up move may mean overbought

        # High volume on a stock = more reliable signal either way
        if volume_anomaly and volume_anomaly.z_score > 3.0:
            # High volume confirms signals — amplify the trend
            if base_score + adjustment > 60:
                anom_adj += 3.0   # Good sign + high volume = stronger
            else:
                anom_adj -= 3.0   # Bad sign + high volume = stronger

        adjustment += anom_adj
        signals["anomaly"] = {
            "price_anomaly": price_anomaly.severity if price_anomaly else None,
            "price_z_score": price_anomaly.z_score if price_anomaly else 0,
            "volume_anomaly": volume_anomaly.severity if volume_anomaly else None,
            "adjustment": anom_adj,
        }

    except Exception as e:
        logger.debug(f"Anomaly enhancement failed for {ticker}: {e}")
        signals["anomaly"] = {"error": str(e)}

    # ── FINAL CALCULATION ──────────────────────────────────────────────────
    # Cap total adjustment at ±15 points
    adjustment = max(-15.0, min(15.0, adjustment))

    enhanced_score = round(base_score + adjustment, 1)
    enhanced_score = max(0.0, min(100.0, enhanced_score))

    # Overall confidence
    confidence = float(sum(confidence_parts) / len(confidence_parts)) if confidence_parts else 0.5

    logger.debug(
        f"[SCORE ENHANCER] {ticker}: {base_score:.1f} → {enhanced_score:.1f} "
        f"(adj={adjustment:+.1f}, skip={skip_trade})"
    )

    return {
        "original_score": base_score,
        "enhanced_score": enhanced_score,
        "adjustment": adjustment,
        "signals": signals,
        "confidence": confidence,
        "skip_trade": skip_trade,
        "skip_reason": skip_reason,
    }
