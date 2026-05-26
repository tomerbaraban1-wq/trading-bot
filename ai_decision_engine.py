"""
AI Decision Engine - The Brain of the Bot
==========================================

Combines signals from ALL modules into a unified intelligent decision:
1. Pattern Recognition (chart patterns)
2. ML Predictor (win probability)
3. Risk Engine (Sharpe, VaR, Kelly)
4. Adaptive Trader (current params)
5. Market Intelligence (volatility, sectors)
6. Continuous Learner (error patterns)
7. Sentiment (Discord community)
8. Technical Indicators (RSI, MACD)

Output: Final BUY/HOLD/SELL decision with confidence score and explanation.

This is the "brain" - all other modules feed into here.
"""

import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Final AI-driven trading decision."""
    action: str               # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence: float         # 0-1
    expected_return: float    # Expected % return
    risk_score: float         # 0-100 (lower = safer)
    position_size_pct: float  # % of capital to risk
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    holding_period_days: float
    signals: dict             # Individual signal scores
    reasoning: list[str]      # Human-readable reasoning
    warnings: list[str]       # Risk warnings
    final_explanation: str    # Single-sentence summary


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_signals(signals: dict) -> dict:
    """
    Aggregate multiple signals using weighted voting.

    Weights:
    - Pattern Recognition: 15%
    - ML Predictor: 20%
    - Technical (RSI/MACD): 15%
    - Sentiment: 10%
    - Market Intelligence: 15%
    - Risk Adjustment: 10%
    - Continuous Learning: 15%
    """
    weights = {
        "pattern": 0.15,
        "ml": 0.20,
        "technical": 0.15,
        "sentiment": 0.10,
        "market_intel": 0.15,
        "risk": 0.10,
        "learning": 0.15,
    }

    weighted_score = 0
    total_weight = 0

    for key, weight in weights.items():
        if key in signals:
            score = signals[key].get("score", 0.5)  # default neutral
            weighted_score += score * weight
            total_weight += weight

    final_score = weighted_score / total_weight if total_weight > 0 else 0.5

    return {
        "weighted_score": final_score,
        "interpretation": _interpret_score(final_score),
        "weights_used": weights,
    }


def _interpret_score(score: float) -> str:
    """Interpret aggregated score."""
    if score >= 0.75:
        return "STRONG_BUY"
    elif score >= 0.60:
        return "BUY"
    elif score >= 0.40:
        return "HOLD"
    elif score >= 0.25:
        return "SELL"
    else:
        return "STRONG_SELL"


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE DECISION
# ─────────────────────────────────────────────────────────────────────────────

async def make_trading_decision(
    ticker: str,
    current_price: float,
    rsi: float = 50,
    macd: float = 0,
    volume_ratio: float = 1.0,
    sentiment_score: float = 5,
) -> TradingDecision:
    """
    Make a comprehensive trading decision using ALL available intelligence.

    This is the main entry point for any trade decision.
    """
    try:
        signals = {}
        reasoning = []
        warnings = []

        # ── 1. PATTERN RECOGNITION ────────────────────────────────────────
        try:
            from pattern_recognition import analyze_chart_patterns
            patterns = await analyze_chart_patterns(ticker, period_days=60)

            if "error" not in patterns:
                bullish = patterns.get("overall_signal", "").startswith("🟢")
                bearish = patterns.get("overall_signal", "").startswith("🔴")
                trend = patterns.get("trend", {}).get("trend", "sideways")

                if bullish or trend in ("uptrend", "strong_uptrend"):
                    pattern_score = 0.75
                    reasoning.append(f"📊 Patterns: Bullish setup ({trend})")
                elif bearish or trend in ("downtrend", "strong_downtrend"):
                    pattern_score = 0.25
                    reasoning.append(f"📊 Patterns: Bearish setup ({trend})")
                else:
                    pattern_score = 0.5
                    reasoning.append("📊 Patterns: Neutral")

                signals["pattern"] = {"score": pattern_score}
        except Exception as e:
            logger.debug(f"Pattern recognition failed: {e}")
            signals["pattern"] = {"score": 0.5}

        # ── 2. ML PREDICTOR ────────────────────────────────────────────────
        try:
            from ml_predictor import predict_trade_outcome
            ml_pred = await predict_trade_outcome(
                ticker=ticker,
                rsi=rsi,
                macd=macd,
                volume_ratio=volume_ratio,
                sentiment_score=sentiment_score,
            )

            ml_score = ml_pred.win_probability
            signals["ml"] = {
                "score": ml_score,
                "confidence": ml_pred.confidence,
            }

            if ml_score > 0.65:
                reasoning.append(f"🤖 ML: High win probability ({ml_score:.0%})")
            elif ml_score < 0.40:
                warnings.append(f"🤖 ML: Low win probability ({ml_score:.0%})")
                reasoning.append(f"🤖 ML: Bearish prediction")

        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")
            signals["ml"] = {"score": 0.5}

        # ── 3. TECHNICAL INDICATORS ───────────────────────────────────────
        tech_score = 0.5

        # RSI scoring
        if 40 < rsi < 65:
            tech_score += 0.1  # Sweet spot
        elif rsi > 75:
            tech_score -= 0.15
            warnings.append(f"⚠️ RSI overbought ({rsi:.0f})")
        elif rsi < 25:
            tech_score += 0.05  # Oversold can be opportunity

        # MACD scoring
        if macd > 0.05:
            tech_score += 0.1
            reasoning.append("📈 MACD: Bullish momentum")
        elif macd < -0.05:
            tech_score -= 0.1
            reasoning.append("📉 MACD: Bearish momentum")

        # Volume scoring
        if volume_ratio > 1.2:
            tech_score += 0.1
            reasoning.append(f"📊 Volume: Strong ({volume_ratio:.1f}x avg)")
        elif volume_ratio < 0.5:
            tech_score -= 0.1
            warnings.append(f"⚠️ Low volume ({volume_ratio:.1f}x avg)")

        signals["technical"] = {"score": max(0, min(1, tech_score))}

        # ── 4. SENTIMENT ───────────────────────────────────────────────────
        sentiment_normalized = sentiment_score / 10  # 0-1
        signals["sentiment"] = {"score": sentiment_normalized}

        if sentiment_score >= 7:
            reasoning.append(f"💬 Sentiment: Strong bullish ({sentiment_score:.1f}/10)")
        elif sentiment_score <= 3:
            warnings.append(f"💬 Sentiment: Bearish ({sentiment_score:.1f}/10)")

        # ── 5. MARKET INTELLIGENCE ────────────────────────────────────────
        try:
            from market_intelligence import detect_volatility_regime
            vol_regime = await detect_volatility_regime()

            if "CONTRACTION" in vol_regime.regime:
                market_score = 0.65  # Favorable
                reasoning.append("🌍 Market: Low volatility - favorable")
            elif "EXPANSION" in vol_regime.regime:
                market_score = 0.35
                warnings.append("🌍 Market: High volatility - risky")
            else:
                market_score = 0.5

            signals["market_intel"] = {"score": market_score}
        except Exception as e:
            logger.debug(f"Market intelligence signal failed: {e}")
            signals["market_intel"] = {"score": 0.5}

        # ── 6. RISK ADJUSTMENT ────────────────────────────────────────────
        try:
            from risk_engine import analyze_portfolio_risk
            portfolio_risk = await analyze_portfolio_risk()

            if "error" not in portfolio_risk:
                risk_score = portfolio_risk["risk_metrics"]["risk_score"]
                # Invert: low risk score = high trading score
                trade_safety = max(0, 1 - (risk_score / 100))
                signals["risk"] = {"score": trade_safety}

                if risk_score > 60:
                    warnings.append(f"⚠️ Portfolio risk high ({risk_score:.0f}/100)")
                elif risk_score < 25:
                    reasoning.append(f"✅ Portfolio risk low ({risk_score:.0f}/100)")
        except Exception as e:
            logger.debug(f"Risk signal failed: {e}")
            signals["risk"] = {"score": 0.5}

        # ── 7. CONTINUOUS LEARNING ────────────────────────────────────────
        try:
            from continuous_learner import track_live_performance
            perf = await asyncio.to_thread(track_live_performance)

            # If we have many consecutive losses, be cautious
            if perf.consecutive_losses >= 3:
                learning_score = 0.3
                warnings.append(f"🚨 {perf.consecutive_losses} consecutive losses")
            elif perf.win_rate_today > 65:
                learning_score = 0.7
                reasoning.append(f"🎯 Today's win rate: {perf.win_rate_today:.0f}%")
            else:
                learning_score = 0.5

            signals["learning"] = {"score": learning_score}
        except Exception as e:
            logger.debug(f"Learning signal failed: {e}")
            signals["learning"] = {"score": 0.5}

        # ── AGGREGATE ALL SIGNALS ─────────────────────────────────────────
        aggregated = aggregate_signals(signals)
        final_score = aggregated["weighted_score"]
        action = aggregated["interpretation"]

        # ── POSITION SIZING (Kelly-inspired) ──────────────────────────────
        try:
            from adaptive_trader import calculate_adaptive_position_size
            # Use score to determine confidence
            base_size = 0.10  # 10% of capital base
            adjusted_size = base_size * (2 * final_score)  # Scale by confidence
            adjusted_size = max(0.02, min(0.25, adjusted_size))  # 2-25% range
        except Exception as e:
            logger.debug(f"Position sizing failed, using default: {e}")
            adjusted_size = 0.05  # Default 5%

        # ── STOP LOSS / TAKE PROFIT ───────────────────────────────────────
        if final_score > 0.5:  # Buying
            stop_loss_price = current_price * 0.98  # 2% stop
            take_profit_price = current_price * (1 + 0.05 * final_score)
        else:
            stop_loss_price = None
            take_profit_price = None

        # ── EXPECTED RETURN ───────────────────────────────────────────────
        expected_return = (final_score - 0.5) * 10  # Scale to -5% to +5%

        # ── FINAL EXPLANATION ─────────────────────────────────────────────
        if action == "STRONG_BUY":
            explanation = f"🟢 {ticker}: STRONG BUY - {len(reasoning)} bullish signals align"
        elif action == "BUY":
            explanation = f"🟢 {ticker}: BUY - Favorable risk/reward"
        elif action == "HOLD":
            explanation = f"🟡 {ticker}: HOLD - Mixed signals"
        elif action == "SELL":
            explanation = f"🔴 {ticker}: SELL - Bearish signals dominate"
        else:
            explanation = f"🔴 {ticker}: STRONG SELL - Avoid this trade"

        # ── HOLDING PERIOD ESTIMATE ───────────────────────────────────────
        # Strong signals = longer holds, weak signals = shorter
        holding_period = 3 + (abs(final_score - 0.5) * 14)  # 3-10 days

        # Calculate risk score
        risk_score = (1 - final_score) * 100  # Invert

        return TradingDecision(
            action=action,
            confidence=abs(final_score - 0.5) * 2,  # 0-1
            expected_return=expected_return,
            risk_score=risk_score,
            position_size_pct=adjusted_size * 100,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            holding_period_days=holding_period,
            signals=signals,
            reasoning=reasoning,
            warnings=warnings,
            final_explanation=explanation,
        )

    except Exception as e:
        logger.error(f"AI decision making failed: {e}")
        return TradingDecision(
            action="HOLD",
            confidence=0,
            expected_return=0,
            risk_score=50,
            position_size_pct=0,
            stop_loss_price=None,
            take_profit_price=None,
            holding_period_days=0,
            signals={},
            reasoning=[],
            warnings=[f"Error in AI decision: {e}"],
            final_explanation=f"❌ {ticker}: Decision error",
        )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH DECISION MAKING
# ─────────────────────────────────────────────────────────────────────────────

async def rank_trading_opportunities(
    tickers_with_data: list[dict]
) -> list[dict]:
    """
    Rank multiple trading opportunities by AI decision quality.

    Args:
        tickers_with_data: list of dicts with ticker info and indicators

    Returns:
        Sorted list (best opportunities first)
    """
    decisions = []

    for data in tickers_with_data:
        try:
            decision = await make_trading_decision(
                ticker=data["ticker"],
                current_price=data.get("price", 0),
                rsi=data.get("rsi", 50),
                macd=data.get("macd", 0),
                volume_ratio=data.get("volume_ratio", 1.0),
                sentiment_score=data.get("sentiment_score", 5),
            )

            decisions.append({
                "ticker": data["ticker"],
                "action": decision.action,
                "confidence": decision.confidence,
                "expected_return": decision.expected_return,
                "risk_score": decision.risk_score,
                "explanation": decision.final_explanation,
                "warnings_count": len(decision.warnings),
            })

        except Exception as e:
            logger.debug(f"Decision failed for {data.get('ticker')}: {e}")

    # Sort by confidence * expected_return (quality score)
    decisions.sort(
        key=lambda d: d["confidence"] * (d["expected_return"] + 5),
        reverse=True
    )

    return decisions
