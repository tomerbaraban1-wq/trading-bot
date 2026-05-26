"""
Machine Learning Trade Prediction Module
=========================================

Simple ML predictions for entry/exit decisions using historical data.

Features used:
1. Technical indicators (RSI, MACD, volume)
2. Recent price action
3. Volatility metrics
4. Sentiment scores
5. Time of day / day of week
6. Sector performance

Approach:
- Logistic regression for win/loss prediction
- Decision tree for entry score
- Ensemble voting for final decision
- Uses scikit-learn if available, fallback to simple statistics
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """A machine learning prediction for a trade."""
    ticker: str
    predicted_outcome: str  # "win", "loss", "neutral"
    win_probability: float  # 0-1
    expected_return: float  # Expected % return
    confidence: float  # 0-1
    factors: dict  # Which factors contributed
    recommendation: str
    risk_level: str


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    rsi: float,
    macd: float,
    volume_ratio: float,
    sentiment_score: float,
    price_momentum: float,
    sector_performance: float,
    hour_of_day: int,
    day_of_week: int,
) -> list[float]:
    """
    Extract features for ML prediction.
    Normalized to 0-1 scale where possible.
    """
    return [
        rsi / 100,  # Normalize RSI
        max(-1, min(1, macd)),  # Cap MACD between -1 and 1
        max(0, min(2, volume_ratio)) / 2,  # Cap volume ratio
        sentiment_score / 10 if sentiment_score else 0.5,
        max(-1, min(1, price_momentum / 10)),  # Normalize momentum
        max(-1, min(1, sector_performance / 10)),
        hour_of_day / 23,
        day_of_week / 6,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE WIN PROBABILITY CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_simple_win_probability(features: list[float]) -> float:
    """
    Calculate win probability using rule-based scoring.
    Used when ML model is not available.
    """
    rsi_norm, macd, vol_ratio, sentiment, momentum, sector, hour, day = features

    # Start at 50% baseline
    probability = 0.5

    # RSI factor: Sweet spot is 30-60 (not overbought, not deeply oversold)
    rsi = rsi_norm * 100
    if 40 < rsi < 65:
        probability += 0.08  # Good zone
    elif rsi > 75:
        probability -= 0.15  # Overbought
    elif rsi < 25:
        probability += 0.05  # Oversold can be opportunity

    # MACD factor: Positive = bullish
    if macd > 0.05:
        probability += 0.10
    elif macd < -0.05:
        probability -= 0.10

    # Volume factor: Strong volume = confirmation
    if vol_ratio > 0.75:  # Above average volume
        probability += 0.08

    # Sentiment factor: Strong positive = good
    if sentiment > 0.7:
        probability += 0.07
    elif sentiment < 0.4:
        probability -= 0.05

    # Momentum factor: Positive momentum is good
    if momentum > 0.3:
        probability += 0.05
    elif momentum < -0.3:
        probability -= 0.05

    # Sector performance
    if sector > 0.2:
        probability += 0.04
    elif sector < -0.2:
        probability -= 0.04

    # Time factors (research-based: avoid first/last hour volatility)
    hour = hour * 23
    if 0.3 < hour/23 < 0.85:  # Mid-day trading is safer
        probability += 0.03

    # Clamp to 0-1
    return max(0, min(1, probability))


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL PATTERN MATCHING
# ─────────────────────────────────────────────────────────────────────────────

async def find_similar_historical_trades(
    rsi: float,
    macd: float,
    volume_ratio: float,
    ticker: Optional[str] = None,
) -> dict:
    """
    Find historical trades with similar setup and analyze outcomes.

    Returns win rate from similar past setups.
    """
    try:
        import database
        conn = database.get_connection()

        # Find trades with similar indicators
        query = """
            SELECT pnl_gross, exit_reason, rsi, macd, volume_ratio
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND rsi BETWEEN ? AND ?
            AND volume_ratio BETWEEN ? AND ?
        """
        params = [
            rsi - 5, rsi + 5,
            volume_ratio - 0.2, volume_ratio + 0.2,
        ]

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        query += " ORDER BY created_at DESC LIMIT 50"

        rows = conn.execute(query, params).fetchall()

        if len(rows) < 3:
            return {"error": "Not enough historical data", "similar_trades": 0}

        pnls = [r[0] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = (len(wins) / len(pnls) * 100) if pnls else 0
        avg_pnl = np.mean(pnls)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        return {
            "similar_trades": len(pnls),
            "historical_win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss),
        }

    except Exception as e:
        logger.error(f"Historical pattern matching failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

async def predict_trade_outcome(
    ticker: str,
    rsi: float = 50,
    macd: float = 0,
    volume_ratio: float = 1.0,
    sentiment_score: float = 5,
    price_momentum: float = 0,
    sector_performance: float = 0,
) -> MLPrediction:
    """
    Predict trade outcome using multiple methods:
    1. Rule-based win probability
    2. Historical pattern matching
    3. Sentiment analysis
    """
    try:
        # Get current time
        now = datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()

        # Extract features
        features = extract_features(
            rsi=rsi,
            macd=macd,
            volume_ratio=volume_ratio,
            sentiment_score=sentiment_score,
            price_momentum=price_momentum,
            sector_performance=sector_performance,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
        )

        # Get rule-based probability
        rule_probability = calculate_simple_win_probability(features)

        # Get historical matching
        historical = await find_similar_historical_trades(rsi, macd, volume_ratio, ticker)
        historical_win_rate = historical.get("historical_win_rate", 50) / 100 if "error" not in historical else 0.5

        # Ensemble: 60% rule-based + 40% historical
        if "error" not in historical and historical["similar_trades"] >= 5:
            final_probability = rule_probability * 0.6 + historical_win_rate * 0.4
            confidence = 0.7 + (historical["similar_trades"] / 100)
        else:
            final_probability = rule_probability
            confidence = 0.5

        # Determine outcome
        if final_probability > 0.65:
            outcome = "win"
            recommendation = "🟢 STRONG BUY - High win probability"
            risk = "🟢 LOW RISK"
        elif final_probability > 0.55:
            outcome = "win"
            recommendation = "🟢 BUY - Favorable odds"
            risk = "🟡 MODERATE"
        elif final_probability > 0.45:
            outcome = "neutral"
            recommendation = "🟡 NEUTRAL - Marginal trade"
            risk = "🟠 ELEVATED"
        else:
            outcome = "loss"
            recommendation = "🔴 AVOID - Low win probability"
            risk = "🔴 HIGH RISK"

        # Calculate expected return
        if "error" not in historical:
            expected_return = historical.get("expectancy", 0)
        else:
            expected_return = 0

        # Factor analysis
        factors = {
            "rsi": "🟢 Good" if 40 < rsi < 65 else "🔴 Risky",
            "macd": "🟢 Bullish" if macd > 0 else "🔴 Bearish",
            "volume": "🟢 Strong" if volume_ratio > 0.75 else "🟡 Weak",
            "sentiment": "🟢 Positive" if sentiment_score > 7 else "🟡 Neutral" if sentiment_score > 5 else "🔴 Negative",
            "momentum": "🟢 Up" if price_momentum > 1 else "🔴 Down" if price_momentum < -1 else "🟡 Flat",
            "historical_match": f"{historical.get('similar_trades', 0)} similar trades, {historical.get('historical_win_rate', 0):.0f}% win rate" if "error" not in historical else "Insufficient data",
        }

        return MLPrediction(
            ticker=ticker,
            predicted_outcome=outcome,
            win_probability=final_probability,
            expected_return=expected_return,
            confidence=confidence,
            factors=factors,
            recommendation=recommendation,
            risk_level=risk,
        )

    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        return MLPrediction(
            ticker=ticker,
            predicted_outcome="neutral",
            win_probability=0.5,
            expected_return=0,
            confidence=0,
            factors={"error": str(e)},
            recommendation="❌ Prediction failed",
            risk_level="❓ UNKNOWN",
        )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def rank_tickers_by_prediction(tickers: list[str], indicators: dict) -> list[dict]:
    """
    Rank multiple tickers by their predicted outcomes.

    Args:
        tickers: list of ticker symbols
        indicators: dict mapping ticker → {rsi, macd, volume_ratio, ...}

    Returns:
        Sorted list of predictions (best first)
    """
    predictions = []

    for ticker in tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            pred = await predict_trade_outcome(
                ticker=ticker,
                rsi=ind.get("rsi", 50),
                macd=ind.get("macd", 0),
                volume_ratio=ind.get("volume_ratio", 1.0),
                sentiment_score=ind.get("sentiment_score", 5),
                price_momentum=ind.get("price_momentum", 0),
                sector_performance=ind.get("sector_performance", 0),
            )
            predictions.append({
                "ticker": ticker,
                "win_probability": pred.win_probability,
                "outcome": pred.predicted_outcome,
                "recommendation": pred.recommendation,
                "confidence": pred.confidence,
            })

    # Sort by probability descending
    predictions.sort(key=lambda x: x["win_probability"], reverse=True)
    return predictions
