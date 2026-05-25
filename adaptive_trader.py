"""
Adaptive Trading Engine
=======================

Dynamically adjusts trading parameters based on live performance, market conditions,
and continuous learning insights.

Features:
1. Smart Position Sizing - scales based on win rate, volatility, drawdown
2. Dynamic Thresholds - adjusts MIN_BUY_SCORE, RSI bands based on performance
3. Stop Loss/TP Optimization - scales based on volatility and recent errors
4. Time-of-Day Optimization - trades more aggressively during high-performance hours
5. Correlation-Aware Trading - avoids entering correlated tickers
6. Volatility-Adjusted Risk - scales risk based on market volatility
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

# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionSizingParams:
    """Dynamic position sizing calculation."""
    base_quantity: float          # Normal quantity
    adjusted_quantity: float      # After risk adjustment
    risk_factor: float           # 0.5 to 1.5x multiplier
    reason: str                  # Why adjusted (e.g., "low_win_rate", "high_drawdown")
    confidence_level: float      # 0-1, how confident in this trade
    recommended_stop_loss_pct: float  # Stop loss as % of entry
    recommended_tp_pct: float         # TP as % of entry


def calculate_adaptive_position_size(
    base_quantity: float,
    current_win_rate: float,
    current_drawdown: float,
    consecutive_losses: int,
    market_volatility: float,
    performance_streak: int,  # positive = wins, negative = losses
) -> PositionSizingParams:
    """
    Calculate position size based on current performance and market conditions.

    Risk scaling:
    - Win rate < 40% → reduce by 50% (be cautious)
    - Win rate 40-60% → reduce by 25%
    - Win rate 60-75% → normal
    - Win rate > 75% → increase by 25-50%
    - Consecutive losses >= 3 → reduce by 50% (cool off)
    - Drawdown > 10% → reduce by 60% (preservation mode)
    - High volatility → reduce by 25-40%
    """
    risk_factor = 1.0
    adjustments = []

    # Win rate adjustment
    if current_win_rate < 40:
        risk_factor *= 0.5
        adjustments.append("low_win_rate")
    elif current_win_rate < 60:
        risk_factor *= 0.75
        adjustments.append("moderate_win_rate")
    elif current_win_rate > 75:
        risk_factor *= 1.25 + (0.25 * min((current_win_rate - 75) / 25, 1))
        adjustments.append("high_win_rate")

    # Consecutive losses adjustment
    if consecutive_losses >= 3:
        risk_factor *= 0.5
        adjustments.append(f"{consecutive_losses}_consecutive_losses")
    elif consecutive_losses == 2:
        risk_factor *= 0.75
        adjustments.append("2_consecutive_losses")

    # Drawdown adjustment
    if current_drawdown > 0.10:
        risk_factor *= 0.4
        adjustments.append("high_drawdown")
    elif current_drawdown > 0.05:
        risk_factor *= 0.65
        adjustments.append("moderate_drawdown")

    # Volatility adjustment
    if market_volatility > 40:  # High volatility (VIX-like)
        risk_factor *= 0.75
        adjustments.append("high_volatility")
    elif market_volatility > 30:
        risk_factor *= 0.85
        adjustments.append("elevated_volatility")

    # Performance streak adjustment
    if performance_streak >= 5:
        risk_factor *= 1.15
        adjustments.append("winning_streak")
    elif performance_streak <= -4:
        risk_factor *= 0.5
        adjustments.append("losing_streak")

    # Clamp to 0.3 - 1.5x
    risk_factor = max(0.3, min(1.5, risk_factor))

    adjusted_quantity = base_quantity * risk_factor

    # Calculate recommended stops/targets based on volatility
    volatility_multiplier = 0.8 + (0.4 * min(market_volatility / 50, 1))
    recommended_stop_loss = 2.0 * volatility_multiplier  # 2% base, adjusted for volatility
    recommended_tp = 4.5 * volatility_multiplier  # 4.5% base, adjusted for volatility

    # Adjust TP/SL based on recent win rate
    if current_win_rate > 70:
        recommended_tp *= 1.2  # wider targets when winning
    if current_win_rate < 45:
        recommended_stop_loss *= 0.8  # tighter stops when losing

    return PositionSizingParams(
        base_quantity=base_quantity,
        adjusted_quantity=adjusted_quantity,
        risk_factor=risk_factor,
        reason=", ".join(adjustments) or "normal",
        confidence_level=min(1.0, current_win_rate / 70) if current_win_rate > 0 else 0.5,
        recommended_stop_loss_pct=recommended_stop_loss,
        recommended_tp_pct=recommended_tp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveThresholds:
    """Dynamically adjusted entry/exit thresholds."""
    min_buy_score: float         # MIN_BUY_SCORE adjustment
    rsi_overbought: float        # RSI threshold for overbought
    rsi_oversold: float          # RSI threshold for oversold
    volume_ratio_min: float      # Minimum volume ratio
    momentum_factor: float       # Momentum strength requirement
    sentiment_weight: float      # How much to weight sentiment (0-1)
    quality_threshold_floor: float  # Never go below this


def calculate_adaptive_thresholds(
    base_min_buy_score: float,
    current_win_rate: float,
    error_patterns: list[dict],
    recent_performance: float,  # -1 to 1, negative = losses
) -> AdaptiveThresholds:
    """
    Adjust trading thresholds based on performance.

    Logic:
    - Low win rate (< 45%) → raise MIN_BUY_SCORE (be pickier)
    - High win rate (> 65%) → lower MIN_BUY_SCORE (more aggressive)
    - Recent losses → tighten RSI bands
    - Recent wins → loosen RSI bands
    - Adjust based on error patterns
    """
    min_buy_score = base_min_buy_score

    # Win rate adjustment
    if current_win_rate < 45:
        min_buy_score += 3.0  # Be much pickier
    elif current_win_rate < 55:
        min_buy_score += 1.5
    elif current_win_rate > 65:
        min_buy_score -= 2.0  # Be more aggressive
    elif current_win_rate > 75:
        min_buy_score -= 3.0

    # Clamp to reasonable range
    min_buy_score = max(base_min_buy_score - 4, min(base_min_buy_score + 5, min_buy_score))

    # RSI adjustment based on recent performance
    rsi_overbought = 70 - (5 * recent_performance)  # Tighter when losing
    rsi_oversold = 30 + (5 * recent_performance)    # Looser when winning
    rsi_overbought = max(65, min(78, rsi_overbought))
    rsi_oversold = max(22, min(35, rsi_oversold))

    # Volume requirement adjustment
    volume_ratio_min = 0.5
    if current_win_rate < 40:
        volume_ratio_min = 0.7  # Require high volume when underperforming
    elif current_win_rate > 70:
        volume_ratio_min = 0.4  # Relax volume when winning

    # Momentum factor adjustment
    momentum_factor = 1.0
    if current_win_rate > 60:
        momentum_factor = 0.85  # Lower threshold for momentum when winning
    elif current_win_rate < 50:
        momentum_factor = 1.15  # Higher threshold when losing

    # Sentiment weight adjustment based on error patterns
    sentiment_weight = 0.6
    sentiment_error_count = len([p for p in error_patterns if "sentiment" in p.get("type", "").lower()])
    if sentiment_error_count >= 2:
        sentiment_weight = 0.3  # Don't trust sentiment if it's causing losses
    elif sentiment_error_count == 0 and current_win_rate > 65:
        sentiment_weight = 0.8  # Trust sentiment more when it's working

    return AdaptiveThresholds(
        min_buy_score=min_buy_score,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        volume_ratio_min=volume_ratio_min,
        momentum_factor=momentum_factor,
        sentiment_weight=sentiment_weight,
        quality_threshold_floor=base_min_buy_score - 4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TIME-OF-DAY OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimeOptimizationData:
    """Performance by time of day."""
    hour: int  # 0-23 (EST)
    win_rate: float
    avg_return: float
    trade_count: int
    aggressiveness_factor: float  # 0.5 to 1.5x


def analyze_performance_by_hour() -> dict[int, TimeOptimizationData]:
    """
    Analyze win rates and returns by hour of day.
    Returns aggressiveness factor for each hour.
    """
    try:
        import database
        conn = database.get_connection()

        hourly_stats = {}
        rows = conn.execute("""
            SELECT
                CAST(strftime('%H', exit_time, 'localtime') AS INTEGER) as hour,
                COUNT(*) as trade_count,
                SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_gross) as avg_return
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= datetime('now', '-30 days')
            GROUP BY hour
        """).fetchall()

        for hour, count, wins, avg_ret in rows:
            if count < 3:  # Need minimum sample size
                continue

            win_rate = (wins / count * 100) if count > 0 else 0

            # Aggressiveness factor: how much to scale position size this hour
            if win_rate > 70:
                agg_factor = 1.3  # Very aggressive
            elif win_rate > 60:
                agg_factor = 1.15  # Aggressive
            elif win_rate > 50:
                agg_factor = 1.0   # Normal
            elif win_rate > 40:
                agg_factor = 0.8   # Conservative
            else:
                agg_factor = 0.5   # Very conservative

            hourly_stats[hour] = TimeOptimizationData(
                hour=hour,
                win_rate=win_rate,
                avg_return=avg_ret or 0,
                trade_count=count,
                aggressiveness_factor=agg_factor,
            )

        logger.info(f"[TIME OPTIMIZATION] Analyzed {len(hourly_stats)} hours of trading")
        return hourly_stats

    except Exception as e:
        logger.error(f"Time optimization analysis failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def calculate_ticker_correlation(ticker1: str, ticker2: str, days: int = 30) -> float:
    """
    Calculate 30-day price correlation between two tickers.
    Returns correlation coefficient -1 to 1.

    If correlation > 0.8, tickers move together - avoid trading both simultaneously.
    """
    try:
        import yfinance as yf
        import numpy as np

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        data1 = yf.download(ticker1, start=start, end=end, progress=False)["Close"]
        data2 = yf.download(ticker2, start=start, end=end, progress=False)["Close"]

        # Calculate returns
        returns1 = data1.pct_change().dropna()
        returns2 = data2.pct_change().dropna()

        # Align to common dates
        common_idx = returns1.index.intersection(returns2.index)
        if len(common_idx) < 5:
            return 0.0

        correlation = np.corrcoef(returns1[common_idx], returns2[common_idx])[0, 1]
        return float(np.nan_to_num(correlation))

    except Exception as e:
        logger.debug(f"Correlation calc failed for {ticker1}/{ticker2}: {e}")
        return 0.0


async def get_position_correlation_risk(current_position_tickers: list[str]) -> dict:
    """
    Check if any current positions are highly correlated.
    Returns {"correlated": bool, "pairs": [...], "recommendation": "..."}
    """
    if len(current_position_tickers) < 2:
        return {"correlated": False, "pairs": []}

    correlated_pairs = []
    for i in range(len(current_position_tickers)):
        for j in range(i + 1, len(current_position_tickers)):
            ticker1, ticker2 = current_position_tickers[i], current_position_tickers[j]
            corr = await asyncio.to_thread(calculate_ticker_correlation, ticker1, ticker2)

            if corr > 0.8:
                correlated_pairs.append({
                    "ticker1": ticker1,
                    "ticker2": ticker2,
                    "correlation": corr,
                    "risk": "HIGH" if corr > 0.9 else "MEDIUM"
                })

    return {
        "correlated": len(correlated_pairs) > 0,
        "pairs": correlated_pairs,
        "recommendation": (
            f"⚠️ {len(correlated_pairs)} pairs highly correlated — consider tightening stops or reducing position size"
            if correlated_pairs else
            "✅ No high correlations detected"
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY-ADJUSTED RISK
# ─────────────────────────────────────────────────────────────────────────────

async def get_market_volatility_index() -> float:
    """
    Estimate market volatility using VIX or broad market metrics.
    Returns volatility score 0-100 (similar to VIX).

    Uses S&P 500 short-term volatility as proxy.
    """
    try:
        import yfinance as yf
        import numpy as np

        # Get SPY (S&P 500 ETF) intraday data
        spy = yf.download("SPY", period="5d", interval="1h", progress=False)
        if spy.empty:
            return 20.0  # Default neutral volatility

        # Calculate intraday returns
        returns = spy["Close"].pct_change().dropna()

        # Volatility = std dev of returns * sqrt(252*6.5) to annualize hourly data
        volatility = float(np.std(returns) * np.sqrt(252 * 6.5) * 100)

        # Scale to VIX-like scale (typically 10-80)
        volatility = max(10, min(80, volatility))

        logger.debug(f"Current market volatility estimate: {volatility:.1f}")
        return volatility

    except Exception as e:
        logger.debug(f"Volatility calculation failed: {e}")
        return 20.0  # Safe default


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

async def get_adaptive_trading_params(
    base_quantity: float,
    base_min_buy_score: float,
    base_stop_loss_pct: float,
    base_take_profit_pct: float,
) -> dict:
    """
    Get all adaptive trading parameters for current conditions.

    Returns comprehensive trading configuration adjusted for:
    - Current performance (win rate, drawdown)
    - Market conditions (volatility, time of day)
    - Position correlation risks
    - Error patterns from continuous learning
    """
    try:
        from continuous_learner import learn_error_patterns, track_live_performance

        # Get current performance
        perf = await asyncio.to_thread(track_live_performance)
        errors = await asyncio.to_thread(learn_error_patterns)

        # Get market conditions
        market_vol = await get_market_volatility_index()
        hourly_stats = await asyncio.to_thread(analyze_performance_by_hour)

        # Current hour aggressiveness
        from datetime import datetime
        current_hour = datetime.now().hour
        time_factor = hourly_stats.get(current_hour, TimeOptimizationData(
            hour=current_hour, win_rate=50, avg_return=0, trade_count=0, aggressiveness_factor=1.0
        )).aggressiveness_factor

        # Calculate position sizing
        position_params = calculate_adaptive_position_size(
            base_quantity=base_quantity,
            current_win_rate=perf.win_rate_today,
            current_drawdown=perf.current_drawdown,
            consecutive_losses=perf.consecutive_losses,
            market_volatility=market_vol,
            performance_streak=perf.total_trades_today if perf.win_rate_today > 50 else -perf.total_trades_today,
        )

        # Apply time-of-day adjustment
        position_params.adjusted_quantity *= time_factor

        # Calculate adaptive thresholds
        recent_perf = (perf.win_rate_today - 50) / 50 if perf.total_trades_today > 0 else 0
        thresholds = calculate_adaptive_thresholds(
            base_min_buy_score=base_min_buy_score,
            current_win_rate=perf.win_rate_today,
            error_patterns=[{"type": p.error_type} for p in errors[:5]],
            recent_performance=recent_perf,
        )

        # Adjust stops/targets for volatility
        adjusted_stop_loss = base_stop_loss_pct * (position_params.recommended_stop_loss_pct / 2.0)
        adjusted_take_profit = base_take_profit_pct * (position_params.recommended_tp_pct / 4.5)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "position_sizing": {
                "base_quantity": position_params.base_quantity,
                "adjusted_quantity": position_params.adjusted_quantity,
                "risk_factor": position_params.risk_factor,
                "time_of_day_factor": time_factor,
                "confidence_level": position_params.confidence_level,
                "reason": position_params.reason,
            },
            "thresholds": {
                "min_buy_score": thresholds.min_buy_score,
                "rsi_overbought": thresholds.rsi_overbought,
                "rsi_oversold": thresholds.rsi_oversold,
                "volume_ratio_min": thresholds.volume_ratio_min,
                "sentiment_weight": thresholds.sentiment_weight,
            },
            "stop_loss_tp": {
                "stop_loss_pct": adjusted_stop_loss,
                "take_profit_pct": adjusted_take_profit,
                "recommended_stop_loss": position_params.recommended_stop_loss_pct,
                "recommended_tp": position_params.recommended_tp_pct,
            },
            "market_conditions": {
                "volatility": market_vol,
                "volatility_level": (
                    "🔴 EXTREME" if market_vol > 60 else
                    "🟠 HIGH" if market_vol > 40 else
                    "🟡 ELEVATED" if market_vol > 30 else
                    "🟢 NORMAL"
                ),
            },
            "performance": {
                "win_rate": perf.win_rate_today,
                "drawdown": perf.current_drawdown,
                "consecutive_losses": perf.consecutive_losses,
                "trades_today": perf.total_trades_today,
            },
        }

    except Exception as e:
        logger.error(f"Failed to calculate adaptive params: {e}")
        return {"error": str(e)}
