"""
Multi-Timeframe Analysis Engine
================================

Analyzes price action across multiple timeframes:
- 1-minute (scalping signals)
- 5-minute (short-term momentum)
- 15-minute (intraday trends)
- 1-hour (swing setups)
- 4-hour (positional trends)
- 1-day (long-term direction)

When multiple timeframes agree → high-confidence signal.
When timeframes diverge → ambiguous, avoid trade.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str
    trend: str          # "uptrend", "downtrend", "sideways"
    momentum: float     # -1 to 1 (bearish to bullish)
    rsi: float
    macd_signal: str    # "bullish", "bearish", "neutral"
    volume_strength: str  # "strong", "normal", "weak"
    support_level: Optional[float]
    resistance_level: Optional[float]
    signal_strength: float  # 0-1


@dataclass
class MultiTimeframeAnalysis:
    """Complete multi-timeframe analysis."""
    ticker: str
    current_price: float
    timeframes: dict  # {tf: TimeframeSignal}
    alignment_score: float  # 0-1, how well timeframes agree
    overall_trend: str      # Consensus trend
    high_confidence: bool   # Are 4+ TFs aligned?
    actionable_setup: Optional[str]  # If clear setup found
    recommendation: str


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TIMEFRAME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(prices: list[float], period: int = 14) -> float:
    """Calculate RSI for a price series."""
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def calculate_macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD."""
    if len(prices) < slow + signal:
        return {"macd": 0, "signal": 0, "histogram": 0, "interpretation": "neutral"}

    prices_arr = np.array(prices)

    # EMA calculation
    def ema(data, period):
        alpha = 2 / (period + 1)
        result = [data[0]]
        for price in data[1:]:
            result.append(alpha * price + (1 - alpha) * result[-1])
        return np.array(result)

    ema_fast = ema(prices_arr, fast)
    ema_slow = ema(prices_arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    current_hist = histogram[-1]

    if current_hist > 0 and current_macd > 0:
        interpretation = "bullish"
    elif current_hist < 0 and current_macd < 0:
        interpretation = "bearish"
    else:
        interpretation = "neutral"

    return {
        "macd": float(current_macd),
        "signal": float(current_signal),
        "histogram": float(current_hist),
        "interpretation": interpretation,
    }


def analyze_timeframe(prices: list[float], volumes: list[float], timeframe: str) -> TimeframeSignal:
    """Analyze a single timeframe."""
    if len(prices) < 20:
        return TimeframeSignal(
            timeframe=timeframe,
            trend="insufficient_data",
            momentum=0,
            rsi=50,
            macd_signal="neutral",
            volume_strength="normal",
            support_level=None,
            resistance_level=None,
            signal_strength=0,
        )

    # Calculate indicators
    rsi = calculate_rsi(prices)
    macd_data = calculate_macd(prices)

    # Trend: linear regression
    x = np.arange(len(prices[-20:]))
    slope = np.polyfit(x, prices[-20:], 1)[0]
    pct_slope = (slope / np.mean(prices[-20:])) * 100

    if pct_slope > 0.5:
        trend = "uptrend"
    elif pct_slope < -0.5:
        trend = "downtrend"
    else:
        trend = "sideways"

    # Momentum
    momentum = max(-1, min(1, pct_slope / 2))

    # Volume strength
    avg_volume = np.mean(volumes[-20:-1]) if len(volumes) > 20 else 0
    current_volume = volumes[-1] if volumes else 0

    if avg_volume > 0:
        volume_ratio = current_volume / avg_volume
        if volume_ratio > 1.5:
            volume_strength = "strong"
        elif volume_ratio < 0.5:
            volume_strength = "weak"
        else:
            volume_strength = "normal"
    else:
        volume_strength = "normal"

    # Support/Resistance
    recent = prices[-20:]
    resistance = max(recent)
    support = min(recent)

    # Signal strength (combined indicator)
    strength = 0
    if rsi > 40 and rsi < 65:
        strength += 0.25
    if macd_data["interpretation"] == "bullish":
        strength += 0.3
    elif macd_data["interpretation"] == "bearish":
        strength -= 0.3
    if volume_strength == "strong":
        strength += 0.2
    if trend == "uptrend":
        strength += 0.25
    elif trend == "downtrend":
        strength -= 0.25

    strength = max(0, min(1, (strength + 0.5)))  # Normalize 0-1

    return TimeframeSignal(
        timeframe=timeframe,
        trend=trend,
        momentum=momentum,
        rsi=rsi,
        macd_signal=macd_data["interpretation"],
        volume_strength=volume_strength,
        support_level=support,
        resistance_level=resistance,
        signal_strength=strength,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def calculate_alignment(timeframe_signals: dict) -> float:
    """
    Calculate how well timeframes agree.
    Returns 0-1 (1 = perfect alignment).
    """
    trends = [tf.trend for tf in timeframe_signals.values()]

    # Count occurrences
    uptrend_count = trends.count("uptrend")
    downtrend_count = trends.count("downtrend")
    sideways_count = trends.count("sideways")
    total = len(trends)

    if total == 0:
        return 0

    # Maximum alignment = all agree
    max_aligned = max(uptrend_count, downtrend_count, sideways_count)
    return max_aligned / total


def detect_actionable_setup(timeframe_signals: dict) -> Optional[str]:
    """
    Detect specific high-probability setups from MTF analysis.

    Setups:
    1. Higher Timeframe Uptrend + Lower TF Pullback = BUY OPPORTUNITY
    2. All TFs Aligned Bullish = STRONG BUY
    3. HTF Uptrend Breaking Down = WARNING / EXIT
    4. Triple Bottom on Multiple TFs = REVERSAL BUY
    """
    if not timeframe_signals:
        return None

    # Get higher and lower timeframes
    higher_tfs = ["1d", "4h", "1h"]
    lower_tfs = ["15m", "5m", "1m"]

    higher_signals = [timeframe_signals.get(tf) for tf in higher_tfs if tf in timeframe_signals]
    lower_signals = [timeframe_signals.get(tf) for tf in lower_tfs if tf in timeframe_signals]

    # Setup 1: HTF uptrend + LTF pullback (best buy setup)
    if higher_signals and lower_signals:
        htf_uptrend = all(s.trend == "uptrend" for s in higher_signals if s)
        ltf_oversold = any(s.rsi < 35 for s in lower_signals if s)

        if htf_uptrend and ltf_oversold:
            return "🟢 OPTIMAL BUY: HTF uptrend + LTF oversold pullback"

    # Setup 2: All bullish
    all_bullish = all(s.trend == "uptrend" for s in timeframe_signals.values() if s)
    if all_bullish and len(timeframe_signals) >= 3:
        return "🟢🟢 STRONG BUY: All timeframes aligned bullish"

    # Setup 3: All bearish
    all_bearish = all(s.trend == "downtrend" for s in timeframe_signals.values() if s)
    if all_bearish and len(timeframe_signals) >= 3:
        return "🔴🔴 STRONG SELL: All timeframes aligned bearish"

    # Setup 4: HTF breaking down
    if higher_signals:
        htf_breaking = any(
            s.trend == "downtrend" and s.macd_signal == "bearish"
            for s in higher_signals if s
        )
        if htf_breaking:
            return "⚠️ HTF BREAKDOWN: Consider exiting longs"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_multi_timeframe(ticker: str) -> MultiTimeframeAnalysis:
    """
    Run multi-timeframe analysis on a ticker.

    Analyzes: 1d, 4h, 1h, 15m timeframes.
    """
    try:
        import yfinance as yf

        # Define timeframes (using yfinance intervals)
        # Note: yfinance has limits on intraday data depth
        intervals = {
            "1d": ("60d", "1d"),     # 60 days of daily
            "1h": ("60d", "1h"),     # 60 days of hourly (yfinance limit)
            "15m": ("5d", "15m"),    # 5 days of 15-min
            "5m": ("5d", "5m"),      # 5 days of 5-min
        }

        timeframe_signals = {}

        for tf_name, (period, interval) in intervals.items():
            try:
                data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                if data.empty:
                    continue

                prices = [float(v) for v in data["Close"].squeeze().dropna().values]
                volumes = [float(v) for v in data["Volume"].squeeze().dropna().values]

                signal = analyze_timeframe(prices, volumes, tf_name)
                timeframe_signals[tf_name] = signal

            except Exception as e:
                logger.debug(f"Failed to analyze {ticker} on {tf_name}: {e}")

        if not timeframe_signals:
            return MultiTimeframeAnalysis(
                ticker=ticker,
                current_price=0,
                timeframes={},
                alignment_score=0,
                overall_trend="unknown",
                high_confidence=False,
                actionable_setup=None,
                recommendation="❌ Insufficient data",
            )

        # Get current price from latest data
        current_price = list(timeframe_signals.values())[0].resistance_level or 0

        # Calculate alignment
        alignment = calculate_alignment(timeframe_signals)

        # Determine overall trend
        trends = [tf.trend for tf in timeframe_signals.values()]
        uptrend_count = trends.count("uptrend")
        downtrend_count = trends.count("downtrend")

        if uptrend_count > downtrend_count + 1:
            overall_trend = "BULLISH"
        elif downtrend_count > uptrend_count + 1:
            overall_trend = "BEARISH"
        else:
            overall_trend = "NEUTRAL"

        # High confidence if 75%+ alignment
        high_confidence = alignment >= 0.75

        # Detect setups
        setup = detect_actionable_setup(timeframe_signals)

        # Generate recommendation
        if high_confidence and overall_trend == "BULLISH":
            recommendation = f"🟢 BUY - {alignment:.0%} alignment, multiple TFs bullish"
        elif high_confidence and overall_trend == "BEARISH":
            recommendation = f"🔴 SELL - {alignment:.0%} alignment, multiple TFs bearish"
        elif alignment > 0.5:
            recommendation = f"🟡 WATCH - {alignment:.0%} alignment, trend: {overall_trend}"
        else:
            recommendation = f"⚠️ AVOID - Mixed signals across TFs ({alignment:.0%})"

        return MultiTimeframeAnalysis(
            ticker=ticker,
            current_price=current_price,
            timeframes={
                tf_name: {
                    "trend": s.trend,
                    "momentum": s.momentum,
                    "rsi": s.rsi,
                    "macd": s.macd_signal,
                    "volume": s.volume_strength,
                    "signal_strength": s.signal_strength,
                }
                for tf_name, s in timeframe_signals.items()
            },
            alignment_score=alignment,
            overall_trend=overall_trend,
            high_confidence=high_confidence,
            actionable_setup=setup,
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"Multi-timeframe analysis failed for {ticker}: {e}")
        return MultiTimeframeAnalysis(
            ticker=ticker,
            current_price=0,
            timeframes={},
            alignment_score=0,
            overall_trend="ERROR",
            high_confidence=False,
            actionable_setup=None,
            recommendation=f"❌ Analysis error: {e}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONFLUENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

async def find_confluence_opportunities(tickers: list[str]) -> list[dict]:
    """
    Find tickers where multiple timeframes align (high confluence).

    Best for: identifying high-probability trade setups.
    """
    opportunities = []

    for ticker in tickers:
        try:
            analysis = await analyze_multi_timeframe(ticker)

            if analysis.high_confidence and analysis.overall_trend == "BULLISH":
                opportunities.append({
                    "ticker": ticker,
                    "alignment_score": analysis.alignment_score,
                    "trend": analysis.overall_trend,
                    "setup": analysis.actionable_setup,
                    "recommendation": analysis.recommendation,
                })

        except Exception as e:
            logger.debug(f"Confluence check failed for {ticker}: {e}")

    # Sort by alignment score
    opportunities.sort(key=lambda x: x["alignment_score"], reverse=True)

    return opportunities
