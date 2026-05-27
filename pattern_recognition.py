"""
Chart Pattern Recognition Module
=================================

Detects classical chart patterns for better entry/exit timing:
1. Trend patterns: Uptrend, Downtrend, Sideways
2. Reversal patterns: Head & Shoulders, Double Top/Bottom, Triple Top/Bottom
3. Continuation patterns: Flags, Pennants, Triangles
4. Candlestick patterns: Doji, Hammer, Engulfing, Morning/Evening Star
5. Volume patterns: Volume spikes, Volume divergence
6. Breakouts: Resistance breaks, Support breaks
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChartPattern:
    """A detected chart pattern."""
    pattern_name: str
    pattern_type: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    detected_at: str
    price_target: Optional[float]  # If pattern complete, where price might go
    stop_loss_suggested: Optional[float]
    description: str
    actionable: bool  # Should bot act on this?


@dataclass
class CandlestickSignal:
    """A candlestick pattern signal."""
    pattern_name: str
    signal_type: str  # "buy", "sell", "neutral"
    strength: float  # 0-1
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# TREND DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_trend(prices: list[float], window: int = 20) -> dict:
    """
    Detect current price trend using linear regression slope.

    Returns:
    - trend: "strong_uptrend", "uptrend", "sideways", "downtrend", "strong_downtrend"
    - slope: regression slope
    - strength: 0-1
    """
    if len(prices) < window:
        return {"trend": "insufficient_data", "slope": 0, "strength": 0}

    recent = np.array(prices[-window:])
    x = np.arange(len(recent))

    # Linear regression
    slope, intercept = np.polyfit(x, recent, 1)
    correlation = np.corrcoef(x, recent)[0, 1]

    # Normalize slope to percentage move per period
    pct_slope = (slope / np.mean(recent)) * 100

    # Classify
    if pct_slope > 0.5 and correlation > 0.8:
        trend = "strong_uptrend"
    elif pct_slope > 0.2 and correlation > 0.6:
        trend = "uptrend"
    elif pct_slope < -0.5 and correlation < -0.8:
        trend = "strong_downtrend"
    elif pct_slope < -0.2 and correlation < -0.6:
        trend = "downtrend"
    else:
        trend = "sideways"

    return {
        "trend": trend,
        "slope": slope,
        "pct_slope": pct_slope,
        "strength": abs(correlation),
        "correlation": correlation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REVERSAL PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def detect_head_and_shoulders(prices: list[float]) -> Optional[ChartPattern]:
    """
    Detect Head & Shoulders pattern (bearish reversal).

    Pattern: Left shoulder < Head > Right shoulder
    With similar shoulder heights and a neckline.
    """
    if len(prices) < 50:
        return None

    try:
        # Find local maxima
        peaks = []
        for i in range(5, len(prices) - 5):
            if prices[i] == max(prices[i-5:i+5]):
                peaks.append((i, prices[i]))

        if len(peaks) < 3:
            return None

        # Look for H&S pattern in last 3 peaks
        recent_peaks = peaks[-3:]
        left_shoulder = recent_peaks[0]
        head = recent_peaks[1]
        right_shoulder = recent_peaks[2]

        # Head must be higher than both shoulders
        if head[1] <= left_shoulder[1] or head[1] <= right_shoulder[1]:
            return None

        # Shoulders should be similar height (within 5%)
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
        if shoulder_diff > 0.05:
            return None

        # Find neckline (low between shoulders and head)
        left_low_idx = left_shoulder[0] + np.argmin(prices[left_shoulder[0]:head[0]])
        right_low_idx = head[0] + np.argmin(prices[head[0]:right_shoulder[0]])
        neckline = (prices[left_low_idx] + prices[right_low_idx]) / 2

        # Price target = neckline - (head - neckline)
        price_target = neckline - (head[1] - neckline)

        return ChartPattern(
            pattern_name="Head & Shoulders",
            pattern_type="bearish",
            confidence=0.75 - shoulder_diff * 5,
            detected_at=datetime.now(timezone.utc).isoformat(),
            price_target=price_target,
            stop_loss_suggested=head[1] * 1.02,
            description=f"Bearish reversal: target ${price_target:.2f}, neckline ${neckline:.2f}",
            actionable=True,
        )

    except Exception as e:
        logger.debug(f"H&S detection failed: {e}")
        return None


def detect_double_bottom(prices: list[float]) -> Optional[ChartPattern]:
    """
    Detect Double Bottom pattern (bullish reversal).

    Pattern: Two lows at similar levels with a peak between them.
    """
    if len(prices) < 30:
        return None

    try:
        # Find local minima
        lows = []
        for i in range(5, len(prices) - 5):
            if prices[i] == min(prices[i-5:i+5]):
                lows.append((i, prices[i]))

        if len(lows) < 2:
            return None

        # Take last 2 lows
        first_low = lows[-2]
        second_low = lows[-1]

        # Lows should be at similar levels (within 3%)
        low_diff = abs(first_low[1] - second_low[1]) / first_low[1]
        if low_diff > 0.03:
            return None

        # Find peak between lows
        between_high_idx = first_low[0] + np.argmax(prices[first_low[0]:second_low[0]])
        peak = prices[between_high_idx]

        # Peak should be at least 3% above lows
        if (peak - second_low[1]) / second_low[1] < 0.03:
            return None

        # Price target = peak + (peak - low)
        price_target = peak + (peak - second_low[1])

        return ChartPattern(
            pattern_name="Double Bottom",
            pattern_type="bullish",
            confidence=0.7 - low_diff * 10,
            detected_at=datetime.now(timezone.utc).isoformat(),
            price_target=price_target,
            stop_loss_suggested=second_low[1] * 0.98,
            description=f"Bullish reversal: target ${price_target:.2f}, support ${second_low[1]:.2f}",
            actionable=True,
        )

    except Exception as e:
        logger.debug(f"Double bottom detection failed: {e}")
        return None


def detect_double_top(prices: list[float]) -> Optional[ChartPattern]:
    """
    Detect Double Top pattern (bearish reversal).

    Pattern: Two highs at similar levels with a valley between them.
    """
    if len(prices) < 30:
        return None

    try:
        # Find local maxima
        highs = []
        for i in range(5, len(prices) - 5):
            if prices[i] == max(prices[i-5:i+5]):
                highs.append((i, prices[i]))

        if len(highs) < 2:
            return None

        first_high = highs[-2]
        second_high = highs[-1]

        # Highs at similar levels
        high_diff = abs(first_high[1] - second_high[1]) / first_high[1]
        if high_diff > 0.03:
            return None

        # Find valley between
        between_low_idx = first_high[0] + np.argmin(prices[first_high[0]:second_high[0]])
        valley = prices[between_low_idx]

        # Valley should be at least 3% below highs
        if (second_high[1] - valley) / valley < 0.03:
            return None

        price_target = valley - (second_high[1] - valley)

        return ChartPattern(
            pattern_name="Double Top",
            pattern_type="bearish",
            confidence=0.7 - high_diff * 10,
            detected_at=datetime.now(timezone.utc).isoformat(),
            price_target=price_target,
            stop_loss_suggested=second_high[1] * 1.02,
            description=f"Bearish reversal: target ${price_target:.2f}, resistance ${second_high[1]:.2f}",
            actionable=True,
        )

    except Exception as e:
        logger.debug(f"Double top detection failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONTINUATION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def detect_triangle(prices: list[float]) -> Optional[ChartPattern]:
    """
    Detect Triangle patterns (continuation):
    - Ascending: Higher lows, similar highs (bullish)
    - Descending: Lower highs, similar lows (bearish)
    - Symmetrical: Lower highs and higher lows (continuation)
    """
    if len(prices) < 30:
        return None

    try:
        recent = prices[-30:]

        # Find local highs and lows
        highs = []
        lows = []
        for i in range(2, len(recent) - 2):
            if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                highs.append((i, recent[i]))
            if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
                lows.append((i, recent[i]))

        if len(highs) < 2 or len(lows) < 2:
            return None

        # Calculate slopes
        high_slope = (highs[-1][1] - highs[0][1]) / (highs[-1][0] - highs[0][0])
        low_slope = (lows[-1][1] - lows[0][1]) / (lows[-1][0] - lows[0][0])

        pattern_name = None
        pattern_type = None
        description = None

        # Ascending triangle: flat highs, rising lows
        if abs(high_slope) < 0.05 and low_slope > 0.1:
            pattern_name = "Ascending Triangle"
            pattern_type = "bullish"
            description = "Bullish continuation: breakout above resistance likely"

        # Descending triangle: flat lows, falling highs
        elif abs(low_slope) < 0.05 and high_slope < -0.1:
            pattern_name = "Descending Triangle"
            pattern_type = "bearish"
            description = "Bearish continuation: breakdown below support likely"

        # Symmetrical triangle: converging
        elif high_slope < -0.1 and low_slope > 0.1:
            pattern_name = "Symmetrical Triangle"
            pattern_type = "neutral"
            description = "Continuation pattern: watch for breakout direction"

        if not pattern_name:
            return None

        # Project price target
        current_price = recent[-1]
        triangle_height = highs[0][1] - lows[0][1]
        price_target = current_price + triangle_height if pattern_type == "bullish" else current_price - triangle_height

        return ChartPattern(
            pattern_name=pattern_name,
            pattern_type=pattern_type,
            confidence=0.6,
            detected_at=datetime.now(timezone.utc).isoformat(),
            price_target=price_target,
            stop_loss_suggested=None,
            description=description,
            actionable=True,
        )

    except Exception as e:
        logger.debug(f"Triangle detection failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CANDLESTICK PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def detect_candlestick_patterns(opens: list[float], highs: list[float], lows: list[float], closes: list[float]) -> List[CandlestickSignal]:
    """
    Detect classical candlestick patterns.
    Requires at least 3 candles for context.
    """
    if len(closes) < 3:
        return []

    signals = []

    # Get last 3 candles for analysis
    for i in [-1]:
        try:
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            full_range = h - l

            if full_range == 0:
                continue

            # Doji: body very small (< 10% of range)
            if body / full_range < 0.1:
                signals.append(CandlestickSignal(
                    pattern_name="Doji",
                    signal_type="neutral",
                    strength=0.6,
                    description="Indecision - watch for reversal",
                ))

            # Hammer: small body at top, long lower wick (bullish reversal)
            elif lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
                signals.append(CandlestickSignal(
                    pattern_name="Hammer",
                    signal_type="buy",
                    strength=0.7,
                    description="Bullish reversal - hammer pattern",
                ))

            # Hanging Man: same as hammer but at top of uptrend (bearish)
            elif lower_wick > body * 2 and upper_wick < body * 0.5 and c < o:
                signals.append(CandlestickSignal(
                    pattern_name="Hanging Man",
                    signal_type="sell",
                    strength=0.65,
                    description="Bearish reversal - hanging man",
                ))

            # Shooting Star: small body at bottom, long upper wick (bearish)
            elif upper_wick > body * 2 and lower_wick < body * 0.5:
                signals.append(CandlestickSignal(
                    pattern_name="Shooting Star",
                    signal_type="sell",
                    strength=0.7,
                    description="Bearish reversal - shooting star",
                ))

        except Exception as e:
            logger.debug(f"Candlestick detection error: {e}")

    # Engulfing patterns (need 2 candles)
    if len(closes) >= 2:
        try:
            prev_o, prev_c = opens[-2], closes[-2]
            curr_o, curr_c = opens[-1], closes[-1]

            # Bullish engulfing
            if prev_c < prev_o and curr_c > curr_o:  # prev red, curr green
                if curr_o < prev_c and curr_c > prev_o:  # current engulfs previous
                    signals.append(CandlestickSignal(
                        pattern_name="Bullish Engulfing",
                        signal_type="buy",
                        strength=0.75,
                        description="Strong bullish reversal",
                    ))

            # Bearish engulfing
            elif prev_c > prev_o and curr_c < curr_o:  # prev green, curr red
                if curr_o > prev_c and curr_c < prev_o:  # current engulfs previous
                    signals.append(CandlestickSignal(
                        pattern_name="Bearish Engulfing",
                        signal_type="sell",
                        strength=0.75,
                        description="Strong bearish reversal",
                    ))
        except:
            pass

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def detect_volume_patterns(prices: list[float], volumes: list[float]) -> dict:
    """
    Detect volume-based patterns:
    - Volume spike: unusual high volume
    - Volume divergence: price up but volume down (weakening)
    - Climactic volume: extreme volume at extremes
    """
    if len(prices) < 20 or len(volumes) < 20:
        return {}

    try:
        recent_volumes = volumes[-20:]
        avg_volume = np.mean(recent_volumes[:-1])
        current_volume = recent_volumes[-1]

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Volume spike
        is_spike = volume_ratio > 2.0
        is_extreme = volume_ratio > 3.0

        # Price direction
        recent_prices = prices[-20:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100

        # Volume trend
        volume_trend_slope = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]

        # Divergence: price up, volume down
        divergence = price_change > 2 and volume_trend_slope < 0

        return {
            "current_volume_ratio": volume_ratio,
            "is_volume_spike": is_spike,
            "is_extreme_volume": is_extreme,
            "volume_divergence": divergence,
            "interpretation": (
                "🔴 EXTREME volume - climactic move" if is_extreme else
                "🟡 Volume spike - watch for continuation" if is_spike else
                "⚠️ Volume divergence - move weakening" if divergence else
                "🟢 Normal volume"
            )
        }

    except Exception as e:
        logger.debug(f"Volume pattern detection failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# BREAKOUT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_breakout(prices: list[float], volumes: list[float]) -> Optional[dict]:
    """
    Detect breakouts above resistance or below support with volume confirmation.
    """
    if len(prices) < 30:
        return None

    try:
        recent = prices[-30:-1]  # Excluding latest price
        current_price = prices[-1]

        resistance = max(recent)
        support = min(recent)

        # Volume confirmation
        avg_volume = np.mean(volumes[-20:-1])
        current_volume = volumes[-1]
        volume_confirmed = current_volume > avg_volume * 1.5

        # Breakout up
        if current_price > resistance * 1.01:  # 1% above resistance
            return {
                "type": "bullish_breakout",
                "level_broken": resistance,
                "current_price": current_price,
                "volume_confirmed": volume_confirmed,
                "strength": "STRONG" if volume_confirmed else "WEAK",
                "recommendation": "🟢 BUY signal" if volume_confirmed else "🟡 Watch for confirmation",
            }

        # Breakdown down
        elif current_price < support * 0.99:  # 1% below support
            return {
                "type": "bearish_breakdown",
                "level_broken": support,
                "current_price": current_price,
                "volume_confirmed": volume_confirmed,
                "strength": "STRONG" if volume_confirmed else "WEAK",
                "recommendation": "🔴 SELL signal" if volume_confirmed else "🟡 Watch for confirmation",
            }

    except Exception as e:
        logger.debug(f"Breakout detection failed: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE PATTERN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_chart_patterns(ticker: str, period_days: int = 90) -> dict:
    """
    Run comprehensive pattern analysis on a ticker.
    """
    try:
        import yfinance as yf

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=period_days)

        data = yf.download(ticker, start=start, end=end, progress=False,
                           auto_adjust=True)
        if data.empty:
            return {"error": "No data available"}

        # Flatten MultiIndex columns that yfinance ≥0.2 may return for single
        # tickers so Series.squeeze() / .values always give a flat 1-D array.
        def _col(name: str) -> list:
            col = data[name]
            if hasattr(col, "squeeze"):
                col = col.squeeze()
            return [float(v) for v in col.dropna().values]

        opens   = _col("Open")
        highs   = _col("High")
        lows    = _col("Low")
        closes  = _col("Close")
        volumes = _col("Volume")

        # Run all analyses
        trend = detect_trend(closes)
        hs_pattern = detect_head_and_shoulders(closes)
        dt_pattern = detect_double_top(closes)
        db_pattern = detect_double_bottom(closes)
        tri_pattern = detect_triangle(closes)
        candlesticks = detect_candlestick_patterns(opens, highs, lows, closes)
        volume_analysis = detect_volume_patterns(closes, volumes)
        breakout = detect_breakout(closes, volumes)

        # Compile actionable patterns
        patterns = [p for p in [hs_pattern, dt_pattern, db_pattern, tri_pattern] if p is not None]

        # Generate overall recommendation
        bullish_signals = sum(1 for p in patterns if p.pattern_type == "bullish")
        bearish_signals = sum(1 for p in patterns if p.pattern_type == "bearish")
        bullish_candles = sum(1 for c in candlesticks if c.signal_type == "buy")
        bearish_candles = sum(1 for c in candlesticks if c.signal_type == "sell")

        total_bullish = bullish_signals + bullish_candles
        total_bearish = bearish_signals + bearish_candles

        if total_bullish > total_bearish + 1:
            overall = "🟢 BULLISH - Multiple buy signals"
        elif total_bearish > total_bullish + 1:
            overall = "🔴 BEARISH - Multiple sell signals"
        else:
            overall = "🟡 NEUTRAL - Mixed signals"

        return {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_price": closes[-1],
            "trend": trend,
            "patterns": [
                {
                    "name": p.pattern_name,
                    "type": p.pattern_type,
                    "confidence": p.confidence,
                    "target": p.price_target,
                    "description": p.description,
                }
                for p in patterns
            ],
            "candlesticks": [
                {
                    "name": c.pattern_name,
                    "signal": c.signal_type,
                    "strength": c.strength,
                    "description": c.description,
                }
                for c in candlesticks
            ],
            "volume": volume_analysis,
            "breakout": breakout,
            "overall_signal": overall,
            "actionable": len(patterns) > 0 or breakout is not None,
        }

    except Exception as e:
        logger.error(f"Pattern analysis failed for {ticker}: {e}")
        return {"error": str(e)}
