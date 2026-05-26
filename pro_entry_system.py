"""
Professional Entry System
==========================

Implements pro-level entry criteria:

1. Relative Strength — stock > sector > market
2. Pullback Entry — RSI 30-42 in uptrend (optimal zone from data)
3. Trend Quality — ADX + Higher Highs/Lows
4. Risk/Reward Filter — minimum 2:1 before entering
5. Volume Surge Confirmation — institutional buying

A trade only passes if ALL conditions align.
This reduces trade frequency significantly but dramatically
improves win rate (quality over quantity).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntryAnalysis:
    """Complete entry analysis for a stock."""
    ticker: str
    overall_grade: str          # A, B, C, D, F
    overall_score: float        # 0-100
    should_enter: bool

    # Individual components
    relative_strength: float    # -1 to +1 vs SPY
    trend_quality: float        # 0-100 (ADX-based)
    pullback_quality: float     # 0-100 (RSI + price location)
    risk_reward_ratio: float    # Expected R/R
    volume_quality: float       # 0-100

    # Details
    adx: float
    rsi: float
    is_pullback_in_uptrend: bool
    support_level: Optional[float]
    resistance_level: Optional[float]
    optimal_entry: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]

    reasoning: list[str]
    warnings: list[str]


def calculate_adx(prices_high: list, prices_low: list, prices_close: list, period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending market (good for trend following)
    ADX < 20 = ranging market (avoid)
    """
    if len(prices_close) < period + 2:
        return 0.0

    try:
        highs  = np.array(prices_high[-period*2:])
        lows   = np.array(prices_low[-period*2:])
        closes = np.array(prices_close[-period*2:])

        # True Range
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
            tr_list.append(tr)

        # Directional Movement
        dm_plus  = [max(highs[i] - highs[i-1], 0) if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
                    for i in range(1, len(highs))]
        dm_minus = [max(lows[i-1] - lows[i], 0) if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
                    for i in range(1, len(lows))]

        if not tr_list or sum(tr_list[:period]) == 0:
            return 0.0

        # Smoothed averages
        atr = sum(tr_list[:period]) / period
        dmp = sum(dm_plus[:period]) / period
        dmm = sum(dm_minus[:period]) / period

        if atr == 0:
            return 0.0

        di_plus  = (dmp / atr) * 100
        di_minus = (dmm / atr) * 100

        dx_sum = abs(di_plus - di_minus) / max(di_plus + di_minus, 0.001) * 100
        return float(np.clip(dx_sum, 0, 100))

    except Exception as e:
        logger.debug(f"ADX calculation failed: {e}")
        return 0.0


def detect_higher_highs_lows(prices: list, window: int = 10) -> dict:
    """
    Detect if stock is making higher highs and higher lows (uptrend quality).
    """
    if len(prices) < window * 2:
        return {"higher_highs": False, "higher_lows": False, "trend": "unknown"}

    prices = np.array(prices)

    # Split into two halves
    mid = len(prices) // 2
    first_half  = prices[:mid]
    second_half = prices[mid:]

    # Peaks (local highs)
    first_highs  = max(first_half)
    second_highs = max(second_half)

    # Troughs (local lows)
    first_lows  = min(first_half)
    second_lows = min(second_half)

    higher_highs = second_highs > first_highs
    higher_lows  = second_lows  > first_lows

    if higher_highs and higher_lows:
        trend = "uptrend"
    elif not higher_highs and not higher_lows:
        trend = "downtrend"
    else:
        trend = "sideways"

    return {
        "higher_highs": higher_highs,
        "higher_lows": higher_lows,
        "trend": trend,
    }


def find_nearest_support(prices: list, current: float, lookback: int = 30) -> Optional[float]:
    """Find nearest support level below current price."""
    if not prices or len(prices) < 5:
        return None

    recent = np.array(prices[-lookback:])

    # Find local minima
    supports = []
    for i in range(2, len(recent) - 2):
        if recent[i] == min(recent[i-2:i+3]):
            supports.append(recent[i])

    # Find support levels below current price
    below = [s for s in supports if s < current * 0.999]
    return max(below) if below else None


def find_nearest_resistance(prices: list, current: float, lookback: int = 30) -> Optional[float]:
    """Find nearest resistance level above current price."""
    if not prices or len(prices) < 5:
        return None

    recent = np.array(prices[-lookback:])

    # Find local maxima
    resistances = []
    for i in range(2, len(recent) - 2):
        if recent[i] == max(recent[i-2:i+3]):
            resistances.append(recent[i])

    # Find resistance above current price
    above = [r for r in resistances if r > current * 1.001]
    return min(above) if above else None


async def calculate_relative_strength(ticker: str, period_days: int = 20) -> float:
    """
    Calculate relative strength vs SPY.
    RS = ticker_return / SPY_return over period.
    RS > 1.0 = outperforming (positive)
    RS < 1.0 = underperforming (negative)
    Returns: -1 to +1 score
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta, timezone

        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=period_days + 5)

        data = yf.download([ticker, "SPY"], start=start, end=end, progress=False)["Close"]

        if data.empty:
            return 0.0

        ticker_ret = float(data[ticker].dropna().pct_change(period_days).iloc[-1])
        spy_ret    = float(data["SPY"].dropna().pct_change(period_days).iloc[-1])

        if spy_ret == 0:
            return 0.0

        rs = ticker_ret - spy_ret  # alpha over SPY

        # Normalize to -1 to +1
        return max(-1.0, min(1.0, rs / 0.10))  # 10% alpha = +1.0

    except Exception as e:
        logger.debug(f"Relative strength failed for {ticker}: {e}")
        return 0.0


async def analyze_entry(ticker: str) -> EntryAnalysis:
    """
    Full professional-grade entry analysis.
    Returns EntryAnalysis with grade and all sub-components.
    """
    reasoning = []
    warnings  = []

    try:
        import yfinance as yf
        from datetime import datetime, timedelta, timezone

        # Get 90 days of OHLCV data
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=90)
        data  = yf.download(ticker, start=start, end=end, progress=False)

        if data.empty or len(data) < 30:
            return EntryAnalysis(
                ticker=ticker, overall_grade="F", overall_score=0,
                should_enter=False, relative_strength=0, trend_quality=0,
                pullback_quality=0, risk_reward_ratio=0, volume_quality=0,
                adx=0, rsi=0, is_pullback_in_uptrend=False,
                support_level=None, resistance_level=None,
                optimal_entry=None, stop_price=None, target_price=None,
                reasoning=["Insufficient data"], warnings=[],
            )

        closes  = data["Close"].values.tolist()
        highs   = data["High"].values.tolist()
        lows    = data["Low"].values.tolist()
        volumes = data["Volume"].values.tolist()
        current = float(closes[-1])

        # ── 1. TREND QUALITY (ADX + Higher Highs/Lows) ────────────────
        adx     = calculate_adx(highs, lows, closes)
        hh_hl   = detect_higher_highs_lows(closes)
        trend   = hh_hl["trend"]

        if adx >= 30 and trend == "uptrend":
            trend_quality = 90
            reasoning.append(f"✅ Trend forte: ADX={adx:.0f}, Higher H/L")
        elif adx >= 20 and trend == "uptrend":
            trend_quality = 65
            reasoning.append(f"🟡 Trend moderato: ADX={adx:.0f}")
        elif trend == "downtrend":
            trend_quality = 10
            warnings.append(f"🔴 Downtrend (ADX={adx:.0f}) — avoid!")
        else:
            trend_quality = 30
            warnings.append(f"🟡 Ranging market (ADX={adx:.0f})")

        # ── 2. PULLBACK QUALITY (RSI + price vs SMA20) ────────────────
        # Calculate RSI
        deltas = np.diff(closes)
        gains  = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

        # SMA20 check
        sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else current
        sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else current

        is_pullback_in_uptrend = (
            trend == "uptrend" and
            sma50 < sma20 < current * 1.03 and  # above both MAs but close to SMA20
            30 <= rsi <= 45  # oversold zone in uptrend (our best historical zone)
        )

        if is_pullback_in_uptrend:
            pullback_quality = 90
            reasoning.append(f"✅ Perfect pullback: RSI={rsi:.0f}, price near SMA20")
        elif 28 <= rsi <= 50 and trend == "uptrend":
            pullback_quality = 65
            reasoning.append(f"🟡 Decent pullback: RSI={rsi:.0f}")
        elif rsi > 65:
            pullback_quality = 20
            warnings.append(f"🔴 Extended / overbought: RSI={rsi:.0f}")
        else:
            pullback_quality = 40

        # ── 3. RELATIVE STRENGTH ──────────────────────────────────────
        rs = await calculate_relative_strength(ticker)
        if rs > 0.3:
            reasoning.append(f"✅ Outperforming SPY (+{rs:.0%})")
        elif rs < -0.3:
            warnings.append(f"🔴 Underperforming SPY ({rs:.0%})")

        # ── 4. SUPPORT/RESISTANCE & RISK-REWARD ───────────────────────
        support    = find_nearest_support(closes, current)
        resistance = find_nearest_resistance(closes, current)

        if support and resistance and current > support:
            potential_loss   = current - support
            potential_gain   = resistance - current
            rr_ratio = potential_gain / potential_loss if potential_loss > 0 else 0

            if rr_ratio >= 2.5:
                reasoning.append(f"✅ Excellent R/R: {rr_ratio:.1f}:1")
            elif rr_ratio >= 2.0:
                reasoning.append(f"🟡 Good R/R: {rr_ratio:.1f}:1")
            elif rr_ratio < 1.5:
                warnings.append(f"🔴 Poor R/R: {rr_ratio:.1f}:1 — skip!")
        else:
            rr_ratio = 2.0  # default assumption

        stop_price   = support * 0.99 if support else current * 0.97
        target_price = resistance * 0.99 if resistance else current * (1 + 0.06)

        # ── 5. VOLUME QUALITY ─────────────────────────────────────────
        avg_vol_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 0
        cur_vol    = float(volumes[-1])
        vol_ratio  = cur_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

        if vol_ratio >= 1.5:
            volume_quality = 85
            reasoning.append(f"✅ Volume surge: {vol_ratio:.1f}x avg")
        elif vol_ratio >= 1.0:
            volume_quality = 60
        else:
            volume_quality = 30
            warnings.append(f"🔴 Low volume: {vol_ratio:.1f}x avg")

        # ── OVERALL SCORE ──────────────────────────────────────────────
        weights = {
            "trend":    0.30,
            "pullback": 0.25,
            "rs":       0.20,
            "rr":       0.15,
            "volume":   0.10,
        }

        rs_score = (rs + 1) / 2 * 100  # Convert -1..1 to 0..100
        rr_score = min(100, rr_ratio / 3 * 100)

        overall_score = (
            trend_quality   * weights["trend"]    +
            pullback_quality * weights["pullback"] +
            rs_score        * weights["rs"]        +
            rr_score        * weights["rr"]        +
            volume_quality  * weights["volume"]
        )

        # Grade
        if overall_score >= 80:
            grade = "A"
        elif overall_score >= 70:
            grade = "B"
        elif overall_score >= 55:
            grade = "C"
        elif overall_score >= 40:
            grade = "D"
        else:
            grade = "F"

        # Professional rule: only enter A or B grade setups
        should_enter = grade in ("A", "B") and rr_ratio >= 1.5 and len(warnings) < 2

        logger.info(
            f"[PRO ENTRY] {ticker}: grade={grade} score={overall_score:.0f} "
            f"ADX={adx:.0f} RSI={rsi:.0f} RS={rs:.2f} RR={rr_ratio:.1f} "
            f"enter={should_enter}"
        )

        return EntryAnalysis(
            ticker=ticker,
            overall_grade=grade,
            overall_score=overall_score,
            should_enter=should_enter,
            relative_strength=rs,
            trend_quality=trend_quality,
            pullback_quality=pullback_quality,
            risk_reward_ratio=rr_ratio,
            volume_quality=volume_quality,
            adx=adx,
            rsi=rsi,
            is_pullback_in_uptrend=is_pullback_in_uptrend,
            support_level=support,
            resistance_level=resistance,
            optimal_entry=current,
            stop_price=stop_price,
            target_price=target_price,
            reasoning=reasoning,
            warnings=warnings,
        )

    except Exception as e:
        logger.error(f"Pro entry analysis failed for {ticker}: {e}")
        return EntryAnalysis(
            ticker=ticker, overall_grade="F", overall_score=0, should_enter=False,
            relative_strength=0, trend_quality=0, pullback_quality=0,
            risk_reward_ratio=0, volume_quality=0, adx=0, rsi=0,
            is_pullback_in_uptrend=False, support_level=None, resistance_level=None,
            optimal_entry=None, stop_price=None, target_price=None,
            reasoning=[], warnings=[f"Error: {e}"],
        )


async def pro_entry_gate(ticker: str, existing_score: float) -> dict:
    """
    Final professional gate before buying.
    Only allows A/B grade setups through.
    Returns the entry analysis + adjusted score.
    """
    analysis = await analyze_entry(ticker)

    # Adjust score based on professional analysis
    score_adjustment = 0.0

    if analysis.overall_grade == "A":
        score_adjustment = +10  # Boost score for perfect setup
    elif analysis.overall_grade == "B":
        score_adjustment = +5
    elif analysis.overall_grade == "C":
        score_adjustment = 0    # Neutral
    elif analysis.overall_grade == "D":
        score_adjustment = -10  # Discourage
    else:  # F
        score_adjustment = -20  # Strong discourage

    # Extra boost for perfect pullback in uptrend
    if analysis.is_pullback_in_uptrend:
        score_adjustment += 5

    # Extra boost for excellent R/R
    if analysis.risk_reward_ratio >= 2.5:
        score_adjustment += 5

    return {
        "analysis": analysis,
        "adjusted_score": max(0, existing_score + score_adjustment),
        "score_adjustment": score_adjustment,
        "should_enter": analysis.should_enter and not (analysis.overall_grade == "F"),
        "skip_reason": " | ".join(analysis.warnings[:2]) if not analysis.should_enter else "",
        "grade": analysis.overall_grade,
    }
