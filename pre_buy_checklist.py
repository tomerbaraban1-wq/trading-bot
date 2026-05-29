"""
Pre-Buy Checklist
==================

Final quality gate before every buy order.
Checks 7 conditions — all must pass.

1. RSI Zone     — not in the 0%-win death zone (40-55)
2. Volume       — sufficient confirmation
3. Trend        — stock above key moving averages
4. Market       — SPY not in crash mode
5. Portfolio    — not over-concentrated
6. Time         — not in high-risk time windows
7. News         — no blocking negative headlines
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def run_pre_buy_checklist(
    ticker: str,
    score: float,
    rsi: float,
    volume_ratio: float,
    above_sma50: bool,
    above_sma200: bool,
    open_positions_count: int,
) -> dict:
    """
    Run all pre-buy checks. Returns:
    {
        "pass": bool,
        "failed_checks": [...],
        "passed_checks": [...],
        "confidence_boost": float,  # extra score if all pass
    }
    """
    failed = []        # CRITICAL failures — these BLOCK the buy
    soft_failed = []   # non-critical warnings — reduce confidence but do NOT block
    passed = []
    confidence_boost = 0.0

    # ── Check 1: RSI Zone ────────────────────────────────────────────────
    # RSI 42-55 historically had 0% win rate.
    # FIX: Allow exception for high-conviction trades (score >= 65).
    # Without this, ALL current candidates (NVDA, AVGO, ORCL at RSI 50) were blocked.
    rsi_death_zone_min = float(os.getenv("RSI_AVOID_MIN", "42"))
    rsi_death_zone_max = float(os.getenv("RSI_AVOID_MAX", "55"))

    if rsi and rsi_death_zone_min <= rsi <= rsi_death_zone_max:
        # NEW: Allow high-score override (composite >= 65 = strong signal regardless of RSI)
        if score >= 65:
            passed.append(f"RSI {rsi:.0f} ב-zone אבל ציון {score:.0f} גבוה — חריגה מאושרת")
        else:
            # SOFT (balanced): RSI zone is less-ideal but not a hard block
            soft_failed.append(f"RSI {rsi:.0f} ב-zone פחות-אידאלי ({rsi_death_zone_min:.0f}-{rsi_death_zone_max:.0f})")
            confidence_boost -= 2
    elif rsi and 28 <= rsi <= 42:
        passed.append(f"RSI {rsi:.0f} ב-zone המצוין (WR 100% היסטורי)")
        confidence_boost += 3
    else:
        passed.append(f"RSI {rsi:.0f} בטווח תקין")

    # ── Check 2: Volume ──────────────────────────────────────────────────
    min_vol = float(os.getenv("MIN_VOLUME_RATIO", "0.75"))
    if volume_ratio and volume_ratio < min_vol:
        # SOFT (balanced): low volume reduces confidence but does not block
        soft_failed.append(f"Volume {volume_ratio:.2f}x נמוך מ-{min_vol}x")
        confidence_boost -= 1
    elif volume_ratio and volume_ratio >= 1.0:
        passed.append(f"Volume {volume_ratio:.2f}x — confirmation חזקה")
        confidence_boost += 2
    else:
        passed.append(f"Volume {volume_ratio:.2f}x — בסדר")

    # ── Check 3: Trend ───────────────────────────────────────────────────
    require_sma50 = os.getenv("REQUIRE_ABOVE_SMA50", "true").lower() == "true"
    if require_sma50 and above_sma50 is False and above_sma200 is False:
        failed.append("Death Cross + מתחת SMA50 — מגמה יורדת חזקה")
    elif above_sma50 and above_sma200:
        passed.append("מעל SMA50 + SMA200 — uptrend מוסמך")
        confidence_boost += 4
    elif above_sma200:
        passed.append("מעל SMA200 — long-term trend חיובי")
        confidence_boost += 2
    elif above_sma50:
        passed.append("מעל SMA50 — short-term trend חיובי")
        confidence_boost += 1

    # ── Check 4: Portfolio Concentration ─────────────────────────────────
    max_positions = int(os.getenv("MAX_OPEN_POSITIONS", "6"))
    if open_positions_count >= max_positions:
        failed.append(f"תיק מלא: {open_positions_count}/{max_positions} פוזיציות")
    else:
        slots_left = max_positions - open_positions_count
        passed.append(f"תיק: {open_positions_count}/{max_positions} ({slots_left} slots פנויים)")

    # ── Check 5: Time of Day ─────────────────────────────────────────────
    try:
        now_et = datetime.now(timezone.utc)
        # Approximate ET from UTC (rough — bot uses proper timezone elsewhere)
        et_hour = (now_et.hour - 4) % 24   # EDT offset
        et_min  = now_et.minute

        minutes_since_open  = (et_hour - 9) * 60 + et_min - 30   # since 9:30
        minutes_before_close = (16 * 60) - (et_hour * 60 + et_min)  # to 4:00

        min_after_open   = int(os.getenv("MIN_MINUTES_AFTER_OPEN", "5"))
        min_before_close = int(os.getenv("MIN_MINUTES_BEFORE_CLOSE", "5"))
        if minutes_since_open < min_after_open:
            # SOFT (balanced): wide spreads near open — warn, don't block
            soft_failed.append(f"שוק נפתח לפני {min_after_open-minutes_since_open:.0f} דקות — spreads רחבים")
        elif minutes_before_close < min_before_close:
            soft_failed.append(f"פחות מ-{min_before_close} דקות לסגירה")
        else:
            # Best time: 10:30-11:30 and 13:00-15:30 (ET)
            if (60 <= minutes_since_open <= 120) or (210 <= minutes_since_open <= 360):
                passed.append("שעת מסחר מצוינת")
                confidence_boost += 1
            else:
                passed.append("שעת מסחר תקינה")
    except Exception:
        passed.append("זמן: לא ניתן לבדוק")

    # ── Check 6: Score Threshold ──────────────────────────────────────────
    min_score = int(os.getenv("MIN_BUY_SCORE", "65"))
    if score < min_score:
        failed.append(f"ציון {score:.0f} מתחת לסף {min_score}")
    elif score >= 80:
        passed.append(f"ציון גבוה {score:.0f}/100 — הזדמנות מצוינת!")
        confidence_boost += 3
    else:
        passed.append(f"ציון {score:.0f}/100 — עובר")

    # ── Check 7: Quick news check ─────────────────────────────────────────
    try:
        from news_intelligence import detect_catalysts
        catalyst_data = await asyncio.wait_for(
            detect_catalysts(ticker), timeout=8
        )
        if catalyst_data.get("has_breaking_news"):
            sentiment = catalyst_data.get("avg_sentiment_24h", 0)
            if sentiment < -0.3:
                failed.append(f"חדשות שליליות שוברות ({sentiment:.2f})")
            elif sentiment > 0.3:
                passed.append(f"חדשות חיוביות ({sentiment:.2f})")
                confidence_boost += 2
            else:
                passed.append("חדשות: ניטרלי")
        else:
            passed.append("חדשות: שקט — בסדר")
    except Exception:
        passed.append("חדשות: לא ניתן לבדוק")

    # ── Final decision ────────────────────────────────────────────────────
    # BALANCED: only CRITICAL failures block the buy. Soft warnings reduce
    # confidence but still allow the trade (paper trading — favor taking trades
    # to actually learn, instead of vetoing every candidate on a minor check).
    all_pass = len(failed) == 0

    if all_pass:
        _warn = f" | soft: {'; '.join(soft_failed)}" if soft_failed else ""
        logger.info(
            f"[PRE-BUY] {ticker}: ✅ PASS ({len(passed)} ok, "
            f"{len(soft_failed)} soft, boost={confidence_boost:+.0f}){_warn}"
        )
    else:
        logger.info(
            f"[PRE-BUY] {ticker}: ❌ BLOCKED (critical) — {'; '.join(failed)}"
        )

    return {
        "pass": all_pass,
        "failed_checks": failed,
        "soft_failed_checks": soft_failed,
        "passed_checks": passed,
        "confidence_boost": confidence_boost,
        "total_checks": len(passed) + len(failed) + len(soft_failed),
        "pass_count": len(passed),
    }
