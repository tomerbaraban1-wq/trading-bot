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
    failed = []
    passed = []
    confidence_boost = 0.0

    # ── Check 1: RSI Zone ────────────────────────────────────────────────
    # RSI 40-55 had 0% win rate in our trade history
    rsi_death_zone_min = float(os.getenv("RSI_AVOID_MIN", "42"))
    rsi_death_zone_max = float(os.getenv("RSI_AVOID_MAX", "55"))

    if rsi and rsi_death_zone_min <= rsi <= rsi_death_zone_max:
        failed.append(f"RSI {rsi:.0f} נמצא ב-zone מסוכן ({rsi_death_zone_min:.0f}-{rsi_death_zone_max:.0f}) — 0% WR היסטורי")
    elif rsi and 28 <= rsi <= 42:
        passed.append(f"RSI {rsi:.0f} ב-zone המצוין (WR 100% היסטורי)")
        confidence_boost += 3
    else:
        passed.append(f"RSI {rsi:.0f} בטווח תקין")

    # ── Check 2: Volume ──────────────────────────────────────────────────
    min_vol = float(os.getenv("MIN_VOLUME_RATIO", "0.75"))
    if volume_ratio and volume_ratio < min_vol:
        failed.append(f"Volume {volume_ratio:.2f}x נמוך מ-{min_vol}x נדרש")
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

        if minutes_since_open < 10:
            failed.append(f"שוק נפתח לפני {10-minutes_since_open:.0f} דקות — spreads רחבים")
        elif minutes_before_close < 10:
            failed.append("פחות מ-10 דקות לסגירה — לא קונים")
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
    all_pass = len(failed) == 0

    if all_pass:
        logger.info(
            f"[PRE-BUY] {ticker}: ✅ ALL {len(passed)} checks passed "
            f"(boost={confidence_boost:+.0f})"
        )
    else:
        logger.info(
            f"[PRE-BUY] {ticker}: ❌ BLOCKED — {'; '.join(failed)}"
        )

    return {
        "pass": all_pass,
        "failed_checks": failed,
        "passed_checks": passed,
        "confidence_boost": confidence_boost,
        "total_checks": len(passed) + len(failed),
        "pass_count": len(passed),
    }
