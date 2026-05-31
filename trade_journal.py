"""
Automated Trade Journal
========================

Auto-generates detailed reviews for each trade:
1. Entry analysis (why did we enter?)
2. Exit analysis (why did we exit? Good or bad?)
3. Lesson learned
4. What to do differently
5. Pattern identification
6. Mistake categorization
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeReview:
    """Comprehensive review of a single trade."""
    trade_id: int
    ticker: str
    entry_date: str
    exit_date: str
    pnl: float
    pnl_pct: float
    outcome: str  # "win", "loss", "breakeven"
    quality_score: float  # 0-100, how well executed

    entry_grade: str  # "A", "B", "C", "D", "F"
    exit_grade: str
    overall_grade: str

    entry_analysis: list[str]
    exit_analysis: list[str]
    lessons_learned: list[str]
    mistakes_made: list[str]
    what_to_repeat: list[str]
    what_to_avoid: list[str]


@dataclass
class TradeJournalSummary:
    """Summary of multiple trade reviews."""
    period_days: int
    total_trades_reviewed: int
    avg_quality_score: float
    grade_distribution: dict
    most_common_mistakes: list[dict]
    most_repeated_patterns: list[dict]
    improvement_areas: list[str]
    strengths: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# TRADE GRADING
# ─────────────────────────────────────────────────────────────────────────────

def grade_entry(rsi: float, macd: float, volume_ratio: float, quality_score: float) -> tuple[str, list[str]]:
    """Grade the entry quality."""
    points = 0
    analysis = []

    # Quality score (40 points)
    if quality_score >= 80:
        points += 40
        analysis.append(f"✅ Excellent setup score ({quality_score:.0f}/100)")
    elif quality_score >= 70:
        points += 30
        analysis.append(f"✅ Good setup score ({quality_score:.0f}/100)")
    elif quality_score >= 60:
        points += 20
        analysis.append(f"🟡 Marginal setup score ({quality_score:.0f}/100)")
    else:
        points += 5
        analysis.append(f"❌ Weak setup score ({quality_score:.0f}/100)")

    # RSI timing (20 points)
    if rsi is not None:
        if 30 < rsi < 65:
            points += 20
            analysis.append(f"✅ Good RSI level ({rsi:.0f})")
        elif rsi > 75:
            analysis.append(f"❌ Overbought entry (RSI {rsi:.0f})")
        else:
            points += 10
            analysis.append(f"🟡 RSI {rsi:.0f}")

    # Volume confirmation (20 points)
    if volume_ratio:
        if volume_ratio > 1.0:
            points += 20
            analysis.append(f"✅ Volume confirmation ({volume_ratio:.1f}x)")
        elif volume_ratio > 0.7:
            points += 10
            analysis.append(f"🟡 Average volume ({volume_ratio:.1f}x)")
        else:
            analysis.append(f"❌ Low volume entry ({volume_ratio:.1f}x)")

    # MACD (20 points)
    if macd is not None:
        if macd > 0:
            points += 20
            analysis.append("✅ MACD bullish")
        else:
            analysis.append("🔴 MACD bearish")

    # Determine grade
    if points >= 80:
        grade = "A"
    elif points >= 65:
        grade = "B"
    elif points >= 50:
        grade = "C"
    elif points >= 35:
        grade = "D"
    else:
        grade = "F"

    return grade, analysis


def grade_exit(pnl_pct: float, exit_reason: str, holding_days: float) -> tuple[str, list[str]]:
    """Grade the exit quality."""
    points = 0
    analysis = []

    # Profit/Loss execution (40 points)
    if pnl_pct >= 4:
        points += 40
        analysis.append(f"✅ Strong win ({pnl_pct:.1f}%)")
    elif pnl_pct >= 1:
        points += 30
        analysis.append(f"✅ Profitable ({pnl_pct:.1f}%)")
    elif pnl_pct >= 0:
        points += 15
        analysis.append(f"🟡 Breakeven ({pnl_pct:.1f}%)")
    elif pnl_pct >= -2:
        points += 10
        analysis.append(f"🟡 Small loss ({pnl_pct:.1f}%) - stop loss worked")
    else:
        analysis.append(f"❌ Large loss ({pnl_pct:.1f}%) - stop loss failed?")

    # Exit reason (30 points)
    if "take_profit" in (exit_reason or "").lower():
        points += 30
        analysis.append("✅ Hit take profit target")
    elif "stop_loss" in (exit_reason or "").lower():
        points += 15
        analysis.append("🛑 Stop loss triggered (good discipline)")
    elif "signal" in (exit_reason or "").lower():
        points += 25
        analysis.append("📊 Exit on signal change")
    elif "time" in (exit_reason or "").lower():
        points += 10
        analysis.append("⏱️ Time-based exit")
    else:
        points += 5
        analysis.append(f"❓ Exit reason: {exit_reason}")

    # Holding period (30 points)
    if 2 <= holding_days <= 10:
        points += 30
        analysis.append(f"✅ Optimal holding period ({holding_days:.1f} days)")
    elif holding_days < 1:
        points += 10
        analysis.append(f"🔥 Very short hold ({holding_days:.1f} days)")
    elif holding_days <= 21:
        points += 20
        analysis.append(f"🟡 Longer hold ({holding_days:.1f} days)")
    else:
        points += 5
        analysis.append(f"⚠️ Very long hold ({holding_days:.1f} days)")

    # Determine grade
    if points >= 80:
        grade = "A"
    elif points >= 65:
        grade = "B"
    elif points >= 50:
        grade = "C"
    elif points >= 35:
        grade = "D"
    else:
        grade = "F"

    return grade, analysis


# ─────────────────────────────────────────────────────────────────────────────
# LESSON EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_lessons(
    pnl_pct: float,
    rsi: float,
    macd: float,
    volume_ratio: float,
    exit_reason: str,
    holding_days: float,
) -> dict:
    """Extract lessons from a trade."""
    lessons = []
    mistakes = []
    repeat = []
    avoid = []

    # Win analysis
    if pnl_pct > 0:
        if rsi and 30 < rsi < 65:
            repeat.append("Entered with good RSI level (30-65)")

        if volume_ratio and volume_ratio > 1.0:
            repeat.append("Confirmed entry with strong volume")

        if "take_profit" in (exit_reason or "").lower():
            lessons.append("✅ Take profit strategy worked - stick with it")

        if 2 <= holding_days <= 10:
            repeat.append(f"Optimal holding period: {holding_days:.1f} days")

    # Loss analysis
    else:
        if rsi and rsi > 70:
            mistakes.append("Entered while overbought (RSI > 70)")
            avoid.append("Skip entries when RSI > 70")

        if volume_ratio and volume_ratio < 0.5:
            mistakes.append("Entered with weak volume")
            avoid.append("Require minimum 0.5x average volume")

        if macd is False or (macd is not None and macd < 0):
            mistakes.append("Ignored bearish MACD")
            avoid.append("Require MACD bullish at entry")

        if holding_days > 14:
            mistakes.append("Held too long without cutting loss")
            avoid.append("Set maximum holding period of 14 days")

        if pnl_pct < -5:
            mistakes.append("Loss exceeded stop loss - slippage or gap?")
            avoid.append("Use guaranteed stop loss orders")

        # Lessons
        if not mistakes:
            lessons.append("Loss with good execution - random market noise")
        else:
            lessons.append(f"Identified {len(mistakes)} specific mistake(s)")

    return {
        "lessons": lessons,
        "mistakes": mistakes,
        "repeat": repeat,
        "avoid": avoid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRADE REVIEW GENERATION
# ─────────────────────────────────────────────────────────────────────────────

async def review_recent_trades(days: int = 7) -> list[TradeReview]:
    """Generate reviews for recent trades."""
    try:
        import database
        conn = database.get_connection()

        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = conn.execute("""
            SELECT id, ticker, created_at, exit_time, pnl_gross,
                   entry_price, exit_price, rsi, macd, volume_ratio,
                   exit_reason
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= ?
            ORDER BY exit_time DESC
            LIMIT 20
        """, (start_date,)).fetchall()

        reviews = []
        for row in rows:
            try:
                (trade_id, ticker, entry_date, exit_date, pnl, entry_p, exit_p,
                 rsi, macd, vol_ratio, exit_reason) = row
                quality = 70  # quality_score not stored in trade_log — default (code uses 'quality or 70')

                # Calculate metrics
                pnl_pct = ((exit_p - entry_p) / entry_p * 100) if entry_p else 0

                try:
                    entry_dt = datetime.fromisoformat(entry_date.replace("Z", "+00:00"))
                    exit_dt = datetime.fromisoformat(exit_date.replace("Z", "+00:00"))
                    holding_days = (exit_dt - entry_dt).total_seconds() / 86400
                except:
                    holding_days = 0

                # Grade entry and exit
                entry_grade, entry_analysis = grade_entry(rsi, macd, vol_ratio, quality or 70)
                exit_grade, exit_analysis = grade_exit(pnl_pct, exit_reason, holding_days)

                # Overall grade (average)
                grade_values = {"A": 90, "B": 80, "C": 70, "D": 60, "F": 50}
                avg_score = (grade_values.get(entry_grade, 50) + grade_values.get(exit_grade, 50)) / 2
                if avg_score >= 85:
                    overall_grade = "A"
                elif avg_score >= 75:
                    overall_grade = "B"
                elif avg_score >= 65:
                    overall_grade = "C"
                elif avg_score >= 55:
                    overall_grade = "D"
                else:
                    overall_grade = "F"

                # Determine outcome
                if pnl > 0.5:
                    outcome = "win"
                elif pnl < -0.5:
                    outcome = "loss"
                else:
                    outcome = "breakeven"

                # Extract lessons
                lessons_data = extract_lessons(pnl_pct, rsi, macd, vol_ratio, exit_reason, holding_days)

                reviews.append(TradeReview(
                    trade_id=trade_id,
                    ticker=ticker,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                    quality_score=avg_score,
                    entry_grade=entry_grade,
                    exit_grade=exit_grade,
                    overall_grade=overall_grade,
                    entry_analysis=entry_analysis,
                    exit_analysis=exit_analysis,
                    lessons_learned=lessons_data["lessons"],
                    mistakes_made=lessons_data["mistakes"],
                    what_to_repeat=lessons_data["repeat"],
                    what_to_avoid=lessons_data["avoid"],
                ))

            except Exception as e:
                logger.debug(f"Trade review error: {e}")

        return reviews

    except Exception as e:
        logger.error(f"Trade reviews failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

async def generate_journal_summary(days: int = 30) -> TradeJournalSummary:
    """Generate summary of all trade reviews over a period."""
    try:
        reviews = await review_recent_trades(days)

        if not reviews:
            return TradeJournalSummary(
                period_days=days,
                total_trades_reviewed=0,
                avg_quality_score=0,
                grade_distribution={},
                most_common_mistakes=[],
                most_repeated_patterns=[],
                improvement_areas=[],
                strengths=[],
            )

        # Aggregate stats
        avg_quality = sum(r.quality_score for r in reviews) / len(reviews)

        # Grade distribution
        grade_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for r in reviews:
            grade_dist[r.overall_grade] = grade_dist.get(r.overall_grade, 0) + 1

        # Common mistakes
        all_mistakes = []
        for r in reviews:
            all_mistakes.extend(r.mistakes_made)

        from collections import Counter
        mistake_counts = Counter(all_mistakes)
        most_common = [
            {"mistake": m, "count": c} for m, c in mistake_counts.most_common(5)
        ]

        # Common patterns to repeat
        all_repeats = []
        for r in reviews:
            all_repeats.extend(r.what_to_repeat)
        repeat_counts = Counter(all_repeats)
        most_repeated = [
            {"pattern": p, "count": c} for p, c in repeat_counts.most_common(5)
        ]

        # Improvement areas
        improvement_areas = []
        if avg_quality < 70:
            improvement_areas.append(f"📈 Average grade too low ({avg_quality:.0f}/100)")
        if grade_dist["F"] > len(reviews) * 0.2:
            improvement_areas.append(f"❌ Too many F grades ({grade_dist['F']})")
        if mistake_counts:
            improvement_areas.append(f"🚫 Address top mistake: {mistake_counts.most_common(1)[0][0]}")

        # Strengths
        strengths = []
        if avg_quality >= 75:
            strengths.append(f"✅ Strong average grade ({avg_quality:.0f}/100)")
        if grade_dist["A"] + grade_dist["B"] >= len(reviews) * 0.6:
            strengths.append("✅ Most trades are A/B grade")
        if repeat_counts:
            strengths.append(f"✅ Successful pattern: {repeat_counts.most_common(1)[0][0]}")

        return TradeJournalSummary(
            period_days=days,
            total_trades_reviewed=len(reviews),
            avg_quality_score=avg_quality,
            grade_distribution=grade_dist,
            most_common_mistakes=most_common,
            most_repeated_patterns=most_repeated,
            improvement_areas=improvement_areas,
            strengths=strengths,
        )

    except Exception as e:
        logger.error(f"Journal summary failed: {e}")
        return TradeJournalSummary(
            period_days=days, total_trades_reviewed=0, avg_quality_score=0,
            grade_distribution={}, most_common_mistakes=[],
            most_repeated_patterns=[], improvement_areas=[], strengths=[],
        )
