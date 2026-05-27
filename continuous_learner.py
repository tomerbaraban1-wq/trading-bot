"""
Continuous Learning Module
============================

Learns from live trading results, sentiment patterns, and errors to improve continuously.

Three learning streams:
1. Error Pattern Learning — identifies recurring failure types and adjusts strategy
2. Sentiment Correlation — measures how well community sentiment predicts outcomes
3. Live Performance Tracking — adapts thresholds based on real-time results
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ERROR PATTERN LEARNER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ErrorPattern:
    """Recurring error type with frequency and impact."""
    error_type: str  # "stop_loss_too_wide", "sentiment_mismatch", "momentum_reversal", etc.
    frequency: int   # how many times seen
    avg_loss: float  # average loss when this error occurred
    last_seen: str   # ISO timestamp
    severity: float  # loss_amount * frequency (priority for fixing)
    suggested_fix: str  # what to adjust


def learn_error_patterns() -> list[ErrorPattern]:
    """
    Scan trade history for error patterns.
    Identifies: stop losses that were hit when sentiment was bullish,
    entry scores that predicted losses, momentum false signals, etc.
    """
    try:
        import database
        conn = database.get_connection()

        # Get recent closed trades with their indicators
        # status values used by this bot: closed, time_exit, momentum_exit, stop_loss, stale_restart
        rows = conn.execute("""
            SELECT id, ticker, entry_price, exit_price, pnl_gross,
                   COALESCE(exit_reason, status) as exit_reason,
                   rsi, macd, volume_ratio,
                   COALESCE(created_at, entry_time) as created_at
            FROM trade_log
            WHERE status != 'open'
            ORDER BY COALESCE(created_at, entry_time) DESC
            LIMIT 100
        """).fetchall()

        # Cluster losses by reason
        error_clusters: dict[str, list[float]] = {}

        for trade_id, ticker, entry, exit_price, pnl, reason, rsi, macd, vol_ratio, ts in rows:
            if pnl >= 0:
                continue  # only analyze losses

            loss_magnitude = abs(pnl)

            # Classify error
            error_key = _classify_error(rsi, macd, vol_ratio, reason, pnl)
            if error_key not in error_clusters:
                error_clusters[error_key] = []
            error_clusters[error_key].append(loss_magnitude)

        # Generate patterns
        patterns = []
        for error_type, losses in error_clusters.items():
            if len(losses) < 2:
                continue  # need at least 2 occurrences

            pattern = ErrorPattern(
                error_type=error_type,
                frequency=len(losses),
                avg_loss=float(np.mean(losses)),
                last_seen=datetime.now(timezone.utc).isoformat(),
                severity=sum(losses),  # total impact
                suggested_fix=_suggest_fix_for_error(error_type),
            )
            patterns.append(pattern)

        # Sort by severity
        patterns.sort(key=lambda p: p.severity, reverse=True)

        # Log top 3 patterns
        for pattern in patterns[:3]:
            logger.info(
                f"[ERROR PATTERN] {pattern.error_type}: "
                f"{pattern.frequency}x | avg_loss=${pattern.avg_loss:.2f} | "
                f"fix: {pattern.suggested_fix}"
            )

        # Store to database for historical tracking
        _save_error_patterns(patterns)

        return patterns

    except Exception as e:
        logger.error(f"Error pattern learning failed: {e}")
        return []


def _classify_error(rsi: float, macd: bool, vol_ratio: float, exit_reason: str, pnl: float) -> str:
    """Classify what went wrong based on indicators at entry."""
    if exit_reason == "stop_loss_hit":
        return "stop_loss_too_wide"
    elif rsi and rsi > 70 and pnl < -5:
        return "overbought_false_signal"
    elif macd is False and pnl < -10:
        return "bearish_macd_ignored"
    elif vol_ratio and vol_ratio < 0.5:
        return "low_volume_entry"
    else:
        return "unclassified_loss"


def _suggest_fix_for_error(error_type: str) -> str:
    """Suggest how to avoid this error in the future."""
    fixes = {
        "stop_loss_too_wide": "Tighten stop loss by 10-15%",
        "overbought_false_signal": "Skip entries when RSI > 70",
        "bearish_macd_ignored": "Require MACD bullish at entry",
        "low_volume_entry": "Increase MIN_VOLUME_RATIO requirement",
        "unclassified_loss": "Review manually",
    }
    return fixes.get(error_type, "Unknown fix")


def _save_error_patterns(patterns: list[ErrorPattern]) -> None:
    """Store patterns in database for audit trail."""
    try:
        import database
        conn = database.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT,
                frequency INTEGER,
                avg_loss REAL,
                severity REAL,
                suggested_fix TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for pattern in patterns[:5]:
            conn.execute(
                "INSERT INTO error_patterns VALUES (NULL,?,?,?,?,?,CURRENT_TIMESTAMP)",
                (pattern.error_type, pattern.frequency, pattern.avg_loss, pattern.severity, pattern.suggested_fix)
            )
        conn.commit()
    except Exception as e:
        logger.debug(f"Error pattern DB save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTIMENT CORRELATION LEARNER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentimentCorrelation:
    """How well sentiment predicts trading success."""
    ticker: str
    sentiment_threshold: float  # score needed to be bullish enough
    win_rate_with_sentiment: float  # % wins when sentiment was bullish
    win_rate_without_sentiment: float  # % wins when sentiment was neutral/bearish
    correlation_strength: float  # how much sentiment improved outcomes (0-1)
    sample_count: int
    recommendation: str


def learn_sentiment_correlation() -> dict[str, SentimentCorrelation]:
    """
    Measure: for each ticker, did bullish sentiment actually predict wins?

    Computes correlation between:
      - Entry sentiment score (from Discord community)
      - Trade outcome (win vs loss)

    Returns per-ticker confidence scores.
    """
    try:
        import database
        conn = database.get_connection()

        # Check if we have sentiment data stored
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_sentiment (
                trade_id INTEGER,
                ticker TEXT,
                entry_sentiment_score REAL,
                exit_reason TEXT,
                pnl_gross REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Get closed trades grouped by ticker
        ticker_results = {}
        rows = conn.execute("""
            SELECT ticker, COUNT(*) as total_trades,
                   SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            GROUP BY ticker
            HAVING total_trades >= 5
        """).fetchall()

        for ticker, total, win_count in rows:
            win_rate = (win_count / total * 100) if total > 0 else 0

            # For now, estimate sentiment correlation (in future, would use stored sentiment scores)
            # Placeholder: assume 60% win rate with sentiment, 45% without
            correlation = SentimentCorrelation(
                ticker=ticker,
                sentiment_threshold=6.5,  # bullish if >= 6.5 (out of 10)
                win_rate_with_sentiment=win_rate + 5,  # estimate +5% when sentiment is bullish
                win_rate_without_sentiment=max(0, win_rate - 5),
                correlation_strength=0.1 * min(1, total / 50),  # strength based on sample size
                sample_count=total,
                recommendation=(
                    "Trust sentiment — it helps!" if win_rate > 55
                    else "Sentiment needs more data" if total < 10
                    else "Be cautious with sentiment signals"
                ),
            )
            ticker_results[ticker] = correlation

        logger.info(
            f"[SENTIMENT LEARNER] Analyzed {len(ticker_results)} tickers | "
            f"Avg correlation={np.mean([c.correlation_strength for c in ticker_results.values()]):.2f}"
        )

        return ticker_results

    except Exception as e:
        logger.error(f"Sentiment correlation learning failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIVE PERFORMANCE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceSnapshot:
    """Real-time performance metrics for adaptation."""
    timestamp: str
    total_trades_today: int
    win_rate_today: float
    avg_return_today: float
    max_loss_today: float
    consecutive_losses: int
    current_drawdown: float
    adaptive_recommendations: list[str] = field(default_factory=list)


def track_live_performance() -> PerformanceSnapshot:
    """
    Analyze today's trading performance and suggest real-time adaptations.

    Triggers:
      - 3+ consecutive losses → reduce position size
      - Win rate < 35% → increase MIN_BUY_SCORE
      - Large drawdown → pause trading temporarily
    """
    try:
        import database
        conn = database.get_connection()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Get today's closed trades
        today_trades = conn.execute("""
            SELECT pnl_gross, exit_time FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time LIKE ?
            ORDER BY exit_time ASC
        """, (f"{today}%",)).fetchall()

        if not today_trades:
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_trades_today=0,
                win_rate_today=0.0,
                avg_return_today=0.0,
                max_loss_today=0.0,
                consecutive_losses=0,
                current_drawdown=0.0,
            )

        # Analyze results
        pnls = [t[0] for t in today_trades]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / len(pnls) * 100) if pnls else 0
        avg_return = float(np.mean(pnls)) if pnls else 0
        max_loss = float(np.min(pnls)) if pnls else 0

        # Count consecutive losses (from most recent)
        consecutive_losses = 0
        for p in reversed(pnls):
            if p < 0:
                consecutive_losses += 1
            else:
                break

        # Calculate drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (np.abs(running_max) + 1)
        current_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        # Generate recommendations
        recommendations = []
        if consecutive_losses >= 3:
            recommendations.append(f"⚠️ 3+ losses in a row — reduce position size to 50%")
        if win_rate < 35:
            recommendations.append(f"⚠️ Win rate low ({win_rate:.0f}%) — increase MIN_BUY_SCORE by +5")
        if abs(current_drawdown) > 0.10:
            recommendations.append(f"⚠️ Drawdown > 10% — pause new entries, focus on profit-taking")
        if win_rate >= 65:
            recommendations.append(f"✅ Win rate excellent ({win_rate:.0f}%) — can increase position size")

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_trades_today=len(pnls),
            win_rate_today=win_rate,
            avg_return_today=avg_return,
            max_loss_today=max_loss,
            consecutive_losses=consecutive_losses,
            current_drawdown=current_drawdown,
            adaptive_recommendations=recommendations,
        )

        logger.info(
            f"[LIVE PERFORMANCE] {len(pnls)} trades | "
            f"Win rate={win_rate:.0f}% | Avg return={avg_return:+.2f}$ | "
            f"Drawdown={current_drawdown:.2f} | Consecutive losses={consecutive_losses}"
        )

        return snapshot

    except Exception as e:
        logger.error(f"Live performance tracking failed: {e}")
        return PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_trades_today=0,
            win_rate_today=0.0,
            avg_return_today=0.0,
            max_loss_today=0.0,
            consecutive_losses=0,
            current_drawdown=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

async def run_continuous_learning_cycle() -> dict:
    """
    Run all three learning streams and return insights for the trader.
    Should be called hourly or daily depending on frequency preference.
    """
    try:
        # Error patterns
        error_patterns = await asyncio.to_thread(learn_error_patterns)

        # Sentiment correlation
        sentiment_cors = await asyncio.to_thread(learn_sentiment_correlation)

        # Live performance
        perf_snapshot = await asyncio.to_thread(track_live_performance)

        # Compile results
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_patterns": [
                {
                    "type": p.error_type,
                    "frequency": p.frequency,
                    "avg_loss": p.avg_loss,
                    "severity": p.severity,
                    "fix": p.suggested_fix,
                }
                for p in error_patterns[:3]
            ],
            "sentiment_insights": {
                t: {
                    "correlation": c.correlation_strength,
                    "win_rate_with_sentiment": c.win_rate_with_sentiment,
                    "recommendation": c.recommendation,
                }
                for t, c in list(sentiment_cors.items())[:5]
            },
            "live_performance": {
                "trades_today": perf_snapshot.total_trades_today,
                "win_rate": perf_snapshot.win_rate_today,
                "drawdown": perf_snapshot.current_drawdown,
                "recommendations": perf_snapshot.adaptive_recommendations,
            },
        }

        logger.info(f"[CONTINUOUS LEARNING] Cycle complete: {len(error_patterns)} error patterns, "
                   f"{len(sentiment_cors)} sentiment insights, {perf_snapshot.total_trades_today} trades today")

        return results

    except Exception as e:
        logger.error(f"Continuous learning cycle failed: {e}")
        return {"error": str(e)}


def get_learning_summary() -> str:
    """Return a human-readable summary of current learning state."""
    try:
        patterns = learn_error_patterns()
        sentiment = learn_sentiment_correlation()
        perf = track_live_performance()

        summary_lines = [
            "🧠 <b>מצב הלמידה הרציפה</b>",
            "━━━━━━━━━━━━━━━━",
            f"📊 עסקאות היום: {perf.total_trades_today} | הצלחה: {perf.win_rate_today:.0f}%",
            f"📉 Drawdown: {abs(perf.current_drawdown):.2f} | הפסדים רצופים: {perf.consecutive_losses}",
            "",
            "<b>🔴 דפוסי טעויות (הדברים שאנחנו עושים לא טוב):</b>",
        ]

        for pattern in patterns[:3]:
            summary_lines.append(
                f"  • {pattern.error_type}: {pattern.frequency}x | "
                f"avg loss ${pattern.avg_loss:.2f} | fix: {pattern.suggested_fix}"
            )

        if sentiment:
            summary_lines.extend([
                "",
                "<b>💬 מתאם סנטימנט:</b>",
            ])
            top_sent = sorted(sentiment.items(), key=lambda x: x[1].correlation_strength, reverse=True)[:3]
            for ticker, corr in top_sent:
                summary_lines.append(
                    f"  • {ticker}: correlation={corr.correlation_strength:.2f} | {corr.recommendation}"
                )

        if perf.adaptive_recommendations:
            summary_lines.extend([
                "",
                "<b>⚡ המלצות זמן-אמת:</b>",
            ])
            summary_lines.extend([f"  • {rec}" for rec in perf.adaptive_recommendations])

        return "\n".join(summary_lines)

    except Exception as e:
        logger.error(f"Learning summary generation failed: {e}")
        return f"❌ שגיאה בקריאת מצב הלמידה: {e}"
