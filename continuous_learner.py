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
        # Keep ONE row per error_type. Previously this INSERTed the same patterns
        # every learning cycle with no dedup, which bloated the table with hundreds
        # of identical rows (e.g. 'unclassified_loss' ×100s). Collapse any existing
        # duplicates, enforce uniqueness, then UPSERT the current snapshot.
        conn.execute(
            "DELETE FROM error_patterns WHERE id NOT IN "
            "(SELECT MAX(id) FROM error_patterns GROUP BY error_type)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_error_patterns_type "
            "ON error_patterns(error_type)"
        )
        for pattern in patterns[:5]:
            conn.execute(
                """INSERT INTO error_patterns
                       (error_type, frequency, avg_loss, severity, suggested_fix, discovered_at)
                   VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)
                   ON CONFLICT(error_type) DO UPDATE SET
                       frequency     = excluded.frequency,
                       avg_loss      = excluded.avg_loss,
                       severity      = excluded.severity,
                       suggested_fix = excluded.suggested_fix,
                       discovered_at = CURRENT_TIMESTAMP""",
                (pattern.error_type, pattern.frequency, pattern.avg_loss,
                 pattern.severity, pattern.suggested_fix)
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


# Entry sentiment >= this (out of 10) counts as "bullish news at entry"
SENTIMENT_BULLISH_THRESHOLD = float(os.getenv("SENTIMENT_BULLISH_THRESHOLD", "6"))
# Need at least this many trades before a correlation is meaningful
SENTIMENT_MIN_SAMPLES = int(os.getenv("SENTIMENT_MIN_SAMPLES", "5"))


def learn_sentiment_correlation() -> dict[str, SentimentCorrelation]:
    """
    REAL measurement (no more placeholder): did a higher entry sentiment score
    actually lead to better trade outcomes?

    Uses the sentiment_score logged on every trade at entry
    (trade_log.sentiment_score, 0-10) vs. the realized pnl_gross — i.e. it
    answers "when we bought on good news, did the price actually reward us?"

    Honesty note: this only covers stocks we actually BOUGHT (we have no
    outcome for stocks we skipped), so there is selection bias — but it is
    REAL measured data from the past, not assumed numbers.
    """
    try:
        import database
        from collections import defaultdict
        conn = database.get_connection()

        # Every trade that is no longer open and has a logged entry sentiment.
        # (Old code filtered status IN ('stopped','sold') — status values this bot
        #  actually writes are closed/time_exit/stop_loss/..., so it matched nothing.)
        rows = conn.execute("""
            SELECT ticker, sentiment_score, pnl_gross
            FROM trade_log
            WHERE status != 'open'
              AND sentiment_score IS NOT NULL
              AND pnl_gross   IS NOT NULL
        """).fetchall()

        by_ticker: dict[str, list[tuple[float, float]]] = defaultdict(list)
        all_samples: list[tuple[float, float]] = []
        for ticker, sent, pnl in rows:
            try:
                s, p = float(sent), float(pnl)
            except (TypeError, ValueError):
                continue
            by_ticker[ticker].append((s, p))
            all_samples.append((s, p))

        def _measure(name: str, data: list[tuple[float, float]]) -> SentimentCorrelation | None:
            if len(data) < SENTIMENT_MIN_SAMPLES:
                return None

            sentiments = np.array([d[0] for d in data], dtype=float)
            wins       = np.array([1.0 if d[1] > 0 else 0.0 for d in data], dtype=float)

            bullish  = [p for s, p in data if s >= SENTIMENT_BULLISH_THRESHOLD]
            non_bull = [p for s, p in data if s <  SENTIMENT_BULLISH_THRESHOLD]
            wr_with    = (sum(1 for p in bullish  if p > 0) / len(bullish)  * 100) if bullish  else 0.0
            wr_without = (sum(1 for p in non_bull if p > 0) / len(non_bull) * 100) if non_bull else 0.0

            # Real point-biserial correlation between sentiment and win/loss.
            # corrcoef is undefined if either series has zero variance.
            if sentiments.std() > 0 and wins.std() > 0:
                corr = float(np.corrcoef(sentiments, wins)[0, 1])
            else:
                corr = 0.0

            if corr >= 0.2:
                rec = "✅ הסנטימנט עוזר באמת — אפשר לסמוך עליו"
            elif corr <= -0.2:
                rec = "🔴 סנטימנט גבוה דווקא קשור להפסד — להפחית משקל"
            elif abs(corr) < 0.1:
                rec = "⚪ לסנטימנט אין יתרון מדיד — לא מנבא תוצאה"
            else:
                rec = "🟡 קשר חלש — צריך עוד נתונים"

            return SentimentCorrelation(
                ticker=name,
                sentiment_threshold=SENTIMENT_BULLISH_THRESHOLD,
                win_rate_with_sentiment=round(wr_with, 1),
                win_rate_without_sentiment=round(wr_without, 1),
                correlation_strength=round(abs(corr), 3),
                sample_count=len(data),
                recommendation=rec,
            )

        results: dict[str, SentimentCorrelation] = {}

        # Overall across all trades — the most statistically meaningful figure.
        overall = _measure("OVERALL", all_samples)
        if overall:
            results["OVERALL"] = overall

        # Per-ticker, where we have enough samples.
        for ticker, data in by_ticker.items():
            c = _measure(ticker, data)
            if c:
                results[ticker] = c

        if "OVERALL" in results:
            ov = results["OVERALL"]
            logger.info(
                f"[SENTIMENT LEARNER] REAL: {len(all_samples)} trades | "
                f"corr={ov.correlation_strength:.2f} | "
                f"WR bullish={ov.win_rate_with_sentiment:.0f}% vs "
                f"non={ov.win_rate_without_sentiment:.0f}% | {ov.recommendation}"
            )
        else:
            logger.info(
                f"[SENTIMENT LEARNER] Not enough trades with logged sentiment yet "
                f"({len(all_samples)} samples, need >={SENTIMENT_MIN_SAMPLES})"
            )

        return results

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
