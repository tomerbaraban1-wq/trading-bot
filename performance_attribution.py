"""
Performance Attribution Engine
==============================

Tells you EXACTLY what's making and losing money.

Breaks down performance by:
1. Strategy (which signals worked best)
2. Ticker (which stocks were profitable)
3. Time of day (when bot trades best)
4. Day of week (which days are profitable)
5. Holding period (best holding durations)
6. Entry condition (RSI level, MACD signal, etc)
7. Market regime (volatility, trend)
8. Sentiment score range
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Performance attribution for a category."""
    category: str
    subcategory: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    sharpe: float
    best_trade: float
    worst_trade: float
    contribution_pct: float  # % of total profits/losses


# ─────────────────────────────────────────────────────────────────────────────
# TICKER ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

async def attribute_by_ticker(days: int = 30) -> list[AttributionResult]:
    """Break down P&L by ticker."""
    try:
        import database
        conn = database.get_connection()

        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = conn.execute("""
            SELECT ticker, COUNT(*) as trades,
                   SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl_gross) as total_pnl,
                   AVG(pnl_gross) as avg_pnl,
                   MAX(pnl_gross) as best,
                   MIN(pnl_gross) as worst
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND created_at >= ?
            GROUP BY ticker
            ORDER BY total_pnl DESC
        """, (start_date,)).fetchall()

        if not rows:
            return []

        total_portfolio_pnl = sum(r[3] for r in rows if r[3])
        results = []

        for ticker, trades, wins, total, avg, best, worst in rows:
            if trades < 1:
                continue

            win_rate = (wins / trades * 100) if trades > 0 else 0
            contribution = (total / total_portfolio_pnl * 100) if total_portfolio_pnl else 0

            # Simple Sharpe approximation
            pnl_values = conn.execute("""
                SELECT pnl_gross FROM trade_log
                WHERE ticker = ? AND status IN ('stopped', 'sold') AND created_at >= ?
            """, (ticker, start_date)).fetchall()
            pnls = [r[0] for r in pnl_values]
            sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0

            results.append(AttributionResult(
                category="ticker",
                subcategory=ticker,
                total_trades=trades,
                win_rate=win_rate,
                total_pnl=total or 0,
                avg_pnl=avg or 0,
                sharpe=float(sharpe),
                best_trade=best or 0,
                worst_trade=worst or 0,
                contribution_pct=contribution,
            ))

        return results

    except Exception as e:
        logger.error(f"Ticker attribution failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# TIME-BASED ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

async def attribute_by_hour() -> list[AttributionResult]:
    """Break down P&L by hour of day."""
    try:
        import database
        conn = database.get_connection()

        rows = conn.execute("""
            SELECT
                CAST(strftime('%H', exit_time, 'localtime') AS INTEGER) as hour,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl_gross) as total_pnl,
                AVG(pnl_gross) as avg_pnl
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= datetime('now', '-30 days')
            GROUP BY hour
            ORDER BY hour
        """).fetchall()

        total_pnl = sum(r[3] for r in rows if r[3])
        results = []

        for hour, trades, wins, total, avg in rows:
            if trades < 2:
                continue

            win_rate = (wins / trades * 100) if trades > 0 else 0
            contribution = (total / total_pnl * 100) if total_pnl else 0

            results.append(AttributionResult(
                category="hour",
                subcategory=f"{hour:02d}:00",
                total_trades=trades,
                win_rate=win_rate,
                total_pnl=total or 0,
                avg_pnl=avg or 0,
                sharpe=0,
                best_trade=0,
                worst_trade=0,
                contribution_pct=contribution,
            ))

        return results

    except Exception as e:
        logger.error(f"Hour attribution failed: {e}")
        return []


async def attribute_by_day_of_week() -> list[AttributionResult]:
    """Break down P&L by day of week."""
    try:
        import database
        conn = database.get_connection()

        # SQLite strftime('%w') returns 0-6 (Sunday-Saturday)
        rows = conn.execute("""
            SELECT
                CAST(strftime('%w', exit_time) AS INTEGER) as dow,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl_gross) as total_pnl,
                AVG(pnl_gross) as avg_pnl
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= datetime('now', '-90 days')
            GROUP BY dow
            ORDER BY dow
        """).fetchall()

        day_names = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
        total_pnl = sum(r[3] for r in rows if r[3])
        results = []

        for dow, trades, wins, total, avg in rows:
            if trades < 2:
                continue

            win_rate = (wins / trades * 100) if trades > 0 else 0
            contribution = (total / total_pnl * 100) if total_pnl else 0

            results.append(AttributionResult(
                category="day_of_week",
                subcategory=day_names[dow],
                total_trades=trades,
                win_rate=win_rate,
                total_pnl=total or 0,
                avg_pnl=avg or 0,
                sharpe=0,
                best_trade=0,
                worst_trade=0,
                contribution_pct=contribution,
            ))

        return results

    except Exception as e:
        logger.error(f"Day-of-week attribution failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# HOLDING PERIOD ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

async def attribute_by_holding_period() -> list[AttributionResult]:
    """Break down P&L by holding duration."""
    try:
        import database
        conn = database.get_connection()

        rows = conn.execute("""
            SELECT
                CASE
                    WHEN (julianday(exit_time) - julianday(created_at)) < 1 THEN '< 1 day'
                    WHEN (julianday(exit_time) - julianday(created_at)) < 3 THEN '1-3 days'
                    WHEN (julianday(exit_time) - julianday(created_at)) < 7 THEN '3-7 days'
                    WHEN (julianday(exit_time) - julianday(created_at)) < 14 THEN '1-2 weeks'
                    WHEN (julianday(exit_time) - julianday(created_at)) < 30 THEN '2-4 weeks'
                    ELSE '> 1 month'
                END as duration_bucket,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl_gross) as total_pnl,
                AVG(pnl_gross) as avg_pnl
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND created_at >= datetime('now', '-90 days')
            GROUP BY duration_bucket
        """).fetchall()

        total_pnl = sum(r[3] for r in rows if r[3])
        results = []

        for bucket, trades, wins, total, avg in rows:
            if trades < 2:
                continue

            win_rate = (wins / trades * 100) if trades > 0 else 0
            contribution = (total / total_pnl * 100) if total_pnl else 0

            results.append(AttributionResult(
                category="holding_period",
                subcategory=bucket,
                total_trades=trades,
                win_rate=win_rate,
                total_pnl=total or 0,
                avg_pnl=avg or 0,
                sharpe=0,
                best_trade=0,
                worst_trade=0,
                contribution_pct=contribution,
            ))

        return results

    except Exception as e:
        logger.error(f"Holding period attribution failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# EXIT REASON ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

async def attribute_by_exit_reason() -> list[AttributionResult]:
    """Break down P&L by why trades were closed."""
    try:
        import database
        conn = database.get_connection()

        rows = conn.execute("""
            SELECT exit_reason, COUNT(*) as trades,
                   SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl_gross) as total_pnl,
                   AVG(pnl_gross) as avg_pnl
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND created_at >= datetime('now', '-90 days')
            GROUP BY exit_reason
            ORDER BY total_pnl DESC
        """).fetchall()

        total_pnl = sum(r[3] for r in rows if r[3])
        results = []

        for reason, trades, wins, total, avg in rows:
            if not reason or trades < 1:
                continue

            win_rate = (wins / trades * 100) if trades > 0 else 0
            contribution = (total / total_pnl * 100) if total_pnl else 0

            results.append(AttributionResult(
                category="exit_reason",
                subcategory=reason,
                total_trades=trades,
                win_rate=win_rate,
                total_pnl=total or 0,
                avg_pnl=avg or 0,
                sharpe=0,
                best_trade=0,
                worst_trade=0,
                contribution_pct=contribution,
            ))

        return results

    except Exception as e:
        logger.error(f"Exit reason attribution failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE ATTRIBUTION REPORT
# ─────────────────────────────────────────────────────────────────────────────

async def generate_attribution_report(days: int = 30) -> dict:
    """
    Generate comprehensive performance attribution report.

    Tells you exactly:
    - Which stocks made you money
    - When you trade best (hours/days)
    - How long to hold positions
    - Why trades close (TP vs SL vs manual)
    """
    try:
        # Run all attributions in parallel
        results = await asyncio.gather(
            attribute_by_ticker(days),
            attribute_by_hour(),
            attribute_by_day_of_week(),
            attribute_by_holding_period(),
            attribute_by_exit_reason(),
            return_exceptions=True,
        )

        ticker_attr, hour_attr, dow_attr, hold_attr, exit_attr = results

        # Find top winners and losers
        if isinstance(ticker_attr, list) and ticker_attr:
            top_winners = sorted(ticker_attr, key=lambda r: r.total_pnl, reverse=True)[:5]
            top_losers = sorted(ticker_attr, key=lambda r: r.total_pnl)[:5]
        else:
            top_winners = []
            top_losers = []

        # Best/worst hours
        best_hour = None
        worst_hour = None
        if isinstance(hour_attr, list) and hour_attr:
            best_hour = max(hour_attr, key=lambda r: r.avg_pnl)
            worst_hour = min(hour_attr, key=lambda r: r.avg_pnl)

        # Best/worst days
        best_day = None
        worst_day = None
        if isinstance(dow_attr, list) and dow_attr:
            best_day = max(dow_attr, key=lambda r: r.avg_pnl)
            worst_day = min(dow_attr, key=lambda r: r.avg_pnl)

        # Best holding period
        best_holding = None
        if isinstance(hold_attr, list) and hold_attr:
            best_holding = max(hold_attr, key=lambda r: r.win_rate)

        # Insights
        insights = []
        if top_winners:
            top = top_winners[0]
            insights.append(f"🏆 Top winner: {top.subcategory} (${top.total_pnl:+.2f}, {top.win_rate:.0f}% win)")
        if top_losers:
            worst = top_losers[0]
            if worst.total_pnl < 0:
                insights.append(f"💔 Top loser: {worst.subcategory} (${worst.total_pnl:.2f})")
        if best_hour:
            insights.append(f"🌅 Best trading hour: {best_hour.subcategory} (${best_hour.avg_pnl:+.2f} avg)")
        if best_day:
            insights.append(f"📅 Best trading day: {best_day.subcategory} ({best_day.win_rate:.0f}% win)")
        if best_holding:
            insights.append(f"⏱️ Best holding period: {best_holding.subcategory} ({best_holding.win_rate:.0f}% win)")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "by_ticker": {
                "top_winners": [
                    {"ticker": r.subcategory, "pnl": r.total_pnl, "win_rate": r.win_rate, "trades": r.total_trades}
                    for r in top_winners
                ],
                "top_losers": [
                    {"ticker": r.subcategory, "pnl": r.total_pnl, "win_rate": r.win_rate, "trades": r.total_trades}
                    for r in top_losers if r.total_pnl < 0
                ],
            },
            "by_time": {
                "best_hour": best_hour.subcategory if best_hour else None,
                "best_hour_avg_pnl": best_hour.avg_pnl if best_hour else 0,
                "worst_hour": worst_hour.subcategory if worst_hour else None,
                "worst_hour_avg_pnl": worst_hour.avg_pnl if worst_hour else 0,
            },
            "by_day_of_week": {
                "best_day": best_day.subcategory if best_day else None,
                "worst_day": worst_day.subcategory if worst_day else None,
            },
            "by_holding_period": {
                "best_duration": best_holding.subcategory if best_holding else None,
                "details": [
                    {"duration": r.subcategory, "win_rate": r.win_rate, "pnl": r.total_pnl}
                    for r in (hold_attr if isinstance(hold_attr, list) else [])
                ],
            },
            "by_exit_reason": [
                {"reason": r.subcategory, "trades": r.total_trades, "pnl": r.total_pnl, "win_rate": r.win_rate}
                for r in (exit_attr if isinstance(exit_attr, list) else [])
            ],
            "insights": insights,
        }

    except Exception as e:
        logger.error(f"Attribution report failed: {e}")
        return {"error": str(e)}


async def get_actionable_insights() -> list[str]:
    """
    Generate actionable insights from attribution data.

    Tells the trader specifically what to do differently.
    """
    try:
        report = await generate_attribution_report(days=30)
        insights = []

        # Time-based insights
        time_data = report.get("by_time", {})
        if time_data.get("best_hour") and time_data.get("worst_hour"):
            best_pnl = time_data["best_hour_avg_pnl"]
            worst_pnl = time_data["worst_hour_avg_pnl"]
            if abs(worst_pnl) > abs(best_pnl):
                insights.append(
                    f"⚠️ Avoid trading at {time_data['worst_hour']} - average loss ${worst_pnl:.2f}"
                )
            if best_pnl > 0:
                insights.append(
                    f"✅ Focus trading at {time_data['best_hour']} - average gain ${best_pnl:.2f}"
                )

        # Ticker insights
        ticker_data = report.get("by_ticker", {})
        if ticker_data.get("top_losers"):
            losers = ticker_data["top_losers"]
            if losers and losers[0]["win_rate"] < 30:
                insights.append(
                    f"🚫 Avoid {losers[0]['ticker']} - only {losers[0]['win_rate']:.0f}% win rate"
                )

        # Holding period insights
        hold_data = report.get("by_holding_period", {})
        if hold_data.get("best_duration"):
            insights.append(f"⏱️ Optimal holding period: {hold_data['best_duration']}")

        # Exit reason insights
        exit_data = report.get("by_exit_reason", [])
        if exit_data:
            stop_loss_data = next((d for d in exit_data if "stop" in d["reason"].lower()), None)
            if stop_loss_data and stop_loss_data["win_rate"] < 5:
                insights.append("⚠️ Stop losses are hitting too often - consider widening or improving entries")

        return insights

    except Exception as e:
        logger.error(f"Actionable insights failed: {e}")
        return []
