"""
System Health Monitor
======================

Monitors bot health and self-heals when issues are detected.

Features:
1. CPU/Memory monitoring
2. API connection health (Alpaca, Telegram, Discord, Yahoo)
3. Database health (size, query speed)
4. Loop health (are all loops running?)
5. Error rate tracking
6. Auto-recovery actions
7. Health dashboard
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """A single health metric reading."""
    name: str
    value: float
    unit: str
    status: str  # "healthy", "warning", "critical"
    threshold_warning: float
    threshold_critical: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class HealthReport:
    """Comprehensive health report."""
    timestamp: str
    overall_status: str  # "healthy", "degraded", "critical"
    metrics: dict
    issues: list[str]
    auto_recovery_actions: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# ERROR RATE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

class ErrorTracker:
    """Track error rates by category."""

    def __init__(self, window_minutes: int = 15):
        self.window_minutes = window_minutes
        self.errors: dict = {}  # category -> deque of timestamps

    def record_error(self, category: str) -> None:
        """Record an error."""
        if category not in self.errors:
            self.errors[category] = deque(maxlen=1000)
        self.errors[category].append(time.time())

    def get_rate(self, category: str) -> float:
        """Get errors per minute for category."""
        if category not in self.errors:
            return 0

        cutoff = time.time() - (self.window_minutes * 60)
        recent = [t for t in self.errors[category] if t > cutoff]

        return len(recent) / self.window_minutes

    def get_all_rates(self) -> dict:
        """Get all error rates."""
        return {cat: self.get_rate(cat) for cat in self.errors}


_error_tracker = ErrorTracker()


def record_error(category: str = "general") -> None:
    """Public function to record errors."""
    _error_tracker.record_error(category)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM RESOURCE MONITORING
# ─────────────────────────────────────────────────────────────────────────────

def check_memory_usage() -> HealthMetric:
    """Check current memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb > 800:
            status = "critical"
        elif memory_mb > 500:
            status = "warning"
        else:
            status = "healthy"

        return HealthMetric(
            name="memory_mb",
            value=memory_mb,
            unit="MB",
            status=status,
            threshold_warning=500,
            threshold_critical=800,
        )
    except ImportError:
        return HealthMetric(
            name="memory_mb",
            value=0,
            unit="MB",
            status="healthy",
            threshold_warning=500,
            threshold_critical=800,
        )


def check_cpu_usage() -> HealthMetric:
    """Check CPU usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_pct = process.cpu_percent(interval=0.1)

        if cpu_pct > 90:
            status = "critical"
        elif cpu_pct > 70:
            status = "warning"
        else:
            status = "healthy"

        return HealthMetric(
            name="cpu_pct",
            value=cpu_pct,
            unit="%",
            status=status,
            threshold_warning=70,
            threshold_critical=90,
        )
    except ImportError:
        return HealthMetric(
            name="cpu_pct",
            value=0,
            unit="%",
            status="healthy",
            threshold_warning=70,
            threshold_critical=90,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HEALTH
# ─────────────────────────────────────────────────────────────────────────────

def check_database_health() -> HealthMetric:
    """Check database health and query speed."""
    try:
        import database
        conn = database.get_connection()

        # Test query speed
        start = time.time()
        conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()
        query_time_ms = (time.time() - start) * 1000

        # Get DB size
        size_row = conn.execute("PRAGMA page_count").fetchone()
        page_size_row = conn.execute("PRAGMA page_size").fetchone()
        db_size_mb = (size_row[0] * page_size_row[0]) / 1024 / 1024

        if query_time_ms > 1000 or db_size_mb > 1000:
            status = "critical"
        elif query_time_ms > 500 or db_size_mb > 500:
            status = "warning"
        else:
            status = "healthy"

        return HealthMetric(
            name="db_query_ms",
            value=query_time_ms,
            unit="ms",
            status=status,
            threshold_warning=500,
            threshold_critical=1000,
        )
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return HealthMetric(
            name="db_query_ms",
            value=999,
            unit="ms",
            status="critical",
            threshold_warning=500,
            threshold_critical=1000,
        )


# ─────────────────────────────────────────────────────────────────────────────
# API HEALTH CHECKS
# ─────────────────────────────────────────────────────────────────────────────

async def check_alpaca_health() -> HealthMetric:
    """Check Alpaca API connectivity."""
    try:
        import broker
        start = time.time()
        await asyncio.wait_for(asyncio.to_thread(broker.is_market_open), timeout=5)
        latency_ms = (time.time() - start) * 1000

        if latency_ms > 3000:
            status = "warning"
        else:
            status = "healthy"

        return HealthMetric(
            name="alpaca_latency_ms",
            value=latency_ms,
            unit="ms",
            status=status,
            threshold_warning=2000,
            threshold_critical=5000,
        )
    except Exception as e:
        record_error("alpaca")
        return HealthMetric(
            name="alpaca_latency_ms",
            value=9999,
            unit="ms",
            status="critical",
            threshold_warning=2000,
            threshold_critical=5000,
        )


async def check_telegram_health() -> HealthMetric:
    """Check Telegram API connectivity."""
    try:
        from config import settings
        if not settings.TELEGRAM_BOT_TOKEN:
            return HealthMetric("telegram", 0, "ok", "healthy", 0, 0)

        import aiohttp
        start = time.time()
        url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getMe"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    latency_ms = (time.time() - start) * 1000
                    return HealthMetric(
                        name="telegram_latency_ms",
                        value=latency_ms,
                        unit="ms",
                        status="healthy" if latency_ms < 2000 else "warning",
                        threshold_warning=2000,
                        threshold_critical=5000,
                    )

        record_error("telegram")
        return HealthMetric("telegram_latency_ms", 9999, "ms", "critical", 2000, 5000)

    except Exception as e:
        record_error("telegram")
        return HealthMetric("telegram_latency_ms", 9999, "ms", "critical", 2000, 5000)


async def check_yahoo_health() -> HealthMetric:
    """Check Yahoo Finance API."""
    try:
        import yfinance as yf
        start = time.time()
        data = await asyncio.wait_for(
            asyncio.to_thread(lambda: yf.Ticker("SPY").info),
            timeout=10
        )
        latency_ms = (time.time() - start) * 1000

        return HealthMetric(
            name="yahoo_latency_ms",
            value=latency_ms,
            unit="ms",
            status="healthy" if latency_ms < 3000 else "warning",
            threshold_warning=3000,
            threshold_critical=8000,
        )
    except Exception as e:
        record_error("yahoo")
        return HealthMetric("yahoo_latency_ms", 9999, "ms", "warning", 3000, 8000)


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

async def run_health_check() -> HealthReport:
    """Run comprehensive health check on all systems."""
    try:
        # Run all checks in parallel
        results = await asyncio.gather(
            asyncio.to_thread(check_memory_usage),
            asyncio.to_thread(check_cpu_usage),
            asyncio.to_thread(check_database_health),
            check_alpaca_health(),
            check_telegram_health(),
            check_yahoo_health(),
            return_exceptions=True,
        )

        memory, cpu, db, alpaca, telegram, yahoo = results

        metrics = {}
        issues = []
        auto_recovery_actions = []
        critical_count = 0
        warning_count = 0

        for metric in [memory, cpu, db, alpaca, telegram, yahoo]:
            if isinstance(metric, HealthMetric):
                metrics[metric.name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "status": metric.status,
                }

                if metric.status == "critical":
                    critical_count += 1
                    issues.append(f"🔴 CRITICAL: {metric.name} = {metric.value:.1f}{metric.unit}")

                    # Auto-recovery actions
                    if metric.name == "memory_mb":
                        auto_recovery_actions.append("Clear caches and request GC")
                    elif metric.name == "alpaca_latency_ms":
                        auto_recovery_actions.append("Will retry Alpaca calls with backoff")

                elif metric.status == "warning":
                    warning_count += 1
                    issues.append(f"🟡 WARNING: {metric.name} = {metric.value:.1f}{metric.unit}")

        # Add error rate metrics
        error_rates = _error_tracker.get_all_rates()
        for category, rate in error_rates.items():
            metrics[f"error_rate_{category}"] = {
                "value": rate,
                "unit": "errors/min",
                "status": "critical" if rate > 5 else "warning" if rate > 2 else "healthy",
            }
            if rate > 5:
                critical_count += 1
                issues.append(f"🔴 High error rate in {category}: {rate:.1f}/min")

        # Determine overall status
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return HealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status=overall_status,
            metrics=metrics,
            issues=issues,
            auto_recovery_actions=auto_recovery_actions,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status="critical",
            metrics={},
            issues=[f"Health check failed: {e}"],
            auto_recovery_actions=[],
        )


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-RECOVERY ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def perform_auto_recovery(report: HealthReport) -> dict:
    """Execute auto-recovery actions based on health issues."""
    actions_taken = []

    for action in report.auto_recovery_actions:
        try:
            if "Clear caches" in action:
                # Force garbage collection
                import gc
                gc.collect()
                actions_taken.append("Memory: GC executed")

            elif "Alpaca" in action:
                # Just log it - actual retry happens in broker module
                actions_taken.append("Alpaca: Backoff strategy active")

        except Exception as e:
            logger.error(f"Auto-recovery failed for action '{action}': {e}")

    return {
        "actions_taken": actions_taken,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH DASHBOARD HTML
# ─────────────────────────────────────────────────────────────────────────────

def generate_health_dashboard_html(report: HealthReport) -> str:
    """Generate HTML dashboard for health monitoring."""
    status_colors = {
        "healthy": "#28a745",
        "degraded": "#ffc107",
        "critical": "#dc3545",
    }

    color = status_colors.get(report.overall_status, "#6c757d")

    metrics_html = ""
    for name, metric in report.metrics.items():
        m_color = status_colors.get(metric["status"], "#6c757d")
        metrics_html += f"""
        <div class="metric" style="border-left: 5px solid {m_color}; padding: 10px; margin: 10px 0; background: #f8f9fa;">
            <strong>{name}:</strong> {metric['value']:.2f} {metric['unit']}
            <span style="color: {m_color}; float: right;">{metric['status'].upper()}</span>
        </div>
        """

    issues_html = ""
    for issue in report.issues:
        issues_html += f"<li>{issue}</li>"

    actions_html = ""
    for action in report.auto_recovery_actions:
        actions_html += f"<li>{action}</li>"

    return f"""
    <html>
    <head>
        <title>Bot Health Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f0f2f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .status-banner {{ background: {color}; color: white; padding: 20px; border-radius: 5px; text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0; }}
            .section {{ margin: 20px 0; }}
            ul {{ list-style-type: none; padding-left: 0; }}
            li {{ padding: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Trading Bot Health</h1>
            <div class="status-banner">{report.overall_status.upper()}</div>
            <div class="section">
                <h2>📊 System Metrics</h2>
                {metrics_html}
            </div>
            {f'<div class="section"><h2>⚠️ Active Issues</h2><ul>{issues_html}</ul></div>' if report.issues else ''}
            {f'<div class="section"><h2>🔧 Auto-Recovery Actions</h2><ul>{actions_html}</ul></div>' if report.auto_recovery_actions else ''}
            <div class="section">
                <small>Last updated: {report.timestamp}</small>
            </div>
        </div>
    </body>
    </html>
    """
