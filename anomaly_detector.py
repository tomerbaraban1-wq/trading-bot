"""
Anomaly Detection Module
=========================

Detects unusual market behavior and bot performance anomalies.

Features:
1. Statistical anomaly detection (Z-score based)
2. Unusual volume detection
3. Unusual price movement detection
4. Bot performance anomalies (sudden drawdown, etc.)
5. Correlation breaks (when usually-correlated stocks diverge)
6. Sector anomalies
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """A detected anomaly."""
    timestamp: str
    ticker: str
    anomaly_type: str  # "price", "volume", "correlation", "performance"
    severity: str      # "low", "medium", "high", "critical"
    z_score: float
    current_value: float
    expected_value: float
    deviation_pct: float
    description: str
    actionable: bool


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_z_score_anomaly(
    series: list[float],
    threshold: float = 3.0,
    window: int = 20,
) -> Optional[dict]:
    """
    Detect anomaly using Z-score (number of standard deviations from mean).

    Z-score > 3 = highly anomalous (only 0.27% of normal distribution).
    """
    if len(series) < window:
        return None

    historical = series[-window:-1]  # Excluding current
    current = series[-1]

    mean = np.mean(historical)
    std = np.std(historical)

    if std == 0:
        return None

    z_score = (current - mean) / std

    if abs(z_score) < threshold:
        return None

    # Determine severity
    if abs(z_score) > 5:
        severity = "critical"
    elif abs(z_score) > 4:
        severity = "high"
    elif abs(z_score) > 3:
        severity = "medium"
    else:
        severity = "low"

    return {
        "z_score": float(z_score),
        "current": float(current),
        "mean": float(mean),
        "std": float(std),
        "severity": severity,
        "deviation_pct": float((current - mean) / mean * 100) if mean else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRICE ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

async def detect_price_anomaly(ticker: str, threshold: float = 3.0) -> Optional[Anomaly]:
    """Detect unusual price movements."""
    try:
        import yfinance as yf

        # Get 30 days of daily data
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        data = yf.download(ticker, start=start, end=end, progress=False)

        if data.empty or len(data) < 20:
            return None

        # Calculate daily returns
        returns = data["Close"].pct_change().dropna().tolist()

        # Detect anomaly in recent return
        result = detect_z_score_anomaly(returns, threshold=threshold)

        if not result:
            return None

        direction = "up" if result["current"] > 0 else "down"
        return Anomaly(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            anomaly_type="price",
            severity=result["severity"],
            z_score=result["z_score"],
            current_value=result["current"] * 100,  # As percentage
            expected_value=result["mean"] * 100,
            deviation_pct=result["deviation_pct"],
            description=(
                f"Unusual price movement {direction}: "
                f"{result['current']*100:+.2f}% "
                f"(Z={result['z_score']:.1f}σ, normal: {result['mean']*100:+.2f}%)"
            ),
            actionable=result["severity"] in ("high", "critical"),
        )

    except Exception as e:
        logger.debug(f"Price anomaly detection failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

async def detect_volume_anomaly(ticker: str, threshold: float = 3.0) -> Optional[Anomaly]:
    """Detect unusual trading volume."""
    try:
        import yfinance as yf

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        data = yf.download(ticker, start=start, end=end, progress=False)

        if data.empty or len(data) < 20:
            return None

        volumes = data["Volume"].tolist()
        result = detect_z_score_anomaly(volumes, threshold=threshold)

        if not result:
            return None

        return Anomaly(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            anomaly_type="volume",
            severity=result["severity"],
            z_score=result["z_score"],
            current_value=result["current"],
            expected_value=result["mean"],
            deviation_pct=result["deviation_pct"],
            description=(
                f"Unusual volume: {result['current']:,.0f} "
                f"(Z={result['z_score']:.1f}σ, normal: {result['mean']:,.0f})"
            ),
            actionable=result["severity"] in ("high", "critical"),
        )

    except Exception as e:
        logger.debug(f"Volume anomaly detection failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANOMALY (Bot performance)
# ─────────────────────────────────────────────────────────────────────────────

async def detect_performance_anomaly() -> list[Anomaly]:
    """
    Detect anomalies in the bot's trading performance.

    Looks for:
    - Sudden drawdown spike
    - Win rate plummeting
    - Loss rate unusual
    """
    try:
        import database
        conn = database.get_connection()

        # Get recent trades
        rows = conn.execute("""
            SELECT date(exit_time) as day,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END) as wins,
                   AVG(pnl_gross) as avg_pnl
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= datetime('now', '-60 days')
            GROUP BY day
            ORDER BY day DESC
            LIMIT 30
        """).fetchall()

        if len(rows) < 7:
            return []

        anomalies = []

        # Daily win rates
        win_rates = []
        for day, trades, wins, avg_pnl in rows:
            if trades > 0:
                win_rates.append((wins / trades) * 100)

        if len(win_rates) >= 7:
            # Reverse so latest is at end
            win_rates_chronological = win_rates[::-1]
            result = detect_z_score_anomaly(win_rates_chronological, threshold=2.5)

            if result and result["current"] < result["mean"]:  # Only flag bad anomalies
                anomalies.append(Anomaly(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    ticker="PORTFOLIO",
                    anomaly_type="performance",
                    severity=result["severity"],
                    z_score=result["z_score"],
                    current_value=result["current"],
                    expected_value=result["mean"],
                    deviation_pct=result["deviation_pct"],
                    description=(
                        f"Win rate anomaly: today {result['current']:.1f}% "
                        f"vs normal {result['mean']:.1f}% (Z={result['z_score']:.1f}σ)"
                    ),
                    actionable=True,
                ))

        # Daily P&L
        pnls = []
        for day, trades, wins, avg_pnl in rows:
            if avg_pnl is not None:
                pnls.append(avg_pnl)

        if len(pnls) >= 7:
            pnls_chronological = pnls[::-1]
            result = detect_z_score_anomaly(pnls_chronological, threshold=2.5)

            if result and result["current"] < result["mean"]:  # Only flag bad
                anomalies.append(Anomaly(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    ticker="PORTFOLIO",
                    anomaly_type="performance",
                    severity=result["severity"],
                    z_score=result["z_score"],
                    current_value=result["current"],
                    expected_value=result["mean"],
                    deviation_pct=result["deviation_pct"],
                    description=(
                        f"P&L anomaly: today ${result['current']:.2f} "
                        f"vs normal ${result['mean']:.2f}"
                    ),
                    actionable=True,
                ))

        return anomalies

    except Exception as e:
        logger.error(f"Performance anomaly detection failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION BREAK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

async def detect_correlation_break(ticker1: str, ticker2: str, threshold: float = 2.5) -> Optional[Anomaly]:
    """
    Detect when two usually-correlated stocks diverge.

    E.g., AAPL and MSFT usually move together. If they suddenly diverge,
    it's a signal of company-specific news.
    """
    try:
        from pairs_trading import calculate_pair_correlation

        # Get recent correlation
        recent_data = await calculate_pair_correlation(ticker1, ticker2, days=30)
        if "error" in recent_data:
            return None

        # Get historical correlation (longer period)
        historical_data = await calculate_pair_correlation(ticker1, ticker2, days=180)
        if "error" in historical_data:
            return None

        recent_corr = recent_data["correlation"]
        historical_corr = historical_data["correlation"]

        # Significant break: correlation dropped significantly
        deviation = abs(recent_corr - historical_corr)

        if deviation > 0.3 and historical_corr > 0.7:
            severity = "critical" if deviation > 0.5 else "high" if deviation > 0.4 else "medium"

            return Anomaly(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ticker=f"{ticker1}/{ticker2}",
                anomaly_type="correlation",
                severity=severity,
                z_score=0,  # not applicable
                current_value=recent_corr,
                expected_value=historical_corr,
                deviation_pct=deviation * 100,
                description=(
                    f"Correlation break: {ticker1}/{ticker2} "
                    f"correlation dropped from {historical_corr:.2f} to {recent_corr:.2f}"
                ),
                actionable=True,
            )

        return None

    except Exception as e:
        logger.debug(f"Correlation break detection failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE ANOMALY SCAN
# ─────────────────────────────────────────────────────────────────────────────

async def scan_portfolio_anomalies() -> dict:
    """
    Scan all positions for various types of anomalies.
    """
    try:
        import broker

        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return {"anomalies": [], "count": 0}

        tickers = [p.symbol for p in positions[:10]]  # Limit
        anomalies = []

        # Scan each position
        for ticker in tickers:
            try:
                # Price anomaly
                price_anomaly = await detect_price_anomaly(ticker)
                if price_anomaly:
                    anomalies.append(price_anomaly)

                # Volume anomaly
                volume_anomaly = await detect_volume_anomaly(ticker)
                if volume_anomaly:
                    anomalies.append(volume_anomaly)

            except Exception as e:
                logger.debug(f"Anomaly scan failed for {ticker}: {e}")

        # Performance anomalies (portfolio-wide)
        perf_anomalies = await detect_performance_anomaly()
        anomalies.extend(perf_anomalies)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 99))

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_scanned": len(tickers),
            "count": len(anomalies),
            "critical_count": sum(1 for a in anomalies if a.severity == "critical"),
            "high_count": sum(1 for a in anomalies if a.severity == "high"),
            "anomalies": [
                {
                    "ticker": a.ticker,
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "z_score": a.z_score,
                    "deviation_pct": a.deviation_pct,
                    "description": a.description,
                    "actionable": a.actionable,
                }
                for a in anomalies
            ],
        }

    except Exception as e:
        logger.error(f"Portfolio anomaly scan failed: {e}")
        return {"error": str(e), "anomalies": [], "count": 0}
