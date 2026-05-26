"""
Benchmark Comparison Engine
============================

Compare bot performance vs market benchmarks:
- S&P 500 (SPY)
- Nasdaq 100 (QQQ)
- Dow Jones (DIA)
- Russell 2000 (IWM)
- Sector ETFs

Generates "alpha" metrics - excess returns over benchmark.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkComparison:
    """Comparison vs a benchmark."""
    benchmark_ticker: str
    benchmark_name: str
    bot_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float  # Excess return
    beta: float
    correlation: float
    sharpe_difference: float
    win_rate_vs_market: str  # "outperforming" or "underperforming"
    information_ratio: float  # Risk-adjusted alpha
    tracking_error: float  # Volatility of difference


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK DATA
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000",
    "VTI": "Total Stock Market",
}


# ─────────────────────────────────────────────────────────────────────────────
# BOT PERFORMANCE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

async def get_bot_returns(days: int = 90) -> list:
    """Get bot's daily returns from trade history."""
    try:
        import database
        conn = database.get_connection()

        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Get daily P&L aggregated
        rows = conn.execute("""
            SELECT date(exit_time) as day, SUM(pnl_gross) as daily_pnl
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= ?
            GROUP BY day
            ORDER BY day
        """, (start_date,)).fetchall()

        # Convert to daily returns (assuming $10k starting capital)
        starting_capital = 10000
        current = starting_capital
        daily_returns = []

        for day, pnl in rows:
            if pnl:
                ret = pnl / current
                daily_returns.append(float(ret))
                current += pnl

        return daily_returns

    except Exception as e:
        logger.error(f"Failed to get bot returns: {e}")
        return []


async def get_benchmark_returns(benchmark_ticker: str, days: int = 90) -> list:
    """Get daily returns for a benchmark."""
    try:
        import yfinance as yf

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return []

        returns = data["Close"].pct_change().dropna()
        return [float(r) for r in returns]

    except Exception as e:
        logger.error(f"Failed to get benchmark returns for {benchmark_ticker}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA & BETA CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_alpha_beta(bot_returns: list, benchmark_returns: list) -> dict:
    """
    Calculate alpha and beta using CAPM.

    Alpha: Excess return over what beta would predict
    Beta: Sensitivity to benchmark movements
    """
    if len(bot_returns) < 5 or len(benchmark_returns) < 5:
        return {"alpha": 0, "beta": 0, "correlation": 0}

    # Align lengths
    min_len = min(len(bot_returns), len(benchmark_returns))
    bot = np.array(bot_returns[-min_len:])
    bench = np.array(benchmark_returns[-min_len:])

    # Beta = covariance / variance
    covariance = np.cov(bot, bench)[0, 1]
    variance = np.var(bench)
    beta = covariance / variance if variance > 0 else 0

    # Alpha = bot_return - (beta * benchmark_return)
    mean_bot = np.mean(bot)
    mean_bench = np.mean(bench)
    alpha = mean_bot - (beta * mean_bench)

    # Correlation
    correlation = np.corrcoef(bot, bench)[0, 1]
    correlation = float(np.nan_to_num(correlation))

    # Annualize alpha (252 trading days)
    annualized_alpha = alpha * 252

    return {
        "alpha": float(annualized_alpha),
        "beta": float(beta),
        "correlation": correlation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

async def compare_to_benchmark(benchmark_ticker: str = "SPY", days: int = 90) -> BenchmarkComparison:
    """Generate comprehensive comparison vs a benchmark."""
    try:
        # Get both return series
        bot_returns = await get_bot_returns(days)
        benchmark_returns = await get_benchmark_returns(benchmark_ticker, days)

        if not bot_returns or not benchmark_returns:
            return BenchmarkComparison(
                benchmark_ticker=benchmark_ticker,
                benchmark_name=BENCHMARKS.get(benchmark_ticker, benchmark_ticker),
                bot_return_pct=0,
                benchmark_return_pct=0,
                alpha_pct=0,
                beta=0,
                correlation=0,
                sharpe_difference=0,
                win_rate_vs_market="insufficient_data",
                information_ratio=0,
                tracking_error=0,
            )

        # Total returns
        bot_total = (np.prod([1 + r for r in bot_returns]) - 1) * 100
        bench_total = (np.prod([1 + r for r in benchmark_returns]) - 1) * 100

        # Alpha/Beta
        ab = calculate_alpha_beta(bot_returns, benchmark_returns)

        # Sharpe ratios
        risk_free_daily = 0.04 / 252  # 4% annual
        bot_sharpe = (np.mean(bot_returns) - risk_free_daily) / np.std(bot_returns) * np.sqrt(252) if np.std(bot_returns) > 0 else 0
        bench_sharpe = (np.mean(benchmark_returns) - risk_free_daily) / np.std(benchmark_returns) * np.sqrt(252) if np.std(benchmark_returns) > 0 else 0

        # Information ratio (alpha / tracking error)
        min_len = min(len(bot_returns), len(benchmark_returns))
        if min_len > 5:
            differences = np.array(bot_returns[-min_len:]) - np.array(benchmark_returns[-min_len:])
            tracking_error = float(np.std(differences) * np.sqrt(252) * 100)
            info_ratio = (ab["alpha"] * 100) / tracking_error if tracking_error > 0 else 0
        else:
            tracking_error = 0
            info_ratio = 0

        # Determine status
        if bot_total > bench_total + 5:
            status = "🟢 Strongly outperforming"
        elif bot_total > bench_total:
            status = "🟡 Outperforming"
        elif bot_total > bench_total - 5:
            status = "🟠 Roughly tracking"
        else:
            status = "🔴 Underperforming"

        return BenchmarkComparison(
            benchmark_ticker=benchmark_ticker,
            benchmark_name=BENCHMARKS.get(benchmark_ticker, benchmark_ticker),
            bot_return_pct=float(bot_total),
            benchmark_return_pct=float(bench_total),
            alpha_pct=ab["alpha"] * 100,
            beta=ab["beta"],
            correlation=ab["correlation"],
            sharpe_difference=float(bot_sharpe - bench_sharpe),
            win_rate_vs_market=status,
            information_ratio=float(info_ratio),
            tracking_error=tracking_error,
        )

    except Exception as e:
        logger.error(f"Benchmark comparison failed: {e}")
        return BenchmarkComparison(
            benchmark_ticker=benchmark_ticker,
            benchmark_name=BENCHMARKS.get(benchmark_ticker, benchmark_ticker),
            bot_return_pct=0, benchmark_return_pct=0, alpha_pct=0, beta=0,
            correlation=0, sharpe_difference=0,
            win_rate_vs_market=f"error: {e}",
            information_ratio=0, tracking_error=0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-BENCHMARK COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

async def compare_to_all_benchmarks(days: int = 90) -> dict:
    """Compare bot to all major benchmarks."""
    try:
        comparisons = {}

        for ticker, name in BENCHMARKS.items():
            comparison = await compare_to_benchmark(ticker, days)
            comparisons[ticker] = {
                "name": name,
                "bot_return": comparison.bot_return_pct,
                "benchmark_return": comparison.benchmark_return_pct,
                "alpha": comparison.alpha_pct,
                "beta": comparison.beta,
                "correlation": comparison.correlation,
                "status": comparison.win_rate_vs_market,
                "info_ratio": comparison.information_ratio,
            }

        # Find best/worst comparisons
        best_alpha = max(comparisons.values(), key=lambda x: x["alpha"])
        worst_alpha = min(comparisons.values(), key=lambda x: x["alpha"])

        return {
            "period_days": days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "comparisons": comparisons,
            "summary": {
                "best_relative_performance": {
                    "benchmark": best_alpha["name"],
                    "alpha": best_alpha["alpha"],
                    "status": best_alpha["status"],
                },
                "worst_relative_performance": {
                    "benchmark": worst_alpha["name"],
                    "alpha": worst_alpha["alpha"],
                    "status": worst_alpha["status"],
                },
                "primary_benchmark_alpha": comparisons.get("SPY", {}).get("alpha", 0),
            },
        }

    except Exception as e:
        logger.error(f"Multi-benchmark comparison failed: {e}")
        return {"error": str(e)}
