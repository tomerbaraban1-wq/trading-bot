"""
Performance Analytics — Hedge Fund KPI Engine

Computes the three metrics that institutional desks use to judge any strategy:

  1. Sharpe Ratio  — risk-adjusted return
        = (mean_daily_return - rf_daily) / std_daily_return × √252
        Benchmark: > 1.0 is acceptable, > 2.0 is good, > 3.0 is excellent

  2. Maximum Drawdown — worst peak-to-trough equity decline (%)
        = max((peak_equity - trough_equity) / peak_equity × 100)
        Benchmark: < 10% is conservative, < 20% is normal for equity strategies

  3. Win Rate per Strategy — % of trades closed profitably, broken down by exit type
        Strategies: stop_loss | take_profit | smart_sell | emergency_exit | closed

All metrics are computed from the local SQLite trade_log.
No broker API calls — always available even when market is closed.

Public API
----------
  compute(weeks=4)            → PerformanceReport (dict-like)
  export_csv(report, path)    → str  (path of written file)
  format_telegram(report)     → str  (HTML for Telegram)

Environment vars
----------------
  PERF_RISK_FREE_RATE  float  default 0.045  (annualised, e.g. US 10-Y T-bill)
"""

import csv
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import settings
import database

logger = logging.getLogger(__name__)

RISK_FREE_RATE: float = float(os.getenv("PERF_RISK_FREE_RATE", "0.045"))
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyStats:
    strategy:    str
    total:       int   = 0
    wins:        int   = 0
    losses:      int   = 0
    win_rate:    float = 0.0   # %
    total_pnl:   float = 0.0
    avg_pnl:     float = 0.0
    best_trade:  float = 0.0
    worst_trade: float = 0.0

    def finalise(self):
        self.win_rate   = round(self.wins / self.total * 100, 1) if self.total else 0.0
        self.avg_pnl    = round(self.total_pnl / self.total, 2) if self.total else 0.0
        self.total_pnl  = round(self.total_pnl, 2)
        self.best_trade = round(self.best_trade, 2)
        self.worst_trade = round(self.worst_trade, 2)


@dataclass
class PerformanceReport:
    generated_at:   str   = ""
    period_weeks:   int   = 4
    period_start:   str   = ""
    period_end:     str   = ""

    # Core KPIs
    sharpe_ratio:    float | None = None   # None = insufficient data (< 2 trading days)
    max_drawdown_pct: float       = 0.0   # always computed; 0 if no trades

    # Aggregate
    total_trades:    int   = 0
    total_wins:      int   = 0
    total_losses:    int   = 0
    overall_win_rate: float = 0.0   # %
    total_pnl_gross: float = 0.0
    total_pnl_net:   float = 0.0
    avg_pnl_per_trade: float = 0.0
    best_trade:      float = 0.0
    worst_trade:     float = 0.0

    # Per-strategy breakdown
    by_strategy: list[StrategyStats] = field(default_factory=list)

    # Equity curve (daily, for charting)
    daily_equity: list[dict] = field(default_factory=list)  # [{date, equity, pnl}]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["by_strategy"] = [asdict(s) for s in self.by_strategy]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute(weeks: int = 4) -> PerformanceReport:
    """
    Compute all performance KPIs for the last `weeks` calendar weeks.

    Parameters
    ----------
    weeks : how many weeks of history to analyse (default 4 = ~1 month)

    Returns
    -------
    PerformanceReport — fully populated, safe to serialise as JSON or CSV
    """
    now   = datetime.now(timezone.utc)
    since = now - timedelta(weeks=weeks)

    # Pull all closed trades in the window
    trades = _fetch_closed_trades(since)

    report = PerformanceReport(
        generated_at = now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        period_weeks = weeks,
        period_start = since.strftime("%Y-%m-%d"),
        period_end   = now.strftime("%Y-%m-%d"),
    )

    if not trades:
        logger.info(f"[PERF] No closed trades in the last {weeks} weeks")
        return report

    # ── Sort by exit_time ─────────────────────────────────────────────────────
    trades.sort(key=lambda t: t.get("exit_time") or "")

    # ── Aggregate totals ──────────────────────────────────────────────────────
    pnl_values = [t["pnl_gross"] for t in trades if t.get("pnl_gross") is not None]
    net_values = [t["pnl_net"]   for t in trades if t.get("pnl_net")   is not None]

    report.total_trades     = len(trades)
    report.total_wins       = sum(1 for p in pnl_values if p > 0)
    report.total_losses     = sum(1 for p in pnl_values if p <= 0)
    report.overall_win_rate = round(report.total_wins / report.total_trades * 100, 1)
    report.total_pnl_gross  = round(sum(pnl_values), 2)
    report.total_pnl_net    = round(sum(net_values), 2)
    report.avg_pnl_per_trade = round(report.total_pnl_gross / report.total_trades, 2)
    report.best_trade        = round(max(pnl_values), 2)
    report.worst_trade       = round(min(pnl_values), 2)

    # ── Equity curve ──────────────────────────────────────────────────────────
    daily = _build_daily_series(trades)
    report.daily_equity = daily

    # ── Sharpe Ratio ─────────────────────────────────────────────────────────
    report.sharpe_ratio = _compute_sharpe(daily)

    # ── Maximum Drawdown ─────────────────────────────────────────────────────
    report.max_drawdown_pct = _compute_max_drawdown(daily)

    # ── Per-strategy breakdown ────────────────────────────────────────────────
    report.by_strategy = _compute_strategy_stats(trades)

    logger.info(
        f"[PERF] {report.total_trades} trades | "
        f"Sharpe={report.sharpe_ratio} | "
        f"MaxDD={report.max_drawdown_pct:.2f}% | "
        f"WinRate={report.overall_win_rate:.1f}%"
    )
    return report


def export_csv(report: PerformanceReport, output_dir: str | None = None) -> str:
    """
    Write two CSV files to output_dir:
      1. trades_YYYYMMDD.csv  — one row per trade
      2. summary_YYYYMMDD.csv — KPI summary + per-strategy table

    Returns the path of the summary file.
    """
    output_dir = output_dir or str(Path(settings.DATABASE_PATH).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tag   = datetime.now(timezone.utc).strftime("%Y%m%d")
    since = (datetime.now(timezone.utc) - timedelta(weeks=report.period_weeks))
    trades = _fetch_closed_trades(since)

    # ── 1. Trades CSV ─────────────────────────────────────────────────────────
    trades_path = str(Path(output_dir) / f"trades_{tag}.csv")
    trade_fields = [
        "id", "ticker", "action", "qty",
        "entry_price", "entry_time",
        "exit_price",  "exit_time",
        "pnl_gross", "pnl_net", "tax_reserved",
        "status", "sentiment_score",
        "rsi", "macd",
    ]
    with open(trades_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trade_fields, extrasaction="ignore")
        writer.writeheader()
        for t in trades:
            writer.writerow(t)

    # ── 2. Summary CSV ───────────────────────────────────────────────────────
    summary_path = str(Path(output_dir) / f"summary_{tag}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header block
        writer.writerow(["=== PERFORMANCE REPORT ==="])
        writer.writerow(["Generated",    report.generated_at])
        writer.writerow(["Period",       f"{report.period_start} → {report.period_end}"])
        writer.writerow([])

        # KPIs
        writer.writerow(["=== KEY METRICS ==="])
        writer.writerow(["Total Trades",      report.total_trades])
        writer.writerow(["Win Rate",          f"{report.overall_win_rate:.1f}%"])
        writer.writerow(["Total PnL (Gross)", f"${report.total_pnl_gross:+.2f}"])
        writer.writerow(["Total PnL (Net)",   f"${report.total_pnl_net:+.2f}"])
        writer.writerow(["Avg PnL / Trade",   f"${report.avg_pnl_per_trade:+.2f}"])
        writer.writerow(["Best Trade",        f"${report.best_trade:+.2f}"])
        writer.writerow(["Worst Trade",       f"${report.worst_trade:+.2f}"])
        writer.writerow(["Sharpe Ratio",      report.sharpe_ratio if report.sharpe_ratio is not None else "N/A (< 2 days)"])
        writer.writerow(["Max Drawdown",      f"{report.max_drawdown_pct:.2f}%"])
        writer.writerow([])

        # Per-strategy table
        writer.writerow(["=== STRATEGY BREAKDOWN ==="])
        writer.writerow(["Strategy", "Total", "Wins", "Losses", "Win Rate", "Total PnL", "Avg PnL", "Best", "Worst"])
        for s in report.by_strategy:
            writer.writerow([
                s.strategy, s.total, s.wins, s.losses,
                f"{s.win_rate:.1f}%",
                f"${s.total_pnl:+.2f}",
                f"${s.avg_pnl:+.2f}",
                f"${s.best_trade:+.2f}",
                f"${s.worst_trade:+.2f}",
            ])
        writer.writerow([])

        # Daily equity curve
        if report.daily_equity:
            writer.writerow(["=== DAILY EQUITY CURVE ==="])
            writer.writerow(["Date", "Equity ($)", "Daily PnL ($)", "Drawdown (%)"])
            for row in report.daily_equity:
                writer.writerow([
                    row["date"],
                    f"{row['equity']:.2f}",
                    f"{row['daily_pnl']:+.2f}",
                    f"{row.get('drawdown_pct', 0):.2f}%",
                ])

    logger.info(f"[PERF] CSV exported → {trades_path}, {summary_path}")
    return summary_path


def format_telegram(report: PerformanceReport) -> str:
    """Format a concise weekly report for Telegram (HTML mode)."""

    sharpe_str = f"{report.sharpe_ratio:.2f}" if report.sharpe_ratio is not None else "N/A"
    sharpe_grade = _grade_sharpe(report.sharpe_ratio)
    dd_grade = "🟢" if report.max_drawdown_pct < 5 else ("🟡" if report.max_drawdown_pct < 15 else "🔴")
    pnl_emoji = "📈" if report.total_pnl_gross >= 0 else "📉"

    lines = [
        f"📊 <b>Weekly Performance Report</b>",
        f"📅 {report.period_start} → {report.period_end}",
        "",
        f"<b>── Key Metrics ──</b>",
        f"🔄 Trades: <b>{report.total_trades}</b>  (W: {report.total_wins} / L: {report.total_losses})",
        f"🎯 Win Rate: <b>{report.overall_win_rate:.1f}%</b>",
        f"{pnl_emoji} Total PnL: <b>${report.total_pnl_gross:+.2f}</b>  (net ${report.total_pnl_net:+.2f})",
        f"⚡ Avg / Trade: ${report.avg_pnl_per_trade:+.2f}",
        f"🏆 Best: ${report.best_trade:+.2f}   💀 Worst: ${report.worst_trade:+.2f}",
        "",
        f"<b>── Risk ──</b>",
        f"{sharpe_grade} Sharpe: <b>{sharpe_str}</b>",
        f"{dd_grade} Max Drawdown: <b>{report.max_drawdown_pct:.2f}%</b>",
        "",
        f"<b>── Strategy Breakdown ──</b>",
    ]

    for s in report.by_strategy:
        pnl_icon = "🟢" if s.total_pnl >= 0 else "🔴"
        lines.append(
            f"{pnl_icon} <b>{s.strategy}</b>: {s.total} trades | "
            f"WR {s.win_rate:.0f}% | PnL ${s.total_pnl:+.2f}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_closed_trades(since: datetime) -> list[dict]:
    """Pull all trades with a non-null exit_time after `since`."""
    conn = database.get_connection()
    rows = conn.execute(
        """SELECT * FROM trade_log
           WHERE status IN ('closed','stop_loss','take_profit','smart_sell','emergency_exit')
             AND exit_time IS NOT NULL
             AND exit_time >= ?
           ORDER BY exit_time ASC""",
        (since.strftime("%Y-%m-%d %H:%M:%S"),),
    ).fetchall()
    return [dict(r) for r in rows]


def _build_daily_series(trades: list[dict]) -> list[dict]:
    """
    Aggregate trades by exit calendar day and build a running equity curve.

    Returns a list of dicts: {date, daily_pnl, equity, drawdown_pct}
    """
    # Group pnl by date
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        exit_date = (t.get("exit_time") or "")[:10]   # "YYYY-MM-DD"
        if exit_date and t.get("pnl_gross") is not None:
            daily_pnl[exit_date] += t["pnl_gross"]

    start_equity = float(settings.MAX_BUDGET)
    equity       = start_equity
    peak         = start_equity

    series = []
    for d in sorted(daily_pnl.keys()):
        pnl = daily_pnl[d]
        equity += pnl
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0.0
        series.append({
            "date":         d,
            "daily_pnl":    round(pnl, 2),
            "equity":       round(equity, 2),
            "drawdown_pct": round(drawdown, 2),
        })

    return series


def _compute_sharpe(daily: list[dict]) -> float | None:
    """
    Annualised Sharpe Ratio from the daily equity series.

    Returns None if fewer than 2 trading days exist (can't compute std).
    """
    if len(daily) < 2:
        return None

    start_equity = float(settings.MAX_BUDGET)
    rf_daily     = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

    # Daily return = daily_pnl / equity at START of that day
    # We approximate start-of-day equity as current equity minus today's pnl
    returns = []
    for row in daily:
        start_of_day = row["equity"] - row["daily_pnl"]
        denom = start_of_day if start_of_day > 0 else start_equity
        returns.append(row["daily_pnl"] / denom)

    n    = len(returns)
    mean = sum(returns) / n
    var  = sum((r - mean) ** 2 for r in returns) / (n - 1)   # sample variance
    std  = math.sqrt(var)

    if std == 0:
        return None   # all returns identical — undefined Sharpe

    sharpe = (mean - rf_daily) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
    return round(sharpe, 3)


def _compute_max_drawdown(daily: list[dict]) -> float:
    """
    Maximum peak-to-trough drawdown percentage from the daily equity curve.
    The drawdown is already computed per row in _build_daily_series.
    """
    if not daily:
        return 0.0
    return round(max(row["drawdown_pct"] for row in daily), 2)


def _compute_strategy_stats(trades: list[dict]) -> list[StrategyStats]:
    """
    Break down win/loss/pnl by exit strategy (status field).
    Returns list sorted by total_pnl descending.
    """
    buckets: dict[str, StrategyStats] = {}

    for t in trades:
        strat = t.get("status") or "closed"
        pnl   = t.get("pnl_gross")
        if pnl is None:
            continue

        if strat not in buckets:
            buckets[strat] = StrategyStats(strategy=strat)

        s = buckets[strat]
        s.total      += 1
        s.total_pnl  += pnl
        s.best_trade  = max(s.best_trade,  pnl)
        s.worst_trade = min(s.worst_trade, pnl)
        if pnl > 0:
            s.wins   += 1
        else:
            s.losses += 1

    result = list(buckets.values())
    for s in result:
        s.finalise()

    result.sort(key=lambda s: s.total_pnl, reverse=True)
    return result


def _grade_sharpe(sharpe: float | None) -> str:
    if sharpe is None:
        return "⚪"
    if sharpe >= 3.0:
        return "🟢"
    if sharpe >= 1.0:
        return "🟡"
    return "🔴"
