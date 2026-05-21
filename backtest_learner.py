"""
Historical Backtest Learner
============================

Learns from historical chart data to improve future trading decisions.

What this module does:
1. Downloads 1 year of daily price data for watchlist stocks
2. Simulates every possible entry point using current indicators
3. Measures which indicators/conditions predicted profitable outcomes
4. Stores learned insights to improve MIN_BUY_SCORE and indicator focus

How it works:
  For each stock, for each day in the past year:
    - Compute all indicators (RSI, MACD, BB, etc.)
    - Check: if we had bought here, what happened in the next 5/10/20 days?
    - Track: which indicator combinations led to >3% gain? Which led to losses?

Results are stored in SQLite and used to:
  - Adjust MIN_BUY_SCORE dynamically
  - Identify which indicators are most predictive right now
  - Block stocks with historically poor performance

Public API
----------
  run_backtest(tickers, lookback_days=252)   → BacktestResult
  get_insights()                              → dict (best indicators, optimal score)
  apply_insights()                            → updates scoring thresholds
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS:    int   = int(os.getenv("BACKTEST_LOOKBACK_DAYS",  "252"))  # 1 year
HOLD_PERIOD:      int   = int(os.getenv("BACKTEST_HOLD_DAYS",      "5"))    # 5-day forward return (shorter = more signals)
WIN_THRESHOLD:    float = float(os.getenv("BACKTEST_WIN_PCT",       "1.5"))  # 1.5% = win (realistic)
LOSS_THRESHOLD:   float = float(os.getenv("BACKTEST_LOSS_PCT",      "-1.5"))  # -1.5% = loss (symmetric)
MIN_SAMPLES:      int   = int(os.getenv("BACKTEST_MIN_SAMPLES",    "20"))   # min entries for insight
CACHE_TTL:        int   = int(os.getenv("BACKTEST_CACHE_TTL",      "1800"))   # 30min — re-trains every 30min while market closed

# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class IndicatorInsight:
    name: str
    win_rate: float       # % of times this indicator value led to a win
    avg_return: float     # average forward return when condition was met
    sample_count: int     # how many data points
    predictive_power: float  # win_rate × sqrt(samples) — reliability score


@dataclass
class BacktestResult:
    tickers_analyzed:  int   = 0
    total_signals:     int   = 0
    win_signals:       int   = 0
    loss_signals:      int   = 0
    overall_win_rate:  float = 0.0
    avg_return:        float = 0.0
    best_indicators:   list  = field(default_factory=list)
    worst_conditions:  list  = field(default_factory=list)
    optimal_min_score: int   = 58
    computed_at:       str   = ""
    ticker_stats:      list  = field(default_factory=list)  # [{ticker, signals, win_rate, avg_return}]

    def to_dict(self) -> dict:
        return {
            "tickers_analyzed":  self.tickers_analyzed,
            "total_signals":     self.total_signals,
            "win_signals":       self.win_signals,
            "loss_signals":      self.loss_signals,
            "overall_win_rate":  round(self.overall_win_rate, 1),
            "avg_return":        round(self.avg_return, 2),
            "best_indicators":   self.best_indicators,
            "worst_conditions":  self.worst_conditions,
            "optimal_min_score": self.optimal_min_score,
            "computed_at":       self.computed_at,
        }


# ── Cache ─────────────────────────────────────────────────────────────────────
_result_cache: BacktestResult | None = None
_cache_ts: float = 0.0
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(tickers: list[str], lookback_days: int = LOOKBACK_DAYS) -> BacktestResult:
    """
    Run historical simulation on a list of tickers.
    Analyzes which indicator conditions predicted profitable outcomes.
    """
    global _result_cache, _cache_ts
    now = time.time()

    with _lock:
        if _result_cache and now - _cache_ts < CACHE_TTL:
            logger.info("[BACKTEST] Returning cached result")
            return _result_cache

    logger.info(f"[BACKTEST] Starting analysis: {len(tickers)} tickers, {lookback_days} days")
    result = BacktestResult(computed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    all_signals: list[dict] = []
    indicator_wins: dict[str, list[float]] = {}  # indicator_condition -> list of forward returns
    per_ticker: list[dict] = []  # per-ticker stats for Telegram reporting

    for ticker in tickers[:30]:  # cap at 30 to save memory/time
        try:
            signals = _analyze_ticker(ticker, lookback_days)
            all_signals.extend(signals)

            # Per-ticker stats
            if signals:
                t_returns = [s["forward_return"] for s in signals]
                t_wins = sum(1 for r in t_returns if r >= WIN_THRESHOLD)
                per_ticker.append({
                    "ticker":     ticker,
                    "signals":    len(signals),
                    "win_rate":   round(t_wins / len(signals) * 100, 1),
                    "avg_return": round(float(np.mean(t_returns)), 2),
                })

            # Track indicator performance
            for sig in signals:
                for ind_key, ind_val in sig.get("indicators", {}).items():
                    k = f"{ind_key}={ind_val}"
                    indicator_wins.setdefault(k, []).append(sig["forward_return"])

        except Exception as exc:
            logger.debug(f"[BACKTEST] {ticker}: {exc}")
            continue

    if not all_signals:
        logger.warning("[BACKTEST] No signals generated")
        return result

    result.tickers_analyzed = len(set(s["ticker"] for s in all_signals))
    result.total_signals    = len(all_signals)
    # Sort per-ticker by win_rate descending for readable Telegram display
    result.ticker_stats = sorted(per_ticker, key=lambda x: x["win_rate"], reverse=True)

    returns = [s["forward_return"] for s in all_signals]
    result.win_signals  = sum(1 for r in returns if r >= WIN_THRESHOLD)
    result.loss_signals = sum(1 for r in returns if r <= LOSS_THRESHOLD)
    result.avg_return   = round(float(np.mean(returns)), 2) if returns else 0.0
    result.overall_win_rate = (
        round(result.win_signals / result.total_signals * 100, 1)
        if result.total_signals > 0 else 0.0
    )

    # Compute indicator insights
    insights: list[IndicatorInsight] = []
    for cond, ret_list in indicator_wins.items():
        if len(ret_list) < MIN_SAMPLES:
            continue
        wins = sum(1 for r in ret_list if r >= WIN_THRESHOLD)
        wr   = wins / len(ret_list) * 100
        avg  = float(np.mean(ret_list))
        power = wr * (len(ret_list) ** 0.5) / 100  # reliability-weighted score
        insights.append(IndicatorInsight(
            name=cond, win_rate=wr, avg_return=round(avg, 2),
            sample_count=len(ret_list), predictive_power=round(power, 3),
        ))

    insights.sort(key=lambda x: x.predictive_power, reverse=True)
    result.best_indicators  = [
        {"condition": i.name, "win_rate": round(i.win_rate, 1),
         "avg_return": i.avg_return, "samples": i.sample_count}
        for i in insights[:5] if i.win_rate > 55
    ]
    result.worst_conditions = [
        {"condition": i.name, "win_rate": round(i.win_rate, 1), "samples": i.sample_count}
        for i in sorted(insights, key=lambda x: x.win_rate)[:3] if i.win_rate < 40
    ]

    # Compute optimal MIN_BUY_SCORE using score-bucketed win rates
    result.optimal_min_score = _find_optimal_threshold(all_signals)

    with _lock:
        # Double-check cache wasn't populated by a concurrent call while we computed
        if _cache_ts == 0 or now - _cache_ts >= 3600:
            _save_to_db(result)   # inside lock — prevents duplicate DB writes
        _result_cache = result
        _cache_ts = now

    logger.info(
        f"[BACKTEST] Done: {result.tickers_analyzed} tickers | "
        f"{result.total_signals} signals | WR={result.overall_win_rate}% | "
        f"optimal_score={result.optimal_min_score}"
    )
    return result


def get_insights() -> dict:
    """Return cached backtest insights (or empty dict if not run yet)."""
    with _lock:
        if _result_cache:
            return _result_cache.to_dict()
    return {"status": "not_run_yet", "message": "Call /backtest to start analysis"}


def apply_insights() -> dict:
    """
    Apply learned insights to improve scoring.
    Updates MIN_BUY_SCORE environment variable based on optimal threshold.
    """
    with _lock:
        if not _result_cache:
            return {"applied": False, "reason": "No backtest results available"}
        optimal = _result_cache.optimal_min_score

    # Only apply if statistically meaningful and different from current
    current_score = int(os.getenv("MIN_BUY_SCORE", "58"))
    if abs(optimal - current_score) < 3:
        return {"applied": False, "reason": f"Optimal ({optimal}) close to current ({current_score}) — no change needed"}

    os.environ["MIN_BUY_SCORE"] = str(optimal)
    logger.info(f"[BACKTEST] Applied: MIN_BUY_SCORE {current_score} → {optimal}")
    return {
        "applied": True,
        "old_score": current_score,
        "new_score": optimal,
        "win_rate": _result_cache.overall_win_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_ticker(ticker: str, lookback_days: int) -> list[dict]:
    """Download historical data for one ticker and compute backtest signals."""
    from indicators import add_all_indicators

    t = yf.Ticker(ticker)
    hist = t.history(period=f"{lookback_days + 30}d", auto_adjust=True)
    if hist.empty or len(hist) < 60:
        return []

    hist.columns = [c.lower() for c in hist.columns]
    hist = hist[["open", "high", "low", "close", "volume"]].dropna()
    hist = add_all_indicators(hist)

    signals = []
    # Iterate each day (skip last HOLD_PERIOD days — no forward data yet)
    for i in range(len(hist) - HOLD_PERIOD - 1):
        row = hist.iloc[i]
        future_close = float(hist.iloc[i + HOLD_PERIOD]["close"])
        cur_close = float(row["close"])
        if cur_close <= 0:
            continue
        forward_return = (future_close - cur_close) / cur_close * 100

        # Extract indicator conditions as discrete buckets
        rsi = row.get("rsi_14")
        macd_bullish = row.get("macd", 0) > row.get("macd_signal", 0)
        # bb_position not stored as a column — compute inline from Bollinger Band bounds
        _bb_upper = row.get("bb_upper")
        _bb_lower = row.get("bb_lower")
        if (_bb_upper and _bb_lower and _bb_upper > _bb_lower
                and not np.isnan(_bb_upper) and not np.isnan(_bb_lower)):
            bb_pos = (cur_close - _bb_lower) / (_bb_upper - _bb_lower)
        else:
            bb_pos = None
        vol_ratio = row.get("volume_ratio", 1.0)
        above_sma50 = row.get("close", 0) > row.get("sma_50", 0) if row.get("sma_50") else None

        conditions = {}
        if rsi is not None and not np.isnan(rsi):
            if rsi < 30:      conditions["rsi"] = "oversold"
            elif rsi < 50:    conditions["rsi"] = "below_mid"
            elif rsi < 65:    conditions["rsi"] = "healthy"
            else:             conditions["rsi"] = "overbought"

        if macd_bullish is not None:
            conditions["macd"] = "bullish" if macd_bullish else "bearish"

        if vol_ratio and not np.isnan(vol_ratio):
            conditions["volume"] = "high" if vol_ratio >= 1.5 else ("normal" if vol_ratio >= 0.8 else "low")

        if above_sma50 is not None:
            conditions["trend"] = "above_sma50" if above_sma50 else "below_sma50"

        # Compute a simple composite score for threshold optimization
        simple_score = _quick_score(row)

        # Only include days where the bot would actually consider buying
        # (score > 45 = above random, filters out clearly bad days)
        if simple_score < 45:
            continue

        signals.append({
            "ticker":         ticker,
            "date":           str(hist.index[i])[:10],
            "forward_return": round(forward_return, 2),
            "simple_score":   simple_score,
            "indicators":     conditions,
        })

    return signals


def _quick_score(row) -> int:
    """Fast composite score (0-100) — mimics the bot's real buy criteria.
    Only days with score>=45 are included in backtest signals.
    """
    score = 50
    try:
        rsi = float(row.get("rsi_14") or 50)
        if 38 <= rsi <= 65:  score += 12   # sweet spot — not oversold or overbought
        elif rsi < 30:        score -= 5    # knife-catching
        elif rsi > 72:        score -= 20   # overbought — skip

        macd = float(row.get("macd") or 0)
        sig  = float(row.get("macd_signal") or 0)
        if macd > sig:   score += 12    # bullish momentum
        else:            score -= 8     # bearish momentum

        vol  = float(row.get("volume_ratio") or 1)
        if vol >= 1.5:   score += 10    # strong volume = conviction
        elif vol >= 1.0: score += 5     # average volume
        elif vol < 0.7:  score -= 12    # weak volume = no interest

        close  = float(row.get("close") or 0)
        sma20  = float(row.get("sma_20") or 0)
        sma50  = float(row.get("sma_50") or 0)
        sma200 = float(row.get("sma_200") or 0)
        if sma50  > 0 and close > sma50:             score += 8   # above medium trend
        if sma200 > 0 and close > sma200:            score += 5   # above long trend
        if sma50  > 0 and sma200 > 0 and sma50 > sma200: score += 5  # golden cross

    except Exception:
        pass
    return max(0, min(100, score))


def _find_optimal_threshold(signals: list[dict]) -> int:
    """Find the MIN_BUY_SCORE that maximizes win rate with sufficient sample size."""
    if not signals:
        return 58

    best_score = 58
    best_metric = 0.0

    for threshold in range(45, 75):
        bucket = [s for s in signals if s["simple_score"] >= threshold]
        if len(bucket) < MIN_SAMPLES:
            continue
        wins = sum(1 for s in bucket if s["forward_return"] >= WIN_THRESHOLD)
        wr   = wins / len(bucket)
        # Metric: win_rate × sqrt(sample_size) — rewards both quality and quantity
        metric = wr * (len(bucket) ** 0.4)
        if metric > best_metric:
            best_metric = metric
            best_score  = threshold

    return best_score


def _explain_chart(row: "pd.Series", ticker: str, outcome: str, ret_10d: float,
                   results_at: dict) -> str:
    """
    Build a human-readable explanation of WHY the chart behaved this way.
    Correlates indicator readings at entry with the actual price outcome.
    """
    reasons_bullish = []
    reasons_bearish = []

    try:
        rsi = float(row.get("rsi_14") or 50)
        if rsi < 30:
            reasons_bullish.append(f"RSI={rsi:.0f} (קניית יתר — זול מאוד)")
        elif rsi < 45:
            reasons_bullish.append(f"RSI={rsi:.0f} (בריא, לא קנוי)")
        elif rsi > 70:
            reasons_bearish.append(f"RSI={rsi:.0f} (מכירת יתר — יקר)")
        elif rsi > 60:
            reasons_bearish.append(f"RSI={rsi:.0f} (מתחיל להתחמם)")

        macd = float(row.get("macd") or 0)
        macd_sig = float(row.get("macd_signal") or 0)
        macd_hist = float(row.get("macd_hist") or 0)
        if macd > macd_sig:
            reasons_bullish.append(f"MACD חיובי (+{macd_hist:.2f})")
        else:
            reasons_bearish.append(f"MACD שלילי ({macd_hist:.2f})")

        vol_ratio = float(row.get("volume_ratio") or 1)
        if vol_ratio >= 1.8:
            reasons_bullish.append(f"נפח גבוה מאוד (×{vol_ratio:.1f})")
        elif vol_ratio >= 1.2:
            reasons_bullish.append(f"נפח גבוה (×{vol_ratio:.1f})")
        elif vol_ratio < 0.7:
            reasons_bearish.append(f"נפח נמוך (×{vol_ratio:.1f})")

        close = float(row.get("close") or 0)
        sma50 = float(row.get("sma_50") or 0)
        sma200 = float(row.get("sma_200") or 0)
        if sma50 > 0 and close > sma50:
            reasons_bullish.append("מעל SMA50 (מגמה עולה)")
        elif sma50 > 0:
            reasons_bearish.append("מתחת SMA50 (מגמה יורדת)")
        if sma50 > 0 and sma200 > 0:
            if sma50 > sma200:
                reasons_bullish.append("SMA50>SMA200 (גולדן קרוס)")
            else:
                reasons_bearish.append("SMA50<SMA200 (דת קרוס)")

        bb_upper = float(row.get("bb_upper") or 0)
        bb_lower = float(row.get("bb_lower") or 0)
        if bb_upper > bb_lower > 0:
            bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
            if bb_pct > 0.85:
                reasons_bearish.append(f"קרוב לגג בולינגר ({bb_pct:.0%})")
            elif bb_pct < 0.20:
                reasons_bullish.append(f"קרוב לרצפת בולינגר ({bb_pct:.0%})")

        momentum = float(row.get("momentum_10") or 0)
        if momentum > 0:
            reasons_bullish.append(f"מומנטום חיובי (+{momentum:.1f}$)")
        else:
            reasons_bearish.append(f"מומנטום שלילי ({momentum:.1f}$)")

    except Exception:
        pass

    bull_str = " | ".join(reasons_bullish) if reasons_bullish else "אין"
    bear_str = " | ".join(reasons_bearish) if reasons_bearish else "אין"
    verdict = "✅ הגרף עלה" if outcome == "win" else ("❌ הגרף ירד" if outcome == "loss" else "➡️ ניטרלי")

    return (
        f"{ticker} | {verdict} ({ret_10d:+.1f}% ב-10י')\n"
        f"  📈 סיבות עלייה בכניסה: {bull_str}\n"
        f"  📉 סיבות ירידה בכניסה: {bear_str}\n"
        f"  תשואות: 5י'={results_at.get('ret_5d',0):+.1f}% | "
        f"10י'={results_at.get('ret_10d',0):+.1f}% | "
        f"20י'={results_at.get('ret_20d',0):+.1f}%"
    )


def simulate_own_trade_history() -> dict:
    """
    Simulate outcomes of the bot's own past trades using real historical data.

    For each trade the bot actually made:
      1. Download real price history from yfinance (entry_date → today)
      2. Compute what P&L would have been at 5 / 10 / 20 trading days hold
      3. Update pnl_gross in trade_log so the learning system can use it
      4. Save insights to learning_log

    Returns a summary dict.
    """
    import sqlite3
    import database as _db
    from datetime import datetime, timedelta, timezone

    logger.info("[OWN-TRADES] Starting simulation of bot's own trade history...")

    conn = _db.get_connection()

    # Get all closed trades (including ghost-closed with pnl=0)
    rows = conn.execute("""
        SELECT id, ticker, entry_price, entry_time, qty
        FROM trade_log
        WHERE status = 'closed'
        ORDER BY entry_time DESC
        LIMIT 50
    """).fetchall()

    if not rows:
        logger.info("[OWN-TRADES] No closed trades to simulate")
        return {"simulated": 0}

    simulated = 0
    wins = losses = 0
    returns_list = []

    for trade_id, ticker, entry_price, entry_time_str, qty in rows:
        try:
            from indicators import add_all_indicators

            # Parse entry date
            entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
            entry_date = entry_dt.date()

            # Download 300 days before entry to compute indicators properly
            start_str = str(entry_date - timedelta(days=300))
            t = yf.Ticker(ticker)
            hist = t.history(start=start_str, auto_adjust=True)
            if hist.empty or len(hist) < 30:
                logger.debug(f"[OWN-TRADES] {ticker}: no historical data")
                continue

            hist.columns = [c.lower() for c in hist.columns]
            hist = hist[["open","high","low","close","volume"]].dropna()

            # Add all indicators to the full history
            hist = add_all_indicators(hist)

            # Find row closest to entry date
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            entry_ts = pd.Timestamp(entry_date)
            future_hist = hist[hist.index >= entry_ts]

            if future_hist.empty or len(future_hist) < 2:
                logger.debug(f"[OWN-TRADES] {ticker}: no data from {entry_date}")
                continue

            entry_row  = future_hist.iloc[0]
            entry_close = float(entry_row["close"])

            # Forward returns at 5, 10, 20 trading days
            results_at = {}
            for days in [5, 10, 20]:
                idx = min(days, len(future_hist) - 1)
                future_price = float(future_hist.iloc[idx]["close"])
                ret_pct = (future_price - entry_close) / entry_close * 100
                results_at[f"ret_{days}d"] = round(ret_pct, 2)

            ret_10d   = results_at.get("ret_10d", 0.0)
            sim_exit  = entry_close * (1 + ret_10d / 100)
            sim_pnl   = (sim_exit - entry_price) * qty
            returns_list.append(ret_10d)

            outcome_str = (
                "win"  if ret_10d >= WIN_THRESHOLD else
                "loss" if ret_10d <= LOSS_THRESHOLD else "neutral"
            )
            if outcome_str == "win":   wins   += 1
            elif outcome_str == "loss": losses += 1

            # Build chart explanation — WHY did it move?
            explanation = _explain_chart(entry_row, ticker, outcome_str, ret_10d, results_at)
            logger.info(f"[OWN-TRADES] {explanation.splitlines()[0]}")
            for line in explanation.splitlines()[1:]:
                logger.info(f"[OWN-TRADES]   {line.strip()}")

            # Update trade_log pnl
            conn.execute("""
                UPDATE trade_log
                SET pnl_gross  = ?,
                    exit_price = ?,
                    rsi        = ?,
                    macd       = ?,
                    macd_signal = ?,
                    volume_ratio = ?
                WHERE id = ? AND (pnl_gross IS NULL OR pnl_gross = 0)
            """, (
                round(sim_pnl, 2),
                round(sim_exit, 2),
                round(float(entry_row.get("rsi_14") or 0), 1),
                round(float(entry_row.get("macd") or 0), 4),
                round(float(entry_row.get("macd_signal") or 0), 4),
                round(float(entry_row.get("volume_ratio") or 1), 2),
                trade_id,
            ))

            # Save full explanation to learning_log
            import json as _json
            indicators_snap = _json.dumps({
                "rsi":          round(float(entry_row.get("rsi_14") or 0), 1),
                "macd_bull":    bool(float(entry_row.get("macd") or 0) > float(entry_row.get("macd_signal") or 0)),
                "vol_ratio":    round(float(entry_row.get("volume_ratio") or 1), 2),
                "above_sma50":  bool(float(entry_row.get("close") or 0) > float(entry_row.get("sma_50") or 0)),
                "above_sma200": bool(float(entry_row.get("close") or 0) > float(entry_row.get("sma_200") or 0)),
                "momentum_10":  round(float(entry_row.get("momentum_10") or 0), 2),
                "ret_5d":       results_at.get("ret_5d", 0),
                "ret_10d":      ret_10d,
                "ret_20d":      results_at.get("ret_20d", 0),
            }, ensure_ascii=False)

            conn.execute("""
                INSERT INTO learning_log
                    (trade_id, pattern_type, description,
                     indicators_snapshot, outcome, pnl, created_at)
                VALUES (?, 'own_trade_chart_analysis', ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (trade_id, explanation, indicators_snap, outcome_str, round(sim_pnl, 2)))

            simulated += 1

        except Exception as exc:
            logger.debug(f"[OWN-TRADES] {ticker}: {exc}")
            continue

    conn.commit()

    avg_ret = float(np.mean(returns_list)) if returns_list else 0.0
    wr      = wins / simulated * 100 if simulated > 0 else 0.0

    summary = {
        "simulated": simulated,
        "wins":      wins,
        "losses":    losses,
        "win_rate":  round(wr, 1),
        "avg_return": round(avg_ret, 2),
    }
    logger.info(
        f"[OWN-TRADES] Done: {simulated} trades simulated | "
        f"WR={wr:.1f}% | avg={avg_ret:+.2f}%"
    )
    return summary


def _save_to_db(result: BacktestResult) -> None:
    """Persist backtest result summary to SQLite for audit trail."""
    try:
        import database as _db
        import json
        conn = _db.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at TEXT,
                tickers    INTEGER,
                total_signals INTEGER,
                win_rate   REAL,
                avg_return REAL,
                optimal_score INTEGER,
                insights   TEXT
            )
        """)
        conn.execute(
            "INSERT INTO backtest_results VALUES (NULL,?,?,?,?,?,?,?)",
            (
                result.computed_at,
                result.tickers_analyzed,
                result.total_signals,
                result.overall_win_rate,
                result.avg_return,
                result.optimal_min_score,
                json.dumps(result.best_indicators, ensure_ascii=False),
            )
        )
        conn.commit()
        logger.info("[BACKTEST] Saved to database")
    except Exception as exc:
        logger.debug(f"[BACKTEST] DB save failed (non-critical): {exc}")
