"""
Earnings Intelligence Module
=============================

Analyzes earnings reports and their market impact to improve trading decisions.

What this module does:
1. UPCOMING EARNINGS GUARD — blocks buys 3 days before earnings (high volatility risk)
2. HISTORICAL EARNINGS IMPACT — analyzes how much a stock typically moves on earnings
3. EARNINGS SCORE — adds context to the composite score

Why earnings matter:
  A stock that beats earnings can jump 5-15% overnight.
  A miss can crash 10-20%. Buying right before earnings = gambling.
  But AFTER a good earnings print → strong momentum signal.

Data source: yfinance earnings calendar
Cache: 24 hours (earnings dates don't change daily)

Public API
----------
  check_earnings_risk(ticker)     → (risky: bool, reason: str, days_until: int | None)
  get_earnings_impact(ticker)     → dict  (historical avg move, beat rate, etc.)
  get_earnings_score(ticker)      → float 0-10  (for composite scoring)

Environment variables
---------------------
  EARNINGS_BLACKOUT_DAYS  int  default 3   (days before earnings to block buys)
  EARNINGS_CACHE_TTL      int  default 86400 (24 hours)
"""

import logging
import math
import os
import threading
import time
from datetime import datetime, timezone, timedelta

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BLACKOUT_DAYS: int = int(os.getenv("EARNINGS_BLACKOUT_DAYS", "3"))
CACHE_TTL:     int = int(os.getenv("EARNINGS_CACHE_TTL",     "86400"))   # 24h

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache:      dict[str, tuple[dict, float]] = {}
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_earnings_risk(ticker: str) -> tuple[bool, str, int | None]:
    """
    Check if the stock is within the earnings blackout window.

    Returns
    -------
    (risky, reason, days_until_earnings)
      risky=True  → do NOT buy (earnings too soon)
      risky=False → safe to buy
    """
    ticker = ticker.upper()
    data = _get_earnings_data(ticker)
    if not data:
        return False, "earnings data unavailable — fail open", None

    next_earnings = data.get("next_earnings_date")
    if not next_earnings:
        return False, "no upcoming earnings found", None

    try:
        if isinstance(next_earnings, str):
            next_dt = datetime.fromisoformat(next_earnings).replace(tzinfo=timezone.utc)
        else:
            next_dt = next_earnings
        now = datetime.now(timezone.utc)
        days_until = (next_dt.date() - now.date()).days
    except Exception:
        return False, "could not parse earnings date", None

    if 0 <= days_until <= BLACKOUT_DAYS:
        reason = (
            f"earnings in {days_until} day(s) on {next_dt.strftime('%Y-%m-%d')} "
            f"— blackout {BLACKOUT_DAYS}d window"
        )
        logger.info(f"[EARNINGS] {ticker}: BLOCKED — {reason}")
        return True, reason, days_until

    logger.debug(f"[EARNINGS] {ticker}: next earnings in {days_until}d — OK to buy")
    return False, f"next earnings in {days_until}d", days_until


def get_earnings_impact(ticker: str) -> dict:
    """
    Analyze historical price movement around past earnings dates.

    Returns dict with:
      avg_move_pct      — average absolute % move on earnings day
      beat_rate         — % of quarters where EPS beat estimate
      post_beat_avg     — avg % gain after a beat
      post_miss_avg     — avg % loss after a miss
      quarters_analyzed — number of quarters used
    """
    ticker = ticker.upper()
    data = _get_earnings_data(ticker)
    if not data:
        return {"error": "no data", "quarters_analyzed": 0}

    return {
        "avg_move_pct":    data.get("avg_move_pct",    0.0),
        "beat_rate":       data.get("beat_rate",       0.0),
        "post_beat_avg":   data.get("post_beat_avg",   0.0),
        "post_miss_avg":   data.get("post_miss_avg",   0.0),
        "quarters_analyzed": data.get("quarters_analyzed", 0),
        "next_earnings":   data.get("next_earnings_date"),
        "ticker":          ticker,
    }


def get_earnings_score(ticker: str) -> float:
    """
    Return an earnings quality score 0-10 for use in composite scoring.

    Logic:
      +5 base if earnings data is available
      +3 if beat rate > 70%
      +2 if avg post-beat move > 3%
      -3 if in blackout window (earnings too soon)
      -2 if beat rate < 40% (miss-prone stock)
    """
    ticker = ticker.upper()
    data = _get_earnings_data(ticker)
    if not data:
        return 5.0   # neutral if no data

    score = 5.0
    beat_rate      = data.get("beat_rate", 0.5)
    post_beat_avg  = data.get("post_beat_avg", 0.0)
    _, _, days_until = check_earnings_risk(ticker)

    if beat_rate > 0.70:
        score += 3
    elif beat_rate < 0.40:
        score -= 2

    if post_beat_avg > 3.0:
        score += 2

    if days_until is not None and 0 <= days_until <= BLACKOUT_DAYS:
        score -= 3

    return round(min(10.0, max(0.0, score)), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_earnings_data(ticker: str) -> dict | None:
    """Fetch and cache earnings data for a ticker."""
    now = time.time()
    with _cache_lock:
        cached = _cache.get(ticker)
    if cached and now - cached[1] < CACHE_TTL:
        return cached[0]

    try:
        data = _fetch_earnings_data(ticker)
    except Exception as exc:
        logger.warning(f"[EARNINGS] {ticker}: fetch failed — {exc}")
        data = None

    with _cache_lock:
        _cache[ticker] = (data or {}, now)
    return data


def _fetch_earnings_data(ticker: str) -> dict:
    """
    Fetch earnings calendar + historical EPS surprises from yfinance.
    Analyzes price movement around past earnings dates.
    """
    t = yf.Ticker(ticker)
    result: dict = {}

    # ── Next earnings date ────────────────────────────────────────────────────
    try:
        cal = t.calendar
        if cal is not None and not (isinstance(cal, dict) and len(cal) == 0):
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earnings_date")
                if ed is not None:
                    if hasattr(ed, '__iter__') and not isinstance(ed, str):
                        ed = list(ed)[0]
                    result["next_earnings_date"] = str(ed)[:10]
            elif hasattr(cal, "iloc"):
                # DataFrame
                if "Earnings Date" in cal.index and len(cal.columns) > 0:
                    ed = cal.loc["Earnings Date"].iloc[0]
                    result["next_earnings_date"] = str(ed)[:10]
    except Exception as e:
        logger.debug(f"[EARNINGS] {ticker}: calendar fetch failed: {e}")

    # ── Historical EPS surprises ──────────────────────────────────────────────
    try:
        hist_earnings = t.earnings_history
        if hist_earnings is not None and len(hist_earnings) > 0:
            beats = 0
            total = 0
            post_beat_moves, post_miss_moves, all_moves = [], [], []

            # Get price history for move analysis
            price_hist = t.history(period="2y", interval="1d", auto_adjust=True)

            for idx, row in hist_earnings.iterrows():
                try:
                    eps_est   = float(row.get("epsEstimate") or row.get("EPS Estimate") or 0)
                    eps_act   = float(row.get("epsActual")   or row.get("Reported EPS") or 0)
                    if eps_est == 0:
                        continue

                    beat = eps_act >= eps_est
                    total += 1
                    if beat:
                        beats += 1

                    # Price move on earnings day
                    if price_hist is not None and not price_hist.empty:
                        try:
                            date_str = str(idx)[:10]
                            date_dt  = pd.Timestamp(date_str)
                            # Find the next trading day's price change
                            close_before = price_hist["Close"].asof(date_dt - pd.Timedelta(days=1))
                            close_after  = price_hist["Close"].asof(date_dt + pd.Timedelta(days=1))
                            if close_before and close_before > 0 and close_after and close_after > 0:
                                move_pct = (close_after - close_before) / close_before * 100
                                all_moves.append(abs(move_pct))
                                if beat:
                                    post_beat_moves.append(move_pct)
                                else:
                                    post_miss_moves.append(move_pct)
                        except Exception:
                            pass
                except Exception:
                    continue

            if total > 0:
                result["beat_rate"]          = round(beats / total, 3)
                result["quarters_analyzed"]  = total
            if all_moves:
                result["avg_move_pct"]       = round(sum(all_moves) / len(all_moves), 2)
            if post_beat_moves:
                result["post_beat_avg"]      = round(sum(post_beat_moves) / len(post_beat_moves), 2)
            if post_miss_moves:
                result["post_miss_avg"]      = round(sum(post_miss_moves) / len(post_miss_moves), 2)

    except Exception as e:
        logger.debug(f"[EARNINGS] {ticker}: EPS history failed: {e}")

    return result


def check_dividend_opportunity(ticker: str) -> dict:
    """
    Check if a dividend payment is coming soon (within 7 days).
    Buying before ex-dividend date captures the dividend.
    Returns: {"has_dividend": bool, "days_to_ex": int, "dividend_yield": float}
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        ex_date = info.get("exDividendDate")
        div_yield = float(info.get("dividendYield") or 0) * 100
        if ex_date:
            from datetime import datetime, timezone
            ex_dt = datetime.fromtimestamp(ex_date, tz=timezone.utc)
            days = (ex_dt.date() - datetime.now(timezone.utc).date()).days
            return {
                "has_dividend": True,
                "days_to_ex": days,
                "dividend_yield": round(div_yield, 2),
                "ex_date": str(ex_dt.date()),
                "capture_opportunity": 0 < days <= 7 and div_yield >= 1.0,
            }
    except Exception:
        pass
    return {"has_dividend": False, "days_to_ex": None, "dividend_yield": 0}


def get_status() -> dict:
    """Return cache stats for /status endpoint."""
    with _cache_lock:
        return {
            "cached_tickers": len(_cache),
            "blackout_days":  BLACKOUT_DAYS,
            "cache_ttl_hours": CACHE_TTL // 3600,
        }
