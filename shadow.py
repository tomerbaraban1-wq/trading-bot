"""
Shadow Mode — Parallel Paper-Trading Engine

What it does:
  Runs a secondary strategy in parallel with the live bot.
  Every evaluated signal is also assessed by the shadow strategy.
  Shadow trades are logged to SQLite but NEVER sent to the broker.

Why it's useful:
  The live bot has many guardrails (volume filter, correlation filter,
  sentiment threshold, composite score floor). Shadow mode answers:

    "Would we have made money if we'd been less conservative?"

  After 2-4 weeks of data you can compare:
  - Shadow win rate vs live win rate
  - Signals live blocked that shadow caught (and their outcomes)
  - Whether each filter is additive or destructive to P&L

Shadow strategy (intentionally more aggressive than live):
  - Lower composite score floor   (SHADOW_MIN_SCORE,     default 45)
  - Lower sentiment floor         (SHADOW_MIN_SENTIMENT, default 3)
  - Skips volume confirmation     (SHADOW_SKIP_VOLUME,   default False)
  - Skips correlation filter      (SHADOW_SKIP_CORR,     default False)
  - Independent paper capital     (SHADOW_CAPITAL,       default $10,000)
  - Max position size as % of capital (SHADOW_POSITION_PCT, default 20%)

Shadow position lifecycle:
  open  → shadow_monitor_loop checks price every 5 min
        → closes at ATR trailing stop OR take-profit ceiling
        → pnl_gross / pnl_pct computed on hypothetical fill prices

Public API:
  evaluate(ticker, price, score, sentiment, live_blocked_by, live_reason, source)
      → logs shadow trade if shadow strategy approves; returns shadow_trade_id | None

  close_position(shadow_trade_id, exit_price, reason, status)
      → marks shadow trade as closed with P&L

  compare()
      → dict comparing shadow vs live performance metrics

  get_trades(limit)
      → list of shadow_trades rows

Environment variables:
  SHADOW_MODE_ENABLED     bool   default true
  SHADOW_MIN_SCORE        float  default 45
  SHADOW_MIN_SENTIMENT    int    default 3
  SHADOW_SKIP_VOLUME      bool   default false
  SHADOW_SKIP_CORR        bool   default false
  SHADOW_CAPITAL          float  default 10000
  SHADOW_POSITION_PCT     float  default 20    (% of capital per position)
  SHADOW_TAKE_PROFIT_PCT  float  default 10
"""

import logging
import math
import os
import threading
import time
from datetime import datetime, timezone

import yfinance as yf

import database

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ENABLED:           bool  = os.getenv("SHADOW_MODE_ENABLED",    "true").lower() == "true"
MIN_SCORE:         float = float(os.getenv("SHADOW_MIN_SCORE",        "45"))
MIN_SENTIMENT:     int   = int(os.getenv("SHADOW_MIN_SENTIMENT",       "3"))
SKIP_VOLUME:       bool  = os.getenv("SHADOW_SKIP_VOLUME",    "false").lower() == "true"
SKIP_CORR:         bool  = os.getenv("SHADOW_SKIP_CORR",      "false").lower() == "true"
CAPITAL:           float = float(os.getenv("SHADOW_CAPITAL",         "10000"))
POSITION_PCT:      float = float(os.getenv("SHADOW_POSITION_PCT",       "20"))
TAKE_PROFIT_PCT:   float = float(os.getenv("SHADOW_TAKE_PROFIT_PCT",    "10"))

# ── Price cache for monitor loop (ticker → (price, ts)) ───────────────────────
_price_cache:  dict[str, tuple[float, float]] = {}
_price_lock    = threading.Lock()
_PRICE_TTL     = 60   # 1 minute


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    ticker:           str,
    price:            float,
    composite_score:  float,
    sentiment_score:  int,
    volume_ratio:     float | None,
    live_blocked_by:  str | None,
    live_block_reason: str,
    signal_source:    str,
) -> int | None:
    """
    Evaluate a signal against the shadow strategy thresholds.
    If the shadow strategy approves, log to shadow_trades and return the row id.
    Returns None if shadow also rejects or if shadow mode is disabled.

    Parameters
    ----------
    live_blocked_by   : which live filter rejected this signal
                        None = live strategy also traded (both agree)
    live_block_reason : human-readable reason the live bot rejected
    signal_source     : 'webhook' | 'auto_invest'
    """
    if not ENABLED:
        return None

    ticker = ticker.upper()

    # ── Shadow strategy evaluation ────────────────────────────────────────────
    shadow_reject_reason = _shadow_check(
        ticker, composite_score, sentiment_score, volume_ratio
    )
    if shadow_reject_reason:
        logger.debug(f"[SHADOW] {ticker}: rejected — {shadow_reject_reason}")
        return None

    # ── Compute hypothetical position size ────────────────────────────────────
    max_notional = CAPITAL * POSITION_PCT / 100
    qty = max(1, int(max_notional / price)) if price > 0 else 0
    if qty == 0:
        logger.debug(f"[SHADOW] {ticker}: qty=0 at price=${price:.2f}")
        return None

    # ── Compute initial ATR stop ──────────────────────────────────────────────
    try:
        from atr_stop import compute_initial_stop
        atr_stop_price, stop_meta = compute_initial_stop(ticker, price)
    except Exception:
        atr_stop_price = price * 0.95   # fallback: 5% stop
        stop_meta = {}

    # ── Persist to shadow_trades ──────────────────────────────────────────────
    row = {
        "ticker":            ticker,
        "signal_source":     signal_source,
        "entry_price":       price,
        "qty":               qty,
        "composite_score":   composite_score,
        "sentiment_score":   sentiment_score,
        "volume_ratio":      volume_ratio,
        "atr_stop_price":    atr_stop_price,
        "high_watermark":    price,
        "live_blocked_by":   live_blocked_by or "none_live_also_traded",
        "live_block_reason": live_block_reason,
    }
    trade_id = database.save_shadow_trade(row)

    verdict = (
        f"SHADOW AGREES (live also traded)"
        if live_blocked_by is None
        else f"SHADOW WOULD TRADE — live blocked by [{live_blocked_by}]"
    )
    logger.info(
        f"[SHADOW] {ticker}: #{trade_id} | score={composite_score:.0f}/100 | "
        f"qty={qty} @ ${price:.2f} | stop=${atr_stop_price:.2f} | {verdict}"
    )
    return trade_id


def close_position(
    shadow_trade_id: int,
    exit_price:      float,
    reason:          str,
    status:          str = "closed",
) -> None:
    """
    Mark a shadow trade as closed.
    Computes hypothetical P&L from entry price stored in DB.
    """
    try:
        trade = database.get_shadow_trade(shadow_trade_id)
        if not trade:
            logger.warning(f"[SHADOW] close_position: trade #{shadow_trade_id} not found")
            return

        pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]
        pnl_pct   = (exit_price - trade["entry_price"]) / trade["entry_price"] * 100 \
                    if trade["entry_price"] else 0

        database.close_shadow_trade(
            shadow_trade_id, exit_price,
            round(pnl_gross, 4), round(pnl_pct, 3),
            status, reason
        )
        logger.info(
            f"[SHADOW] {trade['ticker']} #{shadow_trade_id} CLOSED — "
            f"exit=${exit_price:.2f} | pnl=${pnl_gross:+.2f} ({pnl_pct:+.2f}%) | "
            f"reason={reason}"
        )
    except Exception as exc:
        logger.error(f"[SHADOW] close_position failed for #{shadow_trade_id}: {exc}")


def compare() -> dict:
    """
    Generate a performance comparison between shadow and live strategies.

    Returns a dict with:
      shadow     : aggregate stats for shadow trades
      live       : aggregate stats for live closed trades
      shadow_only: trades shadow caught that live blocked (and their P&L outcomes)
      agreement  : trades both strategies took (both live and shadow executed)
      filters    : per-filter analysis (how often each filter blocked a trade + outcome)
    """
    shadow_trades = database.get_shadow_trade_history(limit=1000)
    live_trades   = database.get_trade_history(limit=1000)

    closed_shadow = [t for t in shadow_trades if t["status"] != "open"]
    closed_live   = [t for t in live_trades   if t.get("status") not in (None, "open")]

    shadow_stats = _aggregate_stats(closed_shadow, pnl_key="pnl_gross")
    live_stats   = _aggregate_stats(closed_live,   pnl_key="pnl_gross")

    # Shadow-only trades (live was blocked, shadow ran)
    shadow_only = [
        t for t in closed_shadow
        if t.get("live_blocked_by") and t["live_blocked_by"] != "none_live_also_traded"
    ]
    shadow_only_stats = _aggregate_stats(shadow_only, pnl_key="pnl_gross")

    # Agreement trades (both took the trade)
    agreement = [
        t for t in closed_shadow
        if t.get("live_blocked_by") == "none_live_also_traded"
    ]

    # Per-filter breakdown: which filter blocked the most, and were those trades winners?
    filter_stats: dict[str, dict] = {}
    for t in shadow_only:
        flt = t.get("live_blocked_by", "unknown")
        if flt not in filter_stats:
            filter_stats[flt] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        pnl = t.get("pnl_gross") or 0
        filter_stats[flt]["count"]     += 1
        filter_stats[flt]["total_pnl"] += pnl
        if pnl > 0:
            filter_stats[flt]["wins"] += 1

    for flt, st in filter_stats.items():
        n = st["count"]
        st["win_rate"]  = round(st["wins"] / n * 100, 1) if n else 0
        st["avg_pnl"]   = round(st["total_pnl"] / n, 2) if n else 0
        st["total_pnl"] = round(st["total_pnl"], 2)

    open_shadow = [t for t in shadow_trades if t["status"] == "open"]

    return {
        "shadow":         shadow_stats,
        "live":           live_stats,
        "shadow_only":    shadow_only_stats,
        "agreement":      {"count": len(agreement)},
        "open_shadow":    len(open_shadow),
        "filter_analysis": filter_stats,
        "generated_at":   _utcnow(),
        "note": (
            "shadow_only = trades shadow took but live blocked. "
            "Positive avg_pnl here means the live filters are HELPING. "
            "Negative avg_pnl means the filters are blocking profitable trades."
        ),
    }


def get_trades(limit: int = 100) -> list[dict]:
    return database.get_shadow_trade_history(limit=limit)


# ─────────────────────────────────────────────────────────────────────────────
# Shadow monitor — called from heartbeat.py shadow_monitor_loop
# ─────────────────────────────────────────────────────────────────────────────

def tick_open_positions() -> None:
    """
    Check all open shadow positions and close those that hit stop or take-profit.
    Called from the background shadow_monitor_loop every 5 minutes.
    """
    from atr_stop import update_trailing_stop, should_exit

    open_trades = database.get_open_shadow_trades()
    if not open_trades:
        return

    for trade in open_trades:
        ticker = trade["ticker"]
        try:
            cur_price = _fetch_price(ticker)
            if not cur_price:
                continue

            entry       = trade["entry_price"]
            atr_stop    = trade.get("atr_stop_price")
            high_wm     = trade.get("high_watermark") or entry

            # Trail stop
            new_stop, new_wm, raised = update_trailing_stop(
                ticker, cur_price, atr_stop, high_wm, entry
            )
            if raised or new_wm != high_wm:
                database.update_shadow_stop(trade["id"], new_stop, new_wm)
                atr_stop = new_stop
                high_wm  = new_wm

            plpc = (cur_price - entry) / entry * 100 if entry else 0

            # ATR trailing stop hit
            if atr_stop and should_exit(cur_price, atr_stop):
                close_position(
                    trade["id"], cur_price,
                    f"ATR trailing stop (stop=${atr_stop:.2f} | {plpc:.1f}%)",
                    "stop_loss",
                )
                continue

            # Take-profit ceiling
            if plpc >= TAKE_PROFIT_PCT:
                close_position(
                    trade["id"], cur_price,
                    f"Take profit ({plpc:.1f}%)",
                    "take_profit",
                )

        except Exception as exc:
            logger.warning(f"[SHADOW] tick error for {ticker} #{trade['id']}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shadow_check(
    ticker:          str,
    composite_score: float,
    sentiment_score: int,
    volume_ratio:    float | None,
) -> str | None:
    """
    Return a rejection reason string, or None if shadow approves the trade.
    """
    if composite_score < MIN_SCORE:
        return f"score {composite_score:.0f} < shadow floor {MIN_SCORE}"

    if sentiment_score < MIN_SENTIMENT:
        return f"sentiment {sentiment_score} < shadow floor {MIN_SENTIMENT}"

    if not SKIP_VOLUME and volume_ratio is not None and volume_ratio < 1.0:
        # Shadow uses a softer 1.0× floor (vs live 1.5×)
        return f"volume ratio {volume_ratio:.2f} < 1.0"

    # Check if shadow already has an open position in this ticker
    existing = database.get_open_shadow_trade_by_ticker(ticker)
    if existing:
        return f"already open shadow position for {ticker}"

    return None   # approved


def _aggregate_stats(trades: list[dict], pnl_key: str = "pnl_gross") -> dict:
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "total_pnl": 0, "avg_pnl": 0,
            "best": 0, "worst": 0,
        }
    pnls  = [t.get(pnl_key) or 0 for t in trades]
    wins  = sum(1 for p in pnls if p > 0)
    total = len(pnls)
    return {
        "total":     total,
        "wins":      wins,
        "losses":    total - wins,
        "win_rate":  round(wins / total * 100, 1) if total else 0,
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl":   round(sum(pnls) / total, 2) if total else 0,
        "best":      round(max(pnls), 2) if pnls else 0,
        "worst":     round(min(pnls), 2) if pnls else 0,
    }


def _fetch_price(ticker: str) -> float | None:
    now = time.time()
    with _price_lock:
        cached = _price_cache.get(ticker)
    if cached and now - cached[1] < _PRICE_TTL:
        return cached[0]
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            if price > 0:
                with _price_lock:
                    _price_cache[ticker] = (price, now)
                return price
    except Exception as exc:
        logger.warning(f"[SHADOW] price fetch failed for {ticker}: {exc}")
    return None


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
