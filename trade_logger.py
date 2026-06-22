"""
Trade Logger — Structured Execution Audit Trail

Amateur approach: plain-text log lines
    → impossible to aggregate, query, or feed into analytics

Hedge fund approach: every trade event emits a structured JSON record
    containing the complete lifecycle:
    signal → sizing → execution → pnl → tax → learning

This module writes two outputs simultaneously:
    1. Python logger (INFO) — structured JSON string for log aggregators
    2. SQLite (via database.py) — queryable persistent store

The JSON schema is fixed so downstream tools (Grafana, pandas, etc.)
can consume it without parsing.
"""

import json
import logging
from datetime import datetime, timezone
from models import WebhookPayload, SentimentResult
from database import save_trade, close_trade, save_learning_entry

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _emit(event_type: str, payload: dict) -> None:
    """Emit a structured JSON log record."""
    record = {
        "ts":         _utcnow(),
        "event":      event_type,
        **payload,
    }
    logger.info(json.dumps(record, default=str))


def log_trade_open(
    payload:      WebhookPayload,
    sentiment:    SentimentResult | None,
    order_result: dict,
    qty:          float,
    sizing_meta:  dict | None = None,
    slippage_meta: dict | None = None,
) -> int:
    """
    Record a new trade opening.

    Parameters
    ----------
    payload       : incoming webhook signal
    sentiment     : Groq sentiment result (or None)
    order_result  : broker order response
    qty           : shares actually bought
    sizing_meta   : output of budget.compute_position_size() — full sizing breakdown
    slippage_meta : output of slippage.estimate() — execution cost estimate

    Returns
    -------
    trade_id : int  (SQLite primary key)
    """
    _broker_price = order_result.get("price")
    actual_price  = float(_broker_price if _broker_price and _broker_price > 0 else payload.price)

    # Capture the FULL indicator snapshot at entry so the bot can later learn
    # which indicators actually predict outcomes. Previously macd_signal /
    # bb_position / volume_ratio were hardcoded None → the learner was blind to
    # them (bb_position was 0% populated). Best-effort + cached (5min), so it
    # adds no real latency at buy time and never blocks a trade.
    _ind: dict = {}
    try:
        from indicators import get_current_indicators
        _ind = get_current_indicators(payload.ticker.upper()) or {}
    except Exception as _ie:
        logger.debug(f"[TRADE-LOG] indicator snapshot failed for {payload.ticker}: {_ie}")

    trade = {
        "ticker":               payload.ticker.upper(),
        "action":               payload.action.value,
        "qty":                  qty,
        "entry_price":          actual_price,
        "trailing_stop_pct":    None,
        "rsi":                  payload.rsi  if payload.rsi  is not None else _ind.get("rsi"),
        "macd":                 payload.macd if payload.macd is not None else _ind.get("macd"),
        "macd_signal":          _ind.get("macd_signal"),
        "bb_position":          _ind.get("bb_position"),
        "volume_ratio":         _ind.get("volume_ratio"),
        "sentiment_score":      sentiment.score     if sentiment else None,
        "sentiment_reasoning":  sentiment.reasoning if sentiment else None,
    }
    trade_id = save_trade(trade)

    _emit("trade_open", {
        "trade_id":     trade_id,
        "ticker":       trade["ticker"],
        "qty":          qty,
        "entry_price":  actual_price,
        "notional":     round(actual_price * qty, 2),
        "order_id":     order_result.get("order_id"),
        "sentiment":    {
            "score":     sentiment.score     if sentiment else None,
            "reasoning": sentiment.reasoning if sentiment else None,
        },
        "sizing":    sizing_meta   or {},
        "slippage":  slippage_meta or {},
        "signal": {
            "source":   "webhook",
            "interval": getattr(payload, "interval", None),
            "price":    payload.price,
        },
    })

    return trade_id


def log_trade_close(
    trade_id:     int,
    exit_price:   float,
    pnl_gross:    float,
    pnl_net:      float,
    tax_reserved: float,
    fees:         float = 0.0,
    status:       str   = "closed",
    reason:       str   = "",
    slippage_meta: dict | None = None,
) -> None:
    """
    Record a trade closing with full PnL attribution.

    Parameters
    ----------
    trade_id      : SQLite row id of the opening trade
    exit_price    : actual fill price
    pnl_gross     : gross profit/loss in USD
    pnl_net       : net profit/loss after tax
    tax_reserved  : tax amount reserved
    fees          : broker commissions (0 for Alpaca stocks)
    status        : 'closed' | 'stop_loss' | 'take_profit' | 'smart_sell' | 'emergency_exit'
    reason        : human-readable exit reason (e.g. "Stop Loss (-5.2%)")
    slippage_meta : execution cost estimate at exit
    """
    close_trade(trade_id, exit_price, pnl_gross, pnl_net, tax_reserved, fees, status)

    _emit("trade_close", {
        "trade_id":     trade_id,
        "exit_price":   exit_price,
        "status":       status,
        "reason":       reason,
        "pnl": {
            "gross":        round(pnl_gross, 4),
            "net":          round(pnl_net, 4),
            "tax_reserved": round(tax_reserved, 4),
            "fees":         round(fees, 4),
        },
        "slippage": slippage_meta or {},
    })


def log_learning(
    trade_id:   int,
    description: str,
    pattern_type: str,
    indicators:  dict,
    outcome:     str,
    pnl:         float,
) -> None:
    """Record a learning entry after trade close."""
    entry = {
        "trade_id":           trade_id,
        "pattern_type":       pattern_type,
        "description":        description,
        "indicators_snapshot": indicators,
        "outcome":            outcome,
        "pnl":                pnl,
    }
    save_learning_entry(entry)

    _emit("learning", {
        "trade_id":     trade_id,
        "pattern_type": pattern_type,
        "outcome":      outcome,
        "pnl":          round(pnl, 4),
        "description":  description,
    })
