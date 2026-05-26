"""
Circuit Breaker — stops all trading if daily loss exceeds the configured limit.

How it works:
  - Every time a trade closes, call record_trade_result(pnl_gross)
  - Before every buy, call check_circuit_breaker()
  - If total realized PnL today drops below -MAX_DAILY_LOSS_PCT of MAX_BUDGET → OPEN circuit
  - Circuit resets automatically at midnight UTC (new trading day)
"""
import logging
import threading
from datetime import datetime, timezone
from config import settings

logger = logging.getLogger(__name__)

# Max daily loss as % of total budget (configurable via env var)
import os
MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))  # default 5%

MAX_DAILY_LOSSES: int = int(os.getenv("MAX_DAILY_LOSSES", "3"))  # stop after N consecutive losses

_lock = threading.Lock()
_state = {
    "tripped": False,         # True = circuit is open (no trading)
    "daily_pnl": 0.0,        # today's realized PnL
    "trade_date": None,       # date string (UTC) this state is for
    "trip_reason": "",
    "consecutive_losses": 0,  # count of consecutive losses today
    "daily_loss_count": 0,    # total losses today
}


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_daily_pnl_from_db() -> float | None:
    """Load today's realized PnL from database (survives restarts).
    Returns None on DB failure so callers can distinguish 'no losses' from 'DB error'."""
    try:
        from database import get_trade_history
        today = _today_utc()
        trades = get_trade_history(limit=500)
        return sum(
            t.get("pnl_gross") or 0
            for t in trades
            if t.get("exit_time") and str(t["exit_time"])[:10] == today
            and t.get("status") in ("closed", "emergency_exit", "stop_loss", "take_profit", "smart_sell", "time_exit", "stale_restart")
        )
    except Exception as exc:
        logger.warning(f"Circuit Breaker: DB load failed ({exc}) — preserving current state")
        return None  # signal failure: caller must not clear existing tripped state


def _reset_if_new_day():
    """Reset circuit breaker state if it's a new trading day."""
    today = _today_utc()
    if _state["trade_date"] != today:
        # Load today's PnL from DB so circuit breaker survives restarts
        daily_pnl = _load_daily_pnl_from_db()

        if daily_pnl is None:
            # DB query failed — update the date but preserve existing tripped/pnl state
            # so we never silently un-trip a live circuit breaker on a DB glitch
            logger.warning(
                f"Circuit Breaker: DB unavailable on day reset — "
                f"keeping tripped={_state['tripped']}, pnl=${_state['daily_pnl']:.2f}"
            )
            _state["trade_date"] = today  # advance the date to avoid infinite retry
            return

        max_loss = settings.MAX_BUDGET * (MAX_DAILY_LOSS_PCT / 100)
        if max_loss <= 0:
            _state["trade_date"] = today  # advance date to avoid infinite retry loop
            return   # misconfigured MAX_BUDGET — don't trip
        # Determine new tripped state BEFORE writing it (atomic decision)
        should_trip = daily_pnl <= -max_loss
        trip_reason = (
            f"Daily loss ${abs(daily_pnl):.2f} exceeded "
            f"limit ${max_loss:.2f} ({MAX_DAILY_LOSS_PCT}% of ${settings.MAX_BUDGET:,.0f})"
        ) if should_trip else ""

        import time as _time_cb
        _state["daily_pnl"]     = daily_pnl
        _state["trade_date"]    = today
        _state["tripped"]       = should_trip
        _state["trip_reason"]   = trip_reason
        _state["_db_loaded_at"] = _time_cb.time()  # timestamp so record_trade_result can detect fresh load

        logger.info(f"Circuit Breaker: initialized for {today} | daily_pnl=${daily_pnl:.2f}")
        if should_trip:
            logger.warning(f"🚨 CIRCUIT BREAKER: tripped on startup from DB data — {trip_reason}")


def record_trade_result(pnl_gross: float):
    """Call this after every trade closes. Updates daily PnL and trips breaker if needed."""
    with _lock:
        _was_uninitialized = _state["trade_date"] is None
        _db_load_ts_before = _state.get("_db_loaded_at", 0)
        _reset_if_new_day()
        _db_load_ts_after  = _state.get("_db_loaded_at", 0)
        # If DB was just loaded successfully, it includes ALL today's trades — don't add again.
        # If DB failed (_db_loaded_at unchanged AND we were uninitialized), add the trade
        # to keep the in-memory counter accurate even without DB data.
        _db_just_loaded = (_db_load_ts_after != _db_load_ts_before) and (_db_load_ts_after > 0)
        if not _db_just_loaded:
            _state["daily_pnl"] += pnl_gross

        # Track consecutive losses for max-daily-losses circuit
        if pnl_gross < 0:
            _state["consecutive_losses"] += 1
            _state["daily_loss_count"] += 1
        else:
            _state["consecutive_losses"] = 0  # reset on win

        max_loss = settings.MAX_BUDGET * (MAX_DAILY_LOSS_PCT / 100)
        if not _state["tripped"] and _state["daily_pnl"] <= -max_loss:
            _state["tripped"] = True
            _state["trip_reason"] = (
                f"Daily loss ${abs(_state['daily_pnl']):.2f} exceeded "
                f"limit ${max_loss:.2f} ({MAX_DAILY_LOSS_PCT}% of ${settings.MAX_BUDGET:,.0f})"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {_state['trip_reason']}")

        # NEW: trip on consecutive losses — prevents averaging down into a crashing market
        if not _state["tripped"] and _state["consecutive_losses"] >= MAX_DAILY_LOSSES:
            _state["tripped"] = True
            _state["trip_reason"] = (
                f"{_state['consecutive_losses']} consecutive losses today — pausing buying"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED (consecutive losses): {_state['trip_reason']}")


def check_circuit_breaker() -> tuple[bool, str]:
    """
    Returns (ok_to_trade, reason).
    ok_to_trade = False means DO NOT place any new buy orders.
    """
    with _lock:
        _reset_if_new_day()
        if _state["tripped"]:
            return False, f"Circuit breaker open: {_state['trip_reason']}"
        return True, ""


def get_status() -> dict:
    """Return current circuit breaker state (for /status endpoint)."""
    with _lock:
        _reset_if_new_day()
        max_loss = settings.MAX_BUDGET * (MAX_DAILY_LOSS_PCT / 100)
        return {
            "tripped": _state["tripped"],
            "daily_pnl": round(_state["daily_pnl"], 2),
            "max_daily_loss": round(max_loss, 2),
            "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
            "trip_reason": _state["trip_reason"],
            "trade_date": _state["trade_date"],
        }
