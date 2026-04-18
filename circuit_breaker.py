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

_lock = threading.Lock()
_state = {
    "tripped": False,         # True = circuit is open (no trading)
    "daily_pnl": 0.0,        # today's realized PnL
    "trade_date": None,       # date string (UTC) this state is for
    "trip_reason": "",
}


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reset_if_new_day():
    """Reset circuit breaker state if it's a new trading day."""
    today = _today_utc()
    if _state["trade_date"] != today:
        _state["tripped"] = False
        _state["daily_pnl"] = 0.0
        _state["trade_date"] = today
        _state["trip_reason"] = ""
        logger.info(f"Circuit Breaker: new day {today} — state reset")


def record_trade_result(pnl_gross: float):
    """Call this after every trade closes. Updates daily PnL and trips breaker if needed."""
    with _lock:
        _reset_if_new_day()
        _state["daily_pnl"] += pnl_gross

        max_loss = settings.MAX_BUDGET * (MAX_DAILY_LOSS_PCT / 100)
        if not _state["tripped"] and _state["daily_pnl"] <= -max_loss:
            _state["tripped"] = True
            _state["trip_reason"] = (
                f"Daily loss ${abs(_state['daily_pnl']):.2f} exceeded "
                f"limit ${max_loss:.2f} ({MAX_DAILY_LOSS_PCT}% of ${settings.MAX_BUDGET:,.0f})"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {_state['trip_reason']}")


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
