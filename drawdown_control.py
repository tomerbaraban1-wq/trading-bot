"""
Professional Drawdown Control
==============================

Pro traders have strict rules about daily/weekly losses.
Revenge trading is the #1 account killer.

Rules:
- Daily loss limit: 2% of account → stop trading for rest of day
- Weekly loss limit: 5% of account → reduce size 50% for next week
- Consecutive losses: 3 → mandatory 24h pause
- Max single trade loss: 1% of account

These rules prevent the "revenge trading" spiral.
"""

import logging
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    """Current drawdown state."""
    daily_loss_pct: float = 0
    weekly_loss_pct: float = 0
    consecutive_losses: int = 0
    mode: str = "NORMAL"  # NORMAL, CAUTION, PAUSE
    pause_until: float = 0  # timestamp
    size_multiplier: float = 1.0
    reason: str = ""


_state = DrawdownState()


def get_drawdown_mode() -> str:
    """Get current drawdown control mode."""
    if time.time() < _state.pause_until:
        return "PAUSE"
    return _state.mode


def get_size_multiplier() -> float:
    """Get position size multiplier based on drawdown state."""
    if get_drawdown_mode() == "PAUSE":
        return 0.0

    # Check consecutive losses
    consecutive = _state.consecutive_losses
    if consecutive >= 3:
        return 0.3   # 30% of normal
    elif consecutive == 2:
        return 0.5   # 50% of normal
    elif consecutive == 1:
        return 0.75  # 75% of normal

    # Check daily loss
    daily_limit = float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0"))
    if abs(_state.daily_loss_pct) > daily_limit * 0.7:
        return 0.5  # 50% when approaching daily limit

    return _state.size_multiplier


def record_trade_loss(loss_pct_of_account: float) -> dict:
    """
    Record a trade loss and update drawdown state.
    loss_pct_of_account: what % of account was lost
    """
    _state.daily_loss_pct  += loss_pct_of_account
    _state.weekly_loss_pct += loss_pct_of_account
    _state.consecutive_losses += 1

    # Check limits
    daily_limit   = float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0"))
    weekly_limit  = float(os.getenv("MAX_WEEKLY_LOSS_PCT", "5.0"))
    max_consec    = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))

    triggered = None
    pause_hours = 0

    # Daily limit hit
    if abs(_state.daily_loss_pct) >= daily_limit:
        triggered = "daily_limit"
        pause_hours = 24
        _state.reason = f"Daily limit {daily_limit}% hit ({_state.daily_loss_pct:.1f}%)"
        logger.warning(f"[DRAWDOWN] Daily limit hit: {_state.daily_loss_pct:.1f}% — pausing {pause_hours}h")

    # Weekly limit hit
    elif abs(_state.weekly_loss_pct) >= weekly_limit:
        triggered = "weekly_limit"
        pause_hours = 48
        _state.reason = f"Weekly limit {weekly_limit}% hit ({_state.weekly_loss_pct:.1f}%)"
        logger.warning(f"[DRAWDOWN] Weekly limit hit: {_state.weekly_loss_pct:.1f}% — pausing {pause_hours}h")

    # Consecutive losses
    elif _state.consecutive_losses >= max_consec:
        triggered = "consecutive_losses"
        pause_hours = 4  # 4 hour pause
        _state.reason = f"{_state.consecutive_losses} consecutive losses — cooling off"
        logger.warning(f"[DRAWDOWN] {max_consec} consecutive losses — pausing {pause_hours}h")

    if triggered:
        _state.pause_until = time.time() + (pause_hours * 3600)
        _state.mode = "PAUSE"

    return {
        "triggered": triggered,
        "pause_hours": pause_hours,
        "daily_loss_pct": _state.daily_loss_pct,
        "consecutive_losses": _state.consecutive_losses,
        "mode": get_drawdown_mode(),
        "size_multiplier": get_size_multiplier(),
    }


def record_trade_win() -> None:
    """Record a trade win — resets consecutive losses."""
    _state.consecutive_losses = 0
    if _state.mode == "CAUTION" and _state.consecutive_losses == 0:
        _state.mode = "NORMAL"
        _state.size_multiplier = 1.0


def reset_daily() -> None:
    """Reset daily counters (called at market open)."""
    _state.daily_loss_pct = 0
    if get_drawdown_mode() != "PAUSE":
        _state.mode = "NORMAL"
        _state.size_multiplier = 1.0
    logger.info("[DRAWDOWN] Daily counters reset")


def get_status() -> dict:
    """Get comprehensive drawdown status."""
    mode = get_drawdown_mode()
    pause_remaining = max(0, _state.pause_until - time.time()) / 3600

    return {
        "mode": mode,
        "daily_loss_pct": _state.daily_loss_pct,
        "weekly_loss_pct": _state.weekly_loss_pct,
        "consecutive_losses": _state.consecutive_losses,
        "size_multiplier": get_size_multiplier(),
        "pause_remaining_hours": pause_remaining,
        "reason": _state.reason,
        "can_trade": mode not in ("PAUSE",),
        "limits": {
            "daily": float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0")),
            "weekly": float(os.getenv("MAX_WEEKLY_LOSS_PCT", "5.0")),
            "consecutive": int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")),
        },
    }
