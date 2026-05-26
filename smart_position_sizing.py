"""
Smart Position Sizing
======================

Uses Kelly Criterion + confidence score to calculate optimal position size.

Instead of fixed MAX_POSITION_PCT:
  - Score 65-70 → 8% of budget
  - Score 70-80 → 12% of budget
  - Score 80-90 → 16% of budget
  - Score 90+   → 20% of budget (maximum)

Also considers:
  - Current portfolio drawdown (smaller positions when losing)
  - Market volatility (smaller when VIX is high)
  - Consecutive losses (shrink after losses)
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    notional_usd: float      # $ amount to invest
    position_pct: float      # % of budget
    qty: float               # shares to buy
    reasoning: str


def calculate_smart_position(
    score: float,
    price: float,
    available_budget: float,
    total_budget: float,
    consecutive_losses: int = 0,
    market_vix: float = 20.0,
    daily_pnl_pct: float = 0.0,  # today's P&L as % of budget
) -> PositionSizeResult:
    """
    Calculate optimal position size using a tiered score system.

    Returns the dollar amount to invest and share quantity.
    """
    max_pct = float(os.getenv("MAX_POSITION_PCT", "15"))
    min_pct = 5.0  # never go below 5%

    # ── 1. Score-based tier ─────────────────────────────────────────────
    if score >= 90:
        base_pct = max_pct           # Max confidence → max size
    elif score >= 80:
        base_pct = max_pct * 0.85   # ~12.75% of budget
    elif score >= 75:
        base_pct = max_pct * 0.70   # ~10.5%
    elif score >= 70:
        base_pct = max_pct * 0.55   # ~8.25%
    else:
        base_pct = max_pct * 0.40   # ~6% (minimum viable)

    # ── 2. Drawdown protection ──────────────────────────────────────────
    # Reduce size if today has been bad
    if daily_pnl_pct < -3.0:
        base_pct *= 0.5
        logger.info(f"[SIZING] Daily drawdown {daily_pnl_pct:.1f}% → position halved")
    elif daily_pnl_pct < -1.5:
        base_pct *= 0.75

    # ── 3. Consecutive losses shrink ────────────────────────────────────
    if consecutive_losses >= 3:
        base_pct *= 0.4   # 40% of base after 3 losses
        logger.info(f"[SIZING] {consecutive_losses} consecutive losses → 40% size")
    elif consecutive_losses == 2:
        base_pct *= 0.65
    elif consecutive_losses == 1:
        base_pct *= 0.85

    # ── 4. VIX volatility adjustment ────────────────────────────────────
    if market_vix > 30:
        base_pct *= 0.6   # High fear = much smaller
    elif market_vix > 25:
        base_pct *= 0.75
    elif market_vix > 22:
        base_pct *= 0.85
    elif market_vix < 15:
        base_pct *= 1.1   # Very calm market = slightly larger

    # ── 5. Clamp and apply ───────────────────────────────────────────────
    final_pct = max(min_pct, min(max_pct, base_pct))
    notional = total_budget * (final_pct / 100)

    # Never invest more than available cash
    notional = min(notional, available_budget * 0.95)  # keep 5% buffer

    if price > 0:
        qty = notional / price
    else:
        qty = 0.0

    # Build reasoning string
    reasons = []
    if score >= 80:
        reasons.append(f"ציון גבוה {score:.0f}")
    if consecutive_losses > 0:
        reasons.append(f"הפסדים רצופים: {consecutive_losses}")
    if market_vix > 22:
        reasons.append(f"VIX={market_vix:.0f}")
    if daily_pnl_pct < -1.5:
        reasons.append(f"יום גרוע ({daily_pnl_pct:.1f}%)")
    reasoning = " | ".join(reasons) if reasons else "גודל סטנדרטי"

    logger.info(
        f"[SIZING] score={score:.0f} → {final_pct:.1f}% of budget "
        f"= ${notional:.0f} ({qty:.4f} shares @ ${price:.2f}) | {reasoning}"
    )

    return PositionSizeResult(
        notional_usd=notional,
        position_pct=final_pct,
        qty=qty,
        reasoning=reasoning,
    )


def get_consecutive_losses() -> int:
    """Read consecutive losses from circuit breaker state."""
    try:
        from circuit_breaker import _state
        return _state.get("consecutive_losses", 0)
    except Exception:
        return 0


def get_today_pnl_pct() -> float:
    """Get today's P&L as % of budget."""
    try:
        from circuit_breaker import _state
        from config import settings
        daily_pnl = _state.get("daily_pnl", 0)
        return (daily_pnl / settings.MAX_BUDGET * 100) if settings.MAX_BUDGET > 0 else 0
    except Exception:
        return 0
