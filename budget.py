"""
Position Sizing — Risk-per-Trade Model

Amateur approach (what we had before):
    qty = int(available_cash / price)
    → ignores stop-loss, ignores actual risk

Hedge fund approach (what this module does):
    The position size is derived from HOW MUCH WE ARE WILLING TO LOSE,
    not from how much cash we have available.

    qty = dollar_risk_per_trade / risk_per_share
    where:
        dollar_risk_per_trade = account_equity × RISK_PER_TRADE_PCT
        risk_per_share        = entry_price × STOP_LOSS_PCT / 100

    Example:
        account_equity     = $1,000
        RISK_PER_TRADE_PCT = 1%      →  $10 max loss per trade
        entry_price        = $100
        stop_loss          = 5%      →  $5 risk per share
        qty                = $10 / $5 = 2 shares ($200 notional)

    Hard limits still apply on top:
        - Max notional per position  ≤ MAX_POSITION_PCT of budget
        - Max total open positions   ≤ MAX_OPEN_POSITIONS
        - Must have enough cash

Environment variables:
    RISK_PER_TRADE_PCT    float   default 1.0   (% of equity risked per trade)
    MAX_POSITION_PCT      float   default 20.0  (max notional % of total budget)
    MAX_OPEN_POSITIONS    int     default 5
"""

import os
import logging
from config import settings
import broker

logger = logging.getLogger(__name__)

# ── Risk parameters ────────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT:  float = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
MAX_OPEN_POSITIONS:  int   = int(os.getenv("MAX_OPEN_POSITIONS", "5"))


def _get_account_equity() -> tuple[float, float]:
    """
    Returns (equity, cash).
    equity = cash + open positions market value.
    Raises on broker failure (caller handles).
    """
    account   = broker.get_account()
    positions = broker.get_positions()
    cash      = float(account["cash"])
    pos_value = sum(float(p["market_value"]) for p in positions)
    equity    = cash + pos_value
    return equity, cash


def compute_position_size(entry_price: float) -> tuple[int, dict]:
    """
    Compute the correct number of shares to buy using risk-per-trade sizing.

    Returns:
        (qty, metadata_dict)
        qty = 0 means "do not trade"

    metadata_dict contains the full sizing breakdown for logging/audit.
    """
    if entry_price <= 0:
        return 0, {"rejected": "invalid_price"}

    equity, cash = _get_account_equity()
    stop_loss_pct = settings.STOP_LOSS_PCT  # e.g. 5.0

    # ── Step 1: Dollar risk budget for this trade ──────────────────────────────
    dollar_risk = equity * (RISK_PER_TRADE_PCT / 100)

    # ── Step 2: Risk per share (distance to stop loss) ────────────────────────
    risk_per_share = entry_price * (stop_loss_pct / 100)
    if risk_per_share <= 0:
        return 0, {"rejected": "zero_risk_per_share"}

    # ── Step 3: Risk-derived qty ───────────────────────────────────────────────
    risk_qty = int(dollar_risk / risk_per_share)

    # ── Step 4: Hard cap — max notional per position ──────────────────────────
    max_notional   = settings.MAX_BUDGET * (settings.MAX_POSITION_PCT / 100)
    notional_qty   = int(max_notional / entry_price)

    # ── Step 5: Cash constraint ────────────────────────────────────────────────
    cash_qty = int(cash / entry_price)

    # ── Step 6: Take the most conservative ────────────────────────────────────
    qty = min(risk_qty, notional_qty, cash_qty)

    metadata = {
        "equity":           round(equity, 2),
        "cash":             round(cash, 2),
        "dollar_risk":      round(dollar_risk, 2),
        "risk_pct":         RISK_PER_TRADE_PCT,
        "stop_loss_pct":    stop_loss_pct,
        "risk_per_share":   round(risk_per_share, 4),
        "risk_qty":         risk_qty,
        "notional_cap_qty": notional_qty,
        "cash_qty":         cash_qty,
        "final_qty":        qty,
        "notional":         round(qty * entry_price, 2),
        "binding_constraint": (
            "risk"     if qty == risk_qty     else
            "notional" if qty == notional_qty else
            "cash"
        ),
    }

    logger.info(
        f"[SIZING] entry=${entry_price:.2f} | risk=${dollar_risk:.2f} "
        f"({RISK_PER_TRADE_PCT}% of ${equity:.0f}) | "
        f"stop=${risk_per_share:.2f}/share | "
        f"qty={qty} (bound by {metadata['binding_constraint']}) | "
        f"notional=${metadata['notional']:.2f}"
    )

    return qty, metadata


def check_can_buy(price: float) -> tuple[bool, int, str]:
    """
    Public API — drop-in replacement for the old check_can_buy.
    Returns (can_buy, qty, reason).
    """
    try:
        from database import get_open_trades
        open_trades = get_open_trades()
        if len(open_trades) >= MAX_OPEN_POSITIONS:
            return False, 0, (
                f"Max open positions reached ({len(open_trades)}/{MAX_OPEN_POSITIONS})"
            )

        qty, meta = compute_position_size(price)

        if qty <= 0:
            return False, 0, (
                f"Position size = 0 | risk=${meta.get('dollar_risk', 0):.2f} "
                f"risk_per_share=${meta.get('risk_per_share', 0):.2f} "
                f"cash=${meta.get('cash', 0):.2f}"
            )

        return True, qty, (
            f"qty={qty} | notional=${meta['notional']:.2f} | "
            f"max_loss=${meta['dollar_risk']:.2f} | "
            f"bound_by={meta['binding_constraint']}"
        )

    except Exception as e:
        logger.error(f"check_can_buy failed: {e}")
        return False, 0, f"Sizing error: {e}"


def get_budget_status() -> dict:
    """Full portfolio snapshot for /status endpoint and heartbeat."""
    try:
        account   = broker.get_account()
        positions = broker.get_positions()
    except Exception as e:
        logger.error(f"get_budget_status broker error: {e}")
        return {
            "total_budget":      settings.MAX_BUDGET,
            "cash_available":    0,
            "positions_value":   0,
            "open_pnl":          0,
            "budget_used_pct":   0,
            "error":             str(e),
        }

    pos_value  = sum(float(p["market_value"])    for p in positions)
    open_pnl   = sum(float(p["unrealized_pl"])   for p in positions)
    cash       = float(account["cash"])
    equity     = cash + pos_value

    from database import get_tax_summary
    tax = get_tax_summary()

    return {
        "total_budget":        settings.MAX_BUDGET,
        "cash_available":      round(cash, 2),
        "positions_value":     round(pos_value, 2),
        "open_pnl":            round(open_pnl, 2),
        "equity":              round(equity, 2),
        "realized_pnl_gross":  round(tax["realized_pnl_gross"], 2),
        "realized_pnl_net":    round(tax["realized_pnl_net"], 2),
        "tax_reserved":        round(tax["tax_reserved"], 2),
        "tax_credit":          round(tax["tax_credit"], 2),
        "budget_used_pct":     round(pos_value / settings.MAX_BUDGET * 100, 2)
                               if settings.MAX_BUDGET > 0 else 0,
        "risk_per_trade_pct":  RISK_PER_TRADE_PCT,
        "max_open_positions":  MAX_OPEN_POSITIONS,
    }


def calculate_position_size(price: float) -> int:
    """Backward-compat alias."""
    qty, _ = compute_position_size(price)
    return qty
