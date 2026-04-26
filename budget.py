"""
Position Sizing — Risk-per-Trade + Kelly Criterion

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

    Kelly Criterion overlay (optional, kicks in after 10+ closed trades):
        f* = (b×p − q) / b       (full Kelly)
        f  = f* / 2              (half-Kelly for risk control)
        kelly_notional = equity × f
        kelly_qty      = kelly_notional / price

    Final qty = min(risk_qty, notional_cap_qty, cash_qty, kelly_qty)
    Kelly only constrains — it never sizes UP beyond risk-per-trade.

    Example — edge deteriorates (win_rate falls from 60% → 45%):
        b=1.5, p=0.45, q=0.55 → f*=(1.5×0.45−0.55)/1.5=0.08 → f=0.04
        On $10,000 account: kelly_notional=$400 → 4 shares at $100
        Risk-per-trade at 1% risk: $100/5=$20/share → 5 shares
        Kelly caps at 4 shares — automatically reducing size as edge shrinks.

    Hard limits still apply on top:
        - Max notional per position  ≤ MAX_POSITION_PCT of budget
        - Max total open positions   ≤ MAX_OPEN_POSITIONS
        - Must have enough cash

Environment variables:
    RISK_PER_TRADE_PCT    float   default 1.0   (% of equity risked per trade)
    MAX_POSITION_PCT      float   default 20.0  (max notional % of total budget)
    MAX_OPEN_POSITIONS    int     default 5
    KELLY_ENABLED         bool    default true  (use Kelly overlay)
    KELLY_MIN_TRADES      int     default 10    (min closed trades for Kelly to activate)
"""

import os
import logging
from config import settings
import broker

logger = logging.getLogger(__name__)

# ── Risk parameters ────────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT:  float = float(os.getenv("RISK_PER_TRADE_PCT",  "10.0"))
# MAX_OPEN_POSITIONS read from settings to ensure same default as config.py (6)
MAX_OPEN_POSITIONS:  int   = settings.MAX_OPEN_POSITIONS
KELLY_ENABLED:       bool  = os.getenv("KELLY_ENABLED", "true").lower() == "true"
KELLY_MIN_TRADES:    int   = int(os.getenv("KELLY_MIN_TRADES",       "10"))


def kelly_fraction() -> float:
    """
    Compute the Half-Kelly optimal bet fraction using closed trade history.

    Formula: f* = (b×p − q) / b  then halved for conservatism.
      b = avg_win / avg_loss  (win-to-loss dollar ratio)
      p = historical win rate
      q = 1 − p

    Returns f ∈ [0, MAX_POSITION_PCT/100].
    Returns 0.0 if there are fewer than KELLY_MIN_TRADES closed trades
    (not enough history for a reliable estimate).

    Notes
    -----
    - Negative f* (no edge) → returns 0.0 (don't size based on Kelly)
    - Half-Kelly halves both the expected gain AND variance vs. full Kelly
    - Always capped at MAX_POSITION_PCT to respect hard notional limits
    """
    if not KELLY_ENABLED:
        return 0.0

    try:
        from database import get_win_trades, get_loss_trades
        wins   = get_win_trades(limit=200)
        losses = get_loss_trades(limit=200)

        n_wins   = len(wins)
        n_losses = len(losses)
        total    = n_wins + n_losses

        if total < KELLY_MIN_TRADES:
            logger.debug(
                f"[KELLY] only {total} closed trades — need {KELLY_MIN_TRADES} "
                f"for reliable estimate; skipping"
            )
            return 0.0

        p = n_wins / total
        q = 1.0 - p

        avg_win  = (
            sum(t.get("pnl_gross", 0) or 0 for t in wins)  / n_wins
            if n_wins > 0 else 0.0
        )
        avg_loss = (
            sum(abs(t.get("pnl_gross", 0) or 0) for t in losses) / n_losses
            if n_losses > 0 else 0.0
        )

        if avg_loss <= 0 or avg_win <= 0:
            return 0.0

        b      = avg_win / avg_loss      # win/loss ratio
        f_full = (b * p - q) / b        # full Kelly fraction

        if f_full <= 0:
            logger.info(
                f"[KELLY] negative edge detected: f*={f_full:.4f} "
                f"(p={p:.2%}, b={b:.2f}) — no Kelly sizing"
            )
            return 0.0

        f_half  = f_full / 2.0
        max_f   = settings.MAX_POSITION_PCT / 100.0
        f_final = min(f_half, max_f)

        logger.info(
            f"[KELLY] p={p:.2%} | W/L={b:.2f} | f*={f_full:.4f} | "
            f"half-f*={f_half:.4f} → capped={f_final:.4f} "
            f"({n_wins}W/{n_losses}L | ⌀win=${avg_win:.2f} ⌀loss=${avg_loss:.2f})"
        )
        return f_final

    except Exception as exc:
        logger.warning(f"[KELLY] computation failed: {exc}")
        return 0.0


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


def compute_position_size(entry_price: float) -> tuple[float, dict]:
    """
    Compute fractional shares to buy using risk-per-trade sizing.
    Supports fractional quantities (e.g. 0.5, 2.37) for high-priced stocks.

    Returns:
        (qty, metadata_dict)   — qty is float, 0.0 means "do not trade"
    """
    if entry_price <= 0:
        return 0.0, {"rejected": "invalid_price"}

    equity, cash = _get_account_equity()
    stop_loss_pct = settings.STOP_LOSS_PCT  # e.g. 5.0

    # ── Step 1: Dollar risk budget for this trade ──────────────────────────────
    dollar_risk = equity * (RISK_PER_TRADE_PCT / 100)

    # ── Step 2: Risk per share (distance to stop loss) ────────────────────────
    risk_per_share = entry_price * (stop_loss_pct / 100)
    if risk_per_share <= 0:
        return 0.0, {"rejected": "zero_risk_per_share"}

    # ── Step 3: Risk-derived qty (fractional) ─────────────────────────────────
    risk_qty = dollar_risk / risk_per_share

    # ── Step 4: Hard cap — max notional per position ──────────────────────────
    max_notional   = settings.MAX_BUDGET * (settings.MAX_POSITION_PCT / 100)
    notional_qty   = max_notional / entry_price

    # ── Step 5: Cash constraint ────────────────────────────────────────────────
    cash_qty = cash / entry_price

    # ── Step 6: Take the most conservative (risk / notional cap / cash) ─────────
    qty = min(risk_qty, notional_qty, cash_qty)

    # ── Step 6b: Minimum 0.01 share fallback ─────────────────────────────────
    # Ensure we always invest something meaningful if we have cash
    MIN_NOTIONAL = 10.0  # minimum $10 investment
    if qty * entry_price < MIN_NOTIONAL and cash >= MIN_NOTIONAL:
        qty = MIN_NOTIONAL / entry_price
        logger.info(f"[SIZING] using minimum notional ${MIN_NOTIONAL} → qty={qty:.4f} @ ${entry_price:.2f}")

    qty = round(qty, 6)

    # ── Step 7: Kelly Criterion overlay ──────────────────────────────────────
    kelly_qty  = qty      # start with unconstrained qty
    kelly_f    = kelly_fraction()
    kelly_note = "disabled"
    if kelly_f > 0:
        kelly_notional = equity * kelly_f
        kelly_qty      = round(kelly_notional / entry_price, 6) if entry_price > 0 else qty
        kelly_note     = f"f={kelly_f:.4f} notional=${kelly_notional:.2f}"
        if kelly_qty < qty:
            qty = kelly_qty   # Kelly tightens the size

    binding = (
        "kelly"    if kelly_f > 0 and qty == kelly_qty and kelly_qty < min(risk_qty, notional_qty, cash_qty) else
        "risk"     if qty == risk_qty     else
        "notional" if qty == notional_qty else
        "cash"
    )

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
        "kelly_qty":        kelly_qty,
        "kelly":            kelly_note,
        "final_qty":        qty,
        "notional":         round(qty * entry_price, 2),
        "binding_constraint": binding,
    }

    logger.info(
        f"[SIZING] entry=${entry_price:.2f} | risk=${dollar_risk:.2f} "
        f"({RISK_PER_TRADE_PCT}% of ${equity:.0f}) | "
        f"stop=${risk_per_share:.2f}/share | kelly={kelly_note} | "
        f"qty={qty} (bound by {binding}) | "
        f"notional=${metadata['notional']:.2f}"
    )

    return qty, metadata


def check_can_buy(price: float) -> tuple[bool, float, str]:
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
