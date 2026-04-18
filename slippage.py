"""
Slippage & Limit Order Module

Instead of plain Market Orders (fill at any price), the bot now submits
"aggressive limit orders" — limit orders priced slightly above/below market
to guarantee fast execution while controlling worst-case slippage.

  BUY  limit = market_price × (1 + BUY_OFFSET_PCT)   → e.g. $100 → $100.10
  SELL limit = market_price × (1 - SELL_OFFSET_PCT)  → e.g. $100 → $99.90

For paper trading (TVPaperBroker) this simulates realistic fill costs.
For real brokers (Alpaca, etc.) the limit price is passed to the order.

Configure via env vars:
  SLIPPAGE_BUY_PCT   (default 0.10%)
  SLIPPAGE_SELL_PCT  (default 0.10%)
"""
import os
import logging

logger = logging.getLogger(__name__)

# Default 0.10% offset — tight enough to fill quickly, wide enough to avoid rejection
BUY_OFFSET_PCT:  float = float(os.getenv("SLIPPAGE_BUY_PCT",  "0.001"))
SELL_OFFSET_PCT: float = float(os.getenv("SLIPPAGE_SELL_PCT", "0.001"))


def limit_buy_price(market_price: float) -> float:
    """Return limit buy price: slightly above market to ensure fast fill."""
    return round(market_price * (1 + BUY_OFFSET_PCT), 4)


def limit_sell_price(market_price: float) -> float:
    """Return limit sell price: slightly below market to ensure fast fill."""
    return round(market_price * (1 - SELL_OFFSET_PCT), 4)


def estimate(market_price: float, qty: int, side: str) -> dict:
    """
    Estimate slippage cost for a trade.

    Returns a dict with:
      market_price, limit_price, slippage_per_share,
      total_slippage_usd, slippage_pct, qty, side
    """
    if side == "buy":
        lp = limit_buy_price(market_price)
    else:
        lp = limit_sell_price(market_price)

    slip_per_share = abs(lp - market_price)
    total_slip     = round(slip_per_share * qty, 4)
    slip_pct       = round((slip_per_share / market_price) * 100, 4) if market_price else 0

    result = {
        "side":               side,
        "qty":                qty,
        "market_price":       round(market_price, 4),
        "limit_price":        lp,
        "slippage_per_share": round(slip_per_share, 4),
        "total_slippage_usd": total_slip,
        "slippage_pct":       slip_pct,
    }

    logger.info(
        f"[SLIPPAGE] {side.upper()} {qty}× | "
        f"market=${market_price:.4f} → limit=${lp:.4f} | "
        f"cost=${total_slip:.4f} ({slip_pct:.3f}%)"
    )
    return result
