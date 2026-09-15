"""
Broker cost model — what a round trip actually costs, per venue.

Built because the raw strategy numbers were being read as if trading were free.
Measured gross return is -0.031% per trade; at $1,000 positions that is -$0.31,
while a round trip costs $5.00 at IBKR and $2.00 at Colmex. Fees are not a
rounding error here, they are several times the edge, so any "is this
profitable?" question has to be asked net of them.

Rates below are user-supplied (2026-08-12), confirmed with each broker:

  IBKR:   $2.50 per order for the first 250 shares, then $0.01/share
  Colmex: $1.00 per order for the first 200 shares, then $0.005/share
          (on a $10k+ account; smaller accounts are quoted differently)

CFD caveat: Colmex trades US stocks largely as CFDs, which charge overnight
financing for every night a position is held. This bot holds up to 72h, so that
can cost more than the commission it saves. The rate is NOT known — set
COLMEX_OVERNIGHT_PCT once Colmex confirms it, or leave it at 0 and treat Colmex
figures as a best case.
"""

from dataclasses import dataclass

# Annualised overnight financing on CFD positions, as a decimal (0.06 = 6%/yr).
# 0.0 = unknown/not applicable. Colmex has not confirmed a rate yet, so Colmex
# numbers are optimistic until this is filled in.
COLMEX_OVERNIGHT_PCT: float = 0.0


@dataclass(frozen=True)
class TradeCost:
    broker: str
    commission: float      # both legs
    financing: float       # overnight/swap for the holding period
    total: float
    breakeven_pct: float   # gross % move needed just to cover cost

    def __str__(self) -> str:
        return (f"{self.broker}: ${self.total:.2f} "
                f"(commission ${self.commission:.2f}, financing ${self.financing:.2f}) "
                f"→ breakeven {self.breakeven_pct:+.3f}%")


def _ibkr_commission(shares: float) -> float:
    """One order. $2.50 covers the first 250 shares, $0.01 each beyond."""
    return 2.50 + max(0.0, shares - 250) * 0.01


def _colmex_commission(shares: float) -> float:
    """One order. $1.00 covers the first 200 shares, $0.005 each beyond."""
    return 1.00 + max(0.0, shares - 200) * 0.005


def round_trip(broker: str, price: float, notional: float,
               hold_hours: float = 24.0) -> TradeCost:
    """
    Full cost of buying and later selling `notional` dollars of a `price` stock.

    hold_hours drives CFD financing only; it costs nothing on real shares.
    """
    if price <= 0 or notional <= 0:
        raise ValueError("price and notional must be positive")

    shares = int(notional / price)          # whole shares — IBKR rejects fractions
    if shares < 1:
        raise ValueError(
            f"${notional:,.0f} cannot buy even one share at ${price:,.2f}"
        )
    actual = shares * price                 # what is really deployed after rounding

    b = broker.lower()
    if b == "ibkr":
        commission = _ibkr_commission(shares) * 2
        financing = 0.0                     # real shares, no carry
    elif b == "colmex":
        commission = _colmex_commission(shares) * 2
        nights = max(0.0, hold_hours / 24.0)
        financing = actual * COLMEX_OVERNIGHT_PCT / 365.0 * nights
    else:
        raise ValueError(f"unknown broker '{broker}' — expected 'ibkr' or 'colmex'")

    total = commission + financing
    return TradeCost(
        broker=b,
        commission=commission,
        financing=financing,
        total=total,
        breakeven_pct=total / actual * 100,
    )


def net_return_pct(gross_pct: float, broker: str, price: float,
                   notional: float, hold_hours: float = 24.0) -> float:
    """Gross % return minus this venue's costs — the number that matters."""
    return gross_pct - round_trip(broker, price, notional, hold_hours).breakeven_pct


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    GROSS = -0.031          # measured over 218 closed trades
    NOTIONAL = 1000.0

    print(f"Measured gross return: {GROSS:+.3f}% per trade\n")
    for price in (50, 100, 200, 368):
        print(f"--- ${price} stock, ${NOTIONAL:,.0f} position "
              f"({int(NOTIONAL / price)} shares) ---")
        for br in ("ibkr", "colmex"):
            c = round_trip(br, price, NOTIONAL, hold_hours=24)
            print(f"  {c}")
            print(f"    net: {net_return_pct(GROSS, br, price, NOTIONAL):+.3f}%")
        print()
