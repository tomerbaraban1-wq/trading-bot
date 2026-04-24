import time
import threading
import logging
try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required: pip install yfinance")

from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)


class TVPaperBroker(BrokerBase):
    """
    Built-in paper trading simulation — no API keys needed.

    Mimics TradingView's own paper trading feature. Prices are fetched
    in real time via yfinance. All state (positions, cash) is kept in
    class-level variables so it survives broker hot-swaps within the
    same Python process.
    """

    # Class-level state — shared across all instances
    _positions: dict = {}   # ticker -> {"qty": int, "avg_cost": float}
    _cash: float = None     # initialised lazily from settings.MAX_BUDGET
    _lock = threading.Lock()  # guards buy/sell mutations against race conditions

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self):
        if TVPaperBroker._cash is None:
            TVPaperBroker._cash = float(settings.MAX_BUDGET)
        logger.info(
            "TradingView Paper Broker ready | "
            f"Cash: ${TVPaperBroker._cash:,.2f} | "
            f"Positions: {list(TVPaperBroker._positions.keys()) or 'none'}"
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_price(self, ticker: str) -> float:
        """Fetch the latest market price for *ticker* via yfinance."""
        ticker = ticker.upper()
        try:
            t = yf.Ticker(ticker)

            # Try intraday history first (most accurate live price)
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                if price > 0:
                    return price

            # Fall back to .info dict
            info = t.info
            price = float(
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
                or 0
            )
            if price > 0:
                return price

        except Exception as exc:
            raise RuntimeError(
                f"yfinance failed to fetch price for '{ticker}': {exc}"
            ) from exc

        raise RuntimeError(
            f"Could not obtain a valid price for '{ticker}' from yfinance. "
            "The ticker may be invalid or the market data feed is unavailable."
        )

    @staticmethod
    def _order_id(ticker: str) -> str:
        return f"TV_{ticker.upper()}_{int(time.time())}"

    def _portfolio_value(self) -> float:
        """Current market value of all virtual positions."""
        total = 0.0
        for ticker, pos in TVPaperBroker._positions.items():
            try:
                price = self._get_price(ticker)
            except Exception:
                price = pos["avg_cost"]   # fall back to cost if price unavailable
            total += price * pos["qty"]
        return total

    # ------------------------------------------------------------------ #
    #  BrokerBase implementation                                           #
    # ------------------------------------------------------------------ #

    def get_account(self) -> dict:
        cash = TVPaperBroker._cash
        portfolio_value = self._portfolio_value()
        equity = cash + portfolio_value
        return {
            "equity": round(equity, 2),
            "buying_power": round(cash, 2),
            "cash": round(cash, 2),
            "portfolio_value": round(portfolio_value, 2),
            "status": "ACTIVE",
        }

    def get_positions(self) -> list[dict]:
        result = []
        for ticker, pos in TVPaperBroker._positions.items():
            try:
                current_price = self._get_price(ticker)
            except Exception:
                current_price = pos["avg_cost"]

            qty = pos["qty"]
            avg_cost = pos["avg_cost"]
            market_value = current_price * qty
            unrealized_pl = (current_price - avg_cost) * qty
            unrealized_plpc = (
                (current_price - avg_cost) / avg_cost if avg_cost else 0.0
            )

            result.append(
                {
                    "ticker": ticker,
                    "qty": qty,
                    "avg_entry_price": round(avg_cost, 4),
                    "current_price": round(current_price, 4),
                    "market_value": round(market_value, 2),
                    "unrealized_pl": round(unrealized_pl, 2),
                    "unrealized_plpc": round(unrealized_plpc, 6),
                }
            )
        return result

    def get_position(self, ticker: str) -> dict | None:
        ticker = ticker.upper()
        pos = TVPaperBroker._positions.get(ticker)
        if not pos:
            return None

        try:
            current_price = self._get_price(ticker)
        except Exception:
            current_price = pos["avg_cost"]

        qty = pos["qty"]
        avg_cost = pos["avg_cost"]
        market_value = current_price * qty
        unrealized_pl = (current_price - avg_cost) * qty
        unrealized_plpc = (current_price - avg_cost) / avg_cost if avg_cost else 0.0

        return {
            "ticker": ticker,
            "qty": qty,
            "avg_entry_price": round(avg_cost, 4),
            "current_price": round(current_price, 4),
            "market_value": round(market_value, 2),
            "unrealized_pl": round(unrealized_pl, 2),
            "unrealized_plpc": round(unrealized_plpc, 6),
        }

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        """Supports fractional shares — qty can be e.g. 0.5 or 2.37."""
        ticker = ticker.upper()

        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        # Use provided price or fetch from yfinance (outside lock - I/O)
        if price and price > 0:
            current_price = float(price)
        else:
            current_price = self._get_price(ticker)  # raises if price = 0

        # Round fractional qty to 6 decimal places
        qty = round(float(qty), 6)
        cost = current_price * qty

        # Atomic cash check + mutation under lock to prevent over-spending race
        with TVPaperBroker._lock:
            if cost > TVPaperBroker._cash:
                raise ValueError(
                    f"Insufficient virtual cash: need ${cost:,.2f} "
                    f"but only ${TVPaperBroker._cash:,.2f} available"
                )

            # Update position (weighted average cost if already held)
            if ticker in TVPaperBroker._positions:
                existing = TVPaperBroker._positions[ticker]
                old_qty = existing["qty"]
                old_cost = existing["avg_cost"]
                new_qty = old_qty + qty
                new_avg = (old_cost * old_qty + current_price * qty) / new_qty
                TVPaperBroker._positions[ticker] = {"qty": new_qty, "avg_cost": new_avg}
            else:
                TVPaperBroker._positions[ticker] = {"qty": qty, "avg_cost": current_price}

            TVPaperBroker._cash -= cost

        order_id = self._order_id(ticker)
        logger.info(
            f"[TVPaper] BUY {ticker} x{qty:.4f} @ ${current_price:.4f} "
            f"| cost=${cost:,.2f} | cash_remaining=${TVPaperBroker._cash:,.2f}"
        )
        return {
            "order_id": order_id,
            "symbol": ticker,
            "qty": qty,
            "price": round(current_price, 4),
            "cost": round(cost, 2),
            "status": "filled",
            "type": "market",
        }

    def submit_sell(self, ticker: str, qty: int | None = None) -> dict:
        ticker = ticker.upper()

        pos = TVPaperBroker._positions.get(ticker)
        if not pos:
            raise ValueError(f"No open virtual position for {ticker}")

        held_qty = pos["qty"]

        if qty is None or qty >= held_qty:
            # Sell the entire position
            sell_qty = held_qty
        else:
            if qty <= 0:
                raise ValueError(f"qty must be positive, got {qty}")
            sell_qty = qty

        current_price = self._get_price(ticker)  # raises if price = 0
        proceeds = current_price * sell_qty

        # Atomic state mutation under lock
        with TVPaperBroker._lock:
            if sell_qty >= held_qty:
                TVPaperBroker._positions.pop(ticker, None)
            else:
                TVPaperBroker._positions[ticker]["qty"] = held_qty - sell_qty

            TVPaperBroker._cash += proceeds

        order_id = self._order_id(ticker)
        logger.info(
            f"[TVPaper] SELL {ticker} x{sell_qty} @ ${current_price:.4f} "
            f"| proceeds=${proceeds:,.2f} | cash_now=${TVPaperBroker._cash:,.2f}"
        )
        return {
            "order_id": order_id,
            "symbol": ticker,
            "qty": sell_qty,
            "price": round(current_price, 4),
            "proceeds": round(proceeds, 2),
            "status": "filled",
        }

    def is_market_open(self) -> bool:
        """
        Check US equity market hours using UTC time (fast, no API call).
        Accounts for EDT (summer) and EST (winter) automatically.
        EDT: 13:30-20:00 UTC | EST: 14:30-21:00 UTC
        """
        import datetime
        now = datetime.datetime.utcnow()
        # Skip weekends
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        # Determine if US is on EDT (2nd Sun Mar - 1st Sun Nov) or EST
        # Simple approximation: EDT runs roughly March 8 – November 1
        month = now.month
        is_edt = 3 <= month <= 10  # close enough for trading purposes
        if is_edt:
            open_hour, open_min = 13, 30   # 9:30 ET = 13:30 UTC
            close_hour, close_min = 20, 0  # 4:00 ET = 20:00 UTC
        else:
            open_hour, open_min = 14, 30   # 9:30 ET = 14:30 UTC
            close_hour, close_min = 21, 0  # 4:00 ET = 21:00 UTC
        market_open  = now.replace(hour=open_hour,  minute=open_min,  second=0, microsecond=0)
        market_close = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)
        return market_open <= now <= market_close

    def get_asset(self, ticker: str) -> dict | None:
        """Always return tradable — paper trading accepts any ticker."""
        return {
            "symbol": ticker.upper(),
            "name": ticker.upper(),
            "tradable": True,
            "fractionable": False,
        }
