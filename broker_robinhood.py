import logging
from datetime import date
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    import robin_stocks.robinhood as r
except ImportError:
    raise ImportError("pip install robin_stocks>=2.1.0")


class RobinhoodBroker(BrokerBase):
    """Robinhood broker via robin_stocks library."""

    def __init__(self):
        self._email = settings.ROBINHOOD_EMAIL
        self._password = settings.ROBINHOOD_PASSWORD

        if not self._email or not self._password:
            raise ValueError(
                "Robinhood credentials not configured. Set ROBINHOOD_EMAIL and ROBINHOOD_PASSWORD."
            )

        self._logged_in = False

    def _ensure_login(self):
        if not self._logged_in:
            r.login(self._email, self._password)
            self._logged_in = True
            logger.info("Robinhood login successful.")

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            self._ensure_login()
            profile = r.load_account_profile()
            portfolio = r.load_portfolio_profile()
            cash = float(profile.get("cash", 0) or 0)
            buying_power = float(profile.get("buying_power", 0) or 0)
            portfolio_value = float(portfolio.get("market_value", 0) or 0)
            return {
                "cash": cash,
                "buying_power": buying_power,
                "portfolio_value": portfolio_value,
            }
        except Exception as e:
            logger.error(f"Robinhood get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}

    def get_positions(self) -> list[dict]:
        try:
            self._ensure_login()
            raw = r.get_open_stock_positions()
            result = []
            for pos in raw:
                qty = float(pos.get("quantity", 0) or 0)
                if qty <= 0:
                    continue
                ticker = pos.get("symbol", "")
                if not ticker:
                    # Resolve instrument URL to ticker
                    try:
                        instrument = r.get_instrument_by_url(pos.get("instrument", ""))
                        ticker = instrument.get("symbol", "")
                    except Exception:
                        ticker = ""
                avg_cost = float(pos.get("average_buy_price", 0) or 0)
                try:
                    quote = r.get_latest_price(ticker)
                    current_price = float(quote[0]) if quote else 0.0
                except Exception:
                    current_price = 0.0
                market_value = qty * current_price
                unrealized_pl = (current_price - avg_cost) * qty
                result.append({
                    "ticker": ticker,
                    "qty": qty,
                    "avg_cost": avg_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                })
            return result
        except Exception as e:
            logger.error(f"Robinhood get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"Robinhood get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        self._ensure_login()
        try:
            order = r.order_buy_market(ticker.upper(), qty)
        except Exception as e:
            logger.error(f"Robinhood BUY failed: {e}")
            raise
        order_id = str(order.get("id", "unknown"))
        status = str(order.get("state", "submitted")).lower()
        logger.info(f"Robinhood BUY submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "qty": float(qty),
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        self._ensure_login()
        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]
        try:
            order = r.order_sell_market(ticker.upper(), qty)
        except Exception as e:
            logger.error(f"Robinhood SELL failed: {e}")
            raise
        order_id = str(order.get("id", "unknown"))
        status = str(order.get("state", "submitted")).lower()
        logger.info(f"Robinhood SELL submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "qty": float(qty),
            "status": status,
        }

    def is_market_open(self) -> bool:
        try:
            self._ensure_login()
            today = date.today().isoformat()
            hours = r.get_market_hours("XNYS", today)
            if not hours:
                return False
            return bool(hours.get("is_open", False))
        except Exception as e:
            logger.error(f"Robinhood is_market_open failed: {e}")
            return False

    def get_asset(self, ticker: str) -> dict | None:
        try:
            self._ensure_login()
            instrument = r.get_instruments_by_symbols(ticker.upper())
            if not instrument:
                return None
            item = instrument[0] if isinstance(instrument, list) else instrument
            tradable = item.get("tradability", "") == "tradable"
            return {"tradable": tradable}
        except Exception as e:
            logger.warning(f"Robinhood get_asset({ticker}) failed: {e}")
            return None
