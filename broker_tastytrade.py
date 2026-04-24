import logging
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    import tastytrade
    from tastytrade import Session, Account
    from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType, InstrumentType
    from tastytrade.instruments import Equity
except ImportError:
    raise ImportError("pip install tastytrade>=8.0")


class TastytradeBroker(BrokerBase):
    """Tastytrade broker implementation via the tastytrade Python SDK."""

    def __init__(self):
        self._username = settings.TASTYTRADE_USERNAME
        self._password = settings.TASTYTRADE_PASSWORD
        self._account_number = settings.TASTYTRADE_ACCOUNT

        if not self._username or not self._password:
            raise ValueError("Tastytrade credentials not configured. Set TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD.")
        if not self._account_number:
            raise ValueError("Tastytrade account not configured. Set TASTYTRADE_ACCOUNT.")

        self._session: Session | None = None
        self._account: Account | None = None

    def _get_session(self) -> Session:
        if self._session is None:
            try:
                self._session = Session(self._username, self._password)
                logger.info("Tastytrade session established.")
            except Exception as e:
                logger.error(f"Tastytrade login failed: {e}")
                raise ConnectionError(f"Tastytrade login failed: {e}") from e
        return self._session

    def _get_account(self) -> Account:
        if self._account is None:
            session = self._get_session()
            self._account = Account.get_account(session, self._account_number)
        return self._account

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            account = self._get_account()
            session = self._get_session()
            balances = account.get_balances(session)
            return {
                "cash": float(getattr(balances, "cash_balance", 0) or 0),
                "buying_power": float(getattr(balances, "equity_buying_power", 0) or 0),
                "portfolio_value": float(getattr(balances, "net_liquidating_value", 0) or 0),
                "equity": float(getattr(balances, "net_liquidating_value", 0) or 0),
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Tastytrade get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0,
                    "equity": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            account = self._get_account()
            session = self._get_session()
            positions = account.get_positions(session)
            result = []
            for p in positions:
                qty = int(getattr(p, "quantity", 0) or 0)
                avg_cost = float(getattr(p, "average_open_price", 0) or 0)
                current_price = float(getattr(p, "close_price", avg_cost) or avg_cost)
                market_value = qty * current_price
                unrealized_pl = (current_price - avg_cost) * qty
                result.append({
                    "ticker": str(p.symbol),
                    "qty": qty,
                    "avg_cost": avg_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                })
            return result
        except Exception as e:
            logger.error(f"Tastytrade get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"Tastytrade get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        session = self._get_session()
        account = self._get_account()
        symbol = ticker.upper()

        equity = Equity.get_equity(session, symbol)
        leg = equity.build_leg(quantity=qty, action=OrderAction.BUY_TO_OPEN)
        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )
        response = account.place_order(session, order, dry_run=False)
        order_id = str(getattr(response.order, "id", "unknown"))
        status = str(getattr(response.order, "status", "submitted"))
        logger.info(f"Tastytrade BUY submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": symbol,
            "qty": qty,
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        session = self._get_session()
        account = self._get_account()
        symbol = ticker.upper()

        if qty is None:
            position = self.get_position(symbol)
            if not position:
                raise ValueError(f"No open position for {symbol}")
            qty = position["qty"]

        equity = Equity.get_equity(session, symbol)
        leg = equity.build_leg(quantity=qty, action=OrderAction.SELL_TO_CLOSE)
        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )
        response = account.place_order(session, order, dry_run=False)
        order_id = str(getattr(response.order, "id", "unknown"))
        status = str(getattr(response.order, "status", "submitted"))
        logger.info(f"Tastytrade SELL submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": symbol,
            "qty": qty,
            "status": status,
        }

    def is_market_open(self) -> bool:
        from datetime import datetime, time as dtime
        try:
            now = datetime.utcnow()
            if now.weekday() >= 5:
                return False
            return dtime(13, 30) <= now.time() < dtime(20, 0)
        except Exception as e:
            logger.error(f"Tastytrade is_market_open failed: {e}")
            return False

    def get_asset(self, ticker: str) -> dict | None:
        try:
            session = self._get_session()
            symbol = ticker.upper()
            equity = Equity.get_equity(session, symbol)
            if equity is None:
                return None
            return {
                "symbol": symbol,
                "name": getattr(equity, "description", symbol),
                "tradable": True,
                "fractionable": False,
            }
        except Exception as e:
            logger.warning(f"Tastytrade get_asset({ticker}) returned None: {e}")
            return None
