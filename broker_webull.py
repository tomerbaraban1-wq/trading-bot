import logging
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    from webull import webull as wb_class
except ImportError:
    raise ImportError("pip install webull>=0.1.0")


class WebullBroker(BrokerBase):
    """Webull broker via webull library."""

    def __init__(self):
        self._email = settings.WEBULL_EMAIL
        self._password = settings.WEBULL_PASSWORD
        self._device_id = settings.WEBULL_DEVICE_ID
        self._trading_pin = settings.WEBULL_TRADING_PIN

        if not self._email or not self._password:
            raise ValueError(
                "Webull credentials not configured. Set WEBULL_EMAIL and WEBULL_PASSWORD."
            )

        self._wb = None

    def _get_client(self):
        if self._wb is None:
            self._wb = wb_class()
            self._wb.login(
                self._email,
                self._password,
                device_id=self._device_id or None,
                mfa=None,
            )
            logger.info("Webull login successful.")
        return self._wb

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            wb = self._get_client()
            account = wb.get_account()
            net_liq = float(account.get("netLiquidation", 0) or 0)
            cash = float(account.get("cashBalance", 0) or 0)
            buying_power = float(account.get("buyingPower", 0) or 0)
            return {
                "cash": cash,
                "buying_power": buying_power,
                "portfolio_value": net_liq,
            }
        except Exception as e:
            logger.error(f"Webull get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}

    def get_positions(self) -> list[dict]:
        try:
            wb = self._get_client()
            raw = wb.get_positions()
            result = []
            for pos in (raw or []):
                qty = float(pos.get("quantity", 0) or 0)
                if qty <= 0:
                    continue
                ticker = pos.get("ticker", {}).get("symbol", "")
                avg_cost = float(pos.get("costPrice", 0) or 0)
                current_price = float(pos.get("lastPrice", 0) or 0)
                market_value = qty * current_price
                unrealized_pl = float(pos.get("unrealizedProfitLoss", 0) or 0)
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
            logger.error(f"Webull get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"Webull get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        wb = self._get_client()
        try:
            order = wb.place_order(
                stock=ticker.upper(),
                action="BUY",
                orderType="MKT",
                enforce="DAY",
                qty=qty,
                quoteType="STOCK",
            )
        except Exception as e:
            logger.error(f"Webull BUY failed: {e}")
            raise
        order_id = str(order.get("orderId", "unknown"))
        status = str(order.get("statusStr", "submitted")).lower()
        logger.info(f"Webull BUY submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "qty": float(qty),
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        wb = self._get_client()
        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]
        try:
            order = wb.place_order(
                stock=ticker.upper(),
                action="SELL",
                orderType="MKT",
                enforce="DAY",
                qty=qty,
                quoteType="STOCK",
            )
        except Exception as e:
            logger.error(f"Webull SELL failed: {e}")
            raise
        order_id = str(order.get("orderId", "unknown"))
        status = str(order.get("statusStr", "submitted")).lower()
        logger.info(f"Webull SELL submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "qty": float(qty),
            "status": status,
        }

    def is_market_open(self) -> bool:
        try:
            wb = self._get_client()
            clock = wb.get_trading_time()
            if isinstance(clock, dict):
                return bool(clock.get("isTrading", False))
            return False
        except Exception as e:
            logger.error(f"Webull is_market_open failed: {e}")
            return False

    def get_asset(self, ticker: str) -> dict | None:
        try:
            wb = self._get_client()
            result = wb.get_ticker(stock=ticker.upper())
            if not result:
                return None
            # webull returns a list
            item = result[0] if isinstance(result, list) else result
            tradable = item.get("disabledTrade", True) is False
            return {"tradable": tradable}
        except Exception as e:
            logger.warning(f"Webull get_asset({ticker}) failed: {e}")
            return None
