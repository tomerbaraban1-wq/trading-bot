import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.common.exceptions import APIError
from config import settings
from utils import retry_sync
from broker_base import BrokerBase

logger = logging.getLogger(__name__)


class AlpacaBroker(BrokerBase):
    """Alpaca broker implementation (paper and live)."""

    def __init__(self, paper: bool = True):
        self._paper = paper
        self._client: TradingClient | None = None

    def _get_client(self) -> TradingClient:
        if self._client is None:
            self._client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=self._paper,
            )
        return self._client

    def get_account(self) -> dict:
        try:
            client = self._get_client()
            account = retry_sync(client.get_account)
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "status": str(account.status),
            }
        except Exception as e:
            logger.error(f"Alpaca get_account failed: {e}")
            return {"equity": 0.0, "buying_power": 0.0, "cash": 0.0,
                    "portfolio_value": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            client = self._get_client()
            positions = retry_sync(client.get_all_positions)
            return [
                {
                    "ticker": p.symbol,
                    "qty": int(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Alpaca get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            client = self._get_client()
            p = client.get_open_position(ticker.upper())
            return {
                "ticker": p.symbol,
                "qty": int(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
        except APIError:
            return None
        except Exception as e:
            logger.error(f"Alpaca get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: int, price: float | None = None) -> dict:
        client = self._get_client()
        stop_pct = settings.TRAILING_STOP_PCT

        order_request = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OTO,
            stop_loss={"trail_percent": str(stop_pct)},
        )

        order = retry_sync(client.submit_order, order_request)
        logger.info(f"BUY order submitted: {ticker} x{qty} (trailing stop: {stop_pct}%)")
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": int(order.qty),
            "status": str(order.status),
            "type": str(order.type),
        }

    def submit_sell(self, ticker: str, qty: int | None = None) -> dict:
        client = self._get_client()

        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]

        order_request = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        order = retry_sync(client.submit_order, order_request)
        logger.info(f"SELL order submitted: {ticker} x{qty}")
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": int(order.qty),
            "status": str(order.status),
        }

    def is_market_open(self) -> bool:
        try:
            client = self._get_client()
            clock = retry_sync(client.get_clock)
            return clock.is_open
        except Exception as e:
            logger.error(f"Alpaca is_market_open failed: {e}")
            return False

    def get_asset(self, ticker: str) -> dict | None:
        try:
            client = self._get_client()
            asset = client.get_asset(ticker.upper())
            if asset.tradable:
                return {
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "tradable": asset.tradable,
                    "fractionable": asset.fractionable,
                }
            return None
        except APIError:
            return None
        except Exception as e:
            logger.error(f"Alpaca get_asset({ticker}) failed: {e}")
            return None
