import logging
import requests
from datetime import datetime, time as dtime
from config import settings
from broker_base import BrokerBase

logger = logging.getLogger(__name__)

_PAPER_BASE = "https://sandbox.tradier.com/v1"
_LIVE_BASE = "https://api.tradier.com/v1"


class TradierBroker(BrokerBase):
    """Tradier REST API broker (paper and live)."""

    def __init__(self, paper: bool = True):
        self._token = settings.TRADIER_TOKEN
        self._account = settings.TRADIER_ACCOUNT
        self._base = _PAPER_BASE if paper else _LIVE_BASE
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        })

    def _get(self, path: str, **params) -> dict:
        url = f"{self._base}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Tradier GET {path} failed: {e}")
            raise

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self._base}/{path.lstrip('/')}"
        try:
            resp = self._session.post(url, data=data, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Tradier POST {path} failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            data = self._get(f"accounts/{self._account}/balances")
            bal = data.get("balances", {})
            return {
                "equity": float(bal.get("total_equity", 0)),
                "buying_power": float(bal.get("stock_buying_power", 0)),
                "cash": float(bal.get("total_cash", 0)),
                "portfolio_value": float(bal.get("market_value", 0)),
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Tradier get_account failed: {e}")
            return {"equity": 0.0, "buying_power": 0.0, "cash": 0.0,
                    "portfolio_value": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            data = self._get(f"accounts/{self._account}/positions")
            positions_raw = data.get("positions", {})
            if positions_raw == "null" or not positions_raw:
                return []
            items = positions_raw.get("position", [])
            if isinstance(items, dict):
                items = [items]
            return [
                {
                    "ticker": p["symbol"],
                    "qty": int(p["quantity"]),
                    "avg_entry_price": float(p["cost_basis"]) / max(int(p["quantity"]), 1),
                    "current_price": float(p.get("current_price", 0)),
                    "market_value": float(p.get("market_value", 0)),
                    "unrealized_pl": float(p.get("gain_loss", 0)),
                    "unrealized_plpc": float(p.get("gain_loss_percent", 0)) / 100,
                }
                for p in items
            ]
        except Exception as e:
            logger.error(f"Tradier get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"Tradier get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        data = self._post(
            f"accounts/{self._account}/orders",
            {
                "class": "equity",
                "symbol": ticker.upper(),
                "side": "buy",
                "quantity": str(qty),
                "type": "market",
                "duration": "day",
            },
        )
        order = data.get("order", {})
        order_id = str(order.get("id", "unknown"))
        status = str(order.get("status", "submitted"))
        logger.info(f"Tradier BUY submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": ticker.upper(),
            "qty": qty,
            "status": status,
            "type": "market",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]

        data = self._post(
            f"accounts/{self._account}/orders",
            {
                "class": "equity",
                "symbol": ticker.upper(),
                "side": "sell",
                "quantity": str(qty),
                "type": "market",
                "duration": "day",
            },
        )
        order = data.get("order", {})
        order_id = str(order.get("id", "unknown"))
        status = str(order.get("status", "submitted"))
        logger.info(f"Tradier SELL submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": ticker.upper(),
            "qty": qty,
            "status": status,
        }

    def is_market_open(self) -> bool:
        try:
            data = self._get("markets/clock")
            clock = data.get("clock", {})
            return clock.get("state", "") in ("open", "premarket")
        except Exception as e:
            logger.error(f"Tradier is_market_open fallback to time check: {e}")
            now = datetime.utcnow()
            if now.weekday() >= 5:
                return False
            return dtime(13, 30) <= now.time() < dtime(20, 0)

    def get_asset(self, ticker: str) -> dict | None:
        try:
            data = self._get("markets/quotes", symbols=ticker.upper())
            quotes = data.get("quotes", {})
            quote = quotes.get("quote", {})
            if not quote or quote.get("type") not in ("stock", "etf"):
                return None
            return {
                "symbol": quote["symbol"],
                "name": quote.get("description", ticker),
                "tradable": True,
                "fractionable": False,
            }
        except Exception as e:
            logger.error(f"Tradier get_asset({ticker}) failed: {e}")
            return None
