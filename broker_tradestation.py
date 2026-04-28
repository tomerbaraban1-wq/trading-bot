import logging
import requests
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.tradestation.com"
_TOKEN_URL = "https://signin.tradestation.com/oauth/token"


class TradeStationBroker(BrokerBase):
    """TradeStation broker via REST API using OAuth2 bearer token."""

    def __init__(self):
        self._api_key = settings.TRADESTATION_API_KEY
        self._secret = settings.TRADESTATION_SECRET
        self._account = settings.TRADESTATION_ACCOUNT
        self._refresh_token = settings.TRADESTATION_REFRESH_TOKEN

        if not self._api_key or not self._secret or not self._account or not self._refresh_token:
            raise ValueError(
                "TradeStation credentials not configured. Set TRADESTATION_API_KEY, "
                "TRADESTATION_SECRET, TRADESTATION_ACCOUNT and TRADESTATION_REFRESH_TOKEN."
            )

        self._access_token: str | None = None
        self._session = requests.Session()

    def _refresh_access_token(self):
        """Obtain a fresh access token using the refresh token."""
        try:
            resp = requests.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self._api_key,
                    "client_secret": self._secret,
                    "refresh_token": self._refresh_token,
                },
                timeout=15,
            )
            resp.raise_for_status()
            self._access_token = resp.json()["access_token"]
            self._session.headers.update(
                {"Authorization": f"Bearer {self._access_token}",
                 "Content-Type": "application/json"}
            )
            logger.info("TradeStation access token refreshed.")
        except Exception as e:
            logger.error(f"TradeStation token refresh failed: {e}")
            raise

    def _ensure_token(self):
        if self._access_token is None:
            self._refresh_access_token()

    def _get(self, path: str, **params) -> dict:
        self._ensure_token()
        url = f"{_BASE_URL}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code == 401:
                self._refresh_access_token()
                resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"TradeStation GET {path} failed: {e}")
            raise

    def _post(self, path: str, body: dict) -> dict:
        self._ensure_token()
        url = f"{_BASE_URL}/{path.lstrip('/')}"
        try:
            resp = self._session.post(url, json=body, timeout=15)
            if resp.status_code == 401:
                self._refresh_access_token()
                resp = self._session.post(url, json=body, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"TradeStation POST {path} failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            data = self._get(f"v3/brokerage/accounts/{self._account}/balances")
            bal = data.get("Balances", [{}])[0] if data.get("Balances") else {}
            cash = float(bal.get("CashBalance", 0) or 0)
            buying_power = float(bal.get("BuyingPower", 0) or 0)
            portfolio_value = float(bal.get("MarketValue", 0) or 0)
            return {
                "cash": cash,
                "buying_power": buying_power,
                "portfolio_value": portfolio_value,
            }
        except Exception as e:
            logger.error(f"TradeStation get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}

    def get_positions(self) -> list[dict]:
        try:
            data = self._get(f"v3/brokerage/accounts/{self._account}/positions")
            positions = data.get("Positions", [])
            result = []
            for pos in positions:
                qty = float(pos.get("Quantity", 0) or 0)
                if qty <= 0:
                    continue
                ticker = pos.get("Symbol", "")
                avg_cost = float(pos.get("AveragePrice", 0) or 0)
                current_price = float(pos.get("Last", 0) or 0)
                market_value = float(pos.get("MarketValue", 0) or 0)
                unrealized_pl = float(pos.get("UnrealizedProfitLoss", 0) or 0)
                result.append({
                    "ticker": ticker,
                    "qty": qty,
                    "avg_entry_price": avg_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                })
            return result
        except Exception as e:
            logger.error(f"TradeStation get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"TradeStation get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        body = {
            "AccountID": self._account,
            "Symbol": ticker.upper(),
            "Quantity": str(qty),
            "OrderType": "Market",
            "TradeAction": "BUY",
            "TimeInForce": {"Duration": "DAY"},
        }
        try:
            resp = self._post("v2/orderexecution/orders", body)
        except Exception as e:
            logger.error(f"TradeStation BUY failed: {e}")
            raise
        orders = resp.get("Orders", [{}])
        order_id = str(orders[0].get("OrderID", "unknown")) if orders else "unknown"
        status = str(orders[0].get("Status", "submitted")).lower() if orders else "submitted"
        logger.info(f"TradeStation BUY submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "qty": float(qty),
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]
        body = {
            "AccountID": self._account,
            "Symbol": ticker.upper(),
            "Quantity": str(qty),
            "OrderType": "Market",
            "TradeAction": "SELL",
            "TimeInForce": {"Duration": "DAY"},
        }
        try:
            resp = self._post("v2/orderexecution/orders", body)
        except Exception as e:
            logger.error(f"TradeStation SELL failed: {e}")
            raise
        orders = resp.get("Orders", [{}])
        order_id = str(orders[0].get("OrderID", "unknown")) if orders else "unknown"
        status = str(orders[0].get("Status", "submitted")).lower() if orders else "submitted"
        logger.info(f"TradeStation SELL submitted: {ticker} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "qty": float(qty),
            "status": status,
        }

    def is_market_open(self) -> bool:
        try:
            data = self._get("v3/marketdata/markets/hours", markets="NYSE")
            hours = data.get("MarketHours", [{}])[0] if data.get("MarketHours") else {}
            return hours.get("Status", "").lower() == "open"
        except Exception as e:
            logger.error(f"TradeStation is_market_open failed: {e}")
            from datetime import datetime, time as dtime
            now = datetime.utcnow()
            if now.weekday() >= 5:
                return False
            return dtime(13, 30) <= now.time() < dtime(20, 0)

    def get_asset(self, ticker: str) -> dict | None:
        try:
            data = self._get(f"v3/marketdata/symbols/{ticker.upper()}")
            symbol_info = data.get("Symbols", [{}])[0] if data.get("Symbols") else {}
            if not symbol_info:
                return None
            tradable = symbol_info.get("Error", "") == ""
            return {"tradable": tradable}
        except Exception as e:
            logger.warning(f"TradeStation get_asset({ticker}) failed: {e}")
            return None
