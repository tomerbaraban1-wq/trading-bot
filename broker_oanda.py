import logging
from datetime import datetime, time as dtime
from config import settings
from broker_base import BrokerBase

logger = logging.getLogger(__name__)

# Oanda forex is open Sun 22:00 UTC - Fri 22:00 UTC (essentially 24/5)
_FOREX_OPEN_WEEKDAY = 6   # Sunday
_FOREX_OPEN_HOUR = 22
_FOREX_CLOSE_WEEKDAY = 4  # Friday
_FOREX_CLOSE_HOUR = 22


def _normalize_forex_pair(ticker: str) -> str:
    """Convert 'EURUSD' -> 'EUR_USD'. Pass-through if already has underscore."""
    ticker = ticker.upper()
    if "_" not in ticker and len(ticker) == 6:
        return f"{ticker[:3]}_{ticker[3:]}"
    return ticker


class OandaBroker(BrokerBase):
    """Oanda REST API v20 broker (forex & CFDs)."""

    def __init__(self):
        self._api_key = settings.OANDA_API_KEY
        self._account_id = settings.OANDA_ACCOUNT_ID
        self._base_url = "https://api-fxtrade.oanda.com/v3"
        self._api = None

    def _get_api(self):
        if self._api is not None:
            return self._api
        try:
            from oandapyV20 import API
            self._api = API(access_token=self._api_key)
            return self._api
        except Exception as e:
            logger.error(f"Oanda API init failed: {e}")
            raise ConnectionError(f"Oanda unavailable: {e}") from e

    def _request(self, endpoint_cls, *args, **kwargs):
        """Run an oandapyV20 request and return the response."""
        import oandapyV20.endpoints as ep_module
        api = self._get_api()
        endpoint = endpoint_cls(*args, **kwargs)
        api.request(endpoint)
        return endpoint.response

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            from oandapyV20.endpoints.accounts import AccountDetails
            api = self._get_api()
            ep = AccountDetails(accountID=self._account_id)
            api.request(ep)
            acc = ep.response["account"]
            return {
                "equity": float(acc.get("NAV", 0)),
                "buying_power": float(acc.get("marginAvailable", 0)),
                "cash": float(acc.get("balance", 0)),
                "portfolio_value": float(acc.get("unrealizedPL", 0)) + float(acc.get("balance", 0)),
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Oanda get_account failed: {e}")
            return {"equity": 0.0, "buying_power": 0.0, "cash": 0.0,
                    "portfolio_value": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            from oandapyV20.endpoints.positions import OpenPositions
            api = self._get_api()
            ep = OpenPositions(accountID=self._account_id)
            api.request(ep)
            result = []
            for pos in ep.response.get("positions", []):
                instrument = pos["instrument"]
                long_units = float(pos["long"]["units"])
                short_units = float(pos["short"]["units"])
                units = long_units if long_units != 0 else short_units
                avg_price = float(pos["long"]["averagePrice"]) if long_units != 0 else float(pos["short"]["averagePrice"])
                unrealized_pl = float(pos.get("unrealizedPL", 0))
                result.append({
                    "ticker": instrument,
                    "qty": int(abs(units)),
                    "avg_entry_price": avg_price,
                    "current_price": avg_price,  # approximate until tick data fetched
                    "market_value": avg_price * abs(units),
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": (unrealized_pl / (avg_price * abs(units))) if units else 0.0,
                })
            return result
        except Exception as e:
            logger.error(f"Oanda get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            from oandapyV20.endpoints.positions import PositionDetails
            api = self._get_api()
            instrument = _normalize_forex_pair(ticker)
            ep = PositionDetails(accountID=self._account_id, instrument=instrument)
            api.request(ep)
            pos = ep.response.get("position", {})
            long_units = float(pos.get("long", {}).get("units", 0))
            short_units = float(pos.get("short", {}).get("units", 0))
            units = long_units if long_units != 0 else short_units
            if units == 0:
                return None
            avg_price = float(
                pos["long"]["averagePrice"] if long_units != 0 else pos["short"]["averagePrice"]
            )
            unrealized_pl = float(pos.get("unrealizedPL", 0))
            return {
                "ticker": instrument,
                "qty": int(abs(units)),
                "avg_entry_price": avg_price,
                "current_price": avg_price,
                "market_value": avg_price * abs(units),
                "unrealized_pl": unrealized_pl,
            }
        except Exception as e:
            logger.error(f"Oanda get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        from oandapyV20.endpoints.orders import OrderCreate
        instrument = _normalize_forex_pair(ticker)
        api = self._get_api()
        data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(qty),  # positive = buy
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        ep = OrderCreate(accountID=self._account_id, data=data)
        api.request(ep)
        resp = ep.response
        fill = resp.get("orderFillTransaction", {})
        order_id = fill.get("id", resp.get("relatedTransactionIDs", ["?"])[0])
        logger.info(f"Oanda BUY submitted: {instrument} x{qty}")
        return {
            "order_id": str(order_id),
            "symbol": instrument,
            "qty": qty,
            "status": "filled",
            "type": "market",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        from oandapyV20.endpoints.orders import OrderCreate
        instrument = _normalize_forex_pair(ticker)

        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]

        api = self._get_api()
        data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(-qty),  # negative = sell
                "timeInForce": "FOK",
                "positionFill": "REDUCE_ONLY",
            }
        }
        ep = OrderCreate(accountID=self._account_id, data=data)
        api.request(ep)
        resp = ep.response
        fill = resp.get("orderFillTransaction", {})
        order_id = fill.get("id", resp.get("relatedTransactionIDs", ["?"])[0])
        logger.info(f"Oanda SELL submitted: {instrument} x{qty}")
        return {
            "order_id": str(order_id),
            "symbol": instrument,
            "qty": qty,
            "status": "filled",
        }

    def is_market_open(self) -> bool:
        """Forex is open Sun 22:00 UTC - Fri 22:00 UTC."""
        now = datetime.utcnow()
        wd = now.weekday()  # 0=Mon ... 6=Sun
        h = now.hour
        # Closed Saturday all day
        if wd == 5:
            return False
        # Closed Friday after 22:00
        if wd == 4 and h >= _FOREX_CLOSE_HOUR:
            return False
        # Closed Sunday before 22:00
        if wd == 6 and h < _FOREX_OPEN_HOUR:
            return False
        return True

    def get_asset(self, ticker: str) -> dict | None:
        try:
            from oandapyV20.endpoints.instruments import InstrumentsCandles
            instrument = _normalize_forex_pair(ticker)
            api = self._get_api()
            params = {"count": "1", "granularity": "S5"}
            ep = InstrumentsCandles(instrument=instrument, params=params)
            api.request(ep)
            return {
                "symbol": instrument,
                "name": instrument,
                "tradable": True,
                "fractionable": True,
            }
        except Exception as e:
            logger.warning(f"Oanda get_asset({ticker}) returned None: {e}")
            return None
