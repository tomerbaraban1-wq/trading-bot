import logging
import uuid
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    import krakenex
except ImportError:
    raise ImportError("pip install krakenex>=2.1.0")

# Kraken uses non-standard ticker conventions.
# BTC is XXBT on Kraken; pairs are typically like XBTUSD, ETHUSD, etc.
_KRAKEN_TICKER_MAP = {
    "BTC": "XBT",
    "BITCOIN": "XBT",
}


def _to_kraken_pair(ticker: str) -> str:
    """Convert ticker to Kraken pair format: BTC -> XBTUSD, ETH -> ETHUSD."""
    ticker = ticker.upper()
    # Strip USDT/USD suffix if already present
    for suffix in ("USDT", "USD"):
        if ticker.endswith(suffix):
            base = ticker[: -len(suffix)]
            base = _KRAKEN_TICKER_MAP.get(base, base)
            return base + "USD"
    base = _KRAKEN_TICKER_MAP.get(ticker, ticker)
    return base + "USD"


def _base_asset(ticker: str) -> str:
    """Return just the base asset (e.g. 'XBT' from 'XBTUSD')."""
    pair = _to_kraken_pair(ticker)
    return pair[:-3]  # strip 'USD'


class KrakenBroker(BrokerBase):
    """Kraken crypto exchange broker via krakenex."""

    def __init__(self):
        self._api_key = settings.KRAKEN_API_KEY
        self._secret = settings.KRAKEN_SECRET

        if not self._api_key or not self._secret:
            raise ValueError(
                "Kraken credentials not configured. Set KRAKEN_API_KEY and KRAKEN_SECRET."
            )

        self._api: krakenex.API | None = None

    def _get_api(self) -> krakenex.API:
        if self._api is None:
            self._api = krakenex.API(key=self._api_key, secret=self._secret)
            logger.info("Kraken API client initialised.")
        return self._api

    def _query_private(self, method: str, data: dict | None = None) -> dict:
        api = self._get_api()
        resp = api.query_private(method, data or {})
        errors = resp.get("error", [])
        if errors:
            raise RuntimeError(f"Kraken API error: {errors}")
        return resp.get("result", {})

    def _query_public(self, method: str, data: dict | None = None) -> dict:
        api = self._get_api()
        resp = api.query_public(method, data or {})
        errors = resp.get("error", [])
        if errors:
            raise RuntimeError(f"Kraken API error: {errors}")
        return resp.get("result", {})

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            result = self._query_private("Balance")
            # result is a dict of asset -> balance string
            usd_balance = float(result.get("ZUSD", result.get("USD", 0)))
            # Estimate portfolio value by including all crypto
            portfolio_value = usd_balance
            for asset, balance in result.items():
                bal = float(balance)
                if bal <= 0 or asset in ("ZUSD", "USD"):
                    continue
                try:
                    pair = asset.lstrip("XZ") + "USD"
                    ticker_data = self._query_public("Ticker", {"pair": pair})
                    price_data = next(iter(ticker_data.values()))
                    price = float(price_data["c"][0])  # last trade price
                    portfolio_value += bal * price
                except Exception:
                    pass
            return {
                "cash": usd_balance,
                "buying_power": usd_balance,
                "portfolio_value": portfolio_value,
                "equity": portfolio_value,
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Kraken get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0,
                    "equity": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            result = self._query_private("Balance")
            positions = []
            for asset, balance in result.items():
                qty = float(balance)
                if qty <= 0 or asset in ("ZUSD", "USD", "ZUSD.HOLD"):
                    continue
                try:
                    base = asset.lstrip("XZ")
                    pair = base + "USD"
                    ticker_data = self._query_public("Ticker", {"pair": pair})
                    price_data = next(iter(ticker_data.values()))
                    current_price = float(price_data["c"][0])
                except Exception:
                    current_price = 0.0
                market_value = qty * current_price
                positions.append({
                    "ticker": asset.lstrip("XZ"),
                    "qty": qty,
                    "avg_cost": 0.0,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": 0.0,
                })
            return positions
        except Exception as e:
            logger.error(f"Kraken get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            base = _base_asset(ticker)
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == base.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"Kraken get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        pair = _to_kraken_pair(ticker)
        data = {
            "pair": pair,
            "type": "buy",
            "ordertype": "market",
            "volume": str(qty),
        }
        result = self._query_private("AddOrder", data)
        txids = result.get("txid", [])
        order_id = txids[0] if txids else str(uuid.uuid4())
        logger.info(f"Kraken BUY submitted: {pair} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": pair,
            "qty": qty,
            "status": "submitted",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        pair = _to_kraken_pair(ticker)

        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]

        data = {
            "pair": pair,
            "type": "sell",
            "ordertype": "market",
            "volume": str(qty),
        }
        result = self._query_private("AddOrder", data)
        txids = result.get("txid", [])
        order_id = txids[0] if txids else str(uuid.uuid4())
        logger.info(f"Kraken SELL submitted: {pair} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": pair,
            "qty": qty,
            "status": "submitted",
        }

    def is_market_open(self) -> bool:
        """Crypto trades 24/7."""
        return True

    def get_asset(self, ticker: str) -> dict | None:
        try:
            pair = _to_kraken_pair(ticker)
            result = self._query_public("AssetPairs", {"pair": pair})
            if not result:
                return None
            return {
                "symbol": pair,
                "name": pair,
                "tradable": True,
                "fractionable": True,
            }
        except Exception as e:
            logger.warning(f"Kraken get_asset({ticker}) returned None: {e}")
            return None
