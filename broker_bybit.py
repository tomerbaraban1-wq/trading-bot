import logging
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    from pybit.unified_trading import HTTP
except ImportError:
    raise ImportError("pip install pybit>=5.0.0")


def _to_bybit_symbol(ticker: str) -> str:
    """Convert ticker to Bybit spot symbol: BTC -> BTCUSDT."""
    ticker = ticker.upper()
    if ticker.endswith("USDT"):
        return ticker
    return ticker + "USDT"


class BybitBroker(BrokerBase):
    """Bybit crypto exchange broker via pybit library."""

    def __init__(self):
        self._api_key = settings.BYBIT_API_KEY
        self._secret = settings.BYBIT_SECRET

        if not self._api_key or not self._secret:
            raise ValueError(
                "Bybit credentials not configured. Set BYBIT_API_KEY and BYBIT_SECRET."
            )

        self._session = None

    def _get_session(self) -> HTTP:
        if self._session is None:
            self._session = HTTP(api_key=self._api_key, api_secret=self._secret)
            logger.info("Bybit session initialised.")
        return self._session

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            session = self._get_session()
            resp = session.get_wallet_balance(accountType="SPOT")
            coins = resp.get("result", {}).get("list", [{}])[0].get("coin", [])
            cash = 0.0
            portfolio_value = 0.0
            for coin in coins:
                if coin.get("coin") == "USDT":
                    cash = float(coin.get("walletBalance", 0) or 0)
                    portfolio_value += cash
                else:
                    qty = float(coin.get("walletBalance", 0) or 0)
                    if qty > 0:
                        symbol = coin["coin"] + "USDT"
                        try:
                            price_resp = session.get_tickers(category="spot", symbol=symbol)
                            price = float(price_resp["result"]["list"][0]["lastPrice"])
                            portfolio_value += qty * price
                        except Exception:
                            pass
            return {
                "cash": cash,
                "buying_power": cash,
                "portfolio_value": portfolio_value,
            }
        except Exception as e:
            logger.error(f"Bybit get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}

    def get_positions(self) -> list[dict]:
        try:
            session = self._get_session()
            resp = session.get_wallet_balance(accountType="SPOT")
            coins = resp.get("result", {}).get("list", [{}])[0].get("coin", [])
            result = []
            for coin in coins:
                qty = float(coin.get("walletBalance", 0) or 0)
                asset = coin.get("coin", "")
                if qty <= 0 or asset == "USDT":
                    continue
                symbol = asset + "USDT"
                try:
                    price_resp = session.get_tickers(category="spot", symbol=symbol)
                    current_price = float(price_resp["result"]["list"][0]["lastPrice"])
                except Exception:
                    current_price = 0.0
                market_value = qty * current_price
                result.append({
                    "ticker": asset,
                    "qty": qty,
                    "avg_entry_price": 0.0,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": 0.0,
                    "unrealized_plpc": 0.0,
                })
            return result
        except Exception as e:
            logger.error(f"Bybit get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            asset = ticker.upper().replace("USDT", "")
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == asset:
                    return p
            return None
        except Exception as e:
            logger.error(f"Bybit get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        session = self._get_session()
        symbol = _to_bybit_symbol(ticker)
        try:
            resp = session.place_order(
                category="spot",
                symbol=symbol,
                side="Buy",
                orderType="Market",
                qty=str(qty),
            )
        except Exception as e:
            logger.error(f"Bybit BUY failed: {e}")
            raise
        result = resp.get("result", {})
        order_id = str(result.get("orderId", "unknown"))
        status = str(result.get("orderStatus", "submitted")).lower()
        avg_price = result.get("avgPrice") or result.get("price")
        logger.info(f"Bybit BUY submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": float(qty),
            "price": float(avg_price) if avg_price else None,
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        session = self._get_session()
        symbol = _to_bybit_symbol(ticker)
        if qty is None:
            asset = ticker.upper().replace("USDT", "")
            position = self.get_position(asset)
            if not position:
                raise ValueError(f"No open position for {asset}")
            qty = position["qty"]
        try:
            resp = session.place_order(
                category="spot",
                symbol=symbol,
                side="Sell",
                orderType="Market",
                qty=str(qty),
            )
        except Exception as e:
            logger.error(f"Bybit SELL failed: {e}")
            raise
        result = resp.get("result", {})
        order_id = str(result.get("orderId", "unknown"))
        status = str(result.get("orderStatus", "submitted")).lower()
        avg_price = result.get("avgPrice") or result.get("price")
        logger.info(f"Bybit SELL submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": float(qty),
            "price": float(avg_price) if avg_price else None,
            "status": status,
        }

    def is_market_open(self) -> bool:
        """Crypto trades 24/7."""
        return True

    def get_asset(self, ticker: str) -> dict | None:
        try:
            session = self._get_session()
            symbol = _to_bybit_symbol(ticker)
            resp = session.get_instruments_info(category="spot", symbol=symbol)
            items = resp.get("result", {}).get("list", [])
            if not items:
                return None
            tradable = items[0].get("status", "") == "Trading"
            return {"tradable": tradable}
        except Exception as e:
            logger.warning(f"Bybit get_asset({ticker}) failed: {e}")
            return None
