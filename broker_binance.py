import logging
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    raise ImportError("pip install python-binance>=1.0.19")


def _to_binance_symbol(ticker: str) -> str:
    """Convert ticker to Binance symbol: BTC -> BTCUSDT (pass-through if already has suffix)."""
    ticker = ticker.upper()
    if ticker.endswith("USDT") or ticker.endswith("BTC") or ticker.endswith("ETH"):
        return ticker
    return ticker + "USDT"


class BinanceBroker(BrokerBase):
    """Binance crypto exchange broker via python-binance SDK."""

    def __init__(self):
        self._api_key = settings.BINANCE_API_KEY
        self._secret = settings.BINANCE_SECRET

        if not self._api_key or not self._secret:
            raise ValueError(
                "Binance credentials not configured. Set BINANCE_API_KEY and BINANCE_SECRET."
            )

        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(self._api_key, self._secret)
            logger.info("Binance client initialised.")
        return self._client

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            client = self._get_client()
            account = client.get_account()
            # Sum all non-zero USDT-equivalent balances
            balances = account.get("balances", [])
            usdt_balance = 0.0
            portfolio_value = 0.0
            for b in balances:
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                total = free + locked
                asset = b.get("asset", "")
                if asset == "USDT":
                    usdt_balance += total
                    portfolio_value += total
                elif total > 0:
                    try:
                        ticker_price = client.get_symbol_ticker(symbol=f"{asset}USDT")
                        portfolio_value += total * float(ticker_price["price"])
                    except Exception:
                        pass

            return {
                "cash": usdt_balance,
                "buying_power": usdt_balance,
                "portfolio_value": portfolio_value,
                "equity": portfolio_value,
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Binance get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0,
                    "equity": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            client = self._get_client()
            account = client.get_account()
            balances = account.get("balances", [])
            result = []
            for b in balances:
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                total = free + locked
                asset = b.get("asset", "")
                if total <= 0 or asset == "USDT":
                    continue
                try:
                    ticker_data = client.get_symbol_ticker(symbol=f"{asset}USDT")
                    current_price = float(ticker_data["price"])
                except Exception:
                    current_price = 0.0
                market_value = total * current_price
                result.append({
                    "ticker": asset,
                    "qty": total,
                    "avg_entry_price": 0.0,   # Binance spot doesn't expose avg cost easily
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": 0.0,
                    "unrealized_plpc": 0.0,
                })
            return result
        except Exception as e:
            logger.error(f"Binance get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            # ticker may be 'BTC' or 'BTCUSDT'
            asset = ticker.upper().replace("USDT", "")
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == asset:
                    return p
            return None
        except Exception as e:
            logger.error(f"Binance get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        client = self._get_client()
        symbol = _to_binance_symbol(ticker)
        try:
            order = client.order_market_buy(symbol=symbol, quantity=qty)
        except BinanceAPIException as e:
            logger.error(f"Binance BUY failed: {e}")
            raise
        order_id = str(order.get("orderId", "unknown"))
        status = str(order.get("status", "submitted")).lower()
        # Binance returns fills array with executed price; compute weighted avg
        fills = order.get("fills", [])
        avg_price = None
        if fills:
            total_qty = sum(float(f.get("qty", 0)) for f in fills)
            total_cost = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
            avg_price = total_cost / total_qty if total_qty > 0 else None
        logger.info(f"Binance BUY submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "price": float(avg_price) if avg_price else None,
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        client = self._get_client()
        symbol = _to_binance_symbol(ticker)

        if qty is None:
            asset = ticker.upper().replace("USDT", "")
            position = self.get_position(asset)
            if not position:
                raise ValueError(f"No open position for {asset}")
            qty = position["qty"]

        try:
            order = client.order_market_sell(symbol=symbol, quantity=qty)
        except BinanceAPIException as e:
            logger.error(f"Binance SELL failed: {e}")
            raise
        order_id = str(order.get("orderId", "unknown"))
        status = str(order.get("status", "submitted")).lower()
        fills = order.get("fills", [])
        avg_price = None
        if fills:
            total_qty = sum(float(f.get("qty", 0)) for f in fills)
            total_cost = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
            avg_price = total_cost / total_qty if total_qty > 0 else None
        logger.info(f"Binance SELL submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "price": float(avg_price) if avg_price else None,
            "status": status,
        }

    def is_market_open(self) -> bool:
        """Crypto trades 24/7."""
        return True

    def get_asset(self, ticker: str) -> dict | None:
        try:
            client = self._get_client()
            symbol = _to_binance_symbol(ticker)
            info = client.get_symbol_info(symbol)
            if info is None:
                return None
            tradable = info.get("status", "") == "TRADING"
            return {
                "symbol": symbol,
                "name": symbol,
                "tradable": tradable,
                "fractionable": True,
            }
        except Exception as e:
            logger.warning(f"Binance get_asset({ticker}) returned None: {e}")
            return None
