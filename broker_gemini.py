import logging
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    import gemini
except ImportError:
    raise ImportError("pip install gemini-python>=0.6.1")


class GeminiBroker(BrokerBase):
    """Gemini crypto exchange broker via gemini-python library."""

    def __init__(self):
        self._api_key = settings.GEMINI_API_KEY
        self._secret = settings.GEMINI_SECRET

        if not self._api_key or not self._secret:
            raise ValueError(
                "Gemini credentials not configured. Set GEMINI_API_KEY and GEMINI_SECRET."
            )

    def _to_symbol(self, ticker: str) -> str:
        """Convert ticker to Gemini symbol: BTC -> btcusd."""
        asset = ticker.upper().replace("USD", "").replace("USDT", "")
        return (asset + "usd").lower()

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            acct = gemini.Account(self._api_key, self._secret)
            balances = acct.balance()
            cash = 0.0
            portfolio_value = 0.0
            for b in (balances or []):
                ccy = b.get("currency", "")
                available = float(b.get("available", 0) or 0)
                amount = float(b.get("amount", 0) or 0)
                if ccy == "USD":
                    cash = available
                    portfolio_value += amount
                elif amount > 0:
                    symbol = (ccy.lower() + "usd")
                    try:
                        pub = gemini.PublicClient()
                        ticker_data = pub.get_ticker(symbol)
                        price = float(ticker_data.get("last", 0) or 0)
                        portfolio_value += amount * price
                    except Exception:
                        pass
            return {
                "cash": cash,
                "buying_power": cash,
                "portfolio_value": portfolio_value,
            }
        except Exception as e:
            logger.error(f"Gemini get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}

    def get_positions(self) -> list[dict]:
        try:
            acct = gemini.Account(self._api_key, self._secret)
            balances = acct.balance()
            result = []
            for b in (balances or []):
                ccy = b.get("currency", "")
                qty = float(b.get("amount", 0) or 0)
                if qty <= 0 or ccy == "USD":
                    continue
                symbol = (ccy.lower() + "usd")
                try:
                    pub = gemini.PublicClient()
                    ticker_data = pub.get_ticker(symbol)
                    current_price = float(ticker_data.get("last", 0) or 0)
                except Exception:
                    current_price = 0.0
                market_value = qty * current_price
                result.append({
                    "ticker": ccy,
                    "qty": qty,
                    "avg_entry_price": 0.0,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": 0.0,
                    "unrealized_plpc": 0.0,
                })
            return result
        except Exception as e:
            logger.error(f"Gemini get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            asset = ticker.upper().replace("USD", "").replace("USDT", "")
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == asset:
                    return p
            return None
        except Exception as e:
            logger.error(f"Gemini get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        symbol = self._to_symbol(ticker)
        try:
            order = gemini.NewOrder(
                self._api_key,
                self._secret,
                symbol,
                str(qty),
                None,
                "market buy",
            )
        except Exception as e:
            logger.error(f"Gemini BUY failed: {e}")
            raise
        order_id = str(order.order_id if hasattr(order, "order_id") else
                       getattr(order, "id", "unknown"))
        status = str(getattr(order, "is_live", True)).lower()
        logger.info(f"Gemini BUY submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": float(qty),
            "price": None,
            "status": "submitted",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        if qty is None:
            asset = ticker.upper().replace("USD", "").replace("USDT", "")
            position = self.get_position(asset)
            if not position:
                raise ValueError(f"No open position for {asset}")
            qty = position["qty"]
        symbol = self._to_symbol(ticker)
        try:
            order = gemini.NewOrder(
                self._api_key,
                self._secret,
                symbol,
                str(qty),
                None,
                "market sell",
            )
        except Exception as e:
            logger.error(f"Gemini SELL failed: {e}")
            raise
        order_id = str(order.order_id if hasattr(order, "order_id") else
                       getattr(order, "id", "unknown"))
        logger.info(f"Gemini SELL submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": float(qty),
            "price": None,
            "status": "submitted",
        }

    def is_market_open(self) -> bool:
        """Crypto trades 24/7."""
        return True

    def get_asset(self, ticker: str) -> dict | None:
        try:
            symbol = self._to_symbol(ticker)
            pub = gemini.PublicClient()
            symbols = pub.symbols()
            tradable = symbol in (symbols or [])
            return {"tradable": tradable}
        except Exception as e:
            logger.warning(f"Gemini get_asset({ticker}) failed: {e}")
            return None
