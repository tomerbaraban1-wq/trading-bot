import logging
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    from kucoin.client import Market, Trade, User
except ImportError:
    raise ImportError("pip install python-kucoin>=2.1.3")


class KuCoinBroker(BrokerBase):
    """KuCoin crypto exchange broker via python-kucoin library."""

    def __init__(self):
        self._api_key = settings.KUCOIN_API_KEY
        self._secret = settings.KUCOIN_SECRET
        self._passphrase = settings.KUCOIN_PASSPHRASE

        if not self._api_key or not self._secret or not self._passphrase:
            raise ValueError(
                "KuCoin credentials not configured. "
                "Set KUCOIN_API_KEY, KUCOIN_SECRET and KUCOIN_PASSPHRASE."
            )

        self._market_client = None
        self._trade_client = None
        self._user_client = None

    def _get_market(self) -> Market:
        if self._market_client is None:
            self._market_client = Market(url="https://api.kucoin.com")
            logger.info("KuCoin Market client initialised.")
        return self._market_client

    def _get_trade(self) -> Trade:
        if self._trade_client is None:
            self._trade_client = Trade(
                key=self._api_key,
                secret=self._secret,
                passphrase=self._passphrase,
                url="https://api.kucoin.com",
            )
            logger.info("KuCoin Trade client initialised.")
        return self._trade_client

    def _get_user(self) -> User:
        if self._user_client is None:
            self._user_client = User(
                key=self._api_key,
                secret=self._secret,
                passphrase=self._passphrase,
                url="https://api.kucoin.com",
            )
        return self._user_client

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            user = self._get_user()
            accounts = user.get_account_list(account_type="trade")
            cash = 0.0
            portfolio_value = 0.0
            mkt = self._get_market()
            for acct in (accounts or []):
                ccy = acct.get("currency", "")
                available = float(acct.get("available", 0) or 0)
                balance = float(acct.get("balance", 0) or 0)
                if ccy == "USDT":
                    cash += available
                    portfolio_value += balance
                elif balance > 0:
                    symbol = ccy + "-USDT"
                    try:
                        ticker = mkt.get_ticker(symbol)
                        price = float(ticker.get("price", 0) or 0)
                        portfolio_value += balance * price
                    except Exception:
                        pass
            return {
                "cash": cash,
                "buying_power": cash,
                "portfolio_value": portfolio_value,
            }
        except Exception as e:
            logger.error(f"KuCoin get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}

    def get_positions(self) -> list[dict]:
        try:
            user = self._get_user()
            accounts = user.get_account_list(account_type="trade")
            mkt = self._get_market()
            result = []
            for acct in (accounts or []):
                ccy = acct.get("currency", "")
                qty = float(acct.get("balance", 0) or 0)
                if qty <= 0 or ccy == "USDT":
                    continue
                symbol = ccy + "-USDT"
                try:
                    ticker = mkt.get_ticker(symbol)
                    current_price = float(ticker.get("price", 0) or 0)
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
            logger.error(f"KuCoin get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            asset = ticker.upper().replace("-USDT", "").replace("USDT", "")
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == asset:
                    return p
            return None
        except Exception as e:
            logger.error(f"KuCoin get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        trade = self._get_trade()
        asset = ticker.upper().replace("-USDT", "").replace("USDT", "")
        symbol = asset + "-USDT"
        try:
            order_id = trade.create_market_order(
                symbol=symbol,
                side="buy",
                size=str(qty),
            )
        except Exception as e:
            logger.error(f"KuCoin BUY failed: {e}")
            raise
        logger.info(f"KuCoin BUY submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": str(order_id),
            "symbol": symbol,
            "qty": float(qty),
            "price": None,
            "status": "submitted",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        trade = self._get_trade()
        asset = ticker.upper().replace("-USDT", "").replace("USDT", "")
        symbol = asset + "-USDT"
        if qty is None:
            position = self.get_position(asset)
            if not position:
                raise ValueError(f"No open position for {asset}")
            qty = position["qty"]
        try:
            order_id = trade.create_market_order(
                symbol=symbol,
                side="sell",
                size=str(qty),
            )
        except Exception as e:
            logger.error(f"KuCoin SELL failed: {e}")
            raise
        logger.info(f"KuCoin SELL submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": str(order_id),
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
            mkt = self._get_market()
            asset = ticker.upper().replace("-USDT", "").replace("USDT", "")
            symbol = asset + "-USDT"
            ticker_data = mkt.get_ticker(symbol)
            if not ticker_data:
                return None
            return {"tradable": True}
        except Exception as e:
            logger.warning(f"KuCoin get_asset({ticker}) failed: {e}")
            return None
