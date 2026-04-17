import logging
import uuid
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    from coinbase.rest import RESTClient
except ImportError:
    raise ImportError("pip install coinbase-advanced-py>=1.3.0")


def _to_coinbase_product(ticker: str) -> str:
    """Convert ticker to Coinbase product ID: BTC -> BTC-USDT."""
    ticker = ticker.upper()
    # Pass-through if it already looks like a product ID (contains '-')
    if "-" in ticker:
        return ticker
    # Strip USDT/USD suffix if already appended
    for suffix in ("USDT", "USD"):
        if ticker.endswith(suffix):
            base = ticker[: -len(suffix)]
            return f"{base}-USDT"
    return f"{ticker}-USDT"


class CoinbaseBroker(BrokerBase):
    """Coinbase Advanced Trade broker via coinbase-advanced-py SDK."""

    def __init__(self):
        self._api_key = settings.COINBASE_API_KEY
        self._secret = settings.COINBASE_SECRET

        if not self._api_key or not self._secret:
            raise ValueError(
                "Coinbase credentials not configured. Set COINBASE_API_KEY and COINBASE_SECRET."
            )

        self._client: RESTClient | None = None

    def _get_client(self) -> RESTClient:
        if self._client is None:
            self._client = RESTClient(api_key=self._api_key, api_secret=self._secret)
            logger.info("Coinbase REST client initialised.")
        return self._client

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            client = self._get_client()
            accounts = client.get_accounts()
            usd_cash = 0.0
            portfolio_value = 0.0

            for acc in accounts.get("accounts", []):
                currency = acc.get("currency", "")
                avail = float(acc.get("available_balance", {}).get("value", 0))
                hold = float(acc.get("hold", {}).get("value", 0))
                total = avail + hold

                if currency in ("USD", "USDT", "USDC"):
                    usd_cash += total
                    portfolio_value += total
                elif total > 0:
                    try:
                        product_id = f"{currency}-USDT"
                        best = client.get_best_bid_ask(product_ids=[product_id])
                        pricebooks = best.get("pricebooks", [])
                        if pricebooks:
                            price = float(pricebooks[0].get("bids", [{}])[0].get("price", 0))
                            portfolio_value += total * price
                    except Exception:
                        pass

            return {
                "cash": usd_cash,
                "buying_power": usd_cash,
                "portfolio_value": portfolio_value,
                "equity": portfolio_value,
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Coinbase get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0,
                    "equity": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            client = self._get_client()
            accounts = client.get_accounts()
            result = []
            for acc in accounts.get("accounts", []):
                currency = acc.get("currency", "")
                avail = float(acc.get("available_balance", {}).get("value", 0))
                hold = float(acc.get("hold", {}).get("value", 0))
                total = avail + hold
                if total <= 0 or currency in ("USD", "USDT", "USDC"):
                    continue
                try:
                    product_id = f"{currency}-USDT"
                    best = client.get_best_bid_ask(product_ids=[product_id])
                    pricebooks = best.get("pricebooks", [])
                    current_price = float(pricebooks[0].get("bids", [{}])[0].get("price", 0)) if pricebooks else 0.0
                except Exception:
                    current_price = 0.0
                market_value = total * current_price
                result.append({
                    "ticker": currency,
                    "qty": total,
                    "avg_cost": 0.0,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": 0.0,
                })
            return result
        except Exception as e:
            logger.error(f"Coinbase get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            # Normalise: 'BTC-USDT' -> 'BTC', 'BTCUSDT' -> 'BTC'
            asset = ticker.upper().split("-")[0].replace("USDT", "")
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == asset:
                    return p
            return None
        except Exception as e:
            logger.error(f"Coinbase get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: int, price: float | None = None) -> dict:
        client = self._get_client()
        product_id = _to_coinbase_product(ticker)
        client_order_id = str(uuid.uuid4())

        order = client.market_order_buy(
            client_order_id=client_order_id,
            product_id=product_id,
            base_size=str(qty),
        )
        order_id = order.get("success_response", {}).get("order_id", client_order_id)
        status = "submitted" if order.get("success") else "failed"
        logger.info(f"Coinbase BUY submitted: {product_id} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": product_id,
            "qty": qty,
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: int | None = None) -> dict:
        client = self._get_client()
        product_id = _to_coinbase_product(ticker)

        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]

        client_order_id = str(uuid.uuid4())
        order = client.market_order_sell(
            client_order_id=client_order_id,
            product_id=product_id,
            base_size=str(qty),
        )
        order_id = order.get("success_response", {}).get("order_id", client_order_id)
        status = "submitted" if order.get("success") else "failed"
        logger.info(f"Coinbase SELL submitted: {product_id} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": product_id,
            "qty": qty,
            "status": status,
        }

    def is_market_open(self) -> bool:
        """Crypto trades 24/7."""
        return True

    def get_asset(self, ticker: str) -> dict | None:
        try:
            client = self._get_client()
            product_id = _to_coinbase_product(ticker)
            product = client.get_product(product_id)
            if product is None:
                return None
            tradable = product.get("status", "") == "online"
            return {
                "symbol": product_id,
                "name": product.get("display_name", product_id),
                "tradable": tradable,
                "fractionable": True,
            }
        except Exception as e:
            logger.warning(f"Coinbase get_asset({ticker}) returned None: {e}")
            return None
