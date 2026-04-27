import logging
from datetime import datetime, time as dtime
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    import schwab
    from schwab.orders.equities import equity_buy_market, equity_sell_market
    from schwab.orders.common import Duration, Session as OrderSession
except ImportError:
    raise ImportError("pip install schwab-py>=1.0.0")


class SchwabBroker(BrokerBase):
    """Charles Schwab broker via schwab-py SDK (OAuth)."""

    def __init__(self):
        self._api_key = settings.SCHWAB_API_KEY
        self._secret = settings.SCHWAB_SECRET
        self._account_hash = settings.SCHWAB_ACCOUNT

        if not self._api_key or not self._secret:
            raise ValueError(
                "Schwab credentials not configured. Set SCHWAB_API_KEY and SCHWAB_SECRET."
            )
        if not self._account_hash:
            raise ValueError("Schwab account not configured. Set SCHWAB_ACCOUNT.")

        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            # schwab-py requires an OAuth token file obtained via the manual flow.
            # The token file path defaults to schwab_token.json in the project directory.
            import os
            from pathlib import Path
            token_path = str(Path(__file__).parent / "schwab_token.json")
            callback_url = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1")

            self._client = schwab.auth.client_from_token_file(
                token_path,
                self._api_key,
                self._secret,
            )
            logger.info("Schwab client initialised from token file.")
        except FileNotFoundError:
            logger.error(
                "schwab_token.json not found. Run the manual OAuth flow once: "
                "schwab.auth.client_from_manual_flow(api_key, secret, callback_url, token_path)"
            )
            raise ConnectionError(
                "Schwab token file missing. Complete OAuth setup first."
            )
        except Exception as e:
            logger.error(f"Schwab client init failed: {e}")
            raise ConnectionError(f"Schwab unavailable: {e}") from e
        return self._client

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            client = self._get_client()
            resp = client.get_account(self._account_hash, fields=[client.Account.Fields.POSITIONS])
            resp.raise_for_status()
            data = resp.json()
            aggregated = data.get("securitiesAccount", {}).get("currentBalances", {})
            return {
                "cash": float(aggregated.get("cashBalance", 0)),
                "buying_power": float(aggregated.get("buyingPower", 0)),
                "portfolio_value": float(aggregated.get("liquidationValue", 0)),
                "equity": float(aggregated.get("equity", 0)),
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Schwab get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0,
                    "equity": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            client = self._get_client()
            resp = client.get_account(self._account_hash, fields=[client.Account.Fields.POSITIONS])
            resp.raise_for_status()
            data = resp.json()
            positions_raw = data.get("securitiesAccount", {}).get("positions", [])
            result = []
            for p in positions_raw:
                instrument = p.get("instrument", {})
                symbol = instrument.get("symbol", "")
                qty = float(p.get("longQuantity", 0) or p.get("shortQuantity", 0))
                avg_cost = float(p.get("averagePrice", 0))
                current_price = float(p.get("marketValue", 0)) / qty if qty > 0 else avg_cost
                market_value = float(p.get("marketValue", 0))
                unrealized_pl = float(p.get("unrealizedPnL", 0))
                unrealized_plpc = (unrealized_pl / (avg_cost * qty)) if avg_cost > 0 and qty > 0 else 0.0
                result.append({
                    "ticker": symbol,
                    "qty": qty,
                    "avg_entry_price": avg_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": unrealized_plpc,
                })
            return result
        except Exception as e:
            logger.error(f"Schwab get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"Schwab get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        client = self._get_client()
        symbol = ticker.upper()
        order = equity_buy_market(symbol, qty).set_duration(Duration.DAY).set_session(OrderSession.NORMAL).build()
        resp = client.place_order(self._account_hash, order)
        resp.raise_for_status()
        order_id = resp.headers.get("Location", "unknown").split("/")[-1]
        logger.info(f"Schwab BUY submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "price": None,          # Schwab async fill — price not available at submission
            "status": "submitted",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        client = self._get_client()
        symbol = ticker.upper()

        if qty is None:
            position = self.get_position(symbol)
            if not position:
                raise ValueError(f"No open position for {symbol}")
            qty = position["qty"]

        order = equity_sell_market(symbol, qty).set_duration(Duration.DAY).set_session(OrderSession.NORMAL).build()
        resp = client.place_order(self._account_hash, order)
        resp.raise_for_status()
        order_id = resp.headers.get("Location", "unknown").split("/")[-1]
        logger.info(f"Schwab SELL submitted: {symbol} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "price": None,          # Schwab async fill — price not available at submission
            "status": "submitted",
        }

    def is_market_open(self) -> bool:
        try:
            now = datetime.utcnow()
            if now.weekday() >= 5:
                return False
            return dtime(13, 30) <= now.time() < dtime(20, 0)
        except Exception as e:
            logger.error(f"Schwab is_market_open failed: {e}")
            return False

    def get_asset(self, ticker: str) -> dict | None:
        try:
            client = self._get_client()
            symbol = ticker.upper()
            resp = client.get_instrument_by_symbol(symbol)
            resp.raise_for_status()
            data = resp.json()
            instruments = data.get("instruments", [])
            if not instruments:
                return None
            inst = instruments[0]
            return {
                "symbol": symbol,
                "name": inst.get("description", symbol),
                "tradable": True,
                "fractionable": False,
            }
        except Exception as e:
            logger.warning(f"Schwab get_asset({ticker}) returned None: {e}")
            return None
