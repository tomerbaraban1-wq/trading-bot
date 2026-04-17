import logging
from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

try:
    import okx.Trade as TradeAPI
    import okx.Account as AccountAPI
    import okx.MarketData as MarketAPI
except ImportError:
    raise ImportError("pip install python-okx>=0.2.0")


class OKXBroker(BrokerBase):
    """OKX crypto exchange broker via python-okx library."""

    def __init__(self):
        self._api_key = settings.OKX_API_KEY
        self._secret = settings.OKX_SECRET
        self._passphrase = settings.OKX_PASSPHRASE

        if not self._api_key or not self._secret or not self._passphrase:
            raise ValueError(
                "OKX credentials not configured. Set OKX_API_KEY, OKX_SECRET and OKX_PASSPHRASE."
            )

        self._trade_api = None
        self._account_api = None
        self._market_api = None

    def _get_trade(self):
        if self._trade_api is None:
            self._trade_api = TradeAPI.TradeAPI(
                self._api_key, self._secret, self._passphrase, False, "0"
            )
            logger.info("OKX TradeAPI initialised.")
        return self._trade_api

    def _get_account(self):
        if self._account_api is None:
            self._account_api = AccountAPI.AccountAPI(
                self._api_key, self._secret, self._passphrase, False, "0"
            )
        return self._account_api

    def _get_market(self):
        if self._market_api is None:
            self._market_api = MarketAPI.MarketAPI(flag="0")
        return self._market_api

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            acct = self._get_account()
            resp = acct.get_account_balance()
            details = resp.get("data", [{}])[0].get("details", [])
            cash = 0.0
            portfolio_value = 0.0
            for d in details:
                if d.get("ccy") == "USDT":
                    cash = float(d.get("cashBal", 0) or 0)
                    portfolio_value += cash
                else:
                    qty = float(d.get("cashBal", 0) or 0)
                    if qty > 0:
                        inst_id = d["ccy"] + "-USDT"
                        try:
                            mkt = self._get_market()
                            price_resp = mkt.get_ticker(instId=inst_id)
                            price = float(price_resp["data"][0]["last"])
                            portfolio_value += qty * price
                        except Exception:
                            pass
            return {
                "cash": cash,
                "buying_power": cash,
                "portfolio_value": portfolio_value,
            }
        except Exception as e:
            logger.error(f"OKX get_account failed: {e}")
            return {"cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}

    def get_positions(self) -> list[dict]:
        try:
            acct = self._get_account()
            resp = acct.get_account_balance()
            details = resp.get("data", [{}])[0].get("details", [])
            result = []
            for d in details:
                qty = float(d.get("cashBal", 0) or 0)
                ccy = d.get("ccy", "")
                if qty <= 0 or ccy == "USDT":
                    continue
                inst_id = ccy + "-USDT"
                try:
                    mkt = self._get_market()
                    price_resp = mkt.get_ticker(instId=inst_id)
                    current_price = float(price_resp["data"][0]["last"])
                except Exception:
                    current_price = 0.0
                market_value = qty * current_price
                result.append({
                    "ticker": ccy,
                    "qty": qty,
                    "avg_cost": 0.0,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": 0.0,
                })
            return result
        except Exception as e:
            logger.error(f"OKX get_positions failed: {e}")
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
            logger.error(f"OKX get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: int, price: float | None = None) -> dict:
        trade = self._get_trade()
        asset = ticker.upper().replace("-USDT", "").replace("USDT", "")
        inst_id = asset + "-USDT"
        try:
            resp = trade.place_order(
                instId=inst_id,
                tdMode="cash",
                side="buy",
                ordType="market",
                sz=str(qty),
            )
        except Exception as e:
            logger.error(f"OKX BUY failed: {e}")
            raise
        data = resp.get("data", [{}])[0]
        order_id = str(data.get("ordId", "unknown"))
        status = "submitted" if resp.get("code") == "0" else "error"
        logger.info(f"OKX BUY submitted: {inst_id} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": inst_id,
            "qty": float(qty),
            "status": status,
        }

    def submit_sell(self, ticker: str, qty: int | None = None) -> dict:
        trade = self._get_trade()
        asset = ticker.upper().replace("-USDT", "").replace("USDT", "")
        inst_id = asset + "-USDT"
        if qty is None:
            position = self.get_position(asset)
            if not position:
                raise ValueError(f"No open position for {asset}")
            qty = position["qty"]
        try:
            resp = trade.place_order(
                instId=inst_id,
                tdMode="cash",
                side="sell",
                ordType="market",
                sz=str(qty),
            )
        except Exception as e:
            logger.error(f"OKX SELL failed: {e}")
            raise
        data = resp.get("data", [{}])[0]
        order_id = str(data.get("ordId", "unknown"))
        status = "submitted" if resp.get("code") == "0" else "error"
        logger.info(f"OKX SELL submitted: {inst_id} x{qty} order_id={order_id}")
        return {
            "order_id": order_id,
            "ticker": inst_id,
            "qty": float(qty),
            "status": status,
        }

    def is_market_open(self) -> bool:
        """Crypto trades 24/7."""
        return True

    def get_asset(self, ticker: str) -> dict | None:
        try:
            mkt = self._get_market()
            asset = ticker.upper().replace("-USDT", "").replace("USDT", "")
            inst_id = asset + "-USDT"
            resp = mkt.get_ticker(instId=inst_id)
            data = resp.get("data", [])
            if not data:
                return None
            return {"tradable": True}
        except Exception as e:
            logger.warning(f"OKX get_asset({ticker}) failed: {e}")
            return None
