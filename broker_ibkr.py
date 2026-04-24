import logging
from datetime import datetime, time as dtime
from broker_base import BrokerBase

logger = logging.getLogger(__name__)


class IBKRBroker(BrokerBase):
    """Interactive Brokers broker via ib_insync."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = None

    def _get_ib(self):
        if self._ib is not None and self._ib.isConnected():
            return self._ib
        try:
            from ib_insync import IB
            ib = IB()
            ib.connect(self._host, self._port, clientId=self._client_id, timeout=10)
            self._ib = ib
            logger.info(f"IBKR connected to {self._host}:{self._port}")
            return self._ib
        except Exception as e:
            logger.error(f"IBKR connection failed ({self._host}:{self._port}): {e}")
            raise ConnectionError(f"IBKR unavailable: {e}") from e

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        try:
            ib = self._get_ib()
            vals = {v.tag: v.value for v in ib.accountValues() if v.currency in ("USD", "")}
            return {
                "equity": float(vals.get("NetLiquidation", 0)),
                "buying_power": float(vals.get("BuyingPower", 0)),
                "cash": float(vals.get("CashBalance", 0)),
                "portfolio_value": float(vals.get("GrossPositionValue", 0)),
                "status": "active",
            }
        except Exception as e:
            logger.error(f"IBKR get_account failed: {e}")
            return {"equity": 0.0, "buying_power": 0.0, "cash": 0.0,
                    "portfolio_value": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        try:
            ib = self._get_ib()
            result = []
            for pos in ib.positions():
                contract = pos.contract
                qty = pos.position
                avg_cost = pos.avgCost
                ticker_data = ib.reqMktData(contract, "", True, False)
                ib.sleep(1)
                current_price = ticker_data.last or ticker_data.close or avg_cost
                market_value = qty * current_price
                unrealized_pl = (current_price - avg_cost) * qty
                unrealized_plpc = ((current_price - avg_cost) / avg_cost) if avg_cost else 0.0
                result.append({
                    "ticker": contract.symbol,
                    "qty": float(qty),
                    "avg_entry_price": float(avg_cost),
                    "current_price": float(current_price),
                    "market_value": float(market_value),
                    "unrealized_pl": float(unrealized_pl),
                    "unrealized_plpc": float(unrealized_plpc),
                })
            return result
        except Exception as e:
            logger.error(f"IBKR get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"IBKR get_position({ticker}) failed: {e}")
            return None

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        from ib_insync import Stock, MarketOrder
        ib = self._get_ib()
        contract = Stock(ticker.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("BUY", qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        logger.info(f"IBKR BUY submitted: {ticker} x{qty}")
        return {
            "order_id": str(trade.order.orderId),
            "symbol": ticker.upper(),
            "qty": qty,
            "status": str(trade.orderStatus.status),
            "type": "market",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        from ib_insync import Stock, MarketOrder
        ib = self._get_ib()

        if qty is None:
            position = self.get_position(ticker)
            if not position:
                raise ValueError(f"No open position for {ticker}")
            qty = position["qty"]

        contract = Stock(ticker.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("SELL", qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        logger.info(f"IBKR SELL submitted: {ticker} x{qty}")
        return {
            "order_id": str(trade.order.orderId),
            "symbol": ticker.upper(),
            "qty": qty,
            "status": str(trade.orderStatus.status),
        }

    def is_market_open(self) -> bool:
        try:
            now = datetime.utcnow()
            # NYSE hours: Mon-Fri 13:30-20:00 UTC
            if now.weekday() >= 5:
                return False
            market_open = dtime(13, 30)
            market_close = dtime(20, 0)
            current = now.time()
            return market_open <= current < market_close
        except Exception as e:
            logger.error(f"IBKR is_market_open failed: {e}")
            return False

    def get_asset(self, ticker: str) -> dict | None:
        try:
            from ib_insync import Stock
            ib = self._get_ib()
            contract = Stock(ticker.upper(), "SMART", "USD")
            details = ib.reqContractDetails(contract)
            if not details:
                return None
            d = details[0]
            return {
                "symbol": ticker.upper(),
                "name": d.longName,
                "tradable": True,
                "fractionable": False,
            }
        except Exception as e:
            logger.error(f"IBKR get_asset({ticker}) failed: {e}")
            return None
