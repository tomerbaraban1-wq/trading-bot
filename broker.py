"""
broker.py — factory module.

Reads settings.ACTIVE_BROKER and exposes the same top-level functions as
before so that the rest of the codebase (webhook.py, budget.py, etc.)
requires zero changes.

Supported values for ACTIVE_BROKER:
  tv_paper      (default — built-in simulation, no API keys needed)
  alpaca_paper
  alpaca_live
  ibkr
  oanda
  tradier
  tradier_live
  tastytrade
  schwab
  binance
  kraken
  coinbase
  robinhood
  webull
  bybit
  okx
  kucoin
  gemini
  tradestation
"""

import logging
from broker_base import BrokerBase

logger = logging.getLogger(__name__)

# Module-level broker instance — replaced at runtime by switch_broker()
_broker: BrokerBase | None = None


def _build_broker(name: str) -> BrokerBase:
    name = name.lower().strip()

    if name == "tv_paper":
        from broker_tv_paper import TVPaperBroker
        return TVPaperBroker()

    if name == "alpaca_paper":
        from broker_alpaca import AlpacaBroker
        return AlpacaBroker(paper=True)

    if name == "alpaca_live":
        from broker_alpaca import AlpacaBroker
        return AlpacaBroker(paper=False)

    if name == "ibkr":
        from broker_ibkr import IBKRBroker
        from config import settings
        return IBKRBroker(
            host=settings.IBKR_HOST,
            port=settings.IBKR_PORT,
            client_id=1,
        )

    if name == "oanda":
        from broker_oanda import OandaBroker
        return OandaBroker()

    if name in ("tradier", "tradier_paper"):
        from broker_tradier import TradierBroker
        return TradierBroker(paper=True)

    if name == "tradier_live":
        from broker_tradier import TradierBroker
        return TradierBroker(paper=False)

    if name == "tastytrade":
        from broker_tastytrade import TastytradeBroker
        return TastytradeBroker()

    if name == "schwab":
        from broker_schwab import SchwabBroker
        return SchwabBroker()

    if name == "binance":
        from broker_binance import BinanceBroker
        return BinanceBroker()

    if name == "kraken":
        from broker_kraken import KrakenBroker
        return KrakenBroker()

    if name == "coinbase":
        from broker_coinbase import CoinbaseBroker
        return CoinbaseBroker()

    if name == "robinhood":
        from broker_robinhood import RobinhoodBroker
        return RobinhoodBroker()

    if name == "webull":
        from broker_webull import WebullBroker
        return WebullBroker()

    if name == "bybit":
        from broker_bybit import BybitBroker
        return BybitBroker()

    if name == "okx":
        from broker_okx import OKXBroker
        return OKXBroker()

    if name == "kucoin":
        from broker_kucoin import KuCoinBroker
        return KuCoinBroker()

    if name == "gemini":
        from broker_gemini import GeminiBroker
        return GeminiBroker()

    if name == "tradestation":
        from broker_tradestation import TradeStationBroker
        return TradeStationBroker()

    raise ValueError(
        f"Unknown broker '{name}'. Valid options: "
        "tv_paper, alpaca_paper, alpaca_live, ibkr, oanda, tradier, tradier_live, "
        "tastytrade, schwab, binance, kraken, coinbase, "
        "robinhood, webull, bybit, okx, kucoin, gemini, tradestation"
    )


def _get_broker() -> BrokerBase:
    global _broker
    if _broker is None:
        from config import settings
        active = getattr(settings, "ACTIVE_BROKER", "alpaca_paper") or "alpaca_paper"
        logger.info(f"Initialising broker: {active}")
        _broker = _build_broker(active)
    return _broker


def switch_broker(name: str) -> str:
    """Hot-swap the active broker at runtime. Returns the new broker name."""
    global _broker
    logger.info(f"Switching broker to: {name}")
    _broker = _build_broker(name)
    # Persist the choice back into settings so other modules see it
    from config import settings
    settings.ACTIVE_BROKER = name
    logger.info(f"Active broker is now: {name}")
    return name


# ------------------------------------------------------------------
# Public API — identical signatures to the original broker.py so
# every import site keeps working without modification.
# ------------------------------------------------------------------

def get_account() -> dict:
    return _get_broker().get_account()


def get_positions() -> list[dict]:
    return _get_broker().get_positions()


def get_position(ticker: str) -> dict | None:
    return _get_broker().get_position(ticker)


def submit_buy(ticker: str, qty: float, price: float | None = None) -> dict:
    return _get_broker().submit_buy(ticker, qty, price)


def submit_sell(ticker: str, qty: float | None = None, price: float | None = None) -> dict:
    return _get_broker().submit_sell(ticker, qty, price)


def is_market_open() -> bool:
    return _get_broker().is_market_open()


def get_clock() -> dict | None:
    """
    Return Alpaca market clock: {is_open, next_open, next_close, timestamp}.
    next_open / next_close are ISO-8601 strings in UTC.
    Falls back to None if broker doesn't support it.
    """
    try:
        b = _get_broker()
        if hasattr(b, "get_clock"):
            return b.get_clock()
        # Fallback: call Alpaca REST directly if broker exposes _api
        if hasattr(b, "_api"):
            clock = b._api.get_clock()
            return {
                "is_open":    clock.is_open,
                "next_open":  str(clock.next_open),
                "next_close": str(clock.next_close),
                "timestamp":  str(clock.timestamp),
            }
    except Exception as e:
        logger.debug(f"get_clock failed: {e}")
    return None


def get_asset(ticker: str) -> dict | None:
    return _get_broker().get_asset(ticker)


def get_price(ticker: str) -> float | None:
    """Get the current market price for a ticker.

    Hard-protected against yfinance hangs via threading timeout (12s).
    NEW: Circuit breaker prevents TCP connection leak when yfinance returns 401.
    """
    # CRITICAL FIX: Check circuit breaker first to prevent socket leaks
    try:
        from yfinance_circuit_breaker import get_breaker
        breaker = get_breaker()
        if breaker.is_open():
            logger.debug(f"get_price({ticker}): yfinance circuit open — skip")
            return None
    except Exception:
        breaker = None

    import threading
    import yfinance as yf
    result: list = [None]
    exc_box: list = [None]

    def _fetch():
        try:
            t = yf.Ticker(ticker.upper())
            hist = t.history(period="1d", interval="1m", timeout=10)
            if not hist.empty:
                p = float(hist["Close"].iloc[-1])
                if p > 0:
                    result[0] = p
                    return
            info = t.info
            p = float(
                info.get("regularMarketPrice") or
                info.get("currentPrice") or
                info.get("previousClose") or 0
            )
            if p > 0:
                result[0] = p
        except Exception as e:
            exc_box[0] = e

    th = threading.Thread(target=_fetch, daemon=True)
    th.start()
    th.join(timeout=12)
    if th.is_alive():
        logger.warning(f"get_price hung >12s for {ticker} — aborted")
        if breaker:
            breaker.record_failure()
        return None
    if exc_box[0] is not None:
        logger.warning(f"get_price failed for {ticker}: {exc_box[0]}")
        if breaker:
            breaker.record_failure()
        return None
    if result[0] is not None and breaker:
        breaker.record_success()
    return result[0]
