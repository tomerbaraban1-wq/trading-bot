from abc import ABC, abstractmethod


class BrokerBase(ABC):
    """Abstract base class that all broker implementations must satisfy."""

    @abstractmethod
    def get_account(self) -> dict:
        """Return account info: equity, buying_power, cash, portfolio_value, status."""

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Return list of open positions."""

    @abstractmethod
    def get_position(self, ticker: str) -> dict | None:
        """Return a single open position or None if not held."""

    @abstractmethod
    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        """Submit a market buy order. Returns order confirmation dict."""

    @abstractmethod
    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        """Submit a market sell order for qty shares (all if qty is None)."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the market is currently open for trading."""

    @abstractmethod
    def get_asset(self, ticker: str) -> dict | None:
        """Return asset metadata or None if not tradable."""
