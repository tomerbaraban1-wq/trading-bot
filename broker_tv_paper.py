import json
import os
import time
import threading
import logging
from pathlib import Path
try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required: pip install yfinance")

from broker_base import BrokerBase
from config import settings

logger = logging.getLogger(__name__)

# ── Persistence ───────────────────────────────────────────────────────────────
# Primary: Postgres (Neon) via DATABASE_URL env var — survives Render restarts.
# Fallback: local JSON file — used only if DATABASE_URL is not set.
DATABASE_URL = "".join(os.getenv("DATABASE_URL", "").split())  # strip all whitespace/newlines

# Support individual connection params as fallback (easier to paste in Render)
_NEON_HOST = os.getenv("NEON_HOST", "").strip()
_NEON_USER = os.getenv("NEON_USER", "neondb_owner").strip()
_NEON_PASSWORD = os.getenv("NEON_PASSWORD", "").strip()
_NEON_DB = os.getenv("NEON_DB", "neondb").strip()

# Build DATABASE_URL from parts if not provided directly
if not DATABASE_URL and _NEON_HOST and _NEON_PASSWORD:
    DATABASE_URL = (
        f"postgresql://{_NEON_USER}:{_NEON_PASSWORD}"
        f"@{_NEON_HOST}/{_NEON_DB}?sslmode=require"
    )

USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    try:
        import psycopg
    except ImportError:
        logger.warning("psycopg not installed — falling back to JSON file persistence")
        USE_POSTGRES = False


def _state_path() -> Path:
    """Path to the JSON file that persists paper broker state across restarts."""
    db_path = Path(settings.DATABASE_PATH)
    return db_path.parent / "paper_broker_state.json"


def _pg_init() -> None:
    """Create the state table in Postgres if it doesn't exist."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paper_broker_state (
                    id INTEGER PRIMARY KEY,
                    cash DOUBLE PRECISION NOT NULL,
                    positions JSONB NOT NULL,
                    saved_at DOUBLE PRECISION NOT NULL
                )
            """)
        conn.commit()


def _pg_load() -> dict | None:
    """Load state from Postgres. Returns None if no row exists."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT cash, positions FROM paper_broker_state WHERE id = 1")
            row = cur.fetchone()
            if not row:
                return None
            return {"cash": float(row[0]), "positions": row[1]}


def _pg_save(cash: float, positions: dict) -> None:
    """Upsert state into Postgres."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO paper_broker_state (id, cash, positions, saved_at)
                VALUES (1, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE SET
                    cash = EXCLUDED.cash,
                    positions = EXCLUDED.positions,
                    saved_at = EXCLUDED.saved_at
            """, (cash, json.dumps(positions), time.time()))
        conn.commit()


class TVPaperBroker(BrokerBase):
    """
    Built-in paper trading simulation — no API keys needed.

    Mimics TradingView's own paper trading feature. Prices are fetched
    in real time via yfinance. All state (positions, cash) is persisted to
    paper_broker_state.json so it survives Render restarts.
    """

    # Class-level state — shared across all instances
    _positions: dict = {}   # ticker -> {"qty": float, "avg_cost": float}
    _cash: float = None     # initialised lazily from settings.MAX_BUDGET
    _lock = threading.Lock()      # guards buy/sell mutations against race conditions
    _init_lock = threading.Lock() # guards one-time state load
    _state_loaded: bool = False   # ensure we only load from disk once

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self):
        with TVPaperBroker._init_lock:
            if not TVPaperBroker._state_loaded:
                TVPaperBroker._load_state()
                TVPaperBroker._state_loaded = True
        logger.info(
            "TradingView Paper Broker ready | "
            f"Cash: ${TVPaperBroker._cash:,.2f} | "
            f"Positions: {list(TVPaperBroker._positions.keys()) or 'none'}"
        )

    # ------------------------------------------------------------------ #
    #  State persistence                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_state(cls) -> None:
        """Load positions and cash from Postgres (primary) or disk (fallback).
        Falls back to MAX_BUDGET if neither source has data."""
        # Try Postgres first
        if USE_POSTGRES:
            try:
                _pg_init()
                data = _pg_load()
                if data is not None:
                    cls._cash = float(data["cash"])
                    raw_pos = data.get("positions", {}) or {}
                    cls._positions = {
                        t: {"qty": float(p["qty"]), "avg_cost": float(p["avg_cost"])}
                        for t, p in raw_pos.items()
                    }
                    logger.info(
                        f"[TVPaper] State loaded from Postgres | "
                        f"cash=${cls._cash:,.2f} | positions={list(cls._positions.keys())}"
                    )
                    return
                else:
                    logger.info("[TVPaper] No state row in Postgres — starting fresh")
                    cls._cash = float(settings.MAX_BUDGET)
                    cls._positions = {}
                    return
            except Exception as e:
                logger.warning(f"[TVPaper] Postgres load failed ({e}) — trying JSON fallback")

        # Fallback: JSON file
        path = _state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                cls._cash = float(data.get("cash", settings.MAX_BUDGET))
                raw_pos = data.get("positions", {})
                cls._positions = {
                    t: {"qty": float(p["qty"]), "avg_cost": float(p["avg_cost"])}
                    for t, p in raw_pos.items()
                }
                logger.info(
                    f"[TVPaper] State loaded from {path} | "
                    f"cash=${cls._cash:,.2f} | positions={list(cls._positions.keys())}"
                )
            else:
                cls._cash = float(settings.MAX_BUDGET)
                cls._positions = {}
                logger.info(f"[TVPaper] No state file found — starting fresh with ${cls._cash:,.2f}")
        except Exception as e:
            logger.warning(f"[TVPaper] Failed to load state ({e}) — starting fresh")
            cls._cash = float(settings.MAX_BUDGET)
            cls._positions = {}

    @classmethod
    def _save_state(cls) -> None:
        """Persist current positions and cash. Writes to Postgres if configured,
        otherwise falls back to a local JSON file."""
        # Primary: Postgres
        if USE_POSTGRES:
            try:
                _pg_save(cls._cash, cls._positions)
                return
            except Exception as e:
                logger.warning(f"[TVPaper] Postgres save failed ({e}) — falling back to JSON file")

        # Fallback: JSON file
        path = _state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "cash": cls._cash,
                "positions": cls._positions,
                "saved_at": time.time(),
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"[TVPaper] Failed to save state: {e}")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_price(self, ticker: str) -> float:
        """Fetch the latest market price for *ticker* via yfinance."""
        ticker = ticker.upper()
        try:
            t = yf.Ticker(ticker)

            # Try intraday history first (most accurate live price)
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                if price > 0:
                    return price

            # Fall back to .info dict
            info = t.info
            price = float(
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
                or 0
            )
            if price > 0:
                return price

        except Exception as exc:
            raise RuntimeError(
                f"yfinance failed to fetch price for '{ticker}': {exc}"
            ) from exc

        raise RuntimeError(
            f"Could not obtain a valid price for '{ticker}' from yfinance. "
            "The ticker may be invalid or the market data feed is unavailable."
        )

    @staticmethod
    def _order_id(ticker: str) -> str:
        return f"TV_{ticker.upper()}_{int(time.time())}"

    def _portfolio_value(self) -> float:
        """Current market value of all virtual positions."""
        total = 0.0
        for ticker, pos in TVPaperBroker._positions.items():
            try:
                price = self._get_price(ticker)
            except Exception:
                price = pos["avg_cost"]   # fall back to cost if price unavailable
            total += price * pos["qty"]
        return total

    # ------------------------------------------------------------------ #
    #  BrokerBase implementation                                           #
    # ------------------------------------------------------------------ #

    def get_account(self) -> dict:
        cash = TVPaperBroker._cash
        portfolio_value = self._portfolio_value()
        equity = cash + portfolio_value
        return {
            "equity": round(equity, 2),
            "buying_power": round(cash, 2),
            "cash": round(cash, 2),
            "portfolio_value": round(portfolio_value, 2),
            "status": "ACTIVE",
        }

    def get_positions(self) -> list[dict]:
        result = []
        for ticker, pos in TVPaperBroker._positions.items():
            try:
                current_price = self._get_price(ticker)
            except Exception:
                current_price = pos["avg_cost"]

            qty = pos["qty"]
            avg_cost = pos["avg_cost"]
            market_value = current_price * qty
            unrealized_pl = (current_price - avg_cost) * qty
            unrealized_plpc = (
                (current_price - avg_cost) / avg_cost if avg_cost else 0.0
            )

            result.append(
                {
                    "ticker": ticker,
                    "qty": qty,
                    "avg_entry_price": round(avg_cost, 4),
                    "current_price": round(current_price, 4),
                    "market_value": round(market_value, 2),
                    "unrealized_pl": round(unrealized_pl, 2),
                    "unrealized_plpc": round(unrealized_plpc, 6),
                }
            )
        return result

    def get_position(self, ticker: str) -> dict | None:
        ticker = ticker.upper()
        pos = TVPaperBroker._positions.get(ticker)
        if not pos:
            return None

        try:
            current_price = self._get_price(ticker)
        except Exception:
            current_price = pos["avg_cost"]

        qty = pos["qty"]
        avg_cost = pos["avg_cost"]
        market_value = current_price * qty
        unrealized_pl = (current_price - avg_cost) * qty
        unrealized_plpc = (current_price - avg_cost) / avg_cost if avg_cost else 0.0

        return {
            "ticker": ticker,
            "qty": qty,
            "avg_entry_price": round(avg_cost, 4),
            "current_price": round(current_price, 4),
            "market_value": round(market_value, 2),
            "unrealized_pl": round(unrealized_pl, 2),
            "unrealized_plpc": round(unrealized_plpc, 6),
        }

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        """Supports fractional shares — qty can be e.g. 0.5 or 2.37.

        Note: For paper trading, we ALWAYS fill at the actual market price
        from yfinance, ignoring any inflated limit price. Using the limit
        price (which is +0.5% above market for slippage protection) would
        make every position appear at an immediate ~0.5% loss, since
        get_positions reads the real market price. The `price` argument is
        accepted for compatibility with real brokers but only used as a
        last-resort fallback if yfinance is unavailable.
        """
        ticker = ticker.upper()

        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        # Always use the live market price for paper fills.
        try:
            current_price = self._get_price(ticker)
        except Exception:
            if price and price > 0:
                logger.warning(
                    f"[TVPaper] yfinance unavailable for {ticker} — falling back to passed price ${price:.4f}"
                )
                current_price = float(price)
            else:
                raise

        # Round fractional qty to 6 decimal places
        qty = round(float(qty), 6)
        cost = current_price * qty

        # Atomic cash check + mutation under lock to prevent over-spending race
        with TVPaperBroker._lock:
            if cost > TVPaperBroker._cash:
                raise ValueError(
                    f"Insufficient virtual cash: need ${cost:,.2f} "
                    f"but only ${TVPaperBroker._cash:,.2f} available"
                )

            # Update position (weighted average cost if already held)
            if ticker in TVPaperBroker._positions:
                existing = TVPaperBroker._positions[ticker]
                old_qty = existing["qty"]
                old_cost = existing["avg_cost"]
                new_qty = old_qty + qty
                new_avg = (old_cost * old_qty + current_price * qty) / new_qty
                TVPaperBroker._positions[ticker] = {"qty": new_qty, "avg_cost": new_avg}
            else:
                TVPaperBroker._positions[ticker] = {"qty": qty, "avg_cost": current_price}

            TVPaperBroker._cash -= cost
            TVPaperBroker._save_state()

        order_id = self._order_id(ticker)
        logger.info(
            f"[TVPaper] BUY {ticker} x{qty:.4f} @ ${current_price:.4f} "
            f"| cost=${cost:,.2f} | cash_remaining=${TVPaperBroker._cash:,.2f}"
        )
        return {
            "order_id": order_id,
            "symbol": ticker,
            "qty": qty,
            "price": round(current_price, 4),
            "cost": round(cost, 2),
            "status": "filled",
            "type": "market",
        }

    def submit_sell(self, ticker: str, qty: float | None = None) -> dict:
        ticker = ticker.upper()

        pos = TVPaperBroker._positions.get(ticker)
        if not pos:
            raise ValueError(f"No open virtual position for {ticker}")

        held_qty = pos["qty"]

        if qty is None or qty >= held_qty:
            # Sell the entire position
            sell_qty = held_qty
        else:
            if qty <= 0:
                raise ValueError(f"qty must be positive, got {qty}")
            sell_qty = qty

        current_price = self._get_price(ticker)  # raises if price = 0
        proceeds = current_price * sell_qty

        # Atomic state mutation under lock
        with TVPaperBroker._lock:
            if sell_qty >= held_qty:
                TVPaperBroker._positions.pop(ticker, None)
            else:
                TVPaperBroker._positions[ticker]["qty"] = held_qty - sell_qty

            TVPaperBroker._cash += proceeds
            TVPaperBroker._save_state()

        order_id = self._order_id(ticker)
        logger.info(
            f"[TVPaper] SELL {ticker} x{sell_qty} @ ${current_price:.4f} "
            f"| proceeds=${proceeds:,.2f} | cash_now=${TVPaperBroker._cash:,.2f}"
        )
        return {
            "order_id": order_id,
            "symbol": ticker,
            "qty": sell_qty,
            "price": round(current_price, 4),
            "proceeds": round(proceeds, 2),
            "status": "filled",
        }

    def is_market_open(self) -> bool:
        """
        Check US equity market hours using UTC time (fast, no API call).
        Accounts for EDT (summer) and EST (winter) automatically.
        EDT: 13:30-20:00 UTC | EST: 14:30-21:00 UTC
        """
        import datetime
        now = datetime.datetime.utcnow()
        # Skip weekends
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        # Determine if US is on EDT (2nd Sun Mar - 1st Sun Nov) or EST
        # Simple approximation: EDT runs roughly March 8 – November 1
        month = now.month
        is_edt = 3 <= month <= 10  # close enough for trading purposes
        if is_edt:
            open_hour, open_min = 13, 30   # 9:30 ET = 13:30 UTC
            close_hour, close_min = 20, 0  # 4:00 ET = 20:00 UTC
        else:
            open_hour, open_min = 14, 30   # 9:30 ET = 14:30 UTC
            close_hour, close_min = 21, 0  # 4:00 ET = 21:00 UTC
        market_open  = now.replace(hour=open_hour,  minute=open_min,  second=0, microsecond=0)
        market_close = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)
        return market_open <= now <= market_close

    def get_asset(self, ticker: str) -> dict | None:
        """Always return tradable — paper trading accepts any ticker."""
        return {
            "symbol": ticker.upper(),
            "name": ticker.upper(),
            "tradable": True,
            "fractionable": False,
        }
