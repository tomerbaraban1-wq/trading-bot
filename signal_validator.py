import logging
import time
import threading
from collections import OrderedDict
from config import settings
import broker
from trading_hours import is_ok_to_trade

logger = logging.getLogger(__name__)

# Duplicate signal filter: ticker:action -> timestamp
_recent_signals: OrderedDict = OrderedDict()
_signals_lock = threading.Lock()  # guards _recent_signals against concurrent webhook threads
DUPLICATE_WINDOW = 300  # 5 minutes


def validate_signal(ticker: str, action: str) -> tuple[bool, str]:
    """
    Validate an incoming signal. Returns (is_valid, reason).
    """
    # Check duplicate
    if _is_duplicate(ticker, action):
        return False, f"Duplicate signal: {ticker} {action} (within {DUPLICATE_WINDOW}s)"

    # Trading hours / liquidity / FOMC blackout guard (BUY only)
    if action.lower() == "buy":
        ok, hours_reason = is_ok_to_trade()
        if not ok:
            logger.info(f"[HOURS] {ticker} BUY blocked: {hours_reason}")
            return False, f"Trading hours: {hours_reason}"

    # Check asset is tradable
    try:
        asset = broker.get_asset(ticker)
        if not asset:
            return False, f"Asset {ticker} not found or not tradable"
    except Exception as e:
        logger.warning(f"Could not verify asset {ticker}: {e}")

    # Record this signal
    _record_signal(ticker, action)
    return True, "Signal valid"


def _is_duplicate(ticker: str, action: str) -> bool:
    key = f"{ticker.upper()}:{action.lower()}"
    now = time.time()

    with _signals_lock:
        # Clean old entries
        while _recent_signals:
            oldest_key, oldest_time = next(iter(_recent_signals.items()))
            if now - oldest_time > DUPLICATE_WINDOW:
                _recent_signals.pop(oldest_key)
            else:
                break

        return key in _recent_signals and (now - _recent_signals[key]) < DUPLICATE_WINDOW


def _record_signal(ticker: str, action: str):
    key = f"{ticker.upper()}:{action.lower()}"
    with _signals_lock:
        _recent_signals[key] = time.time()
