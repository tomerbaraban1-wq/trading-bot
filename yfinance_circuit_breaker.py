"""
yfinance Circuit Breaker — מונע דליפת TCP connections
====================================================

הבעיה האמיתית:
- yfinance נכשל עם 401 Unauthorized
- requests.Session() לא נסגר נכון אחרי כישלון
- 78+ TCP connections תקועים ב-CloseWait
- ב-Windows: file descriptor limit מתמלא → exit=4294967295

הפתרון:
- אם yfinance נכשל N פעמים ב-window זמן → blokuje קריאות לפרק זמן
- מנקה sessions ישנים אוטומטית
- מאזין על rate-limit ב-Yahoo
"""

import logging
import time
import gc
from typing import Optional
from functools import wraps

logger = logging.getLogger(__name__)


class YFinanceCircuitBreaker:
    """Circuit breaker for yfinance API."""

    def __init__(
        self,
        failure_threshold: int = 3,   # AGGRESSIVE: 10→3 (was leaking too fast)
        timeout_seconds: int = 600,   # 10 minutes
        window_seconds: int = 120,    # 60→120 (catch more failures)
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.window_seconds = window_seconds
        self.failures: list[float] = []  # timestamps
        self.circuit_open_until: float = 0
        self.total_failures = 0
        self.total_successes = 0
        self.last_cleanup = time.time()

    def is_open(self) -> bool:
        """Check if circuit is open (blocked)."""
        if time.time() < self.circuit_open_until:
            return True
        # Auto-close after timeout
        if self.circuit_open_until > 0 and time.time() >= self.circuit_open_until:
            logger.info(
                f"[YF_CIRCUIT] Auto-closing after {self.timeout_seconds}s timeout "
                f"(failures={self.total_failures}, successes={self.total_successes})"
            )
            self.circuit_open_until = 0
            self.failures.clear()
        return False

    def record_failure(self) -> None:
        """Record a yfinance failure."""
        now = time.time()
        self.total_failures += 1
        self.failures.append(now)

        # Clean old failures outside window
        self.failures = [t for t in self.failures if now - t < self.window_seconds]

        # Trip the breaker?
        if len(self.failures) >= self.failure_threshold:
            self.circuit_open_until = now + self.timeout_seconds
            logger.warning(
                f"[YF_CIRCUIT] OPEN — {len(self.failures)} failures in "
                f"{self.window_seconds}s. Blocking yfinance for {self.timeout_seconds}s. "
                f"Total: {self.total_failures} fails, {self.total_successes} ok."
            )
            self.failures.clear()
            # Force cleanup of stale sessions
            self._force_cleanup()

    def record_success(self) -> None:
        """Record a yfinance success."""
        self.total_successes += 1
        # On success, reset failure window
        self.failures.clear()

    def _force_cleanup(self) -> None:
        """Force-close any stale yfinance sessions and run gc."""
        try:
            import yfinance as yf
            # yfinance uses requests.Session internally; try to close cached ones
            if hasattr(yf, '_cache') and hasattr(yf._cache, 'close'):
                try:
                    yf._cache.close()
                except Exception:
                    pass

            # Close any data-cache sessions
            try:
                from yfinance import shared as _shared
                if hasattr(_shared, '_DFS'):
                    _shared._DFS.clear()
                if hasattr(_shared, '_ERRORS'):
                    _shared._ERRORS.clear()
            except Exception:
                pass

            # Force garbage collection of dangling sockets
            gc.collect()
            logger.info("[YF_CIRCUIT] Forced cleanup of stale yfinance state")
        except Exception as e:
            logger.debug(f"[YF_CIRCUIT] Cleanup error: {e}")


# Global singleton
_BREAKER = YFinanceCircuitBreaker()


def get_breaker() -> YFinanceCircuitBreaker:
    return _BREAKER


def with_yf_breaker(default_return=None):
    """Decorator that wraps a yfinance call with circuit breaker."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _BREAKER.is_open():
                logger.debug(f"[YF_CIRCUIT] Open — skipping {fn.__name__}")
                return default_return
            try:
                result = fn(*args, **kwargs)
                _BREAKER.record_success()
                return result
            except Exception as e:
                _BREAKER.record_failure()
                logger.debug(f"[YF_CIRCUIT] {fn.__name__} failed: {type(e).__name__}: {str(e)[:80]}")
                return default_return
        return wrapper
    return decorator


def install_session_force_close():
    """
    AGGRESSIVE FIX: Force every HTTP request to add 'Connection: close' header.
    This makes the server (Yahoo) close the connection immediately after response,
    preventing CloseWait state accumulation.

    Side effect: Slightly slower (no keep-alive) but no socket leak.
    """
    try:
        import requests.sessions

        _original_send = requests.sessions.Session.send

        def _send_with_close(self, request, **kwargs):
            # Force Connection: close on every outgoing request
            request.headers['Connection'] = 'close'
            response = _original_send(self, request, **kwargs)
            # Aggressively close the connection on this side too
            try:
                response.close()
            except Exception:
                pass
            return response

        requests.sessions.Session.send = _send_with_close
        logger.info("[YF_PATCH] Forced 'Connection: close' on all HTTP requests")
        return True
    except Exception as e:
        logger.warning(f"[YF_PATCH] Could not install session force-close: {e}")
        return False


def install_global_monkey_patch():
    """
    GLOBAL FIX: Monkey-patch yfinance.Ticker so that EVERY caller
    (heartbeat, scoring, buffett_analysis, atr_stop, etc.) goes through
    the circuit breaker.

    This is the only way to stop the CloseWait leak — there are 20+ files
    that import yfinance directly, wrapping each individually is impractical.

    When circuit is OPEN:
    - yf.Ticker(...) returns a stub that returns empty data
    - yf.download(...) returns empty DataFrame
    - No socket connections are made → no CloseWait accumulation
    """
    try:
        import yfinance as yf

        _real_ticker = yf.Ticker
        _real_download = yf.download

        class _StubInfo(dict):
            """Empty info dict that doesn't raise errors."""
            def get(self, key, default=None):
                return default

        class _StubTicker:
            """Stub Ticker that returns empty data without making network calls."""
            def __init__(self, *args, **kwargs):
                self.ticker = args[0] if args else "?"
                self._info = _StubInfo()

            @property
            def info(self):
                return self._info

            def history(self, *args, **kwargs):
                import pandas as pd
                return pd.DataFrame()

            def __getattr__(self, name):
                # Any other attribute access → empty/None
                if name in ('financials', 'balance_sheet', 'cashflow',
                            'recommendations', 'calendar', 'news'):
                    import pandas as pd
                    return pd.DataFrame()
                return None

        class _MonitoredTicker:
            """Wraps real Ticker to detect empty responses (401 swallowed by yfinance)."""
            __slots__ = ('_real', '_ticker_str')
            def __init__(self, real_ticker, ticker_str):
                object.__setattr__(self, '_real', real_ticker)
                object.__setattr__(self, '_ticker_str', ticker_str)

            @property
            def info(self):
                # FIX: yfinance swallows 401 and returns empty dict — detect it!
                result = self._real.info
                if not result or len(result) < 3:
                    _BREAKER.record_failure()
                    logger.debug(f"[YF_PATCH] {self._ticker_str}.info empty (likely 401)")
                else:
                    _BREAKER.record_success()
                return result

            def history(self, *args, **kwargs):
                result = self._real.history(*args, **kwargs)
                if result is None or (hasattr(result, 'empty') and result.empty):
                    _BREAKER.record_failure()
                    logger.debug(f"[YF_PATCH] {self._ticker_str}.history empty (likely 401)")
                else:
                    _BREAKER.record_success()
                return result

            def __getattr__(self, name):
                return getattr(self._real, name)

        def _safe_ticker(*args, **kwargs):
            if _BREAKER.is_open():
                # Don't make ANY network call when circuit is open
                return _StubTicker(*args, **kwargs)
            try:
                real = _real_ticker(*args, **kwargs)
                # Wrap in monitored wrapper to detect silent 401s
                ticker_str = args[0] if args else "?"
                return _MonitoredTicker(real, ticker_str)
            except Exception as e:
                _BREAKER.record_failure()
                logger.debug(f"[YF_PATCH] Ticker failed: {type(e).__name__}")
                return _StubTicker(*args, **kwargs)

        def _safe_download(*args, **kwargs):
            if _BREAKER.is_open():
                import pandas as pd
                return pd.DataFrame()
            try:
                result = _real_download(*args, **kwargs)
                _BREAKER.record_success()
                return result
            except Exception as e:
                _BREAKER.record_failure()
                logger.debug(f"[YF_PATCH] download failed: {type(e).__name__}")
                import pandas as pd
                return pd.DataFrame()

        # Install the patches
        yf.Ticker = _safe_ticker
        yf.download = _safe_download

        logger.info("[YF_PATCH] Global monkey-patch installed — all yfinance calls go through circuit breaker")
        return True
    except Exception as e:
        logger.error(f"[YF_PATCH] Failed to install: {e}")
        return False


def manual_socket_cleanup():
    """
    Manually clean up stale TCP connections from yfinance.
    Run this periodically (e.g., every 5 minutes) to prevent file descriptor leaks.
    """
    try:
        # Force-close any sockets in CLOSE_WAIT state
        import gc
        gc.collect()

        # Try yfinance internal session cleanup
        try:
            import yfinance.data as _yfdata
            if hasattr(_yfdata, '_requests'):
                sess = _yfdata._requests
                if hasattr(sess, 'close'):
                    sess.close()
                    logger.debug("[YF_CLEANUP] Closed yfinance session")
        except Exception:
            pass

        try:
            from yfinance import utils as _yfutils
            if hasattr(_yfutils, '_TIMEOUTS'):
                _yfutils._TIMEOUTS.clear()
        except Exception:
            pass

        # Run garbage collection multiple times for cyclic references
        for _ in range(3):
            gc.collect()

        logger.debug("[YF_CLEANUP] Forced socket cleanup")
    except Exception as e:
        logger.debug(f"[YF_CLEANUP] Error: {e}")
