"""
Global yfinance hardening patch
================================

Yahoo Finance frequently rejects requests with HTTP 401 "Invalid Crumb". The
root cause (verified against yfinance/data.py): once the per-process crumb goes
stale, yfinance's own retry RE-USES the same stale crumb instead of clearing it,
so every call keeps failing until the whole process restarts.

This module patches the single chokepoint that EVERY yfinance HTTP request flows
through — ``YfData._make_request`` — so the fix covers every call site at once
(``.info``, ``.history``, ``.calendar``, ``.upgrades_downgrades``, ``download``,
…), including the many direct ``yf.Ticker(...)`` calls scattered across the bot
that bypass yfinance_cache / yfinance_safe.

It does two things:
  1. On a 401/403 response, CLEAR the cached crumb+cookie and retry once with
     genuinely fresh credentials (a fresh fetch re-authenticates — verified).
  2. Apply a light global throttle between requests, so bursts don't trip
     Yahoo's rate limit ("User is unable to access this feature").

Safe by design: monkeypatch is idempotent, wrapped in try/except, and on any
unexpected error falls back to yfinance's original behaviour. Importing this
module installs the patch automatically.
"""

import logging
import threading
import time

logger = logging.getLogger("yf_auth_patch")

# Minimum spacing between yfinance HTTP requests (process-wide). Smooths the
# bursts that trip Yahoo's rate limit. Modest so scans stay fast.
_MIN_INTERVAL = 0.15  # seconds (tuned up from 0.10 to cut residual 401 bursts)

_throttle_lock = threading.Lock()
_last_call = [0.0]
_patched = False


def _reset_auth(inst) -> None:
    """Clear ONLY the cached crumb+cookie (never the lock/strategy)."""
    for attr in ("_crumb", "_cookie"):
        try:
            if hasattr(inst, attr):
                setattr(inst, attr, None)
        except Exception:
            pass


def _throttle() -> None:
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def install() -> bool:
    """Install the patch. Returns True if applied, False if already patched/failed."""
    global _patched
    if _patched:
        return False
    try:
        from yfinance import data as yfdata
    except Exception as e:  # pragma: no cover
        logger.warning("yf patch: could not import yfinance.data (%s) — skipped", e)
        return False

    YfData = yfdata.YfData
    _orig_make_request = YfData._make_request

    def _patched_make_request(self, url, request_method, body=None, params=None,
                              timeout=30, data=None):
        _throttle()
        try:
            resp = _orig_make_request(self, url, request_method, body=body,
                                      params=params, timeout=timeout, data=data)
        except Exception:
            # Hard failure (incl. rate-limit). A stale crumb is a common cause —
            # reset, brief backoff, and try once more. If it still fails, re-raise
            # so the caller's own fallback logic kicks in.
            _reset_auth(self)
            time.sleep(0.6)
            return _orig_make_request(self, url, request_method, body=body,
                                      params=params, timeout=timeout, data=data)

        # Still unauthorized after yfinance's own retry → the cached crumb/cookie
        # is stale (or a heavy burst is being rate-limited). Clear it and retry up
        # to TWICE with fresh credentials and a growing backoff, so requests caught
        # in a burst get a couple of chances to recover before giving up.
        attempt = 0
        while getattr(resp, "status_code", 200) in (401, 403) and attempt < 2:
            attempt += 1
            _reset_auth(self)
            time.sleep(0.3 * attempt)  # 0.3s, then 0.6s — let Yahoo settle
            logger.info("yf patch: %s -> reset crumb+cookie, retry #%d",
                        resp.status_code, attempt)
            try:
                resp = _orig_make_request(self, url, request_method, body=body,
                                          params=params, timeout=timeout, data=data)
            except Exception:
                break  # callers already degrade gracefully on failure
        return resp

    try:
        YfData._make_request = _patched_make_request
        _patched = True
        logger.info("yf patch installed: crumb self-heal on 401/403 + %dms throttle",
                    int(_MIN_INTERVAL * 1000))
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("yf patch: failed to install (%s) — yfinance left unpatched", e)
        return False


# Auto-install on import so a single `import yf_auth_patch` early in startup
# hardens every yfinance call in the process.
install()
