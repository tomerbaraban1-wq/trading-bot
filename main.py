import os
# Fix SSL certificate path - support env var for deployment (Render, Docker, etc)
_cert = os.getenv('CERT_PATH', 'C:/certs/cacert.pem')
if os.path.exists(_cert):
    os.environ['REQUESTS_CA_BUNDLE'] = _cert
    os.environ['SSL_CERT_FILE'] = _cert
    os.environ['CURL_CA_BUNDLE'] = _cert
    # Patch certifi directly so yfinance/curl always use correct path
    try:
        import certifi
        certifi.where = lambda: _cert
        import certifi.core
        certifi.core.where = lambda: _cert
    except Exception:
        pass

import asyncio
import logging
import socket
import sys
import time
import threading
import signal
import faulthandler
import traceback

# Quiet benign numpy warnings from indicator math on empty/short data series.
# These produce NaN, which the bot already treats as a safe skip — no need to
# spam the logs (keeps real errors easy to spot).
import warnings as _np_warn
_np_warn.filterwarnings("ignore", message="Mean of empty slice")
_np_warn.filterwarnings("ignore", message="invalid value encountered")
_np_warn.filterwarnings("ignore", message="Degrees of freedom <= 0")
_np_warn.filterwarnings("ignore", message="All-NaN slice encountered")
from pathlib import Path

# ── yfinance hardening ──────────────────────────────────────────────────────
# Install the global yfinance crumb self-heal + request throttle BEFORE any
# module makes a Yahoo request. One chokepoint (YfData._make_request) covers
# every call site, including direct yf.Ticker(...) calls that bypass the
# yfinance_cache / yfinance_safe wrappers. See yf_auth_patch.py.
import yf_auth_patch  # noqa: F401  (auto-installs on import)

# ═══════════════════════════════════════════════════════════════════════════
# CRASH PREVENTION LAYER 0: Process-level safety nets
# These run BEFORE anything else — catch crashes at the lowest level.
# ═══════════════════════════════════════════════════════════════════════════

# 1. faulthandler — dumps Python stack on segfault/abort/SIGSEGV/SIGILL
#    Without this, a C-extension crash gives ZERO information.
try:
    _crash_log = open("crash_traceback.log", "a", encoding="utf-8")
    faulthandler.enable(file=_crash_log, all_threads=True)
except Exception:
    pass

# 2. sys.excepthook — catches ANY uncaught exception in the main thread
#    Without this, uncaught exceptions print to stderr and exit silently.
_original_excepthook = sys.excepthook
def _global_excepthook(exc_type, exc_value, exc_tb):
    """Last-resort handler: log uncaught exceptions before process dies."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow Ctrl+C to work normally
        _original_excepthook(exc_type, exc_value, exc_tb)
        return
    try:
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        # Write to a dedicated crash file (separate from main log)
        with open("uncaught_exceptions.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n{tb_str}\n")
    except Exception:
        pass
    _original_excepthook(exc_type, exc_value, exc_tb)
sys.excepthook = _global_excepthook

# 3. threading.excepthook — catches uncaught exceptions in worker threads
#    Python 3.8+. Without this, thread exceptions disappear silently.
def _thread_excepthook(args):
    """Log exceptions from worker threads."""
    if args.exc_type is SystemExit:
        return
    try:
        tb_str = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        with open("uncaught_exceptions.log", "a", encoding="utf-8") as f:
            f.write(f"\n[THREAD: {args.thread.name}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{tb_str}\n")
    except Exception:
        pass
if hasattr(threading, "excepthook"):
    threading.excepthook = _thread_excepthook

# Global network safety net: any socket call that hangs >30s raises socket.timeout
# instead of hanging forever. Protects yfinance, requests, urllib, httpx — anything
# that uses sockets. Without this, a slow yahoo response can freeze the event loop.
socket.setdefaulttimeout(30)

# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: GLOBAL yfinance MONKEY-PATCH (ROOT CAUSE OF CRASHES)
# Investigation found 82+ TCP CloseWait sockets to Yahoo Finance servers.
# This was leaking ~7 sockets/min → file descriptor exhaustion → exit=4294967295
# 20+ files import yfinance directly. Only way to fix is to patch yf at module level.
# After 10 failures in 60s, circuit opens for 10 min — returns stub instead of real call.
# ═══════════════════════════════════════════════════════════════════════════
try:
    from yfinance_circuit_breaker import install_global_monkey_patch
    # NOTE: install_session_force_close() was tested and REVERTED — it made
    # CloseWait WORSE (9.4/min vs 6.3/min) because yfinance also uses curl_cffi
    # which bypasses requests.Session entirely. Without keep-alive, every call
    # creates a new socket that ends up in CloseWait state.
    # The real fix is scheduled_restart.ps1 (restart every 8 hours).
    install_global_monkey_patch()
except Exception as _yfp_err:
    print(f"[STARTUP] yfinance monkey-patch failed: {_yfp_err}")

# ═══════════════════════════════════════════════════════════════════════════
# PRE-STARTUP PORT RECLAIM — prevents "port busy" crashes on restart
# Even if watchdog releases the port, OS may still hold a TIME_WAIT socket.
# This forces SO_REUSEADDR on the listening socket.
# ═══════════════════════════════════════════════════════════════════════════
def _ensure_port_free(port: int = 8000) -> None:
    """Kill any lingering process on the port before uvicorn binds."""
    try:
        # Try to bind to detect occupancy
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return  # Port is free
        except OSError:
            pass
        s.close()

        # Port is busy — try to find and kill the process holding it
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
                if conn.pid:
                    try:
                        proc = psutil.Process(conn.pid)
                        if "python" in proc.name().lower() or "uvicorn" in (proc.name() or "").lower():
                            proc.kill()
                            time.sleep(1)
                            print(f"[STARTUP] Killed stale process {conn.pid} on port {port}")
                    except Exception:
                        pass
    except Exception:
        pass  # best-effort — uvicorn will report its own errors


_ensure_port_free(int(os.getenv("PORT", "8000")))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from database import init_db, close_connections, flush_database, check_database_integrity

class _SecretMaskingFilter(logging.Filter):
    """Mask API keys and tokens in log output."""
    _PATTERNS = [
        (r'(ghp_)[A-Za-z0-9]{36}',        r'\1***'),
        (r'(gsk_)[A-Za-z0-9]{20,}',        r'\1***'),   # Groq key (relaxed 50→20: catch shorter keys too)
        (r'(\d{9,10}:AA)[A-Za-z0-9_-]{30,}', r'\1***'),   # Telegram token
        (r'(secret=)[^&\s"]+',              r'\1***'),
        (r'(ALPACA_SECRET_KEY=)\S+',        r'\1***'),
        # ── Defense-in-depth: mask DB/API secrets that could otherwise leak in a log
        #    line (e.g. a connection string printed in an error). Targeted so they
        #    never touch legitimate content.
        (r'(postgres(?:ql)?://[^:/\s]+:)[^@\s]+(@)', r'\1***\2'),  # Postgres/Neon URL password
        (r'(?i)(NEON_PASSWORD=)\S+',        r'\1***'),
        (r'(?i)(GROQ_API_KEY=)\S+',         r'\1***'),
        (r'(?i)(WEBHOOK_SECRET=)\S+',       r'\1***'),
        (r'(?i)(TELEGRAM_BOT_TOKEN=)\S+',   r'\1***'),
        (r'(?i)(password=)[^&\s"\']+',      r'\1***'),  # generic key=value passwords
        (r'(Bearer\s+)[A-Za-z0-9._\-]{20,}', r'\1***'),
        (r'(sk-)[A-Za-z0-9]{20,}',          r'\1***'),   # OpenAI-style keys
    ]
    def filter(self, record):
        import re
        msg = str(record.getMessage())
        for pat, repl in self._PATTERNS:
            msg = re.sub(pat, repl, msg)
        record.msg  = msg
        record.args = ()
        return True

from logging.handlers import RotatingFileHandler

# Log rotation: max 10MB per file, keep last 5 files (max 50MB total)
# Prevents log file from growing unbounded (was reaching 12.7MB+ in 24h)
_log_file_handler = RotatingFileHandler(
    "trading_bot.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB per file
    backupCount=5,               # keep last 5 files = max 50 MB
    encoding="utf-8",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        _log_file_handler,
    ],
)
logging.getLogger().addFilter(_SecretMaskingFilter())

# Suppress yfinance noise — 2347+ "401 Invalid Crumb" errors flooding logs
# yfinance API frequently fails with auth errors; we fall back to other sources gracefully
# Setting to WARNING means only critical issues are logged (not every 401)
logging.getLogger("yfinance").setLevel(logging.WARNING)
# Also suppress httpx info-level requests (every API call was being logged)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Suppress peewee/sqlalchemy if used
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"[LOG_ROTATION] Active: max 10MB per file, 5 backups (50MB total cap)")
logger.info("[LOG_FILTER] Suppressed verbose loggers: yfinance, httpx, urllib3")

START_TIME = time.time()


def _asyncio_exception_handler(loop, context):
    """
    Global asyncio exception handler — catches ALL unhandled coroutine exceptions.
    Prevents silent failures and logs with full context.
    Without this, many async errors are swallowed by asyncio silently.

    NOTE: must stay a plain (loop, context) -> None callable. It is registered via
    loop.set_exception_handler() below. It was previously decorated with
    @asynccontextmanager, which turned it into a context-manager factory — so asyncio
    called it, got a CM object back, and the body never ran (the handler was silently
    dead, and benign network-close errors it was meant to filter leaked to the default
    handler as noise). Do NOT re-add that decorator.
    """
    exc = context.get("exception")
    msg = context.get("message", "Unknown asyncio error")

    if exc is None:
        logger.error(f"[ASYNCIO] Unhandled error: {msg}")
        return

    # Filter known/harmless exceptions
    if isinstance(exc, (asyncio.CancelledError, SystemExit, KeyboardInterrupt)):
        return
    if "Event loop is closed" in str(exc):
        return
    # Suppress benign network errors that happen on every connection close
    # ConnectionResetError [WinError 10054] = remote closed connection (NORMAL)
    # ConnectionAbortedError [WinError 10053] = local aborted (NORMAL)
    # BrokenPipeError = client disconnected before response
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        logger.debug(f"[ASYNCIO] Network close (benign): {type(exc).__name__}")
        return
    # Suppress proactor socket shutdown errors (Windows-specific noise)
    _exc_str = str(exc)
    if "WinError 10054" in _exc_str or "WinError 10053" in _exc_str:
        logger.debug(f"[ASYNCIO] Network reset (benign): {_exc_str[:80]}")
        return

    import traceback as _tb
    tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(f"[ASYNCIO] Unhandled exception in task:\n{tb_str}")

    # Send Telegram alert for serious errors (not just debug noise)
    _serious = (
        isinstance(exc, (MemoryError, RuntimeError, ImportError))
        or "cannot schedule" in str(exc).lower()
        or "event loop" in str(exc).lower()
    )
    if _serious:
        try:
            import requests as _req
            _tok = os.getenv("TELEGRAM_BOT_TOKEN", "")
            _chat = os.getenv("TELEGRAM_CHAT_ID", "")
            if _tok and _chat:
                _req.post(
                    f"https://api.telegram.org/bot{_tok}/sendMessage",
                    json={
                        "chat_id": _chat,
                        "text": (
                            f"⚠️ <b>Async Error</b>\n"
                            f"<code>{type(exc).__name__}: {str(exc)[:300]}</code>"
                        ),
                        "parse_mode": "HTML",
                    },
                    timeout=5,
                )
        except Exception:
            pass


async def lifespan(app: FastAPI):
    # Install global asyncio exception handler immediately
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_asyncio_exception_handler)

    # Enlarge the default thread pool so heavy parallel scans (many asyncio.to_thread
    # calls for sentiment/scoring/indicators at once) don't starve housekeeping tasks
    # like database.get_open_trades — which was timing out ~13x/session and causing
    # skipped heartbeat cycles. These tasks are I/O-bound, so extra workers are cheap.
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        import os as _os_tp
        # Right-sized for a 4-core box: enough I/O headroom without CPU thrash.
        # (Scan concurrency is separately capped by a semaphore, so DB tasks
        #  no longer get starved even with a modest pool.)
        _workers = int(_os_tp.getenv("THREAD_POOL_WORKERS", "16"))
        loop.set_default_executor(_TPE(max_workers=_workers, thread_name_prefix="bot-io"))
        logger.info(f"Thread pool set to {_workers} workers (balanced: no DB starvation, no CPU thrash)")
    except Exception as _e:
        logger.warning(f"Could not configure thread pool: {_e}")

    # Prevent Windows from sleeping while bot is running
    # Sleep = port conflicts + crash loops when system wakes up
    try:
        import ctypes
        ES_CONTINUOUS      = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        logger.info("Sleep prevention active — Windows will not suspend this process")
    except Exception:
        pass  # not on Windows, no problem

    # Startup
    settings.validate()

    # ── RUN COMPREHENSIVE STARTUP CHECKLIST ──────────────────────────────────
    try:
        from startup_checklist import run_startup_checklist
        is_safe, checks = await run_startup_checklist()

        if not is_safe:
            logger.critical("⛔ STARTUP CHECKLIST FAILED - BOT CANNOT START")
            logger.critical("Please fix the critical issues listed above before restarting.")
            logger.critical("")
            logger.critical("Critical issues must be resolved:")
            for check in checks:
                if check.status == "fail" and check.severity == "critical":
                    logger.critical(f"  {check.message}")
            logger.critical("")
            logger.critical("For detailed diagnostics, visit: /health")
            raise RuntimeError("Startup checklist failed - critical configuration issues detected")
        else:
            logger.info("✅ Startup checklist PASSED - proceeding with initialization")
    except ImportError:
        logger.warning("⚠️  startup_checklist module not found - skipping comprehensive checks")
    except RuntimeError as e:
        # Re-raise startup failures - don't let bot start with critical issues
        raise
    except Exception as e:
        logger.warning(f"⚠️  Startup checklist error (non-blocking): {e}")

    init_db()

    # Check database integrity
    db_ok = check_database_integrity()
    if not db_ok:
        logger.warning("Database integrity check failed but continuing...")

    # Log durability mode
    durability_mode = "HARDENED" if settings.HARDENED_DURABILITY else "NORMAL"
    logger.info("=== Trading Bot Started ===")
    _broker_info = settings.ACTIVE_BROKER if settings.ACTIVE_BROKER else settings.ALPACA_BASE_URL
    logger.info(f"Budget: ${settings.MAX_BUDGET:,.2f} | Broker: {_broker_info} | DB Mode: {durability_mode}")
    logger.info(f"Config: MIN_BUY_SCORE={settings.MIN_BUY_SCORE} | MAX_POSITIONS={settings.MAX_OPEN_POSITIONS} | MAX_HOLD={settings.MAX_HOLD_HOURS}h")

    # Send startup configuration report to Telegram
    try:
        from startup_validator import send_startup_report as _startup_report
        import asyncio as _asyncio_sr
        _asyncio_sr.create_task(_startup_report())
    except Exception:
        pass

    # ── Pre-warm Telegram context cache (parallel with other startup) ─────────
    # First user message after restart hits a 5-6s cold cache. Pre-build now
    # in background so by the time the user sends a command, cache is hot.
    try:
        import asyncio as _asyncio_pw
        async def _prewarm():
            try:
                import telegram_chat as _tc_pw
                await _asyncio_pw.to_thread(_tc_pw._build_context)
                logger.info("[STARTUP] Telegram context pre-warmed")
            except Exception as _pw_err:
                logger.debug(f"[STARTUP] Context prewarm failed: {_pw_err}")
        _asyncio_pw.create_task(_prewarm())
    except Exception:
        pass

    # ── Start polling loop when running locally (no RENDER_EXTERNAL_URL) ──────
    # On Render: webhook handles incoming messages (registered below).
    # Locally:   no public URL → use getUpdates polling instead.
    _polling_task = None
    try:
        from telegram_polling import is_local_mode, polling_loop as _polling_loop

        async def _polling_guard():
            # Keep Telegram listening alive forever: the loop once died
            # silently and commands went dark until a manual restart. If it
            # ever crashes or exits, relaunch it after a short pause.
            while True:
                try:
                    await _polling_loop()
                    logger.warning("[POLLING] Loop exited unexpectedly — restarting in 30s")
                except asyncio.CancelledError:
                    raise  # clean shutdown
                except Exception as _e:
                    logger.error(f"[POLLING] Loop crashed: {_e} — restarting in 30s")
                await asyncio.sleep(30)

        if is_local_mode():
            logger.info("[POLLING] Local mode — starting Telegram polling loop (guarded)")
            _polling_task = asyncio.create_task(_polling_guard())
        else:
            logger.info("[POLLING] Cloud mode — using webhook")
    except Exception as _poll_err:
        logger.warning(f"Polling setup failed (non-critical): {_poll_err}")

    # ── Auto-register Telegram webhook + command menu ─────────────────────────
    try:
        _render_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        _tg_token   = settings.TELEGRAM_BOT_TOKEN
        if _render_url and _tg_token:
            import aiohttp as _aiohttp
            _webhook_url = f"{_render_url}/telegram/webhook"
            _secret      = settings.WEBHOOK_SECRET
            async with _aiohttp.ClientSession() as _sess:
                # Register webhook WITH secret_token — Telegram will send it in
                # X-Telegram-Bot-Api-Secret-Token header. Otherwise any client
                # can POST forged updates to our webhook.
                async with _sess.post(
                    f"https://api.telegram.org/bot{_tg_token}/setWebhook",
                    json={
                        "url": _webhook_url,
                        "secret_token": _secret,
                        "drop_pending_updates": False,
                    },
                    timeout=_aiohttp.ClientTimeout(total=10),
                ) as _resp:
                    _data = await _resp.json()
                    if _data.get("ok"):
                        logger.info(f"Telegram webhook registered: {_webhook_url}")

                # Set command menu (shows as clickable buttons in Telegram)
                _commands = [
                    {"command": "status",    "description": "📊 מצב התיק המלא"},
                    {"command": "manioth",   "description": "📂 איזה מניות יש לי"},
                    {"command": "revach",    "description": "💰 מה הרווח שלי"},
                    {"command": "shovi",     "description": "💼 מה שווי התיק"},
                    {"command": "mazon",     "description": "💵 כמה מזומן יש לי"},
                    {"command": "biztsuim",  "description": "🏆 ביצועים ואחוז הצלחה"},
                    {"command": "count",     "description": "🔢 כמה עסקאות (סך כל הזמן)"},
                    {"command": "market",    "description": "🌍 מצב השוק (SPY/QQQ/DIA)"},
                    {"command": "trending",   "description": "🔥 מניות בתנופה חזקה"},
                    {"command": "gainers",   "description": "🚀 מניות מובילות היום"},
                    {"command": "exposure",  "description": "🏢 חשיפת תיק לסקטורים"},
                    {"command": "volatility","description": "📐 תנודתיות ו-Beta מניה"},
                    {"command": "morning",   "description": "☀️ תדרוך בוקר ידני"},
                    {"command": "signals",   "description": "📡 הזדמנויות קנייה עכשיו"},
                    {"command": "quick",     "description": "⚡ סקירה מהירה של מניה"},
                    {"command": "position",  "description": "📂 פרטי פוזיציה מלאים"},
                    {"command": "monthly",   "description": "📅 סיכום 30 ימים"},
                    {"command": "watchadd",  "description": "➕ הוסף מניה לרשימה"},
                    {"command": "watchremove","description": "➖ הסר מניה מהרשימה"},
                    {"command": "macro",     "description": "📅 אירועים כלכליים קרובים"},
                    {"command": "sectors",   "description": "📈 דירוג סקטורים"},
                    {"command": "pause",     "description": "⏸️ עצור קניות חדשות"},
                    {"command": "resume",    "description": "▶️ חדש קניות"},
                    {"command": "next",       "description": "🕐 מתי השוק נפתח"},
                    {"command": "portfolio",  "description": "📊 הקצאת תיק"},
                    {"command": "summary",    "description": "📅 סיכום 7 ימים"},
                    {"command": "best",       "description": "🏆 העסקה הטובה ביותר"},
                    {"command": "worst",      "description": "📉 העסקה הגרועה ביותר"},
                    {"command": "uptime",     "description": "🤖 זמן פעילות הבוט"},
                    {"command": "taxes",      "description": "🧾 סיכום מס"},
                    {"command": "risk",       "description": "⚠️ ניתוח סיכון"},
                    {"command": "correlation","description": "📊 קורלציה בין פוזיציות"},
                    {"command": "health",    "description": "🩺 בריאות כל הפוזיציות"},
                    {"command": "pnl",       "description": "💰 רווח/הפסד מהיר"},
                    {"command": "volume",    "description": "📊 נפח מסחר מניה"},
                    {"command": "watchlist",  "description": "👁️ רשימת מניות לסריקה"},
                    {"command": "top",        "description": "🏆 מניות עם ציון גבוה"},
                    {"command": "winners",   "description": "🟢 פוזיציות ברווח"},
                    {"command": "losers",    "description": "🔴 פוזיציות בהפסד"},
                    {"command": "today",     "description": "📅 מה קרה היום"},
                    {"command": "vix",       "description": "🌡️ מדד הפחד VIX"},
                    {"command": "budget",    "description": "⚙️ הגדרות הבוט"},
                    {"command": "history",   "description": "📋 עסקאות אחרונות"},
                    {"command": "fear",      "description": "😨 Fear and Greed Index"},
                    {"command": "newscheck", "description": "📰 בדוק חדשות לכל הפוזיציות"},
                    {"command": "price",     "description": "💲 מחיר מניה"},
                    {"command": "alerts",    "description": "🔔 התראות מחיר פעילות"},
                    {"command": "scan",       "description": "🔍 הפעל סריקה מיידית"},
                    {"command": "chart",      "description": "📊 גרף מחיר 30 ימים"},
                    {"command": "fundamental","description": "📈 נתוני יסוד (P/E, הכנסות)"},
                    {"command": "dividend",  "description": "💰 דיבידנד ותשואה"},
                    {"command": "review",    "description": "🤖 AI סוקר את הפוזיציות"},
                    {"command": "journal",   "description": "📓 יומן עסקאות אישי"},
                    {"command": "whatsnew",  "description": "📋 5 הפעולות האחרונות"},
                    {"command": "levels",    "description": "📐 רמות תמיכה/תנגדות"},
                    {"command": "remind",    "description": "⏰ הגדר תזכורת"},
                    {"command": "quiet",     "description": "🔕 מצב שקט (פחות התראות)"},
                    {"command": "loud",      "description": "🔔 כל ההתראות"},
                    {"command": "ask",       "description": "🤖 שאל שאלה חופשית ל-AI"},
                    {"command": "advice",    "description": "🤖 ייעוץ AI על התיק"},
                    {"command": "explain",   "description": "📚 הסבר מונח פיננסי"},
                    {"command": "streak",    "description": "🔥 רצף ניצחונות"},
                    {"command": "diagnose",  "description": "🔍 למה הבוט לא קונה"},
                    {"command": "backtest",  "description": "🧠 למידה היסטורית"},
                    {"command": "help",      "description": "❓ כל הפקודות"},
                ]
                async with _sess.post(
                    f"https://api.telegram.org/bot{_tg_token}/setMyCommands",
                    json={"commands": _commands},
                    timeout=_aiohttp.ClientTimeout(total=10),
                ) as _resp2:
                    _data2 = await _resp2.json()
                    if _data2.get("ok"):
                        logger.info("Telegram command menu registered")
    except Exception as _e:
        logger.warning(f"Telegram setup failed (non-critical): {_e}")

    # ── Startup state restore + reconciliation ───────────────────────────────
    # Detects and fixes the case where broker has positions but SQLite is empty.
    # This happens when Render redeploys and wipes the ephemeral SQLite file,
    # but the Postgres-backed broker state (TVPaperBroker) still holds positions.
    # Without reconciliation those positions have no stop-loss protection.
    try:
        from database import get_open_trades, save_trade
        import broker as _broker
        open_trades = get_open_trades()

        acct = await asyncio.wait_for(
            asyncio.to_thread(_broker.get_account), timeout=20
        )
        cash = float(acct.get("cash", 0))
        equity = float(acct.get("equity", 0))
        logger.info(f"BROKER: cash=${cash:,.2f} | equity=${equity:,.2f}")

        if open_trades:
            tickers = [t["ticker"] for t in open_trades]
            logger.info(f"RESTORED {len(open_trades)} open position(s): {tickers}")

            # Cross-check: close SQLite records that no longer exist in the broker
            # (prevents stop-loss monitor from trying to sell non-existent positions)
            try:
                from database import close_trade as _close_trade
                broker_positions = await asyncio.wait_for(asyncio.to_thread(_broker.get_positions), timeout=20)
                # Guard: if broker returns empty list (API error / transient failure),
                # skip cross-check entirely to avoid closing ALL valid positions
                if not broker_positions:
                    logger.warning(
                        "RECONCILE: broker returned 0 positions — skipping cross-check "
                        "(could be API error; not closing valid SQLite records)"
                    )
                else:
                    broker_tickers = {p.get("ticker", "").upper() for p in broker_positions}
                    for t in open_trades:
                        if t["ticker"].upper() not in broker_tickers:
                            logger.warning(
                                f"RECONCILE: {t['ticker']} is open in SQLite but NOT in broker — "
                                f"closing as stale_restart"
                            )
                            _close_trade(t["id"], t["entry_price"], 0.0, 0.0, 0.0, 0.0, "stale_restart")
            except Exception as _ce:
                logger.warning(f"Cross-check reconciliation failed (non-critical): {_ce}")
        else:
            # SQLite is empty — check if broker has positions we need to recover
            broker_positions = await asyncio.to_thread(_broker.get_positions)
            if broker_positions:
                logger.warning(
                    f"RECONCILE: SQLite is empty but broker has {len(broker_positions)} position(s) — re-creating records"
                )
                from models import WebhookPayload, TradeAction
                for pos in broker_positions:
                    ticker    = pos.get("ticker", "").upper()
                    qty       = float(pos.get("qty", 0))
                    entry     = float(pos.get("avg_entry_price", 0))
                    if not ticker or qty <= 0 or entry <= 0:
                        continue
                    try:
                        trade = {
                            "ticker":         ticker,
                            "action":         "buy",
                            "qty":            qty,
                            "entry_price":    entry,
                            "trailing_stop_pct": None,
                            "rsi": None, "macd": None, "macd_signal": None,
                            "bb_position": None, "volume_ratio": None,
                            "sentiment_score": 5,
                            "sentiment_reasoning": "Recovered from broker on restart",
                        }
                        trade_id = save_trade(trade)
                        # Set ATR stop immediately
                        from atr_stop import compute_initial_stop
                        from database import update_trade_stop
                        stop_price, stop_meta = await asyncio.wait_for(asyncio.to_thread(compute_initial_stop, ticker, entry), timeout=20)
                        update_trade_stop(trade_id, stop_price, entry)
                        logger.info(
                            f"RECONCILE: restored {ticker} x{qty} @ ${entry:.2f} "
                            f"(trade_id={trade_id}, stop=${stop_price:.2f})"
                        )
                    except Exception as _re:
                        logger.warning(f"RECONCILE: failed to restore {ticker}: {_re}")
            else:
                logger.info("RESTORED: no open positions — clean slate")
    except Exception as _e:
        logger.warning(f"Startup reconciliation failed (non-critical): {_e}")
    # ─────────────────────────────────────────────────────────────────────────

    # Store the running event loop so worker threads can use it (e.g. Discord sentiment)
    try:
        import asyncio as _asyncio
        from discord_bot import set_event_loop as _set_loop
        _set_loop(_asyncio.get_running_loop())
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # CRASH PREVENTION SYSTEM — TaskMonitor Integration
    # All 50+ background tasks now have automatic crash detection + restart
    # ═══════════════════════════════════════════════════════════════════════════
    monitor = None
    try:
        from task_monitor import init_monitor, get_monitor
        monitor = await init_monitor()
        logger.info("[CRASH_PREVENTION] TaskMonitor initialized — auto-restart on any task crash")
    except Exception as _tm_err:
        logger.warning(f"[CRASH_PREVENTION] TaskMonitor failed to init (fallback to plain create_task): {_tm_err}")

    # Helper: use monitored task if available, else plain create_task
    def _spawn(coro, name: str):
        if monitor is not None:
            return asyncio.create_task(_monitored_wrapper(monitor, coro, name), name=name)
        return asyncio.create_task(coro, name=name)

    async def _monitored_wrapper(_mon, _coro, _name):
        """Wrap coroutine to register with TaskMonitor for crash detection."""
        try:
            return await _mon.create_task(_coro, _name)
        except Exception:
            # If monitor fails for any reason, run task directly
            return await _coro

    from heartbeat import (heartbeat_loop, heartbeat_cleanup_loop, sentiment_monitor, stop_loss_monitor,
                           auto_invest_loop, keep_alive_loop, daily_summary_loop, daily_full_report_loop,
                           weekly_report_loop, shadow_monitor_loop, portfolio_update_loop,
                           news_refresh_loop, news_monitor_loop, morning_briefing_loop,
                           position_alert_loop, backtest_learning_loop, eod_sweep_loop,
                           price_alert_loop, market_closed_training_loop,
                           telegram_context_warmup_loop, earnings_monitor_loop,
                           market_pulse_loop, webhook_keeper_loop,
                           golden_opportunity_loop, smart_reentry_loop,
                           weekend_research_loop, daily_ai_insights_loop, global_pulse_loop,
                           self_improvement_loop, rapid_move_alert_loop,
                           drawdown_protection_loop, idle_cash_alert_loop,
                           adaptive_threshold_loop, daily_goal_progress_loop,
                           continuous_learning_loop, adaptive_parameters_monitor_loop,
                           correlation_monitor_loop, market_intelligence_loop,
                           detailed_analytics_loop, ai_decision_loop,
                           attribution_loop, notification_digest_loop,
                           multi_timeframe_loop, health_monitoring_loop,
                           news_catalyst_loop, pairs_trading_loop,
                           benchmark_comparison_loop, trade_journal_loop,
                           anomaly_detection_loop, stale_position_guard_loop,
                           fast_track_progress_loop, volume_surge_loop)
    # ── Core tasks (always run) — wrapped with TaskMonitor ─────────────
    heartbeat_task         = _spawn(heartbeat_loop(), "heartbeat_loop")
    heartbeat_cleanup_task = _spawn(heartbeat_cleanup_loop(), "heartbeat_cleanup_loop")
    stop_loss_task         = _spawn(stop_loss_monitor(), "stop_loss_monitor")
    auto_invest_task       = _spawn(auto_invest_loop(), "auto_invest_loop")
    keep_alive_task        = _spawn(keep_alive_loop(), "keep_alive_loop")
    daily_summary_task     = _spawn(daily_summary_loop(), "daily_summary_loop")
    daily_report_task      = _spawn(daily_full_report_loop(), "daily_full_report_loop")
    weekly_report_task     = _spawn(weekly_report_loop(), "weekly_report_loop")
    backtest_task          = _spawn(backtest_learning_loop(), "backtest_learning_loop")
    training_task          = _spawn(market_closed_training_loop(), "market_closed_training_loop")
    tg_warmup_task         = _spawn(telegram_context_warmup_loop(), "telegram_context_warmup_loop")
    eod_sweep_task         = _spawn(eod_sweep_loop(), "eod_sweep_loop")
    price_alert_task       = _spawn(price_alert_loop(), "price_alert_loop")
    morning_briefing_task  = _spawn(morning_briefing_loop(), "morning_briefing_loop")
    news_refresh_task      = _spawn(news_refresh_loop(), "news_refresh_loop")
    news_monitor_task      = _spawn(news_monitor_loop(), "news_monitor_loop")
    earnings_monitor_task  = _spawn(earnings_monitor_loop(), "earnings_monitor_loop")
    market_pulse_task      = _spawn(market_pulse_loop(), "market_pulse_loop")
    goal_progress_task     = _spawn(daily_goal_progress_loop(), "daily_goal_progress_loop")
    learning_task          = _spawn(continuous_learning_loop(), "continuous_learning_loop")
    adaptive_params_task   = _spawn(adaptive_parameters_monitor_loop(), "adaptive_parameters_monitor_loop")
    correlation_task       = _spawn(correlation_monitor_loop(), "correlation_monitor_loop")
    market_intel_task      = _spawn(market_intelligence_loop(), "market_intelligence_loop")
    analytics_task         = _spawn(detailed_analytics_loop(), "detailed_analytics_loop")
    ai_decision_task       = _spawn(ai_decision_loop(), "ai_decision_loop")
    attribution_task       = _spawn(attribution_loop(), "attribution_loop")
    digest_task            = _spawn(notification_digest_loop(), "notification_digest_loop")
    mtf_task               = _spawn(multi_timeframe_loop(), "multi_timeframe_loop")
    health_task            = _spawn(health_monitoring_loop(), "health_monitoring_loop")
    news_catalyst_task     = _spawn(news_catalyst_loop(), "news_catalyst_loop")
    pairs_task             = _spawn(pairs_trading_loop(), "pairs_trading_loop")
    benchmark_task         = _spawn(benchmark_comparison_loop(), "benchmark_comparison_loop")
    journal_task           = _spawn(trade_journal_loop(), "trade_journal_loop")
    anomaly_task           = _spawn(anomaly_detection_loop(), "anomaly_detection_loop")
    stale_guard_task       = _spawn(stale_position_guard_loop(), "stale_position_guard_loop")
    fast_track_task        = _spawn(fast_track_progress_loop(), "fast_track_progress_loop")
    webhook_keeper_task    = _spawn(webhook_keeper_loop(), "webhook_keeper_loop")
    golden_opp_task        = _spawn(golden_opportunity_loop(), "golden_opportunity_loop")
    reentry_task           = _spawn(smart_reentry_loop(), "smart_reentry_loop")
    weekend_task           = _spawn(weekend_research_loop(), "weekend_research_loop")
    global_pulse_task      = _spawn(global_pulse_loop(), "global_pulse_loop")
    ai_insights_task       = _spawn(daily_ai_insights_loop(), "daily_ai_insights_loop")
    self_improve_task      = _spawn(self_improvement_loop(), "self_improvement_loop")
    rapid_move_task        = _spawn(rapid_move_alert_loop(), "rapid_move_alert_loop")
    drawdown_task          = _spawn(drawdown_protection_loop(), "drawdown_protection_loop")
    idle_cash_task         = _spawn(idle_cash_alert_loop(), "idle_cash_alert_loop")
    adaptive_task          = _spawn(adaptive_threshold_loop(), "adaptive_threshold_loop")
    volume_surge_task      = _spawn(volume_surge_loop(), "volume_surge_loop")

    # ── Resource monitor: alerts on high CPU/memory ───────────────────
    try:
        from resource_monitor import resource_monitor_loop
        resource_monitor_task = _spawn(resource_monitor_loop(), "resource_monitor_loop")
    except ImportError:
        resource_monitor_task = None

    # ── Optional tasks (disabled on free tier to save memory) ────────
    import os as _os
    _full_mode = _os.getenv("FULL_MODE", "false").lower() == "true"
    sentiment_task      = _spawn(sentiment_monitor(), "sentiment_monitor") if _full_mode else None
    shadow_monitor_task = _spawn(shadow_monitor_loop(), "shadow_monitor_loop") if _full_mode else None
    portfolio_update_task = _spawn(portfolio_update_loop(), "portfolio_update_loop") if _full_mode else None
    position_alert_task = _spawn(position_alert_loop(), "position_alert_loop") if _full_mode else None

    if not _full_mode:
        logger.info("Memory-saving mode: shadow, portfolio_update, position_alert, sentiment disabled. Set FULL_MODE=true to enable.")

    # ── Periodic GC + memory health check ───────────────────────────────
    # Adds an extra layer: every 30 min, force gc + check memory.
    # If memory > 500MB, log warning + trigger gc.
    # If memory > 1GB, force-restart via watchdog (kill self).
    async def _memory_guard_loop():
        import gc, psutil, os as _osmem
        proc = psutil.Process(_osmem.getpid())
        while True:
            try:
                await asyncio.sleep(30 * 60)
                mem_mb = proc.memory_info().rss / 1024 / 1024
                gc.collect()
                if mem_mb > 1000:
                    # Critical: kill self to let watchdog restart with clean memory
                    logger.critical(f"[MEMORY_GUARD] CRITICAL: {mem_mb:.0f}MB — triggering self-restart")
                    try:
                        import requests
                        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
                        chat = os.getenv("TELEGRAM_CHAT_ID", "")
                        if token and chat:
                            requests.post(
                                f"https://api.telegram.org/bot{token}/sendMessage",
                                json={
                                    "chat_id": chat,
                                    "text": f"🔴 <b>MEMORY CRITICAL</b>\nזיכרון: {mem_mb:.0f}MB\n♻️ אתחול עצמי",
                                    "parse_mode": "HTML",
                                },
                                timeout=3,
                            )
                    except Exception:
                        pass
                    await asyncio.sleep(2)
                    os._exit(1)  # watchdog will restart
                elif mem_mb > 500:
                    logger.warning(f"[MEMORY_GUARD] High memory: {mem_mb:.0f}MB — gc.collect() ran")
                else:
                    logger.debug(f"[MEMORY_GUARD] Memory OK: {mem_mb:.0f}MB")
            except asyncio.CancelledError:
                raise
            except Exception as _mg_err:
                logger.debug(f"[MEMORY_GUARD] error: {_mg_err}")

    memory_guard_task = _spawn(_memory_guard_loop(), "memory_guard_loop")

    # ── CRITICAL FIX: TCP Connection Leak Prevention ────────────────────
    # ROOT CAUSE FOUND: yfinance 401 errors leave sockets in CLOSE_WAIT state
    # → 78+ CloseWait connections detected → file descriptor exhaustion
    # → exit=4294967295 crashes (process killed by OS)
    # Solution: Force socket cleanup every 3 minutes + circuit breaker
    async def _socket_cleanup_loop():
        while True:
            try:
                await asyncio.sleep(180)  # every 3 minutes
                try:
                    from yfinance_circuit_breaker import manual_socket_cleanup, get_breaker
                    manual_socket_cleanup()
                    breaker = get_breaker()
                    if breaker.is_open():
                        logger.info(
                            f"[SOCKET_CLEANUP] yfinance circuit OPEN until +{int(breaker.circuit_open_until - time.time())}s "
                            f"(saved from {breaker.total_failures} failed calls)"
                        )
                except Exception as _sc_err:
                    logger.debug(f"[SOCKET_CLEANUP] {_sc_err}")
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    socket_cleanup_task = _spawn(_socket_cleanup_loop(), "socket_cleanup_loop")
    logger.info("[CRASH_PREVENTION] TCP socket cleanup loop active (every 3 min)")

    # ── STOP-LOSS HEALTH CHECK — most critical safety net ────────────────
    # If stop_loss_monitor is dead, open positions are UNPROTECTED.
    # Check every 2 minutes that the task is alive. If not, alert + restart.
    async def _stop_loss_watchdog():
        await asyncio.sleep(120)  # initial grace period
        consecutive_failures = 0
        while True:
            try:
                await asyncio.sleep(120)  # check every 2 min
                if monitor is None:
                    continue
                status = await monitor.get_status()
                tasks = status.get("tasks", {})
                sl_info = tasks.get("stop_loss_monitor", {})

                if not sl_info.get("alive", False):
                    consecutive_failures += 1
                    logger.critical(
                        f"[STOP_LOSS_WATCHDOG] stop_loss_monitor DEAD! "
                        f"(failure #{consecutive_failures})"
                    )
                    # Critical alert — positions are at risk
                    try:
                        import requests
                        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
                        chat = os.getenv("TELEGRAM_CHAT_ID", "")
                        if token and chat:
                            requests.post(
                                f"https://api.telegram.org/bot{token}/sendMessage",
                                json={
                                    "chat_id": chat,
                                    "text": (
                                        f"🚨 <b>STOP-LOSS DOWN</b>\n"
                                        f"מנגנון הגנת ההפסדים לא פעיל!\n"
                                        f"פוזיציות פתוחות לא מוגנות.\n"
                                        f"כישלון #{consecutive_failures}"
                                    ),
                                    "parse_mode": "HTML",
                                },
                                timeout=3,
                            )
                    except Exception:
                        pass

                    # After 3 failures — restart whole bot
                    if consecutive_failures >= 3:
                        logger.critical("[STOP_LOSS_WATCHDOG] 3 failures — triggering bot restart")
                        await asyncio.sleep(2)
                        os._exit(3)  # watchdog will restart
                else:
                    consecutive_failures = 0  # reset on success
            except asyncio.CancelledError:
                raise
            except Exception as _sw_err:
                logger.debug(f"[STOP_LOSS_WATCHDOG] error: {_sw_err}")

    stop_loss_watchdog_task = _spawn(_stop_loss_watchdog(), "stop_loss_watchdog")

    # ── DEADMAN SWITCH — detects deadlock/freeze ─────────────────────────
    # Every 60s, writes timestamp to a file. If the timestamp is older than
    # 5 minutes when checked from another thread, force-restart.
    # This catches deadlocks that even TaskMonitor can't detect.
    async def _deadman_switch_loop():
        import os as _osd
        deadman_file = Path("data") / ".deadman_alive"
        deadman_file.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                deadman_file.write_text(str(int(time.time())))
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    # External thread that monitors the deadman file
    def _deadman_watchdog():
        deadman_file = Path("data") / ".deadman_alive"
        STALE_THRESHOLD = 300  # 5 minutes
        while True:
            try:
                time.sleep(60)
                if not deadman_file.exists():
                    continue
                try:
                    last = int(deadman_file.read_text().strip())
                except Exception:
                    continue
                if time.time() - last > STALE_THRESHOLD:
                    logger.critical(
                        f"[DEADMAN] Event loop frozen for {int(time.time()-last)}s — force-restart"
                    )
                    # Force-exit so watchdog process restarts the bot
                    os._exit(2)
            except Exception:
                pass

    deadman_task = _spawn(_deadman_switch_loop(), "deadman_switch_loop")
    _deadman_thread = threading.Thread(target=_deadman_watchdog, daemon=True, name="DeadmanWatchdog")
    _deadman_thread.start()
    logger.info("[CRASH_PREVENTION] Deadman switch active — force-restart if event loop freezes 5+ min")

    yield

    # ═══════════════════════════════════════════════════════════════════════════
    # GRACEFUL SHUTDOWN — Order matters: monitor → tasks → DB
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Initiating graceful shutdown...")

    # Shut down TaskMonitor first (stops health checker, prevents restarts during cleanup)
    try:
        if monitor is not None:
            await asyncio.wait_for(monitor.shutdown(), timeout=5.0)
            logger.info("[CRASH_PREVENTION] TaskMonitor shut down")
    except Exception as _md_err:
        logger.warning(f"TaskMonitor shutdown failed: {_md_err}")
    all_tasks = [t for t in [
        heartbeat_task, heartbeat_cleanup_task, sentiment_task, stop_loss_task, auto_invest_task,
        keep_alive_task, daily_summary_task, weekly_report_task, shadow_monitor_task,
        portfolio_update_task, news_refresh_task, news_monitor_task, morning_briefing_task,
        position_alert_task, backtest_task, training_task, eod_sweep_task, price_alert_task,
        earnings_monitor_task, market_pulse_task, goal_progress_task, learning_task, adaptive_params_task,
        correlation_task, market_intel_task, analytics_task, ai_decision_task,
        attribution_task, digest_task, mtf_task, health_task, news_catalyst_task,
        pairs_task, benchmark_task, journal_task, anomaly_task, stale_guard_task,
        fast_track_task, webhook_keeper_task,
        golden_opp_task, reentry_task, weekend_task, global_pulse_task,
        ai_insights_task, self_improve_task, rapid_move_task,
        drawdown_task, idle_cash_task, adaptive_task, tg_warmup_task, _polling_task,
        memory_guard_task, volume_surge_task, resource_monitor_task,
    ] if t is not None]

    # Cancel all background tasks
    for task in all_tasks:
        if not task.done():
            task.cancel()

    # Wait for tasks to complete with 10-second timeout
    try:
        await asyncio.wait_for(asyncio.gather(*all_tasks, return_exceptions=True), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Background tasks did not complete within 10s timeout, forcing shutdown...")
    except Exception as e:
        logger.warning(f"Exception during task shutdown: {e}")

    # Ensure database is flushed and properly closed
    flush_database()
    close_connections()
    logger.info("=== Trading Bot Stopped ===")


app = FastAPI(title="TradeBot", version="1.0.0", lifespan=lifespan)

# ═══════════════════════════════════════════════════════════════════════════
# CRASH PREVENTION — Health Check Endpoints
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/monitor/health")
async def _monitor_health():
    """Real-time task health: which tasks are alive/dead, restart counts."""
    try:
        from task_monitor import get_monitor
        mon = get_monitor()
        if mon is None:
            return {"status": "uninitialized", "alive": 0, "dead": 0}
        return await mon.get_status()
    except Exception as e:
        logger.warning(f"monitor endpoint error: {e}")
        return {"status": "error", "error": "internal monitor error"}


@app.get("/monitor/alive")
async def _monitor_alive():
    """Lightweight liveness probe for external monitoring."""
    try:
        from task_monitor import get_monitor
        mon = get_monitor()
        if mon is None:
            return {"status": "starting", "uptime_seconds": int(time.time() - START_TIME)}
        status = await mon.get_status()
        alive = status.get("alive", 0)
        dead = status.get("dead", 0)
        return {
            "status": "healthy" if dead == 0 else ("degraded" if dead < 3 else "critical"),
            "alive_tasks": alive,
            "dead_tasks": dead,
            "uptime_seconds": int(time.time() - START_TIME),
        }
    except Exception as e:
        logger.warning(f"monitor endpoint error: {e}")
        return {"status": "error", "error": "internal monitor error"}


# Setup graceful shutdown handlers for SIGTERM and SIGINT
def handle_shutdown(signum, frame):
    """Handle SIGTERM and SIGINT signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    # FastAPI's lifespan context manager will handle cleanup

# Only register signal handlers if not in Windows (Windows handles these differently)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, handle_shutdown)
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, handle_shutdown)

# atexit handler — ensures DB is flushed even on abnormal exit (SystemExit, etc.)
import atexit as _atexit
def _emergency_shutdown():
    """Last-resort cleanup if normal shutdown didn't run."""
    try:
        from database import flush_database, close_connections
        flush_database()
        close_connections()
        logger.info("[ATEXIT] Database flushed on exit")
    except Exception:
        pass
_atexit.register(_emergency_shutdown)

# CORS — restrict to known safe origins (TradingView + bot's own dashboard).
# Custom origins can be added via ALLOWED_ORIGINS env var (comma-separated).
_default_origins = [
    "https://www.tradingview.com",
    "https://tradingview.com",
    "https://tradebot-yc8p.onrender.com",
]
import re as _re
_extra_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip() and o.strip() != "*"  # block wildcard injection
    and _re.match(r"^https://[a-zA-Z0-9.-]+(:[0-9]+)?$", o.strip())  # https only — no http
]
_allowed_origins = _default_origins + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Webhook-Secret", "X-Telegram-Bot-Api-Secret-Token", "X-Admin-Key", "X-Api-Key"],
    allow_credentials=False,
)

# Add security middleware (rate limiting, injection detection, security headers)
try:
    from security_middleware import SecurityMiddleware
    app.add_middleware(SecurityMiddleware)
    logger.info("Security middleware enabled: rate limiting, injection detection, security headers")
except Exception as _sec_err:
    logger.warning(f"Security middleware not available: {_sec_err}")

# Import and include routes
from webhook import router
app.include_router(router)

# Rental/SaaS admin endpoints (dormant until RENTAL_MODE_ENABLED=true)
try:
    from rental_admin import router as rental_router
    app.include_router(rental_router)
    logger.info("Rental admin endpoints registered (currently dormant - set RENTAL_MODE_ENABLED=true to activate)")
except Exception as _rental_err:
    logger.warning(f"Rental admin endpoints not available: {_rental_err}")

# Analytics API endpoints (rich data access for dashboards)
try:
    from analytics_api import router as analytics_router
    app.include_router(analytics_router)
    logger.info("Analytics API endpoints registered at /api/v1/*")
except Exception as _analytics_err:
    logger.warning(f"Analytics API not available: {_analytics_err}")

# Security management endpoints
try:
    from security_endpoints import router as security_router
    app.include_router(security_router)
    logger.info("Security endpoints registered at /security/*")
except Exception as _sec_err:
    logger.warning(f"Security endpoints not available: {_sec_err}")


@app.get("/ping")
async def ping():
    """Ultra-lightweight liveness endpoint for UptimeRobot / external keep-alive.
    No DB calls — returns instantly so Render never marks it as slow."""
    return {"ok": True, "uptime": round(time.time() - START_TIME)}


@app.get("/health")
async def health_endpoint():
    """Comprehensive health check endpoint."""
    try:
        from health_monitor import run_health_check
        report = await run_health_check()
        return {
            "status": report.overall_status,
            "timestamp": report.timestamp,
            "metrics": report.metrics,
            "issues": report.issues,
            "uptime_seconds": round(time.time() - START_TIME),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/health/dashboard", response_class=HTMLResponse)
async def health_dashboard():
    """Visual health monitoring dashboard."""
    try:
        from health_monitor import run_health_check, generate_health_dashboard_html
        report = await run_health_check()
        html = generate_health_dashboard_html(report)
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Health Dashboard Error</h1><p>{e}</p></body></html>",
            status_code=500
        )


@app.get("/dashboard/advanced", response_class=HTMLResponse)
async def advanced_dashboard():
    """Beautiful real-time trading dashboard with all metrics."""
    try:
        from advanced_dashboard import generate_advanced_dashboard_html
        html = await generate_advanced_dashboard_html()
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Dashboard Error</h1><p>{e}</p></body></html>",
            status_code=500
        )


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    import asyncio as _aio
    html_path = Path(__file__).parent / "dashboard.html"
    return await _aio.to_thread(html_path.read_text, encoding="utf-8")


@app.get("/inject.js")
async def inject_js():
    from fastapi.responses import Response
    _expected_root = (Path(__file__).parent.parent / "tradebot-extension").resolve()
    js_path = _expected_root / "content.js"

    # Path traversal guard — ensure resolved path stays inside expected directory
    try:
        js_resolved = js_path.resolve()
        if not str(js_resolved).startswith(str(_expected_root)):
            logger.warning(f"[SECURITY] inject.js path traversal attempt blocked: {js_path}")
            return Response(content="// Access denied", status_code=403, media_type="application/javascript")
    except Exception:
        return Response(content="// Access denied", status_code=403, media_type="application/javascript")

    if not js_path.exists():
        return Response(content="// Inject file not found", status_code=404, media_type="application/javascript")

    code = await asyncio.to_thread(js_path.read_text, encoding="utf-8")
    return Response(content=code, media_type="application/javascript")


@app.get("/tunnel")
async def tunnel_info():
    global TUNNEL_URL
    # Read from file (written by node tunnel.js in parent process) with lock protection
    with TUNNEL_URL_LOCK:
        url = TUNNEL_URL
        url_file = Path(__file__).parent / "tunnel_url.txt"
        if not url and url_file.exists():
            try:
                url = url_file.read_text().strip() or None
                if url:
                    TUNNEL_URL = url  # Cache it
            except Exception:
                pass
    return {"url": url, "webhook": f"{url}/webhook" if url else None}


def get_uptime() -> float:
    return time.time() - START_TIME


TUNNEL_URL = None
TUNNEL_URL_LOCK = threading.Lock()  # Protect race condition on read/write


def start_tunnel():
    """Start localtunnel via node script."""
    global TUNNEL_URL
    try:
        import subprocess, threading

        tunnel_script = Path(__file__).parent / "tunnel.js"
        url_file = Path(__file__).parent / "tunnel_url.txt"

        def run_lt():
            global TUNNEL_URL
            proc = subprocess.Popen(
                ["node", str(tunnel_script), str(settings.PORT)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line:
                    logger.info(f"tunnel: {line}")
                if "TUNNEL_URL=" in line:
                    with TUNNEL_URL_LOCK:
                        TUNNEL_URL = line.split("=", 1)[1]
                    logger.info(f"Webhook URL: {TUNNEL_URL}/webhook")

        # Also check url file as fallback
        def watch_file():
            global TUNNEL_URL
            import time as _t
            for _ in range(30):
                _t.sleep(1)
                if url_file.exists():
                    with TUNNEL_URL_LOCK:
                        TUNNEL_URL = url_file.read_text().strip()
                    logger.info(f"Tunnel URL from file: {TUNNEL_URL}")
                    break

        threading.Thread(target=run_lt, daemon=True).start()
        threading.Thread(target=watch_file, daemon=True).start()
    except Exception as e:
        logger.warning(f"Tunnel failed (bot still works locally): {e}")


if __name__ == "__main__":
    # Only start localtunnel when explicitly requested (local dev). Cloud deployments skip it.
    if os.environ.get("USE_TUNNEL") == "1":
        start_tunnel()
    # Respect PORT env var (Render/Heroku/other PaaS provide this); fall back to configured port.
    port = int(os.environ.get("PORT", settings.PORT if hasattr(settings, "PORT") else 8000))
    host = os.environ.get("HOST", getattr(settings, "HOST", "0.0.0.0"))

    # ── PORT WAIT: don't start if port is still busy from previous instance ──
    # Root cause of exit=1 crashes: watchdog restarts too fast, port still in
    # TIME_WAIT state. Uvicorn fails to bind → Python exit code 1 → watchdog
    # restarts again → same crash loop until port finally clears (~15s on Windows)
    import socket as _socket
    _port_wait_limit = 45  # wait up to 45s for port to free
    _port_start = time.time()
    while time.time() - _port_start < _port_wait_limit:
        try:
            _s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            _s.settimeout(1)
            _r = _s.connect_ex(('127.0.0.1', port))
            _s.close()
            if _r != 0:
                break   # port is free!
            logger.warning(f"[STARTUP] Port {port} still occupied — waiting ({time.time()-_port_start:.0f}s)...")
            time.sleep(2)
        except Exception:
            break  # can't check, just proceed

    # ── CRASH PROTECTION ─────────────────────────────────────────────────
    # Wrap uvicorn.run in a try/except so any unhandled exception is logged
    # in detail (including a full traceback to disk) BEFORE the process exits.
    # The external watchdog will restart us — but we want forensic info.
    try:
        uvicorn.run("main:app", host=host, port=port, reload=False)
    except KeyboardInterrupt:
        logger.info("Bot stopped by KeyboardInterrupt (Ctrl+C)")
    except SystemExit:
        # uvicorn raises SystemExit on graceful shutdown — re-raise as-is
        raise
    except Exception as fatal:
        import traceback as _tb
        # Write a crash report to disk for post-mortem analysis
        try:
            crash_log = Path(__file__).parent / "crash_reports.log"
            with open(crash_log, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 70}\n")
                f.write(f"CRASH @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Type: {type(fatal).__name__}\n")
                f.write(f"Message: {fatal}\n")
                f.write(f"Traceback:\n{_tb.format_exc()}\n")
                f.write(f"{'=' * 70}\n")
        except Exception:
            pass

        logger.critical(f"FATAL CRASH: {type(fatal).__name__}: {fatal}")
        logger.critical(f"Traceback:\n{_tb.format_exc()}")

        # Try to notify via Telegram before exiting (best-effort, non-blocking)
        try:
            import requests as _req
            _tok = os.getenv("TELEGRAM_BOT_TOKEN", "")
            _chat = os.getenv("TELEGRAM_CHAT_ID", "")
            if _tok and _chat:
                _msg = (
                    f"🚨 <b>הבוט קרס</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"<b>סוג:</b> {type(fatal).__name__}\n"
                    f"<b>הודעה:</b> <code>{str(fatal)[:200]}</code>\n"
                    f"\n"
                    f"⚙️ Watchdog מפעיל מחדש תוך 30 שניות..."
                )
                _req.post(
                    f"https://api.telegram.org/bot{_tok}/sendMessage",
                    json={"chat_id": _chat, "text": _msg, "parse_mode": "HTML"},
                    timeout=5,
                )
        except Exception:
            pass

        # Exit with code 1 (real crash) — watchdog will restart us
        sys.exit(1)
