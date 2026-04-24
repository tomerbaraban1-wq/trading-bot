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
import time
import threading
import signal
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from database import init_db, close_connections, flush_database, check_database_integrity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings.validate()
    init_db()

    # Check database integrity
    db_ok = check_database_integrity()
    if not db_ok:
        logger.warning("Database integrity check failed but continuing...")

    # Log durability mode
    durability_mode = "HARDENED" if settings.HARDENED_DURABILITY else "NORMAL"
    logger.info("=== Trading Bot Started ===")
    logger.info(f"Budget: ${settings.MAX_BUDGET:,.2f} | Broker: {settings.ALPACA_BASE_URL} | DB Mode: {durability_mode}")

    # ── Startup state restore log ─────────────────────────────────────────────
    # Show what we're resuming from (SQLite persists across restarts; broker API
    # provides live cash/equity).  This is purely informational — no state is
    # lost because every trade is already in the DB and budget comes from Alpaca.
    try:
        from database import get_open_trades
        import broker as _broker
        open_trades = get_open_trades()
        if open_trades:
            tickers = [t["ticker"] for t in open_trades]
            logger.info(f"RESTORED {len(open_trades)} open position(s): {tickers}")
        else:
            logger.info("RESTORED: no open positions — clean slate")

        acct = _broker.get_account()
        cash = float(acct.get("cash", 0))
        equity = float(acct.get("equity", 0))
        logger.info(f"BROKER: cash=${cash:,.2f} | equity=${equity:,.2f}")
    except Exception as _e:
        logger.warning(f"Startup state log failed (non-critical): {_e}")
    # ─────────────────────────────────────────────────────────────────────────

    from heartbeat import (heartbeat_loop, heartbeat_cleanup_loop, sentiment_monitor, stop_loss_monitor,
                           auto_invest_loop, keep_alive_loop, daily_summary_loop,
                           weekly_report_loop, shadow_monitor_loop, portfolio_update_loop,
                           news_refresh_loop)
    heartbeat_task      = asyncio.create_task(heartbeat_loop())
    heartbeat_cleanup_task = asyncio.create_task(heartbeat_cleanup_loop())
    sentiment_task      = asyncio.create_task(sentiment_monitor())
    stop_loss_task      = asyncio.create_task(stop_loss_monitor())
    auto_invest_task    = asyncio.create_task(auto_invest_loop())
    keep_alive_task     = asyncio.create_task(keep_alive_loop())
    daily_summary_task  = asyncio.create_task(daily_summary_loop())
    weekly_report_task  = asyncio.create_task(weekly_report_loop())
    shadow_monitor_task   = asyncio.create_task(shadow_monitor_loop())
    portfolio_update_task = asyncio.create_task(portfolio_update_loop())
    news_refresh_task     = asyncio.create_task(news_refresh_loop())

    yield

    # Shutdown — Gracefully cancel and await all background tasks with timeout
    logger.info("Initiating graceful shutdown...")
    all_tasks = [heartbeat_task, heartbeat_cleanup_task, sentiment_task, stop_loss_task, auto_invest_task,
                 keep_alive_task, daily_summary_task, weekly_report_task, shadow_monitor_task,
                 portfolio_update_task, news_refresh_task]

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

# Allow TradingView (and any site) to call the bot API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routes
from webhook import router
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/inject.js")
async def inject_js():
    from fastapi.responses import Response
    js_path = Path(__file__).parent.parent / "tradebot-extension" / "content.js"

    # Check if file exists before reading
    if not js_path.exists():
        logger.warning(f"JavaScript inject file not found: {js_path}")
        return Response(
            content="// Inject file not found",
            status_code=404,
            media_type="application/javascript"
        )

    code = js_path.read_text(encoding="utf-8")
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
                    TUNNEL_URL = line.split("=", 1)[1]
                    logger.info(f"Webhook URL: {TUNNEL_URL}/webhook")

        # Also check url file as fallback
        def watch_file():
            global TUNNEL_URL
            import time as _t
            for _ in range(30):
                _t.sleep(1)
                if url_file.exists():
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
    uvicorn.run("main:app", host=host, port=port, reload=False)
