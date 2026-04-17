import os
# Fix SSL certificate path for Hebrew Windows username (local dev only)
_cert = 'C:/certs/cacert.pem'
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
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from database import init_db, close_connections

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
    logger.info("=== Trading Bot Started ===")
    logger.info(f"Budget: ${settings.MAX_BUDGET:,.2f} | Broker: {settings.ALPACA_BASE_URL}")

    from heartbeat import heartbeat_loop, sentiment_monitor, stop_loss_monitor, auto_invest_loop
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    sentiment_task = asyncio.create_task(sentiment_monitor())
    stop_loss_task = asyncio.create_task(stop_loss_monitor())
    auto_invest_task = asyncio.create_task(auto_invest_loop())

    yield

    # Shutdown
    heartbeat_task.cancel()
    sentiment_task.cancel()
    stop_loss_task.cancel()
    auto_invest_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    try:
        await sentiment_task
    except asyncio.CancelledError:
        pass
    try:
        await stop_loss_task
    except asyncio.CancelledError:
        pass
    try:
        await auto_invest_task
    except asyncio.CancelledError:
        pass
    close_connections()
    logger.info("=== Trading Bot Stopped ===")


app = FastAPI(title="TradeBot", version="1.0.0", lifespan=lifespan)

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
    code = js_path.read_text(encoding="utf-8")
    return Response(content=code, media_type="application/javascript")


@app.get("/tunnel")
async def tunnel_info():
    # Read from file (written by node tunnel.js in parent process)
    url = TUNNEL_URL
    url_file = Path(__file__).parent / "tunnel_url.txt"
    if not url and url_file.exists():
        try:
            url = url_file.read_text().strip() or None
        except Exception:
            pass
    return {"url": url, "webhook": f"{url}/webhook" if url else None}


def get_uptime() -> float:
    return time.time() - START_TIME


TUNNEL_URL = None


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
