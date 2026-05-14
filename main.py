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
    _broker_info = settings.ACTIVE_BROKER if settings.ACTIVE_BROKER else settings.ALPACA_BASE_URL
    logger.info(f"Budget: ${settings.MAX_BUDGET:,.2f} | Broker: {_broker_info} | DB Mode: {durability_mode}")

    # ── Auto-register Telegram webhook + command menu ─────────────────────────
    try:
        _render_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        _tg_token   = settings.TELEGRAM_BOT_TOKEN
        if _render_url and _tg_token:
            import aiohttp as _aiohttp
            _webhook_url = f"{_render_url}/telegram/webhook"
            async with _aiohttp.ClientSession() as _sess:
                # Register webhook
                async with _sess.post(
                    f"https://api.telegram.org/bot{_tg_token}/setWebhook",
                    json={"url": _webhook_url},
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
                    {"command": "market",    "description": "🌍 מצב השוק עכשיו"},
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
                    {"command": "advice",    "description": "🤖 ייעוץ AI על התיק"},
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

        acct = _broker.get_account()
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
                broker_positions = _broker.get_positions()
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
            broker_positions = _broker.get_positions()
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
                        stop_price, stop_meta = compute_initial_stop(ticker, entry)
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

    from heartbeat import (heartbeat_loop, heartbeat_cleanup_loop, sentiment_monitor, stop_loss_monitor,
                           auto_invest_loop, keep_alive_loop, daily_summary_loop,
                           weekly_report_loop, shadow_monitor_loop, portfolio_update_loop,
                           news_refresh_loop, news_monitor_loop, morning_briefing_loop,
                           position_alert_loop, backtest_learning_loop, eod_sweep_loop,
                           price_alert_loop)
    # ── Core tasks (always run) ───────────────────────────────────────
    heartbeat_task         = asyncio.create_task(heartbeat_loop())
    heartbeat_cleanup_task = asyncio.create_task(heartbeat_cleanup_loop())
    stop_loss_task         = asyncio.create_task(stop_loss_monitor())
    auto_invest_task       = asyncio.create_task(auto_invest_loop())
    keep_alive_task        = asyncio.create_task(keep_alive_loop())
    daily_summary_task     = asyncio.create_task(daily_summary_loop())
    weekly_report_task     = asyncio.create_task(weekly_report_loop())
    backtest_task          = asyncio.create_task(backtest_learning_loop())
    eod_sweep_task         = asyncio.create_task(eod_sweep_loop())
    price_alert_task       = asyncio.create_task(price_alert_loop())
    morning_briefing_task  = asyncio.create_task(morning_briefing_loop())
    news_refresh_task      = asyncio.create_task(news_refresh_loop())
    news_monitor_task      = asyncio.create_task(news_monitor_loop())   # 🆕 real-time news → sell/tighten

    # ── Optional tasks (disabled on free tier to save memory) ────────
    import os as _os
    _full_mode = _os.getenv("FULL_MODE", "false").lower() == "true"
    sentiment_task      = asyncio.create_task(sentiment_monitor())     if _full_mode else None
    shadow_monitor_task = asyncio.create_task(shadow_monitor_loop())   if _full_mode else None
    portfolio_update_task = asyncio.create_task(portfolio_update_loop()) if _full_mode else None
    position_alert_task = asyncio.create_task(position_alert_loop())   if _full_mode else None

    if not _full_mode:
        logger.info("Memory-saving mode: shadow, portfolio_update, position_alert, sentiment disabled. Set FULL_MODE=true to enable.")

    yield

    # Shutdown — Gracefully cancel and await all background tasks with timeout
    logger.info("Initiating graceful shutdown...")
    all_tasks = [t for t in [
        heartbeat_task, heartbeat_cleanup_task, sentiment_task, stop_loss_task, auto_invest_task,
        keep_alive_task, daily_summary_task, weekly_report_task, shadow_monitor_task,
        portfolio_update_task, news_refresh_task, news_monitor_task, morning_briefing_task,
        position_alert_task, backtest_task, eod_sweep_task, price_alert_task,
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
    allow_headers=["Content-Type", "X-Webhook-Secret"],
    allow_credentials=False,
)

# Import and include routes
from webhook import router
app.include_router(router)


@app.get("/ping")
async def ping():
    """Ultra-lightweight liveness endpoint for UptimeRobot / external keep-alive.
    No DB calls — returns instantly so Render never marks it as slow."""
    return {"ok": True, "uptime": round(time.time() - START_TIME)}


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
    uvicorn.run("main:app", host=host, port=port, reload=False)
