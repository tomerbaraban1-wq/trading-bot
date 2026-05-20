"""
Watchdog v3 — עמיד לתקיעות + heartbeat עצמי
- מזהה קריסה (process מת)
- מזהה תקיעה (אין לוג 10 דקות)
- מגן על עצמו מפני תקיעה עם threads
- כותב heartbeat כל 60 שניות
"""
import subprocess, time, sys, os, logging, threading
from pathlib import Path

BASE_DIR    = Path(__file__).parent
BOT_SCRIPT  = BASE_DIR / "main.py"
TUNNEL_JS   = BASE_DIR / "tunnel.js"
LOG_FILE    = BASE_DIR / "trading_bot.log"
TUNNEL_FILE = BASE_DIR / "tunnel_url.txt"
WD_LOG      = BASE_DIR / "watchdog.log"

CHECK_EVERY  = 30     # בדוק כל 30 שניות
HANG_TIMEOUT = 600    # 10 דקות ללא לוג = תקוע
RESTART_DELAY = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | WATCHDOG | %(message)s",
    handlers=[
        logging.FileHandler(str(WD_LOG), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def _run_with_timeout(fn, timeout=15, default=None):
    """Run fn() in a thread, return default if it times out."""
    result = [default]
    exc    = [None]
    def _worker():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return default   # timed out
    if exc[0]:
        raise exc[0]
    return result[0]


def start_bot():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen([sys.executable, str(BOT_SCRIPT)], env=env, cwd=str(BASE_DIR))
    logger.info(f"הבוט פעיל — PID {proc.pid}")
    return proc


def start_tunnel():
    try:
        proc = subprocess.Popen(
            ["node", str(TUNNEL_JS), "8000"],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        logger.info(f"Tunnel started — PID {proc.pid}")
        return proc
    except Exception as e:
        logger.warning(f"Tunnel start failed: {e}")
        return None


def set_webhook(url: str):
    """Set Telegram webhook — runs with 15s timeout."""
    def _do():
        import requests
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        token  = os.getenv("TELEGRAM_BOT_TOKEN", "")
        secret = os.getenv("WEBHOOK_SECRET", "")
        cert   = os.getenv("REQUESTS_CA_BUNDLE", True)
        if not token:
            return
        r = requests.post(
            f"https://api.telegram.org/bot{token}/setWebhook",
            json={"url": f"{url}/telegram/webhook",
                  "drop_pending_updates": True,
                  "secret_token": secret},
            timeout=10, verify=cert,
        )
        if r.ok:
            logger.info(f"Webhook set → {url}")
    _run_with_timeout(_do, timeout=15)


def send_alert(msg: str):
    """Send Telegram alert — runs with 10s timeout."""
    def _do():
        import requests
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat  = os.getenv("TELEGRAM_CHAT_ID", "")
        cert  = os.getenv("REQUESTS_CA_BUNDLE", True)
        if not token or not chat:
            return
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=8, verify=cert,
        )
    _run_with_timeout(_do, timeout=10)


def log_is_fresh() -> bool:
    try:
        mtime = LOG_FILE.stat().st_mtime
        return (time.time() - mtime) < HANG_TIMEOUT
    except Exception:
        return True


def get_tunnel_url() -> str:
    try:
        if TUNNEL_FILE.exists():
            return TUNNEL_FILE.read_text().strip().replace("TUNNEL_URL=", "")
    except Exception:
        pass
    return ""


def main():
    logger.info("=" * 55)
    logger.info("Watchdog v3 — עמיד לתקיעות")
    logger.info("=" * 55)

    bot_proc   = start_bot()
    tun_proc   = start_tunnel()
    crashes    = 0
    last_url   = ""
    last_hb    = time.time()  # heartbeat timer

    # Wait for tunnel URL (max 30s)
    for _ in range(15):
        time.sleep(2)
        url = get_tunnel_url()
        if url and url != last_url:
            last_url = url
            set_webhook(url)
            break

    while True:
        time.sleep(CHECK_EVERY)

        # ── Watchdog heartbeat ────────────────────────────────────────
        if time.time() - last_hb > 60:
            logger.debug("Watchdog alive ✅")
            last_hb = time.time()

        # ── 1. Bot crash ──────────────────────────────────────────────
        ret = _run_with_timeout(lambda: bot_proc.poll(), timeout=5)
        if ret is not None:
            crashes += 1
            logger.warning(f"הבוט קרס (exit={ret}) | קריסה #{crashes}")
            send_alert(f"⚠️ <b>בוט קרס</b> — מופעל מחדש #{crashes}")
            time.sleep(RESTART_DELAY)
            bot_proc = start_bot()
            continue

        # ── 2. Bot hung ───────────────────────────────────────────────
        if not log_is_fresh():
            crashes += 1
            logger.warning(f"הבוט תקוע {HANG_TIMEOUT//60}min | ריסטרט #{crashes}")
            send_alert(f"🔄 <b>בוט תקוע</b> — ריסטרט #{crashes}")
            try:
                bot_proc.kill()
            except Exception:
                pass
            time.sleep(RESTART_DELAY)
            bot_proc = start_bot()
            continue

        # ── 3. Tunnel health ──────────────────────────────────────────
        tun_dead = _run_with_timeout(
            lambda: tun_proc.poll() is not None if tun_proc else True,
            timeout=3, default=False
        )
        if tun_dead:
            logger.warning("Tunnel נפסק — מפעיל מחדש...")
            try:
                TUNNEL_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            tun_proc = start_tunnel()
            # Wait for new URL (max 30s, non-blocking)
            for _ in range(15):
                time.sleep(2)
                url = get_tunnel_url()
                if url and url != last_url:
                    last_url = url
                    set_webhook(url)
                    break


if __name__ == "__main__":
    main()
