"""
Watchdog v3 — עמיד לתקיעות + heartbeat עצמי
- מזהה קריסה (process מת)
- מזהה תקיעה (אין לוג 10 דקות)
- מגן על עצמו מפני תקיעה עם threads
- כותב heartbeat כל 60 שניות
"""
import subprocess, time, sys, os, logging, threading
from pathlib import Path

# ── Detach from parent process so watchdog survives terminal/IDE close ────
# On Windows, setting a new process group prevents the parent's SIGBREAK
# from propagating to this process.
try:
    import ctypes
    # CTRL_CLOSE_EVENT handler — ignore close signals from parent
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCtrlHandler(None, True)   # ignore Ctrl+C / Ctrl+Break
except Exception:
    pass

BASE_DIR    = Path(__file__).parent
BOT_SCRIPT  = BASE_DIR / "main.py"
TUNNEL_JS   = BASE_DIR / "tunnel.js"
LOG_FILE    = BASE_DIR / "trading_bot.log"
TUNNEL_FILE = BASE_DIR / "tunnel_url.txt"
WD_LOG      = BASE_DIR / "watchdog.log"

CHECK_EVERY   = 30    # בדוק כל 30 שניות
HANG_TIMEOUT  = 600   # 10 דקות ללא לוג = תקוע
RESTART_DELAY = 30    # המתן 30s לפני restart (port release לוקח 15-45s בWindows) - raised from 15s

from logging.handlers import RotatingFileHandler as _RFH

# Log rotation: prevent watchdog.log from growing unbounded
_wd_handler = _RFH(
    str(WD_LOG),
    maxBytes=5 * 1024 * 1024,   # 5 MB per file
    backupCount=3,                # keep last 3 files = max 20 MB
    encoding="utf-8",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | WATCHDOG | %(message)s",
    handlers=[
        _wd_handler,
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


def kill_process_on_port(port: int = 8000) -> bool:
    """
    Force-kill whatever process is holding the port.
    Prevents 'port still busy after 45s' — the old instance might be
    hanging during graceful shutdown (uvicorn timeout).
    Returns True if kill succeeded or port was already free.
    """
    try:
        import subprocess as _sp
        # Find PID owning the port via netstat
        result = _sp.run(
            ['netstat', '-ano'],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        for line in result.stdout.splitlines():
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.strip().split()
                pid = int(parts[-1])
                if pid > 4:  # never kill system processes
                    try:
                        _sp.run(['taskkill', '/F', '/PID', str(pid)],
                                capture_output=True, timeout=5)
                        logger.info(f"Force-killed PID {pid} holding port {port}")
                        time.sleep(2)  # let OS reclaim port
                        return True
                    except Exception as ke:
                        logger.debug(f"Failed to kill PID {pid}: {ke}")
    except Exception as e:
        logger.debug(f"kill_process_on_port error: {e}")
    return False


def wait_for_port_free(port: int = 8000, max_wait: int = 90) -> bool:
    """
    Ensure port is free before starting the bot.
    Strategy:
      1. Quick check — is port already free? (common after clean shutdown)
      2. If busy after 5s → force-kill the hanging process
      3. Wait up to max_wait for OS to fully release (increased from 60s to 90s)
    """
    import socket

    # Quick initial check
    def _is_busy():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            r = s.connect_ex(('127.0.0.1', port))
            s.close()
            return r == 0
        except Exception:
            return False

    if not _is_busy():
        logger.info(f"Port {port} is free — proceeding with restart")
        return True

    # Port busy — wait 5s then force-kill
    time.sleep(5)
    if _is_busy():
        logger.warning(f"Port {port} still busy after 5s — force-killing holder")
        kill_process_on_port(port)

    # Wait for OS to fully release
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if not _is_busy():
            logger.info(f"Port {port} is free — proceeding with restart")
            return True
        time.sleep(2)

    logger.warning(f"Port {port} still busy after {max_wait}s — starting anyway")
    return False


def prevent_sleep():
    """
    Prevent Windows from sleeping while the watchdog is running.
    Uses SetThreadExecutionState to tell Windows we need continuous operation.
    This fixes: 'sleep detection ~1.5 min' causing port conflicts.
    """
    try:
        import ctypes
        ES_CONTINUOUS       = 0x80000000
        ES_SYSTEM_REQUIRED  = 0x00000001
        # Tell Windows: keep system awake, reset on every call
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        logger.info("Sleep prevention: Windows will not suspend the bot process")
    except Exception as e:
        logger.debug(f"Sleep prevention not available: {e}")


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


def _delete_webhook_now():
    """Force-delete any existing webhook so polling can run unobstructed.
    Required because polling and webhook are mutually exclusive."""
    def _do():
        import requests
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        cert  = os.getenv("REQUESTS_CA_BUNDLE", True)
        if not token:
            return
        r = requests.post(
            f"https://api.telegram.org/bot{token}/deleteWebhook",
            json={"drop_pending_updates": False},
            timeout=10, verify=cert,
        )
        if r.ok:
            logger.info("Webhook deleted — polling mode active")
    _run_with_timeout(_do, timeout=12)


def main():
    logger.info("=" * 55)
    logger.info("Watchdog v4 — crash-proof + sleep-proof")
    logger.info("=" * 55)

    # ── PREVENT WINDOWS SLEEP while bot is running ────────────────────
    # Sleep = port conflicts + crash loops when waking up
    prevent_sleep()

    # ── POLLING ONLY: delete any leftover webhook so polling can take over ──
    _delete_webhook_now()

    # ── CLEAN SLATE: kill anything already on port 8000 before starting ──
    kill_process_on_port(8000)
    time.sleep(2)

    bot_proc   = start_bot()
    # No tunnel needed for polling mode — local bot only
    tun_proc   = None
    crashes    = 0
    last_url   = ""
    last_hb    = time.time()  # heartbeat timer

    # ── CRASH LOOP DETECTION ───────────────────────────────────────────
    # Track timestamps of recent crashes. If too many crashes in short time,
    # pause and send critical alert instead of restart-looping forever.
    crash_timestamps = []  # list of recent crash times
    CRASH_LOOP_WINDOW = 600  # 10 minutes
    CRASH_LOOP_MAX = 5       # max crashes per window before pause
    CRASH_LOOP_PAUSE = 300   # pause 5 minutes if loop detected

    def _check_crash_loop() -> bool:
        """Detect crash loop. Returns True if should pause."""
        now = time.time()
        # Remove old timestamps outside window
        crash_timestamps[:] = [t for t in crash_timestamps if now - t < CRASH_LOOP_WINDOW]
        crash_timestamps.append(now)
        if len(crash_timestamps) >= CRASH_LOOP_MAX:
            logger.critical(
                f"🚨 CRASH LOOP DETECTED: {len(crash_timestamps)} crashes in "
                f"{CRASH_LOOP_WINDOW//60} min — pausing {CRASH_LOOP_PAUSE//60} min"
            )
            send_alert(
                f"🚨 <b>CRASH LOOP</b>\n"
                f"{len(crash_timestamps)} קריסות ב-{CRASH_LOOP_WINDOW//60} דקות\n"
                f"⏸️ עוצר {CRASH_LOOP_PAUSE//60} דקות לבדיקה ידנית\n"
                f"בדוק לוגים: trading_bot.log"
            )
            return True
        return False

    last_check_time = time.time()
    while True:
        time.sleep(CHECK_EVERY)

        # ── SLEEP DETECTION: did the clock jump forward unexpectedly? ──
        now = time.time()
        elapsed = now - last_check_time
        if elapsed > CHECK_EVERY * 3:
            # Clock jumped >90s — likely the machine was suspended/hibernated
            jump_minutes = elapsed / 60
            logger.warning(f"⏰ זוהתה השעיית מחשב (~{jump_minutes:.1f} דקות) — מוודא שהבוט פעיל")
            send_alert(
                f"⏰ <b>המחשב התעורר משינה</b>\n"
                f"זמן השעיה: ~{jump_minutes:.0f} דקות\n"
                f"בודק שהבוט עדיין פעיל..."
            )
        last_check_time = now

        # ── Watchdog heartbeat ────────────────────────────────────────
        if time.time() - last_hb > 60:
            logger.debug("Watchdog alive ✅")
            last_hb = time.time()

        # ── 1. Bot crash ──────────────────────────────────────────────
        ret = _run_with_timeout(lambda: bot_proc.poll(), timeout=5)
        if ret is not None:
            crashes += 1
            exit_desc = {
                4294967295: "נהרג חיצונית (SIGKILL)",
                3221225786: "Ctrl+C / terminal סגר",
                1: "Python exception (port busy?)",
                0: "יציאה נקייה",
            }.get(ret, f"exit={ret}")
            logger.warning(f"הבוט קרס ({exit_desc}) | קריסה #{crashes}")
            send_alert(f"⚠️ <b>בוט קרס</b> — {exit_desc}\nמופעל מחדש #{crashes}")

            # NEW: Detect crash loop and pause if needed
            if _check_crash_loop():
                time.sleep(CRASH_LOOP_PAUSE)
                crash_timestamps.clear()  # Reset after pause

            time.sleep(RESTART_DELAY)
            # CRITICAL FIX: wait for port 8000 to be free before restarting
            # Without this, uvicorn crashes immediately with "Address already in use" (exit=1)
            wait_for_port_free(8000, max_wait=60)
            # Force-kill any leftover process on port
            kill_process_on_port(8000)
            time.sleep(1)
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
            wait_for_port_free(8000, max_wait=60)
            bot_proc = start_bot()
            continue

        # ── 3. Tunnel health ──────────────────────────────────────────
        # DISABLED in polling mode — we don't need a public URL.
        # Polling pulls updates directly from Telegram. No tunnel = no 409 conflicts.
        pass


if __name__ == "__main__":
    main()
