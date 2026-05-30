"""
Bot Supervisor — A bulletproof Python wrapper that:
1. Starts the bot
2. Monitors it
3. Restarts if dead (max 5/hour)
4. Restarts proactively every 8 hours (prevents socket leak)
5. Logs everything

NO ADMIN REQUIRED. Just run: python bot_supervisor.py
"""
import subprocess
import sys
import time
import socket
import signal
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable
LOG_FILE = BASE_DIR / "supervisor.log"

# Configuration
HEALTH_CHECK_INTERVAL = 30        # check every 30 sec
PROACTIVE_RESTART_HOURS = 8       # restart every 8 hours
MAX_RESTARTS_PER_HOUR = 5         # crash loop protection
PORT = 8000
LOCK_PORT = 8765                  # single-instance lock for the supervisor itself

# Holds the single-instance lock socket open for the whole process lifetime.
_lock_socket = None


def log(msg: str) -> None:
    """Log to file and stdout."""
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def acquire_single_instance_lock() -> bool:
    """
    Ensure ONLY ONE supervisor runs at a time. Binds a localhost lock port and
    holds it for the whole process lifetime. If another supervisor already holds
    it, we return False so the duplicate can exit cleanly.

    This prevents two supervisors fighting over port 8000 — the root cause of the
    historical 'port busy' crash loop (crashes #1-3 on 2026-05-28).

    Fail-OPEN: on any unexpected error we allow startup, so a bug here can never
    block the bot from running.
    """
    global _lock_socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Do NOT set SO_REUSEADDR — we WANT bind to fail if another instance holds it.
        s.bind(("127.0.0.1", LOCK_PORT))
        s.listen(1)
        _lock_socket = s  # keep reference alive => lock held until process exits
        return True
    except OSError:
        return False          # another supervisor already running
    except Exception:
        return True           # fail-open: never block the bot on a lock bug


def touch_alive() -> None:
    """
    Write a heartbeat timestamp to data/last_alive.txt every health check.
    On the next startup, the bot reads this to compute how long it was DOWN
    (asleep/off) and announces "I'm awake — was down X hours" to Telegram.
    """
    try:
        data_dir = BASE_DIR / "data"
        data_dir.mkdir(exist_ok=True)
        with open(data_dir / "last_alive.txt", "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def prevent_windows_sleep():
    """
    AGGRESSIVE sleep prevention — prevents Windows from sleeping while supervisor runs.

    Uses TWO layers:
    1. SetThreadExecutionState — prevents idle-timeout sleep
    2. powercfg /change standby-timeout-ac 0 — disables AC sleep entirely

    Layer 2 was added because SetThreadExecutionState does NOT block a
    user-mode process that calls SetSuspendState() directly (e.g., Logitech
    software, OEM utilities). The only defense against that is either admin
    rights (to register a power request override) or disabling standby in
    the active power scheme — which we can do without admin.

    Root cause: the bot died at 19:01 on 2026-05-29 because a user-mode process
    called SetSuspendState, ignoring the ES_SYSTEM_REQUIRED flag entirely.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS        = 0x80000000
        ES_SYSTEM_REQUIRED   = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040

        result = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
        if result != 0:
            log(f"[SLEEP_GUARD] SetThreadExecutionState ACTIVE (0x{result:X})")
        else:
            log("[SLEEP_GUARD] WARNING: SetThreadExecutionState failed")
    except Exception as e:
        log(f"[SLEEP_GUARD] ctypes failed: {e}")

    # Layer 2: force the power scheme to never-standby (no admin needed)
    try:
        import subprocess as _sp
        for cmd in [
            "powercfg /change standby-timeout-ac 0",
            "powercfg /change standby-timeout-dc 0",
            "powercfg /change hibernate-timeout-ac 0",
            "powercfg /change hibernate-timeout-dc 0",
        ]:
            _sp.run(cmd, shell=True, capture_output=True, timeout=10)
        log("[SLEEP_GUARD] powercfg: AC/DC standby+hibernate = NEVER")
    except Exception as e:
        log(f"[SLEEP_GUARD] powercfg failed (non-fatal): {e}")


def detect_resume_from_sleep(last_check_time: float) -> bool:
    """Detect if system was suspended (clock jumped forward unexpectedly)."""
    now = time.time()
    gap = now - last_check_time
    # If gap >> expected check interval, system was likely sleeping
    if gap > HEALTH_CHECK_INTERVAL * 3:
        log(f"[SLEEP_GUARD] DETECTED: System was sleeping for ~{gap/60:.1f} min — verifying bot")
        return True
    return False


def is_port_listening(port: int) -> bool:
    """Check if port is in LISTEN state."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def wait_for_port_free(port: int, max_wait: int = 60) -> bool:
    """Wait for port to become available."""
    start = time.time()
    while time.time() - start < max_wait:
        if not is_port_listening(port):
            return True
        time.sleep(2)
    return False


def kill_anything_on_port(port: int) -> None:
    """Try to kill any process holding the port."""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
                if conn.pid:
                    try:
                        psutil.Process(conn.pid).kill()
                        log(f"Killed leftover process {conn.pid} on port {port}")
                    except Exception:
                        pass
    except Exception:
        pass


def start_bot() -> subprocess.Popen:
    """Start the bot as a child process."""
    # Make sure port is free
    if is_port_listening(PORT):
        log(f"Port {PORT} still in use — killing")
        kill_anything_on_port(PORT)
        wait_for_port_free(PORT, max_wait=30)

    log(f"Starting bot...")
    out_file = open(BASE_DIR / "bot_background.log", "a", encoding="utf-8")
    err_file = open(BASE_DIR / "bot_background_err.log", "a", encoding="utf-8")

    # CREATE_NO_WINDOW (0x08000000): no console window.
    # CREATE_NEW_PROCESS_GROUP (0x00000200): isolate from the parent's console
    #   signal group, so Ctrl+C / console-close events can NOT kill the bot.
    #   This was the #1 historical crash ("Ctrl+C / terminal סגר", exit 0xC000013A).
    creation_flags = (0x08000000 | 0x00000200) if os.name == "nt" else 0

    proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=str(BASE_DIR),
        stdout=out_file,
        stderr=err_file,
        creationflags=creation_flags,
    )
    log(f"Bot started — PID {proc.pid}")

    # Wait for it to actually bind to the port
    for i in range(30):
        time.sleep(1)
        if is_port_listening(PORT):
            log(f"Bot is healthy — port {PORT} listening")
            return proc
    log(f"WARNING: Bot started but port {PORT} not listening after 30s")
    return proc


def main():
    # Single-instance guard: if another supervisor is already running, exit
    # cleanly instead of starting a second one that would fight over port 8000.
    if not acquire_single_instance_lock():
        log("Another supervisor already holds the lock — exiting (avoids duplicate/port conflict)")
        return

    log("=" * 60)
    log("BOT SUPERVISOR STARTED")
    log(f"Python: {PYTHON_EXE}")
    log(f"Working dir: {BASE_DIR}")
    log("=" * 60)

    # CRITICAL: Prevent Windows sleep (root cause of last night's outage!)
    prevent_windows_sleep()

    bot_proc = start_bot()
    last_proactive_restart = time.time()
    last_check_time = time.time()
    restart_times: list[float] = []

    # Signal handlers
    def handle_signal(sig, frame):
        log(f"Received signal {sig} — stopping bot and exiting")
        try:
            bot_proc.terminate()
            bot_proc.wait(timeout=10)
        except Exception:
            try:
                bot_proc.kill()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        try:
            time.sleep(HEALTH_CHECK_INTERVAL)
            now = time.time()

            # ── Check 0: Detect Windows sleep/resume ────────────────────
            if detect_resume_from_sleep(last_check_time):
                # Re-enable sleep prevention (in case it was cancelled)
                prevent_windows_sleep()
                # Verify bot is still alive after resume
                if bot_proc.poll() is not None or not is_port_listening(PORT):
                    log("[SLEEP_GUARD] Bot died during sleep — restarting")
                    try:
                        bot_proc.kill()
                    except Exception:
                        pass
                    wait_for_port_free(PORT, max_wait=60)
                    bot_proc = start_bot()
                    last_proactive_restart = now
                    last_check_time = now
                    continue
            last_check_time = now
            touch_alive()   # heartbeat marker for "was-down" detection on next start

            # ── Check 1: Did the bot crash? ─────────────────────────────
            ret = bot_proc.poll()
            if ret is not None:
                log(f"Bot died (exit code {ret}) — restarting")

                # Crash loop protection: max 5 restarts/hour
                restart_times = [t for t in restart_times if now - t < 3600]
                if len(restart_times) >= MAX_RESTARTS_PER_HOUR:
                    log(f"CRASH LOOP DETECTED ({len(restart_times)} restarts/hour) — sleeping 30 min")
                    time.sleep(1800)
                    restart_times.clear()

                restart_times.append(now)

                # Wait for port to free
                wait_for_port_free(PORT, max_wait=60)
                bot_proc = start_bot()
                last_proactive_restart = now
                continue

            # ── Check 2: Proactive restart every 8 hours ─────────────────
            hours_since_restart = (now - last_proactive_restart) / 3600
            if hours_since_restart >= PROACTIVE_RESTART_HOURS:
                log(f"Proactive restart after {hours_since_restart:.1f}h "
                    f"(prevents socket leak from yfinance)")
                try:
                    bot_proc.terminate()
                    bot_proc.wait(timeout=30)
                except Exception:
                    bot_proc.kill()
                    time.sleep(5)

                wait_for_port_free(PORT, max_wait=60)
                bot_proc = start_bot()
                last_proactive_restart = now
                continue

            # ── Check 3: Is port actually listening? ─────────────────────
            if not is_port_listening(PORT):
                log(f"Bot alive but port {PORT} not listening — likely hung, restarting")
                try:
                    bot_proc.kill()
                    bot_proc.wait(timeout=10)
                except Exception:
                    pass
                wait_for_port_free(PORT, max_wait=60)
                bot_proc = start_bot()
                last_proactive_restart = now

        except KeyboardInterrupt:
            log("Ctrl+C received — exiting")
            try:
                bot_proc.terminate()
            except Exception:
                pass
            break
        except Exception as e:
            log(f"Supervisor error (continuing): {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
