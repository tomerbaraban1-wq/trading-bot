#!/usr/bin/env python3
"""
Bot Monitoring & Auto-Restart Service
=======================================

Periodically checks if the trading bot is running and healthy.
Automatically restarts the bot if it goes down.

Usage:
    python monitor_bot.py
    # or in background:
    nohup python monitor_bot.py > monitor.log 2>&1 &
    # or in screen:
    screen -S monitor -d python monitor_bot.py
"""

import requests
import subprocess
import time
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
log_file = Path(__file__).parent / "monitor.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# Configuration
HEALTH_URL = "http://localhost:8000/ping"
HEALTH_CHECK_INTERVAL = 60  # seconds (check every minute)
MAX_CONSECUTIVE_FAILURES = 3  # restart after 3 failed checks
BOT_START_TIMEOUT = 10  # seconds to wait after restart
BOT_DIR = Path(__file__).parent

# Track bot status
class BotStatus:
    def __init__(self):
        self.is_running = False
        self.consecutive_failures = 0
        self.last_check_time = None
        self.last_restart_time = None
        self.restart_count = 0

status = BotStatus()


def is_bot_running() -> bool:
    """Check if bot is responding to health endpoint."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        is_healthy = response.status_code == 200

        if is_healthy:
            try:
                data = response.json()
                uptime = data.get("uptime", 0)
                return True
            except:
                return True
        else:
            return False

    except requests.exceptions.Timeout:
        logger.debug("Bot health check timeout (connection timeout)")
        return False
    except requests.exceptions.ConnectionError:
        logger.debug("Bot health check failed (connection refused)")
        return False
    except Exception as e:
        logger.debug(f"Bot health check failed: {e}")
        return False


def get_bot_process_info() -> dict:
    """Get info about running bot process."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split("\n"):
            if "python main.py" in line and "monitor" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    return {
                        "pid": parts[1],
                        "running": True,
                        "command": " ".join(parts[10:])
                    }
        return {"pid": None, "running": False}
    except:
        return {"pid": None, "running": False}


def check_port_in_use() -> bool:
    """Check if port 8000 is in use."""
    try:
        result = subprocess.run(
            ["netstat", "-tuln"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return ":8000" in result.stdout
    except:
        # Fallback: try to connect
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", 8000))
            sock.close()
            return result == 0
        except:
            return False


def restart_bot() -> bool:
    """Attempt to restart the bot."""
    logger.warning("🔴 Bot is down - attempting restart...")

    try:
        # Kill any existing python main.py processes
        logger.info("Killing existing bot processes...")
        subprocess.run(
            "pkill -f 'python main.py' || true",
            shell=True,
            timeout=5
        )

        time.sleep(2)  # Wait for process to die

        # Check if port is still in use
        if check_port_in_use():
            logger.warning("⚠️  Port 8000 still in use after kill")
            return False

        # Start bot in background
        logger.info("Starting bot process...")
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=BOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
        )

        logger.info(f"Bot started (PID: {process.pid})")
        status.last_restart_time = datetime.now()
        status.restart_count += 1

        # Wait for bot to start
        logger.info(f"Waiting {BOT_START_TIMEOUT}s for bot to initialize...")
        time.sleep(BOT_START_TIMEOUT)

        # Verify bot is running
        if is_bot_running():
            logger.info("✅ Bot successfully restarted and is responding")
            status.consecutive_failures = 0
            return True
        else:
            logger.error("❌ Bot process started but still not responding")
            logger.error("Check logs at: trading_bot.log")
            return False

    except subprocess.TimeoutExpired as e:
        logger.error(f"❌ Restart timeout: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Restart failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_health_check() -> None:
    """Perform health check and restart if needed."""
    status.last_check_time = datetime.now()

    # Quick health check
    if is_bot_running():
        status.consecutive_failures = 0
        status.is_running = True
        logger.debug("✅ Bot is healthy")
    else:
        status.consecutive_failures += 1
        status.is_running = False

        logger.warning(
            f"⚠️  Bot not responding ({status.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
        )

        # Check actual process status
        proc_info = get_bot_process_info()
        if proc_info["running"]:
            logger.warning(f"⚠️  Process is running (PID {proc_info['pid']}) but not responding")
            logger.warning("This suggests the bot is hung or has an error")
            logger.warning("Check logs at: trading_bot.log")
        else:
            logger.warning("⚠️  Bot process is not running")

        # Restart if threshold reached
        if status.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.error(
                f"❌ Bot failed {MAX_CONSECUTIVE_FAILURES} consecutive checks - restarting..."
            )
            success = restart_bot()

            if success:
                status.is_running = True
            else:
                logger.error("⚠️  Restart failed - will retry on next check")


def log_status_summary() -> None:
    """Log a status summary every hour."""
    logger.info("")
    logger.info("═" * 70)
    logger.info("📊 BOT MONITOR STATUS SUMMARY")
    logger.info("═" * 70)
    logger.info(f"Current Time: {datetime.now().isoformat()}")
    logger.info(f"Bot Status: {'✅ RUNNING' if status.is_running else '❌ DOWN'}")
    logger.info(f"Last Check: {status.last_check_time.isoformat() if status.last_check_time else 'Never'}")
    logger.info(f"Consecutive Failures: {status.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")
    logger.info(f"Total Restarts: {status.restart_count}")
    if status.last_restart_time:
        logger.info(f"Last Restart: {status.last_restart_time.isoformat()}")
    logger.info(f"Health Check URL: {HEALTH_URL}")
    logger.info(f"Check Interval: {HEALTH_CHECK_INTERVAL}s")
    logger.info("═" * 70)
    logger.info("")


def main() -> None:
    """Main monitoring loop."""
    logger.info("")
    logger.info("━" * 70)
    logger.info("🤖 TRADING BOT MONITOR STARTED")
    logger.info("━" * 70)
    logger.info(f"Health URL: {HEALTH_URL}")
    logger.info(f"Check Interval: {HEALTH_CHECK_INTERVAL}s")
    logger.info(f"Restart Threshold: {MAX_CONSECUTIVE_FAILURES} failures")
    logger.info(f"Bot Directory: {BOT_DIR}")
    logger.info(f"Monitor Log: {log_file}")
    logger.info("")
    logger.info("Press Ctrl+C to stop monitoring")
    logger.info("━" * 70)
    logger.info("")

    check_count = 0
    hour_counter = 0

    try:
        while True:
            try:
                check_count += 1
                hour_counter += 1

                run_health_check()

                # Log summary every 60 checks (1 hour if check_interval=60s)
                if hour_counter >= 60:
                    log_status_summary()
                    hour_counter = 0

                # Wait before next check
                time.sleep(HEALTH_CHECK_INTERVAL)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Check error: {e}")
                time.sleep(HEALTH_CHECK_INTERVAL)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("━" * 70)
        logger.info("🛑 Monitor stopped by user")
        logger.info(f"Total checks: {check_count}")
        logger.info(f"Total restarts: {status.restart_count}")
        logger.info("━" * 70)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
