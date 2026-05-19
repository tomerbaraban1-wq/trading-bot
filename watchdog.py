"""
Watchdog — מפעיל מחדש את הבוט אוטומטית אם קורס.
הרץ: python watchdog.py
"""
import subprocess
import time
import sys
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | WATCHDOG | %(message)s",
    handlers=[
        logging.FileHandler("watchdog.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

BOT_SCRIPT  = os.path.join(os.path.dirname(__file__), "main.py")
CHECK_EVERY = 30    # בדוק כל 30 שניות
RESTART_DELAY = 5   # המתן 5 שניות לפני הפעלה מחדש

def start_bot():
    logger.info("מפעיל את הבוט...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, BOT_SCRIPT],
        env=env,
        cwd=os.path.dirname(__file__),
    )
    logger.info(f"הבוט פעיל — PID {proc.pid}")
    return proc

def main():
    logger.info("=" * 50)
    logger.info("Watchdog הופעל — מנטר את הבוט")
    logger.info("=" * 50)

    proc = start_bot()
    crashes = 0

    while True:
        time.sleep(CHECK_EVERY)

        ret = proc.poll()
        if ret is None:
            continue  # הבוט רץ — הכל תקין

        # הבוט נפסק
        crashes += 1
        logger.warning(f"הבוט נפסק (exit code={ret}) | קריסה #{crashes}")

        time.sleep(RESTART_DELAY)
        logger.info("מפעיל מחדש...")
        proc = start_bot()

if __name__ == "__main__":
    main()
