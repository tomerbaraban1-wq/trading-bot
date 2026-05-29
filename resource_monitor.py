"""
Resource Monitor
================

Monitors bot resource usage and sends Telegram alerts when thresholds exceeded:
  - CPU > 150% for 3 consecutive checks → alert + log
  - Memory > 500 MB → alert
  - Memory > 750 MB → force garbage collection
  - Disk < 500 MB free → alert

Runs as a background asyncio task.
"""

import asyncio
import gc
import logging
import os
import time

logger = logging.getLogger(__name__)

# Thresholds
CPU_WARN_PCT    = 150.0   # warn if CPU avg > 150%
CPU_CRIT_PCT    = 250.0   # critical if CPU avg > 250%
MEM_WARN_MB     = 500     # warn at 500 MB
MEM_GC_MB       = 750     # force GC at 750 MB
MEM_CRIT_MB     = 1000    # critical at 1 GB
DISK_WARN_MB    = 500     # warn if disk < 500 MB free

# State
_cpu_high_count   = 0      # consecutive high-CPU checks
_last_mem_alert   = 0.0
_last_cpu_alert   = 0.0
_ALERT_COOLDOWN   = 1800   # 30 min between same-type alerts


async def _send_alert(msg: str, force: bool = False) -> None:
    try:
        from telegram_bot import send_message
        await send_message(msg, force=force)
    except Exception as e:
        logger.debug(f"[RESOURCE] alert send failed: {e}")


def _get_bot_process():
    """Return the psutil Process for the current PID."""
    try:
        import psutil
        return psutil.Process(os.getpid())
    except Exception:
        return None


def _get_disk_free_mb() -> float:
    try:
        import psutil, pathlib
        usage = psutil.disk_usage(str(pathlib.Path(__file__).parent))
        return usage.free / (1024 * 1024)
    except Exception:
        return 9999.0


async def resource_monitor_loop() -> None:
    """
    Background task: checks memory, CPU, and disk every 60 seconds.
    Sends Telegram alerts on threshold violations.
    """
    global _cpu_high_count, _last_mem_alert, _last_cpu_alert

    await asyncio.sleep(120)   # 2 min after startup — let bot settle

    proc = _get_bot_process()
    if proc is None:
        logger.warning("[RESOURCE] psutil not available — resource monitoring disabled")
        return

    logger.info("[RESOURCE] Monitor started")

    while True:
        try:
            now = time.time()

            # ── Memory check ─────────────────────────────────────────────
            mem_mb = proc.memory_info().rss / (1024 * 1024)

            if mem_mb > MEM_GC_MB:
                collected = gc.collect()
                mem_after = proc.memory_info().rss / (1024 * 1024)
                logger.info(f"[RESOURCE] GC collected {collected} objects — {mem_mb:.0f}→{mem_after:.0f} MB")
                mem_mb = mem_after

            if mem_mb > MEM_CRIT_MB and now - _last_mem_alert > _ALERT_COOLDOWN:
                _last_mem_alert = now
                await _send_alert(
                    f"🚨 <b>זיכרון קריטי!</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"💾 שימוש: <b>{mem_mb:.0f} MB</b>\n"
                    f"⚠️ מעל 1 GB — הבוט עלול לקרוס!\n"
                    f"💡 נסה להפעיל מחדש: הפעל watchdog.py",
                    force=True,
                )
            elif mem_mb > MEM_WARN_MB and now - _last_mem_alert > _ALERT_COOLDOWN:
                _last_mem_alert = now
                await _send_alert(
                    f"⚠️ <b>זיכרון גבוה</b>\n"
                    f"💾 {mem_mb:.0f} MB (סף: {MEM_WARN_MB} MB)\n"
                    f"🔄 ניקיון זיכרון בוצע"
                )

            # ── CPU check ────────────────────────────────────────────────
            cpu_pct = proc.cpu_percent(interval=1)

            if cpu_pct > CPU_CRIT_PCT:
                _cpu_high_count += 1
            elif cpu_pct > CPU_WARN_PCT:
                _cpu_high_count = max(1, _cpu_high_count)
            else:
                _cpu_high_count = 0

            if _cpu_high_count >= 3 and now - _last_cpu_alert > _ALERT_COOLDOWN:
                _last_cpu_alert = now
                await _send_alert(
                    f"🔥 <b>CPU גבוה</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"⚙️ שימוש: <b>{cpu_pct:.0f}%</b>\n"
                    f"📋 Threads: {proc.num_threads()}\n"
                    f"💡 אולי training loop רץ מהר מדי?"
                )
                _cpu_high_count = 0

            # ── Disk check ───────────────────────────────────────────────
            disk_free = _get_disk_free_mb()
            if disk_free < DISK_WARN_MB:
                await _send_alert(
                    f"💽 <b>דיסק כמעט מלא!</b>\n"
                    f"פנוי: {disk_free:.0f} MB בלבד"
                )

            # ── Log every 10 minutes ─────────────────────────────────────
            if not hasattr(resource_monitor_loop, '_last_log'):
                resource_monitor_loop._last_log = 0
            if now - resource_monitor_loop._last_log > 600:
                resource_monitor_loop._last_log = now
                logger.info(
                    f"[RESOURCE] Memory: {mem_mb:.0f}MB | "
                    f"CPU: {cpu_pct:.0f}% | "
                    f"Threads: {proc.num_threads()} | "
                    f"Disk: {disk_free:.0f}MB free"
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"[RESOURCE] check error: {e}")

        await asyncio.sleep(60)   # check every 60 seconds
