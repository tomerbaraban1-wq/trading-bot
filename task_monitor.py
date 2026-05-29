"""
Task Monitor v2 — זיהוי וריסטרט אוטומטי של tasks שקרסו
ממונע משתיקה: כל task ש-crash מזוהה ומוריסטרט מיד
"""
import asyncio
import logging
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """עקבוי after task"""
    name: str
    task: asyncio.Task
    created_at: float
    restart_fn: Callable
    restart_count: int = 0
    last_restart: float = 0
    last_error: Optional[str] = None


class TaskMonitor:
    """מנהל tasks background — גם קריאוש כללי נשמר"""

    def __init__(self, max_restart_count: int = 5, max_restart_wait: float = 300):
        """
        Args:
            max_restart_count: כמה פעמים תנסה restart לפני surrender
            max_restart_wait: זמן חכיה max בין restarts (exponential backoff)
        """
        self.tasks: Dict[str, TaskInfo] = {}
        self.max_restart_count = max_restart_count
        self.max_restart_wait = max_restart_wait
        self.check_interval = 10  # בדוק every 10 seconds
        self.monitor_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        coro: Any,
        name: str,
        restart_fn: Optional[Callable] = None,
    ) -> asyncio.Task:
        """
        Create monitored task with automatic restart on crash

        Args:
            coro: coroutine to run
            name: unique task name
            restart_fn: callable that returns new coroutine on restart
                       If not provided, will re-run original coro
        """
        async with self._lock:
            # If restart_fn not provided, use a factory
            if restart_fn is None:
                # Capture original coro function
                if hasattr(coro, '__self__'):
                    # It's a bound method
                    original_fn = coro
                    restart_fn = lambda: original_fn()
                else:
                    # Try to extract function
                    logger.warning(f"Task {name}: restart_fn not provided, will attempt re-run")
                    restart_fn = None

            task = asyncio.create_task(coro, name=name)

            task_info = TaskInfo(
                name=name,
                task=task,
                created_at=time.time(),
                restart_fn=restart_fn,
            )
            self.tasks[name] = task_info
            logger.info(f"[MONITOR] Task created: {name}")

            # Start monitor if not running
            if not self.monitor_running:
                await self._start_monitor()

            return task

    async def _start_monitor(self):
        """Start background health check loop"""
        if self.monitor_running:
            return

        self.monitor_running = True
        self.monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="TaskMonitor._monitor_loop"
        )
        logger.info("[MONITOR] Health check loop started")

    async def _monitor_loop(self):
        """Background health monitoring — checks dead tasks every 10s"""
        last_full_status = time.time()

        while self.monitor_running:
            try:
                # Check for dead tasks
                async with self._lock:
                    dead_tasks = []

                    for name, info in self.tasks.items():
                        # Check if task is done (crashed or completed)
                        if info.task.done():
                            try:
                                # Try to get exception
                                exc = info.task.exception()
                                if exc:
                                    info.last_error = str(exc)
                                    logger.error(f"[MONITOR] Task CRASHED: {name}")
                                    logger.error(f"          Error: {exc}")
                                    dead_tasks.append(name)
                            except asyncio.CancelledError:
                                logger.info(f"[MONITOR] Task cancelled: {name}")
                                dead_tasks.append(name)
                            except asyncio.InvalidStateError:
                                pass

                    # Restart dead tasks
                    for name in dead_tasks:
                        await self._restart_task(name)

                # Periodic full status (every 5 minutes)
                now = time.time()
                if now - last_full_status > 300:
                    await self._log_status()
                    last_full_status = now

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("[MONITOR] Monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"[MONITOR] Monitor error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _restart_task(self, name: str):
        """Restart a dead task with exponential backoff"""
        if name not in self.tasks:
            return

        info = self.tasks[name]

        # Check restart limit
        if info.restart_count >= self.max_restart_count:
            logger.critical(f"[MONITOR] Task {name} hit restart limit ({self.max_restart_count})")
            logger.critical(f"          Last error: {info.last_error}")
            # Send Telegram alert
            try:
                import os
                from config import settings
                import requests
                token = settings.TELEGRAM_BOT_TOKEN
                chat = settings.TELEGRAM_CHAT_ID
                if token and chat:
                    requests.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={
                            "chat_id": chat,
                            "text": (
                                f"🆘 <b>CRITICAL</b>\n"
                                f"Task <code>{name}</code> failed {self.max_restart_count}x\n"
                                f"Last error: <code>{info.last_error}</code>\n"
                                f"Watchdog will restart bot"
                            ),
                            "parse_mode": "HTML"
                        },
                        timeout=5
                    )
            except Exception as e:
                logger.warning(f"Failed to send alert: {e}")
            return

        # Exponential backoff: 1s, 2s, 4s, 8s... up to max_restart_wait
        wait_time = min(2 ** info.restart_count, self.max_restart_wait)

        logger.warning(
            f"[MONITOR] Restarting task {name} "
            f"(attempt {info.restart_count + 1}/{self.max_restart_count}) "
            f"after {wait_time}s"
        )

        await asyncio.sleep(wait_time)

        # Create new task
        if info.restart_fn:
            try:
                new_coro = info.restart_fn()
                new_task = asyncio.create_task(new_coro, name=name)

                info.task = new_task
                info.restart_count += 1
                info.last_restart = time.time()
                info.last_error = None

                logger.info(f"[MONITOR] Task restarted: {name} (attempt {info.restart_count})")
            except Exception as e:
                logger.error(f"[MONITOR] Failed to restart {name}: {e}")
                # Try again next cycle
        else:
            logger.error(f"[MONITOR] Cannot restart {name}: no restart_fn provided")

    async def _log_status(self):
        """Log current task status"""
        async with self._lock:
            alive = sum(1 for t in self.tasks.values() if not t.task.done())
            dead = len(self.tasks) - alive

            logger.info(f"[MONITOR] Status: {alive} alive, {dead} dead, total {len(self.tasks)}")

            # List any dead tasks
            for name, info in self.tasks.items():
                if info.task.done():
                    logger.warning(f"  DEAD: {name} (restarts: {info.restart_count}) - {info.last_error}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current health status"""
        async with self._lock:
            tasks_status = {}
            for name, info in self.tasks.items():
                tasks_status[name] = {
                    "alive": not info.task.done(),
                    "uptime": time.time() - info.created_at,
                    "restarts": info.restart_count,
                    "last_error": info.last_error,
                }

            return {
                "total_tasks": len(self.tasks),
                "alive": sum(1 for t in self.tasks.values() if not t.task.done()),
                "dead": sum(1 for t in self.tasks.values() if t.task.done()),
                "tasks": tasks_status,
                "timestamp": datetime.now().isoformat(),
            }

    async def shutdown(self):
        """Graceful shutdown — cancel all tasks"""
        self.monitor_running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            logger.info("[MONITOR] Shutting down all tasks...")

            for name, info in self.tasks.items():
                if not info.task.done():
                    info.task.cancel()

            # Wait for all to complete
            all_tasks = [info.task for info in self.tasks.values()]
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)

            logger.info("[MONITOR] All tasks shut down")


# Global instance
_monitor: Optional[TaskMonitor] = None


async def init_monitor() -> TaskMonitor:
    """Initialize global task monitor"""
    global _monitor
    if _monitor is None:
        _monitor = TaskMonitor(max_restart_count=5, max_restart_wait=300)
    return _monitor


def get_monitor() -> TaskMonitor:
    """Get global task monitor (must be initialized first)"""
    global _monitor
    if _monitor is None:
        raise RuntimeError("TaskMonitor not initialized. Call init_monitor() first")
    return _monitor
