"""
Smart Notifications System
==========================

Intelligent priority-based notification system to prevent alert spam.

Features:
1. Priority levels (CRITICAL, HIGH, MEDIUM, LOW, INFO)
2. Rate limiting per notification type
3. Smart grouping (combine related alerts)
4. Quiet hours (no non-critical alerts at night)
5. User preferences (channel, frequency)
6. Notification batching (combine multiple into digest)
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Notification priority levels."""
    CRITICAL = 5    # Always send immediately (e.g., emergency exit)
    HIGH = 4        # Send unless in quiet hours
    MEDIUM = 3      # Send normally
    LOW = 2         # Batch with others
    INFO = 1        # Daily digest only


@dataclass
class Notification:
    """A pending notification."""
    notification_id: str
    priority: Priority
    category: str        # "trade", "risk", "market", "system", "learning"
    title: str
    message: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    sent: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Rate limit notifications by category."""

    def __init__(self):
        # Default cooldowns (seconds) per category
        self.cooldowns = {
            "trade": 0,           # No cooldown for trades
            "risk": 300,          # 5 min between same risk alert
            "market": 1800,       # 30 min between market updates
            "system": 600,        # 10 min between system alerts
            "learning": 3600,     # 1 hour between learning insights
            "sentiment": 900,     # 15 min between sentiment alerts
            "default": 600,
        }
        self.last_sent: dict = {}  # category_subkey → timestamp

    def can_send(self, category: str, subkey: str = "") -> bool:
        """Check if we can send notification now."""
        key = f"{category}:{subkey}"
        last = self.last_sent.get(key, 0)
        cooldown = self.cooldowns.get(category, self.cooldowns["default"])

        return (time.time() - last) >= cooldown

    def mark_sent(self, category: str, subkey: str = "") -> None:
        """Mark notification as sent."""
        key = f"{category}:{subkey}"
        self.last_sent[key] = time.time()

    def set_cooldown(self, category: str, seconds: int) -> None:
        """Update cooldown for a category."""
        self.cooldowns[category] = seconds


# Global rate limiter
_rate_limiter = RateLimiter()


# ─────────────────────────────────────────────────────────────────────────────
# QUIET HOURS
# ─────────────────────────────────────────────────────────────────────────────

def is_quiet_hours() -> bool:
    """
    Check if we're in user's quiet hours.

    Default: 23:00 - 07:00 (no notifications except CRITICAL)
    Configurable via QUIET_HOURS_START and QUIET_HOURS_END env vars.
    """
    try:
        start_hour = int(os.getenv("QUIET_HOURS_START", "23"))
        end_hour = int(os.getenv("QUIET_HOURS_END", "7"))

        now = datetime.now()
        current_hour = now.hour

        # Handle overnight quiet hours (e.g., 23-7)
        if start_hour > end_hour:
            return current_hour >= start_hour or current_hour < end_hour
        else:
            return start_hour <= current_hour < end_hour

    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION BATCHING
# ─────────────────────────────────────────────────────────────────────────────

class NotificationBatcher:
    """Batch low-priority notifications into digests."""

    def __init__(self):
        self.pending: dict = defaultdict(list)  # category → [notifications]

    def add(self, notification: Notification) -> None:
        """Add notification to batch."""
        self.pending[notification.category].append(notification)

    def get_digest(self, category: Optional[str] = None) -> str:
        """Get formatted digest message."""
        if category:
            categories = [category] if category in self.pending else []
        else:
            categories = list(self.pending.keys())

        lines = []
        for cat in categories:
            notifs = self.pending[cat]
            if not notifs:
                continue

            lines.append(f"📋 <b>{cat.upper()}</b> ({len(notifs)} items)")
            for n in notifs[:5]:  # Show up to 5 per category
                lines.append(f"  • {n.title}")

            if len(notifs) > 5:
                lines.append(f"  ... and {len(notifs) - 5} more")

            # Clear after digesting
            self.pending[cat] = []

        return "\n".join(lines) if lines else ""

    def has_pending(self) -> bool:
        return any(self.pending.values())

    def count_pending(self) -> int:
        return sum(len(notifs) for notifs in self.pending.values())


_batcher = NotificationBatcher()


# ─────────────────────────────────────────────────────────────────────────────
# SMART NOTIFICATION DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

async def send_smart_notification(
    title: str,
    message: str,
    priority: Priority = Priority.MEDIUM,
    category: str = "default",
    subkey: str = "",
    metadata: Optional[dict] = None,
    force: bool = False,
) -> bool:
    """
    Send a notification with smart filtering and batching.

    Returns: True if sent immediately, False if batched/suppressed.

    Logic:
    1. CRITICAL: Always send immediately
    2. HIGH: Send unless in quiet hours
    3. MEDIUM: Send if rate limit allows
    4. LOW: Batch into digest
    5. INFO: Daily digest only
    """
    try:
        notification_id = f"{category}:{int(time.time())}"
        notification = Notification(
            notification_id=notification_id,
            priority=priority,
            category=category,
            title=title,
            message=message,
            metadata=metadata or {},
        )

        from telegram_bot import send_message

        # ── CRITICAL: Always send ─────────────────────────────────────────
        if priority == Priority.CRITICAL or force:
            await send_message(f"🚨 <b>CRITICAL:</b> {title}\n{message}")
            _rate_limiter.mark_sent(category, subkey)
            return True

        # ── Check quiet hours ─────────────────────────────────────────────
        if is_quiet_hours() and priority.value < Priority.HIGH.value:
            logger.debug(f"Notification suppressed (quiet hours): {title}")
            _batcher.add(notification)
            return False

        # ── HIGH priority ─────────────────────────────────────────────────
        if priority == Priority.HIGH:
            if _rate_limiter.can_send(category, subkey):
                await send_message(f"⚠️ <b>{title}</b>\n{message}")
                _rate_limiter.mark_sent(category, subkey)
                return True
            else:
                _batcher.add(notification)
                return False

        # ── MEDIUM priority ───────────────────────────────────────────────
        if priority == Priority.MEDIUM:
            if _rate_limiter.can_send(category, subkey):
                await send_message(f"<b>{title}</b>\n{message}")
                _rate_limiter.mark_sent(category, subkey)
                return True
            else:
                _batcher.add(notification)
                return False

        # ── LOW priority: Always batch ────────────────────────────────────
        if priority == Priority.LOW:
            _batcher.add(notification)
            # Send digest if we have 5+ pending
            if _batcher.count_pending() >= 5:
                digest = _batcher.get_digest()
                if digest:
                    await send_message(f"📋 <b>Digest:</b>\n{digest}")
                    return True
            return False

        # ── INFO: Daily digest only ───────────────────────────────────────
        if priority == Priority.INFO:
            _batcher.add(notification)
            return False

        return False

    except Exception as e:
        logger.error(f"Smart notification failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def notify_critical(title: str, message: str, category: str = "system") -> None:
    """Send critical notification - always sent immediately."""
    await send_smart_notification(title, message, Priority.CRITICAL, category)


async def notify_high(title: str, message: str, category: str = "default", subkey: str = "") -> None:
    """Send high priority notification."""
    await send_smart_notification(title, message, Priority.HIGH, category, subkey)


async def notify_medium(title: str, message: str, category: str = "default", subkey: str = "") -> None:
    """Send medium priority notification."""
    await send_smart_notification(title, message, Priority.MEDIUM, category, subkey)


async def notify_low(title: str, message: str, category: str = "default", subkey: str = "") -> None:
    """Send low priority - will be batched."""
    await send_smart_notification(title, message, Priority.LOW, category, subkey)


async def notify_info(title: str, message: str, category: str = "default", subkey: str = "") -> None:
    """Send info notification - daily digest only."""
    await send_smart_notification(title, message, Priority.INFO, category, subkey)


# ─────────────────────────────────────────────────────────────────────────────
# DIGEST DELIVERY
# ─────────────────────────────────────────────────────────────────────────────

async def send_daily_digest() -> None:
    """Send daily digest of all batched notifications."""
    if not _batcher.has_pending():
        return

    try:
        digest = _batcher.get_digest()
        if digest:
            from telegram_bot import send_message
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            full_message = (
                f"📅 <b>Daily Notification Digest</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"<i>{now}</i>\n\n"
                f"{digest}"
            )
            await send_message(full_message)
            logger.info(f"Sent daily digest with {_batcher.count_pending()} items")

    except Exception as e:
        logger.error(f"Daily digest failed: {e}")


async def send_hourly_digest() -> None:
    """Send hourly digest of medium/low priority items."""
    if _batcher.count_pending() < 3:
        return  # Not enough to warrant a digest

    try:
        digest = _batcher.get_digest()
        if digest:
            from telegram_bot import send_message
            await send_message(f"📋 <b>Hourly Update:</b>\n{digest}")
            logger.info("Sent hourly digest")

    except Exception as e:
        logger.error(f"Hourly digest failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def configure_notifications(
    quiet_hours_start: int = 23,
    quiet_hours_end: int = 7,
    cooldowns: Optional[dict] = None,
) -> None:
    """Configure notification preferences."""
    os.environ["QUIET_HOURS_START"] = str(quiet_hours_start)
    os.environ["QUIET_HOURS_END"] = str(quiet_hours_end)

    if cooldowns:
        for category, seconds in cooldowns.items():
            _rate_limiter.set_cooldown(category, seconds)

    logger.info(f"Notifications configured: quiet={quiet_hours_start}-{quiet_hours_end}")


def get_notification_stats() -> dict:
    """Get statistics about notifications."""
    return {
        "pending_in_batch": _batcher.count_pending(),
        "in_quiet_hours": is_quiet_hours(),
        "quiet_hours": {
            "start": os.getenv("QUIET_HOURS_START", "23"),
            "end": os.getenv("QUIET_HOURS_END", "7"),
        },
        "rate_limits": {
            cat: seconds for cat, seconds in _rate_limiter.cooldowns.items()
        },
    }
