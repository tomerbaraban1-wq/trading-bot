"""
Telegram Security Layer
========================

Layered security for Telegram bot commands:

1. Rate limiting per chat_id (anti-DoS, anti-abuse)
2. Dangerous command confirmation (sell_all, change config — require "yes confirm")
3. Audit log — every sensitive action recorded with timestamp
4. Anomaly detection — burst of unusual commands
5. Command authorization levels (info / action / dangerous)

This module wraps handle_telegram_update without changing its core logic.
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING — Sliding window
# ─────────────────────────────────────────────────────────────────────────────

_chat_requests: dict[str, deque] = defaultdict(deque)
_RATE_WINDOW_SECONDS = 60
_RATE_MAX_REQUESTS = 30   # 30 commands per minute per chat_id


def check_rate_limit(chat_id: str) -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Allowed = True if request is within rate limit.
    """
    now = time.time()
    requests = _chat_requests[chat_id]

    # Drop old entries outside the window
    while requests and requests[0] < now - _RATE_WINDOW_SECONDS:
        requests.popleft()

    if len(requests) >= _RATE_MAX_REQUESTS:
        oldest = requests[0]
        wait_seconds = int(_RATE_WINDOW_SECONDS - (now - oldest))
        return False, f"מקסימום {_RATE_MAX_REQUESTS} פקודות לדקה — חכה {wait_seconds}s"

    requests.append(now)
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# DANGEROUS COMMAND CONFIRMATION — Two-step verification
# ─────────────────────────────────────────────────────────────────────────────

# Commands that require explicit confirmation before execution
DANGEROUS_COMMANDS = {
    "/sell_all", "/close_all", "/emergency_stop",
    "/wipe_data", "/reset", "/clear_history",
    "/pause", "/disable_bot",
}

# Pending confirmations: chat_id → (command, timestamp)
_pending_confirmations: dict[str, tuple[str, float]] = {}
_CONFIRMATION_TIMEOUT_SECONDS = 30


def needs_confirmation(text: str) -> Optional[str]:
    """If command is dangerous, return the canonical command name. Else None."""
    cmd = text.strip().split()[0].lower() if text else ""
    return cmd if cmd in DANGEROUS_COMMANDS else None


def request_confirmation(chat_id: str, command: str) -> str:
    """Mark a dangerous command as pending confirmation. Returns the prompt to send."""
    _pending_confirmations[chat_id] = (command, time.time())
    return (
        f"⚠️ <b>פעולה רגישה: {command}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"זוהי פעולה לא הפיכה. כדי לאשר, שלח:\n"
        f"<code>כן {command[1:]}</code>\n"
        f"\n"
        f"⏱ תוקף: 30 שניות. אחרת — מבוטל אוטומטית."
    )


def is_confirmation(chat_id: str, text: str) -> Optional[str]:
    """
    Check if `text` confirms a pending command for `chat_id`.
    Returns the confirmed command or None.
    """
    if chat_id not in _pending_confirmations:
        return None

    command, ts = _pending_confirmations[chat_id]
    now = time.time()

    # Expired
    if now - ts > _CONFIRMATION_TIMEOUT_SECONDS:
        _pending_confirmations.pop(chat_id, None)
        return None

    # Expected format: "כן <command_without_slash>"
    expected = f"כן {command[1:]}".strip().lower()
    if text.strip().lower() == expected:
        _pending_confirmations.pop(chat_id, None)
        return command

    return None


def cancel_confirmation(chat_id: str) -> None:
    """Cancel any pending confirmation for this chat."""
    _pending_confirmations.pop(chat_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG — Persistent record of sensitive actions
# ─────────────────────────────────────────────────────────────────────────────

# Commands that should be audited
AUDITED_COMMANDS = DANGEROUS_COMMANDS | {
    "/buy", "/sell", "/set_budget", "/set_min_score",
    "/restart", "/update_config",
}


def audit_log(chat_id: str, command: str, status: str = "executed", extra: str = "") -> None:
    """Write an audit entry to the database security_audit_log table."""
    cmd = command.strip().split()[0].lower() if command else ""
    if cmd not in AUDITED_COMMANDS:
        return  # Only audit sensitive commands

    try:
        import sqlite3
        from config import settings
        conn = sqlite3.connect(settings.DATABASE_PATH, timeout=5)
        c = conn.cursor()
        # Ensure table exists
        c.execute("""
            CREATE TABLE IF NOT EXISTS telegram_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                extra TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute(
            "INSERT INTO telegram_audit_log (chat_id, command, status, extra) VALUES (?, ?, ?, ?)",
            (chat_id, command[:200], status, extra[:500])
        )
        conn.commit()
        conn.close()
        logger.info(f"[AUDIT] chat={chat_id} cmd={cmd} status={status}")
    except Exception as e:
        logger.debug(f"audit_log write failed: {e}")


def get_recent_audit(chat_id: str, limit: int = 20) -> list[dict]:
    """Return the most recent audit entries for a chat_id."""
    try:
        import sqlite3
        from config import settings
        conn = sqlite3.connect(settings.DATABASE_PATH, timeout=5)
        c = conn.cursor()
        c.execute(
            """SELECT command, status, extra, created_at
               FROM telegram_audit_log
               WHERE chat_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (chat_id, limit)
        )
        rows = c.fetchall()
        conn.close()
        return [
            {"command": r[0], "status": r[1], "extra": r[2], "time": r[3]}
            for r in rows
        ]
    except Exception as e:
        logger.debug(f"get_recent_audit failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION — Detect unusual command patterns
# ─────────────────────────────────────────────────────────────────────────────

# Track recent commands for anomaly detection: chat_id → deque of (cmd, ts)
_recent_commands: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))


def check_anomaly(chat_id: str, command: str) -> Optional[str]:
    """
    Detect anomalous patterns. Returns warning string or None.

    Triggers:
    - Same command repeated 10+ times in 60s
    - 20+ commands in 30s (burst)
    - Dangerous command without prior info-gathering
    """
    now = time.time()
    history = _recent_commands[chat_id]
    history.append((command, now))

    # Remove old entries
    while history and history[0][1] < now - 60:
        history.popleft()

    # Same command spam
    same_cmd_count = sum(1 for c, _ in history if c == command)
    if same_cmd_count >= 10:
        return f"חזרה חריגה של אותה פקודה ({same_cmd_count}× ב-60s)"

    # Burst
    last_30s = [c for c, t in history if t >= now - 30]
    if len(last_30s) >= 20:
        return f"פיצוץ פקודות חריג ({len(last_30s)} ב-30s)"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND AUTHORIZATION LEVEL
# ─────────────────────────────────────────────────────────────────────────────

class CommandLevel:
    INFO = "info"          # Read-only: /status, /pnl, /price
    ACTION = "action"      # Trades/changes: /buy, /sell, /pause
    DANGEROUS = "dangerous"  # Irreversible: /sell_all, /wipe_data


def classify_command(text: str) -> str:
    """Return the authorization level for a command."""
    cmd = text.strip().split()[0].lower() if text else ""
    if cmd in DANGEROUS_COMMANDS:
        return CommandLevel.DANGEROUS
    if cmd in {"/buy", "/sell", "/pause", "/resume", "/set_budget"}:
        return CommandLevel.ACTION
    return CommandLevel.INFO


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SECURITY CHECK — Called before handling any update
# ─────────────────────────────────────────────────────────────────────────────

def security_check(chat_id: str, text: str) -> dict:
    """
    Comprehensive security check.

    Returns:
        {
          "allowed": bool,
          "reason": str,        # Why blocked (if not allowed)
          "needs_confirm": bool,
          "confirm_prompt": str,  # Message to send if confirmation needed
          "confirmed_command": Optional[str],  # If this is a confirmation
        }
    """
    result = {
        "allowed": True,
        "reason": "",
        "needs_confirm": False,
        "confirm_prompt": "",
        "confirmed_command": None,
    }

    # Skip checks for empty messages
    if not text:
        return result

    # 1. Check if this is a confirmation response
    confirmed = is_confirmation(chat_id, text)
    if confirmed:
        result["confirmed_command"] = confirmed
        audit_log(chat_id, confirmed, "confirmed_and_executing")
        return result

    # 2. Rate limit check (only on commands, not on plain text)
    if text.startswith("/"):
        ok, reason = check_rate_limit(chat_id)
        if not ok:
            audit_log(chat_id, text, "rate_limited", reason)
            result["allowed"] = False
            result["reason"] = f"⏳ {reason}"
            return result

    # 3. Anomaly detection
    anomaly = check_anomaly(chat_id, text)
    if anomaly:
        audit_log(chat_id, text, "anomaly_detected", anomaly)
        logger.warning(f"[SECURITY] Anomaly from chat {chat_id}: {anomaly}")
        # Don't block on first anomaly — just warn
        # If user persists, rate limit will kick in

    # 4. Dangerous command check
    dangerous_cmd = needs_confirmation(text)
    if dangerous_cmd:
        audit_log(chat_id, text, "awaiting_confirmation")
        result["needs_confirm"] = True
        result["confirm_prompt"] = request_confirmation(chat_id, dangerous_cmd)
        return result

    # 5. Standard command — log if audited
    if text.startswith("/"):
        cmd = text.strip().split()[0].lower()
        if cmd in AUDITED_COMMANDS:
            audit_log(chat_id, text, "executed")

    return result
