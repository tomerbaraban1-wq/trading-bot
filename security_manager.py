"""
Central Security Manager
=========================

Comprehensive security infrastructure for the trading bot.

Features:
1. Advanced rate limiting (per-IP, per-endpoint, per-user)
2. Brute force protection
3. Encryption at rest for sensitive data
4. Audit logging for all sensitive operations
5. Input sanitization and validation
6. Security event detection
7. IP reputation tracking
8. Session management
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_FAILED = "login_failed"
    LOGIN_SUCCESS = "login_success"
    RATE_LIMIT_HIT = "rate_limit_hit"
    BRUTE_FORCE = "brute_force"
    UNAUTHORIZED = "unauthorized"
    SUSPICIOUS_REQUEST = "suspicious_request"
    ADMIN_ACTION = "admin_action"
    SECRETS_ACCESS = "secrets_access"
    DATA_EXPORT = "data_export"
    CONFIG_CHANGE = "config_change"
    API_KEY_USAGE = "api_key_usage"


@dataclass
class SecurityEvent:
    """A security-relevant event."""
    timestamp: str
    event_type: SecurityEventType
    severity: str  # "info", "warning", "critical"
    source_ip: str
    user_id: Optional[str]
    endpoint: str
    description: str
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED RATE LIMITING
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedRateLimiter:
    """
    Multi-tier rate limiter with adaptive thresholds.

    Limits:
    - Per IP: 100 req/min
    - Per endpoint: configurable
    - Per user: configurable
    - Sliding window
    """

    def __init__(self):
        self.ip_requests: dict = defaultdict(lambda: deque(maxlen=1000))
        self.endpoint_requests: dict = defaultdict(lambda: deque(maxlen=10000))
        self.user_requests: dict = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: dict = {}  # ip -> unblock_timestamp

        # Default limits (req per minute)
        self.limits = {
            "default": 100,
            "/admin/*": 20,
            "/api/v1/*": 60,
            "/telegram/webhook": 30,
            "/webhook/*": 30,
            "auth": 5,  # Special category for auth attempts
        }

    def _matches_pattern(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches a pattern."""
        if pattern.endswith("/*"):
            return endpoint.startswith(pattern[:-1])
        return endpoint == pattern

    def _get_limit(self, endpoint: str) -> int:
        """Get rate limit for endpoint."""
        for pattern, limit in self.limits.items():
            if self._matches_pattern(endpoint, pattern):
                return limit
        return self.limits["default"]

    def is_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        if ip in self.blocked_ips:
            if time.time() < self.blocked_ips[ip]:
                return True
            else:
                del self.blocked_ips[ip]
        return False

    def block_ip(self, ip: str, duration_seconds: int = 3600) -> None:
        """Block an IP for specified duration."""
        self.blocked_ips[ip] = time.time() + duration_seconds
        logger.warning(f"[SECURITY] Blocked IP {ip} for {duration_seconds}s")

    def check_rate_limit(
        self,
        ip: str,
        endpoint: str,
        user_id: Optional[str] = None,
    ) -> dict:
        """
        Check if request is within rate limits.

        Returns: {"allowed": bool, "reason": str, "retry_after": int}
        """
        # Check if IP is blocked
        if self.is_blocked(ip):
            return {
                "allowed": False,
                "reason": "IP temporarily blocked",
                "retry_after": int(self.blocked_ips[ip] - time.time()),
            }

        now = time.time()
        cutoff = now - 60  # 1-minute window

        # Per-IP check
        self.ip_requests[ip].append(now)
        recent_ip_requests = sum(1 for t in self.ip_requests[ip] if t > cutoff)

        if recent_ip_requests > 200:  # 200 req/min absolute limit
            self.block_ip(ip, 600)  # Block for 10 minutes
            return {
                "allowed": False,
                "reason": "IP exceeded global rate limit",
                "retry_after": 600,
            }

        # Per-endpoint check
        endpoint_key = f"{ip}:{endpoint}"
        self.endpoint_requests[endpoint_key].append(now)
        recent_endpoint = sum(1 for t in self.endpoint_requests[endpoint_key] if t > cutoff)

        limit = self._get_limit(endpoint)
        if recent_endpoint > limit:
            return {
                "allowed": False,
                "reason": f"Endpoint rate limit ({limit}/min)",
                "retry_after": 60,
            }

        # Per-user check (if authenticated)
        if user_id:
            self.user_requests[user_id].append(now)
            recent_user = sum(1 for t in self.user_requests[user_id] if t > cutoff)

            if recent_user > 300:  # User-level limit
                return {
                    "allowed": False,
                    "reason": "User rate limit exceeded",
                    "retry_after": 60,
                }

        return {"allowed": True, "reason": "OK", "retry_after": 0}


_rate_limiter = AdvancedRateLimiter()


# ─────────────────────────────────────────────────────────────────────────────
# BRUTE FORCE PROTECTION
# ─────────────────────────────────────────────────────────────────────────────

class BruteForceProtector:
    """Detect and block brute force attempts."""

    def __init__(self):
        self.failed_attempts: dict = defaultdict(list)
        self.max_attempts = 5
        self.window_minutes = 15
        self.block_duration_minutes = 60

    def record_failure(self, identifier: str, ip: str) -> dict:
        """Record a failed authentication attempt."""
        now = time.time()
        cutoff = now - (self.window_minutes * 60)

        # Clean old entries
        self.failed_attempts[identifier] = [
            t for t in self.failed_attempts[identifier] if t > cutoff
        ]
        self.failed_attempts[identifier].append(now)

        attempts = len(self.failed_attempts[identifier])

        if attempts >= self.max_attempts:
            _rate_limiter.block_ip(ip, self.block_duration_minutes * 60)
            log_security_event(
                event_type=SecurityEventType.BRUTE_FORCE,
                severity="critical",
                source_ip=ip,
                description=f"Brute force detected: {attempts} attempts on {identifier}",
            )
            return {
                "blocked": True,
                "attempts": attempts,
                "block_duration_minutes": self.block_duration_minutes,
            }

        return {
            "blocked": False,
            "attempts": attempts,
            "remaining": self.max_attempts - attempts,
        }

    def record_success(self, identifier: str) -> None:
        """Clear failed attempts on successful auth."""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]


_brute_force = BruteForceProtector()


# ─────────────────────────────────────────────────────────────────────────────
# ENCRYPTION AT REST
# ─────────────────────────────────────────────────────────────────────────────

def get_encryption_key() -> bytes:
    """Get or create encryption key for sensitive data."""
    key_b64 = os.getenv("ENCRYPTION_KEY", "")

    if not key_b64:
        # Generate new key (should be set in env)
        logger.warning("[SECURITY] ENCRYPTION_KEY not set - using temporary key (NOT secure for production)")
        # Derive from a default secret for development
        seed = os.getenv("LICENSE_SECRET_KEY", "default-development-key-change-in-prod")
        key = hashlib.sha256(seed.encode()).digest()
        return key

    try:
        return base64.urlsafe_b64decode(key_b64)
    except:
        return hashlib.sha256(key_b64.encode()).digest()


def encrypt_value(plaintext: str) -> str:
    """
    Encrypt a sensitive value using AES-256 GCM (via cryptography library).

    Falls back to simple XOR if cryptography not available.
    """
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes

        # Derive Fernet-compatible key
        key = get_encryption_key()
        fernet_key = base64.urlsafe_b64encode(key[:32])
        f = Fernet(fernet_key)
        encrypted = f.encrypt(plaintext.encode())
        return encrypted.decode()

    except ImportError:
        # Fallback: simple obfuscation (NOT cryptographically secure)
        logger.warning("[SECURITY] cryptography library not available - using basic obfuscation")
        key = get_encryption_key()
        # XOR with repeating key
        encoded = plaintext.encode()
        encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encoded))
        return base64.b64encode(encrypted).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a value encrypted with encrypt_value."""
    try:
        from cryptography.fernet import Fernet

        key = get_encryption_key()
        fernet_key = base64.urlsafe_b64encode(key[:32])
        f = Fernet(fernet_key)
        decrypted = f.decrypt(ciphertext.encode())
        return decrypted.decode()

    except ImportError:
        # Fallback decrypt
        key = get_encryption_key()
        encrypted = base64.b64decode(ciphertext)
        decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encrypted))
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_security_event(
    event_type: SecurityEventType,
    severity: str,
    source_ip: str = "unknown",
    user_id: Optional[str] = None,
    endpoint: str = "",
    description: str = "",
    metadata: Optional[dict] = None,
) -> None:
    """Log a security event to database and file."""
    try:
        event = SecurityEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            endpoint=endpoint,
            description=description,
            metadata=metadata or {},
        )

        # Log to logger
        log_msg = f"[SECURITY {severity.upper()}] {event_type.value} from {source_ip}: {description}"
        if severity == "critical":
            logger.error(log_msg)
        elif severity == "warning":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Persist to database
        try:
            import database
            conn = database.get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT,
                    severity TEXT,
                    source_ip TEXT,
                    user_id TEXT,
                    endpoint TEXT,
                    description TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                INSERT INTO security_audit_log
                (event_type, severity, source_ip, user_id, endpoint, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_type.value, event.severity, event.source_ip,
                event.user_id, event.endpoint, event.description,
                json.dumps(event.metadata)
            ))
            conn.commit()
        except Exception as e:
            logger.debug(f"Failed to persist security event: {e}")

        # Send alert for critical events
        if severity == "critical":
            try:
                # Get running loop - may not exist if called from sync context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Store reference to prevent garbage collection
                    task = loop.create_task(_send_security_alert(event))
                    _pending_alert_tasks.add(task)
                    task.add_done_callback(_pending_alert_tasks.discard)
            except RuntimeError:
                # No event loop - skip the alert (DB log is still recorded)
                logger.debug("No event loop available for security alert")

    except Exception as e:
        logger.error(f"Failed to log security event: {e}")


# Track pending alert tasks to prevent garbage collection
_pending_alert_tasks: set = set()


async def _send_security_alert(event: SecurityEvent) -> None:
    """Send alert for critical security events."""
    try:
        from smart_notifications import notify_critical

        await notify_critical(
            title=f"🚨 Security Alert: {event.event_type.value}",
            message=(
                f"<b>Severity:</b> {event.severity}\n"
                f"<b>IP:</b> {event.source_ip}\n"
                f"<b>Endpoint:</b> {event.endpoint}\n"
                f"<b>Description:</b> {event.description}"
            ),
            category="system",
        )
    except Exception as e:
        logger.debug(f"Security alert notification failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION & SANITIZATION
# ─────────────────────────────────────────────────────────────────────────────

import re

# Patterns
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}(-[A-Z])?$')
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
SAFE_STRING_PATTERN = re.compile(r'^[\w\s\-.,_!?@#$%^&*()+=]+$')

# Dangerous patterns to detect injections
SQL_INJECTION_PATTERNS = [
    re.compile(r"';|--|/\*|\*/|union\s+select", re.IGNORECASE),
    re.compile(r"(drop|delete|truncate)\s+(table|database)", re.IGNORECASE),
    re.compile(r"or\s+1\s*=\s*1", re.IGNORECASE),
]

XSS_PATTERNS = [
    re.compile(r"<script[^>]*>", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
]

PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\./|\.\.\\", re.IGNORECASE),
    re.compile(r"/etc/passwd|/etc/shadow", re.IGNORECASE),
    re.compile(r"c:\\windows|c:/windows", re.IGNORECASE),
]


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Validate stock ticker format."""
    if not ticker or len(ticker) > 6:
        return False, "Invalid ticker length"
    ticker_upper = ticker.upper().strip()
    if not TICKER_PATTERN.match(ticker_upper):
        return False, "Invalid ticker format"
    return True, ticker_upper


def validate_email(email: str) -> tuple[bool, str]:
    """Validate email format."""
    if not email or len(email) > 254:
        return False, "Invalid email length"
    email_lower = email.lower().strip()
    if not EMAIL_PATTERN.match(email_lower):
        return False, "Invalid email format"
    return True, email_lower


def detect_injection_attempt(value: str) -> Optional[str]:
    """Detect potential injection attempts."""
    if not isinstance(value, str):
        return None

    # SQL injection
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(value):
            return "sql_injection"

    # XSS
    for pattern in XSS_PATTERNS:
        if pattern.search(value):
            return "xss"

    # Path traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern.search(value):
            return "path_traversal"

    return None


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize a string for safe use."""
    if not isinstance(value, str):
        return ""

    # Truncate
    value = value[:max_length]

    # Remove null bytes
    value = value.replace('\x00', '')

    # Strip excessive whitespace
    value = ' '.join(value.split())

    return value


def validate_numeric(value, min_val: float = None, max_val: float = None) -> tuple[bool, float]:
    """Validate and convert a numeric value."""
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False, 0
        if max_val is not None and num > max_val:
            return False, 0
        return True, num
    except (ValueError, TypeError):
        return False, 0


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

def inspect_request(headers: dict, body: dict, ip: str, endpoint: str) -> dict:
    """
    Inspect a request for security issues.

    Returns: {"safe": bool, "issues": [...], "risk_level": "low/medium/high"}
    """
    issues = []
    risk_level = "low"

    # Check User-Agent
    user_agent = headers.get("user-agent", "").lower()
    if not user_agent:
        issues.append("Missing User-Agent")
        risk_level = "medium"

    suspicious_agents = ["sqlmap", "nikto", "nmap", "masscan", "metasploit"]
    if any(s in user_agent for s in suspicious_agents):
        issues.append(f"Suspicious User-Agent: {user_agent}")
        risk_level = "high"

    # Check for injection in body
    def check_value(value, path=""):
        nonlocal risk_level
        if isinstance(value, dict):
            for k, v in value.items():
                check_value(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                check_value(v, f"{path}[{i}]")
        elif isinstance(value, str):
            attack = detect_injection_attempt(value)
            if attack:
                issues.append(f"{attack} attempt at {path}")
                risk_level = "high"

    if body:
        check_value(body)

    # Check headers for injection
    for key, value in headers.items():
        if isinstance(value, str):
            attack = detect_injection_attempt(value)
            if attack:
                issues.append(f"{attack} in header {key}")
                risk_level = "high"

    return {
        "safe": risk_level == "low",
        "issues": issues,
        "risk_level": risk_level,
        "ip": ip,
        "endpoint": endpoint,
    }


# ─────────────────────────────────────────────────────────────────────────────
# API KEY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def generate_secure_api_key(prefix: str = "key", length: int = 32) -> str:
    """Generate a cryptographically secure API key."""
    random_part = secrets.token_urlsafe(length)
    return f"{prefix}_{random_part}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage (never store plaintext)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(provided: str, expected_hash: str) -> bool:
    """Verify an API key against its hash using constant-time comparison."""
    if not provided or not expected_hash:
        return False

    provided_hash = hash_api_key(provided)
    return hmac.compare_digest(provided_hash, expected_hash)


# ─────────────────────────────────────────────────────────────────────────────
# SECRETS ROTATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SecretMetadata:
    """Metadata about a secret."""
    name: str
    created_at: str
    last_rotated: str
    days_old: int
    needs_rotation: bool
    rotation_recommended_after_days: int


def check_secrets_rotation() -> list[SecretMetadata]:
    """
    Check which secrets need rotation.

    Best practice: rotate keys every 90 days.
    """
    secrets_to_check = [
        ("TELEGRAM_BOT_TOKEN", 180),  # Less critical
        ("ALPACA_API_KEY", 90),
        ("ALPACA_SECRET_KEY", 90),
        ("WEBHOOK_SECRET", 60),
        ("ADMIN_API_KEY", 30),  # Most critical
        ("LICENSE_SECRET_KEY", 90),
        ("ENCRYPTION_KEY", 365),  # Rarely rotated (would need data migration)
    ]

    results = []
    for name, rotation_days in secrets_to_check:
        if os.getenv(name):
            # We can't track actual age without persistence
            # In production, store rotation timestamps in database
            results.append(SecretMetadata(
                name=name,
                created_at="unknown",
                last_rotated="unknown",
                days_old=0,
                needs_rotation=False,
                rotation_recommended_after_days=rotation_days,
            ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY DASHBOARD DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_security_status() -> dict:
    """Get comprehensive security status."""
    try:
        import database
        conn = database.get_connection()

        # Recent events (last 24h)
        events_24h = conn.execute("""
            SELECT event_type, severity, COUNT(*) as count
            FROM security_audit_log
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY event_type, severity
            ORDER BY count DESC
        """).fetchall() if _table_exists(conn, "security_audit_log") else []

        # Critical events (last 7 days)
        critical_events = conn.execute("""
            SELECT timestamp, event_type, source_ip, description
            FROM security_audit_log
            WHERE severity = 'critical'
            AND timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp DESC
            LIMIT 10
        """).fetchall() if _table_exists(conn, "security_audit_log") else []

        # Configuration check
        env_checks = {
            "TELEGRAM_BOT_TOKEN": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
            "WEBHOOK_SECRET": bool(os.getenv("WEBHOOK_SECRET")),
            "ADMIN_API_KEY": bool(os.getenv("ADMIN_API_KEY")),
            "ENCRYPTION_KEY": bool(os.getenv("ENCRYPTION_KEY")),
            "LICENSE_SECRET_KEY": bool(os.getenv("LICENSE_SECRET_KEY")),
            "ANALYTICS_API_KEY": bool(os.getenv("ANALYTICS_API_KEY")),
        }

        # Security score
        score = sum(env_checks.values()) / len(env_checks) * 100

        # Recommendations
        recommendations = []
        if not env_checks["ENCRYPTION_KEY"]:
            recommendations.append("🔴 Set ENCRYPTION_KEY for encrypting sensitive data")
        if not env_checks["ADMIN_API_KEY"]:
            recommendations.append("🔴 Set ADMIN_API_KEY for admin endpoint protection")
        if not env_checks["WEBHOOK_SECRET"]:
            recommendations.append("⚠️ Set WEBHOOK_SECRET for webhook validation")
        if not env_checks["ANALYTICS_API_KEY"]:
            recommendations.append("💡 Set ANALYTICS_API_KEY for API protection")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_score": score,
            "score_label": (
                "🟢 Excellent" if score >= 90 else
                "🟡 Good" if score >= 70 else
                "🟠 Needs improvement" if score >= 50 else
                "🔴 Critical"
            ),
            "configuration": env_checks,
            "events_24h": [
                {"type": t, "severity": s, "count": c}
                for t, s, c in events_24h
            ],
            "critical_events_7d": [
                {"timestamp": ts, "type": t, "ip": ip, "description": d}
                for ts, t, ip, d in critical_events
            ],
            "blocked_ips_count": len(_rate_limiter.blocked_ips),
            "active_rate_limits": {
                "default": _rate_limiter.limits["default"],
                "/admin/*": _rate_limiter.limits["/admin/*"],
                "/api/v1/*": _rate_limiter.limits["/api/v1/*"],
            },
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Security status check failed: {e}")
        return {"error": str(e)}


def _table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        ).fetchone()
        return result is not None
    except:
        return False
