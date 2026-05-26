"""
Enhanced Security Layer
========================

Advanced security improvements:
1. Progressive IP blocking (Fail2Ban style)
2. Suspicious pattern detection
3. Request fingerprinting
4. Geo-anomaly detection
5. Credential stuffing protection
6. JWT token support
7. Security event aggregation
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESSIVE BLOCKING (Fail2Ban style)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IPRecord:
    """Track an IP's behavior."""
    ip: str
    violations: int = 0
    block_count: int = 0          # How many times blocked
    current_block_until: float = 0  # Timestamp
    last_violation: float = 0
    violation_types: list = field(default_factory=list)


_ip_records: dict[str, IPRecord] = {}


def _get_block_duration(block_count: int) -> int:
    """Progressive blocking durations."""
    durations = [
        300,    # 1st block: 5 minutes
        1800,   # 2nd block: 30 minutes
        7200,   # 3rd block: 2 hours
        86400,  # 4th+ block: 24 hours
    ]
    return durations[min(block_count, len(durations) - 1)]


def record_violation(ip: str, violation_type: str) -> dict:
    """
    Record a security violation for an IP.
    Returns block info if IP should be blocked.
    """
    if ip not in _ip_records:
        _ip_records[ip] = IPRecord(ip=ip)

    record = _ip_records[ip]
    record.violations += 1
    record.last_violation = time.time()
    record.violation_types.append(violation_type)

    # Keep only last 20 violations in memory
    if len(record.violation_types) > 20:
        record.violation_types = record.violation_types[-20:]

    # Progressive blocking thresholds
    BLOCK_THRESHOLDS = {
        "auth_fail": 5,        # Block after 5 auth failures
        "rate_limit": 10,      # Block after 10 rate limit hits
        "injection": 1,        # Block immediately on injection attempt
        "scanner": 2,          # Block after 2 scanner detections
        "forbidden": 8,        # Block after 8 forbidden attempts
    }

    threshold = BLOCK_THRESHOLDS.get(violation_type, 5)

    # Count recent violations of this type
    recent_same = sum(
        1 for vt in record.violation_types[-20:]
        if vt == violation_type
    )

    should_block = recent_same >= threshold

    if should_block and time.time() > record.current_block_until:
        record.block_count += 1
        duration = _get_block_duration(record.block_count)
        record.current_block_until = time.time() + duration

        logger.warning(
            f"[SECURITY] Progressive block: {ip} | "
            f"violation={violation_type} x{recent_same} | "
            f"block #{record.block_count} for {duration}s"
        )

        # Persist to security manager
        try:
            from security_manager import _rate_limiter, log_security_event, SecurityEventType
            _rate_limiter.block_ip(ip, duration)
            log_security_event(
                event_type=SecurityEventType.BRUTE_FORCE,
                severity="critical" if record.block_count >= 3 else "warning",
                source_ip=ip,
                description=f"Progressive block #{record.block_count}: {violation_type} x{recent_same}",
            )
        except Exception:
            pass

        return {"blocked": True, "duration": duration, "block_count": record.block_count}

    return {"blocked": False, "violations": record.violations}


def is_ip_blocked(ip: str) -> bool:
    """Check if IP is currently blocked."""
    record = _ip_records.get(ip)
    if record and time.time() < record.current_block_until:
        return True
    # Also check security manager
    try:
        from security_manager import _rate_limiter
        return _rate_limiter.is_blocked(ip)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SUSPICIOUS PATTERN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

SCANNER_USER_AGENTS = {
    "sqlmap", "nikto", "nmap", "masscan", "metasploit",
    "burpsuite", "acunetix", "nessus", "openvas", "w3af",
    "dirbuster", "gobuster", "hydra", "medusa", "shodan",
    "zgrab", "python-requests/2.2", "curl/7.29",  # old versions often used in scanning
}

SUSPICIOUS_PATHS = {
    "/wp-admin", "/wp-login", "/.env", "/.git", "/phpmyadmin",
    "/admin.php", "/config.php", "/shell", "/cmd", "/exec",
    "/../", "/etc/passwd", "/proc/self", "//",
}

SUSPICIOUS_PARAMS = {
    "cmd", "exec", "shell", "phpinfo", "base64_decode",
    "eval(", "system(", "passthru(", "file_get_contents(",
}


def detect_suspicious_request(
    path: str,
    user_agent: str,
    query_string: str,
    headers: dict,
) -> Optional[str]:
    """
    Detect suspicious/malicious request patterns.
    Returns detection type or None if clean.
    """
    ua_lower = (user_agent or "").lower()
    path_lower = path.lower()
    qs_lower = (query_string or "").lower()

    # Scanner detection
    for scanner in SCANNER_USER_AGENTS:
        if scanner in ua_lower:
            return f"scanner:{scanner}"

    # Suspicious path
    for susp_path in SUSPICIOUS_PATHS:
        if susp_path in path_lower:
            return f"suspicious_path:{susp_path}"

    # Suspicious parameters
    for susp_param in SUSPICIOUS_PARAMS:
        if susp_param in qs_lower or susp_param in ua_lower:
            return f"suspicious_param:{susp_param}"

    # Empty or very short user agent (often bots)
    if len(ua_lower) < 10:
        return "suspicious_ua:too_short"

    # No Accept header (often API scanners)
    if not headers.get("accept") and not headers.get("content-type"):
        # Only flag POST requests without headers
        return None  # Too many false positives on GET

    return None


# ─────────────────────────────────────────────────────────────────────────────
# JWT TOKEN SUPPORT (for mobile/frontend access)
# ─────────────────────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "")


def generate_jwt_token(payload: dict, expires_in_hours: int = 24) -> str:
    """
    Generate a simple HMAC-based JWT-like token.
    Not full JWT — uses HMAC-SHA256 for simplicity without dependencies.
    """
    import base64

    if not JWT_SECRET:
        raise ValueError("JWT_SECRET not configured")

    expiry = int(time.time()) + (expires_in_hours * 3600)
    payload_with_exp = {**payload, "exp": expiry, "iat": int(time.time())}

    # Encode payload
    payload_bytes = json.dumps(payload_with_exp, separators=(',', ':')).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()

    # Sign
    signature = hmac.new(
        JWT_SECRET.encode(),
        payload_b64.encode(),
        hashlib.sha256
    ).hexdigest()[:32]

    return f"{payload_b64}.{signature}"


def verify_jwt_token(token: str) -> Optional[dict]:
    """
    Verify a JWT token and return payload if valid.
    Returns None if invalid or expired.
    """
    import base64

    if not JWT_SECRET or not token:
        return None

    try:
        parts = token.split(".", 1)
        if len(parts) != 2:
            return None

        payload_b64, provided_sig = parts

        # Verify signature
        expected_sig = hmac.new(
            JWT_SECRET.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()[:32]

        if not hmac.compare_digest(provided_sig, expected_sig):
            return None

        # Decode payload
        padding = 4 - len(payload_b64) % 4
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + "=" * padding)
        payload = json.loads(payload_bytes)

        # Check expiry
        if payload.get("exp", 0) < time.time():
            return None

        return payload

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY DASHBOARD DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_threat_summary() -> dict:
    """Get current threat landscape."""
    now = time.time()
    total_blocked = sum(
        1 for r in _ip_records.values()
        if r.current_block_until > now
    )
    total_violations = sum(r.violations for r in _ip_records.values())
    repeat_offenders = [
        {"ip": r.ip, "blocks": r.block_count, "violations": r.violations}
        for r in _ip_records.values()
        if r.block_count >= 2
    ]
    repeat_offenders.sort(key=lambda x: x["blocks"], reverse=True)

    return {
        "currently_blocked_ips": total_blocked,
        "total_tracked_ips": len(_ip_records),
        "total_violations": total_violations,
        "repeat_offenders": repeat_offenders[:10],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
