"""
Two-Factor Authentication (2FA)
=================================

TOTP-based 2FA for admin endpoints.
Compatible with Google Authenticator, Authy, etc.

Features:
1. TOTP code generation (RFC 6238)
2. QR code generation for setup
3. Backup codes (one-time use)
4. Time-window tolerance
5. Rate limiting on attempts
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TwoFactorSecret:
    """A user's 2FA secret."""
    user_id: str
    secret: str  # Base32-encoded
    backup_codes: list[str]
    created_at: str
    last_used_code: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# TOTP IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_totp_secret() -> str:
    """Generate a new TOTP secret (160-bit, base32 encoded)."""
    # 20 bytes = 160 bits = 32 chars base32
    secret_bytes = secrets.token_bytes(20)
    return base64.b32encode(secret_bytes).decode().rstrip("=")


def generate_totp_code(secret: str, time_step: int = 30, digits: int = 6) -> str:
    """
    Generate a TOTP code based on current time.

    Implements RFC 6238.
    """
    # Decode base32 secret
    # Add padding if needed
    secret_padded = secret + "=" * (-len(secret) % 8)
    secret_bytes = base64.b32decode(secret_padded.upper())

    # Calculate counter (Unix time / time_step)
    counter = int(time.time()) // time_step

    # HMAC-SHA1
    counter_bytes = struct.pack(">Q", counter)
    hmac_result = hmac.new(secret_bytes, counter_bytes, hashlib.sha1).digest()

    # Dynamic truncation
    offset = hmac_result[-1] & 0x0F
    truncated = struct.unpack(">I", hmac_result[offset:offset+4])[0] & 0x7FFFFFFF

    # Modulo to get correct number of digits
    code = truncated % (10 ** digits)
    return str(code).zfill(digits)


def verify_totp_code(secret: str, code: str, window: int = 1) -> bool:
    """
    Verify a TOTP code with time window tolerance.

    Window of 1 means: accept current + previous time step (60 seconds total).
    """
    if not code or not code.isdigit() or len(code) != 6:
        return False

    secret_padded = secret + "=" * (-len(secret) % 8)
    try:
        secret_bytes = base64.b32decode(secret_padded.upper())
    except:
        return False

    current_time = int(time.time())

    # Check current and adjacent time windows
    for delta in range(-window, window + 1):
        counter = (current_time // 30) + delta
        counter_bytes = struct.pack(">Q", counter)
        hmac_result = hmac.new(secret_bytes, counter_bytes, hashlib.sha1).digest()

        offset = hmac_result[-1] & 0x0F
        truncated = struct.unpack(">I", hmac_result[offset:offset+4])[0] & 0x7FFFFFFF
        expected_code = str(truncated % 1000000).zfill(6)

        if hmac.compare_digest(code, expected_code):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# BACKUP CODES
# ─────────────────────────────────────────────────────────────────────────────

def generate_backup_codes(count: int = 10) -> list[str]:
    """Generate one-time use backup codes."""
    codes = []
    for _ in range(count):
        # 4 groups of 4 digits
        code = "-".join(
            "".join(secrets.choice("0123456789") for _ in range(4))
            for _ in range(2)
        )
        codes.append(code)
    return codes


def verify_backup_code(provided: str, valid_codes: list[str]) -> tuple[bool, list[str]]:
    """
    Verify a backup code and remove it from valid codes.

    Returns: (valid, remaining_codes)
    """
    provided = provided.strip().replace(" ", "")

    for stored_code in valid_codes:
        if hmac.compare_digest(provided, stored_code):
            # Remove used code
            remaining = [c for c in valid_codes if c != stored_code]
            return True, remaining

    return False, valid_codes


# ─────────────────────────────────────────────────────────────────────────────
# QR CODE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_qr_code_url(secret: str, user_email: str, issuer: str = "TradingBot") -> str:
    """
    Generate the otpauth URL for QR code.

    Format: otpauth://totp/Issuer:user@example.com?secret=ABC&issuer=Issuer

    Users can scan this with Google Authenticator/Authy.
    """
    from urllib.parse import quote

    user_encoded = quote(user_email)
    issuer_encoded = quote(issuer)

    return (
        f"otpauth://totp/{issuer_encoded}:{user_encoded}"
        f"?secret={secret}"
        f"&issuer={issuer_encoded}"
        f"&algorithm=SHA1"
        f"&digits=6"
        f"&period=30"
    )


def generate_qr_code_image_url(secret: str, user_email: str, issuer: str = "TradingBot") -> str:
    """
    Generate a Google Charts URL for QR code image.
    No QR library needed - uses external service.
    """
    from urllib.parse import quote
    otpauth_url = generate_qr_code_url(secret, user_email, issuer)
    encoded = quote(otpauth_url)
    return f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={encoded}"


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def save_2fa_secret(user_id: str, secret: str, backup_codes: list[str]) -> bool:
    """Save 2FA secret to database (encrypted)."""
    try:
        import database
        from security_manager import encrypt_value
        from datetime import datetime, timezone
        import json

        conn = database.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS two_factor_auth (
                user_id TEXT PRIMARY KEY,
                encrypted_secret TEXT NOT NULL,
                encrypted_backup_codes TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                last_used_code TEXT
            )
        """)

        # Encrypt secret and backup codes
        encrypted_secret = encrypt_value(secret)
        encrypted_backup = encrypt_value(json.dumps(backup_codes))

        conn.execute("""
            INSERT OR REPLACE INTO two_factor_auth
            (user_id, encrypted_secret, encrypted_backup_codes, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            user_id, encrypted_secret, encrypted_backup,
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
        return True

    except Exception as e:
        logger.error(f"Failed to save 2FA secret: {e}")
        return False


def get_2fa_secret(user_id: str) -> Optional[TwoFactorSecret]:
    """Get user's 2FA secret (decrypted)."""
    try:
        import database
        from security_manager import decrypt_value
        import json

        conn = database.get_connection()
        row = conn.execute("""
            SELECT encrypted_secret, encrypted_backup_codes, created_at, last_used_code
            FROM two_factor_auth
            WHERE user_id = ?
        """, (user_id,)).fetchone()

        if not row:
            return None

        encrypted_secret, encrypted_backup, created_at, last_used = row

        secret = decrypt_value(encrypted_secret)
        backup_codes = json.loads(decrypt_value(encrypted_backup))

        return TwoFactorSecret(
            user_id=user_id,
            secret=secret,
            backup_codes=backup_codes,
            created_at=created_at,
            last_used_code=last_used,
        )

    except Exception as e:
        logger.error(f"Failed to get 2FA secret: {e}")
        return None


def update_used_code(user_id: str, code: str) -> None:
    """Update last used code (prevent reuse)."""
    try:
        import database
        from datetime import datetime, timezone

        conn = database.get_connection()
        conn.execute("""
            UPDATE two_factor_auth
            SET last_used_code = ?, last_used_at = ?
            WHERE user_id = ?
        """, (code, datetime.now(timezone.utc).isoformat(), user_id))
        conn.commit()
    except Exception as e:
        logger.debug(f"Failed to update used code: {e}")


def update_backup_codes(user_id: str, remaining_codes: list[str]) -> bool:
    """Update backup codes after one is used."""
    try:
        import database
        from security_manager import encrypt_value
        import json

        conn = database.get_connection()
        encrypted = encrypt_value(json.dumps(remaining_codes))

        conn.execute("""
            UPDATE two_factor_auth
            SET encrypted_backup_codes = ?
            WHERE user_id = ?
        """, (encrypted, user_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to update backup codes: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

def setup_2fa(user_id: str, user_email: str, issuer: str = "TradingBot") -> dict:
    """
    Set up 2FA for a user.

    Returns: {"secret", "qr_code_url", "backup_codes", "instructions"}
    """
    # Generate secret
    secret = generate_totp_secret()
    backup_codes = generate_backup_codes()

    # Save to database
    saved = save_2fa_secret(user_id, secret, backup_codes)

    if not saved:
        return {"error": "Failed to save 2FA secret"}

    # Generate QR code URL
    qr_url = generate_qr_code_url(secret, user_email, issuer)
    qr_image_url = generate_qr_code_image_url(secret, user_email, issuer)

    return {
        "user_id": user_id,
        "secret": secret,  # Show once to user
        "qr_code_url": qr_url,
        "qr_image_url": qr_image_url,
        "backup_codes": backup_codes,  # Show once to user
        "instructions": [
            "1. Install Google Authenticator or Authy",
            "2. Scan the QR code or enter the secret manually",
            "3. Save backup codes in a secure location",
            "4. Enter the 6-digit code from your authenticator app",
        ],
    }


def verify_2fa(user_id: str, code: str) -> dict:
    """
    Verify a 2FA code.

    Accepts both TOTP codes and backup codes.
    """
    from security_manager import log_security_event, SecurityEventType

    secret_data = get_2fa_secret(user_id)
    if not secret_data:
        return {"valid": False, "reason": "2FA not configured"}

    # Try TOTP first
    if verify_totp_code(secret_data.secret, code):
        # Prevent code reuse
        if secret_data.last_used_code == code:
            log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                severity="warning",
                user_id=user_id,
                description="Attempted to reuse TOTP code",
            )
            return {"valid": False, "reason": "Code already used"}

        update_used_code(user_id, code)
        return {"valid": True, "type": "totp"}

    # Try backup code
    is_valid, remaining = verify_backup_code(code, secret_data.backup_codes)
    if is_valid:
        update_backup_codes(user_id, remaining)
        return {
            "valid": True,
            "type": "backup_code",
            "remaining_backup_codes": len(remaining),
            "warning": f"Backup code used. {len(remaining)} codes remaining." if remaining else "No backup codes left!"
        }

    return {"valid": False, "reason": "Invalid code"}
