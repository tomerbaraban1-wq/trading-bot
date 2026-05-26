"""
Security API Endpoints
=======================

Endpoints for security management:
- Status dashboard
- Audit logs
- 2FA setup/verification
- IP management
- Configuration check
"""

import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse, HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/security", tags=["security"])


def _verify_admin_key(provided_key: Optional[str]) -> bool:
    """Verify admin API key."""
    if not provided_key:
        return False
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected:
        return False
    return hmac.compare_digest(provided_key, expected)


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY STATUS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status")
async def security_status(x_admin_key: Optional[str] = Header(None)):
    """Get comprehensive security status."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    from security_manager import get_security_status
    return get_security_status()


@router.get("/audit-log")
async def get_audit_log(
    days: int = 7,
    severity: Optional[str] = None,
    x_admin_key: Optional[str] = Header(None)
):
    """Get security audit log."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        import database
        conn = database.get_connection()

        query = """
            SELECT timestamp, event_type, severity, source_ip, user_id,
                   endpoint, description, metadata
            FROM security_audit_log
            WHERE timestamp >= datetime('now', ?)
        """
        params = [f"-{days} days"]

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY timestamp DESC LIMIT 200"

        rows = conn.execute(query, params).fetchall()

        return {
            "period_days": days,
            "filter_severity": severity,
            "count": len(rows),
            "events": [
                {
                    "timestamp": r[0],
                    "event_type": r[1],
                    "severity": r[2],
                    "source_ip": r[3],
                    "user_id": r[4],
                    "endpoint": r[5],
                    "description": r[6],
                }
                for r in rows
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# IP MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/blocked-ips")
async def get_blocked_ips(x_admin_key: Optional[str] = Header(None)):
    """Get list of currently blocked IPs."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    from security_manager import _rate_limiter
    import time

    blocked = [
        {
            "ip": ip,
            "blocked_until": datetime.fromtimestamp(unblock_ts, timezone.utc).isoformat(),
            "remaining_seconds": int(unblock_ts - time.time()),
        }
        for ip, unblock_ts in _rate_limiter.blocked_ips.items()
        if unblock_ts > time.time()
    ]

    return {"blocked_count": len(blocked), "blocked_ips": blocked}


@router.post("/unblock-ip")
async def unblock_ip(request: Request, x_admin_key: Optional[str] = Header(None)):
    """Unblock a specific IP."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
        ip = data.get("ip")
        if not ip:
            raise HTTPException(status_code=400, detail="IP required")

        from security_manager import _rate_limiter, log_security_event, SecurityEventType

        if ip in _rate_limiter.blocked_ips:
            del _rate_limiter.blocked_ips[ip]
            log_security_event(
                event_type=SecurityEventType.ADMIN_ACTION,
                severity="info",
                description=f"Admin unblocked IP {ip}",
            )
            return {"success": True, "ip": ip}

        return {"success": False, "reason": "IP not in block list"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/block-ip")
async def block_ip(request: Request, x_admin_key: Optional[str] = Header(None)):
    """Manually block an IP."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
        ip = data.get("ip")
        duration_seconds = data.get("duration_seconds", 3600)

        if not ip:
            raise HTTPException(status_code=400, detail="IP required")

        from security_manager import _rate_limiter, log_security_event, SecurityEventType

        _rate_limiter.block_ip(ip, duration_seconds)
        log_security_event(
            event_type=SecurityEventType.ADMIN_ACTION,
            severity="info",
            description=f"Admin manually blocked IP {ip} for {duration_seconds}s",
        )

        return {
            "success": True,
            "ip": ip,
            "duration_seconds": duration_seconds,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2FA ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/2fa/setup")
async def setup_two_factor(request: Request, x_admin_key: Optional[str] = Header(None)):
    """
    Set up 2FA for a user.

    Body: {"user_id": "...", "email": "..."}
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
        user_id = data.get("user_id", "admin")
        email = data.get("email", "admin@tradingbot.local")

        from two_factor_auth import setup_2fa
        result = setup_2fa(user_id, email)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/2fa/verify")
async def verify_two_factor(request: Request):
    """
    Verify a 2FA code.

    Body: {"user_id": "...", "code": "123456"}

    No admin key required - this is the auth endpoint.
    """
    try:
        data = await request.json()
        user_id = data.get("user_id")
        code = data.get("code")

        if not user_id or not code:
            raise HTTPException(status_code=400, detail="user_id and code required")

        from two_factor_auth import verify_2fa
        from security_manager import _brute_force, log_security_event, SecurityEventType

        result = verify_2fa(user_id, code)

        # Get client IP for brute force tracking
        client_ip = "unknown"
        if hasattr(request, "client") and request.client:
            client_ip = request.client.host

        if result["valid"]:
            _brute_force.record_success(user_id)
            log_security_event(
                event_type=SecurityEventType.LOGIN_SUCCESS,
                severity="info",
                source_ip=client_ip,
                user_id=user_id,
                description=f"2FA verified ({result.get('type', 'totp')})",
            )
        else:
            brute_check = _brute_force.record_failure(user_id, client_ip)
            log_security_event(
                event_type=SecurityEventType.LOGIN_FAILED,
                severity="warning" if not brute_check["blocked"] else "critical",
                source_ip=client_ip,
                user_id=user_id,
                description=f"2FA verification failed: {result.get('reason', 'invalid')}",
            )

            if brute_check["blocked"]:
                raise HTTPException(status_code=429, detail="Too many failed attempts - blocked")

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG CHECK
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/config-check")
async def check_security_config(x_admin_key: Optional[str] = Header(None)):
    """Check security configuration completeness."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    from security_manager import check_secrets_rotation

    rotation_info = check_secrets_rotation()

    # Check various security configurations
    checks = {
        "encryption_key_set": bool(os.getenv("ENCRYPTION_KEY")),
        "admin_key_set": bool(os.getenv("ADMIN_API_KEY")),
        "webhook_secret_set": bool(os.getenv("WEBHOOK_SECRET")),
        "license_secret_set": bool(os.getenv("LICENSE_SECRET_KEY")),
        "analytics_key_set": bool(os.getenv("ANALYTICS_API_KEY")),
        "hardened_durability": os.getenv("HARDENED_DURABILITY", "false").lower() == "true",
        "render_url_set": bool(os.getenv("RENDER_EXTERNAL_URL")),
        "tls_enabled": True,  # Render enforces TLS
    }

    # Calculate score
    score = sum(1 for v in checks.values() if v) / len(checks) * 100

    # Recommendations
    recommendations = []
    if not checks["encryption_key_set"]:
        recommendations.append({
            "severity": "critical",
            "issue": "ENCRYPTION_KEY not set",
            "action": "Generate with: python -c 'import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())'",
        })
    if not checks["admin_key_set"]:
        recommendations.append({
            "severity": "critical",
            "issue": "ADMIN_API_KEY not set",
            "action": "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'",
        })
    if not checks["webhook_secret_set"]:
        recommendations.append({
            "severity": "high",
            "issue": "WEBHOOK_SECRET not set",
            "action": "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'",
        })
    if not checks["analytics_key_set"]:
        recommendations.append({
            "severity": "medium",
            "issue": "ANALYTICS_API_KEY not set - API publicly accessible",
            "action": "Set to require authentication for /api/v1/* endpoints",
        })

    return {
        "security_score": score,
        "score_label": (
            "🟢 Excellent" if score >= 90 else
            "🟡 Good" if score >= 70 else
            "🟠 Needs improvement" if score >= 50 else
            "🔴 Critical"
        ),
        "configuration_checks": checks,
        "secret_rotation_status": [
            {
                "name": s.name,
                "rotation_recommended_after_days": s.rotation_recommended_after_days,
                "set": True,
            }
            for s in rotation_info
        ],
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY DASHBOARD HTML
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def security_dashboard(x_admin_key: Optional[str] = Header(None)):
    """Visual security dashboard."""
    if not _verify_admin_key(x_admin_key):
        return HTMLResponse(
            content="<html><body><h1>401 Unauthorized</h1><p>Provide X-Admin-Key header</p></body></html>",
            status_code=401
        )

    from security_manager import get_security_status

    status = get_security_status()

    if "error" in status:
        return HTMLResponse(content=f"<html><body><h1>Error</h1><p>{status['error']}</p></body></html>")

    # Build HTML
    score = status.get("security_score", 0)
    score_color = "#10b981" if score >= 90 else "#f59e0b" if score >= 70 else "#ef4444"

    recommendations_html = ""
    for rec in status.get("recommendations", []):
        recommendations_html += f"<li>{rec}</li>"

    blocked_ips_count = status.get("blocked_ips_count", 0)
    events_html = ""
    for event in status.get("events_24h", [])[:10]:
        sev_color = "#ef4444" if event["severity"] == "critical" else "#f59e0b" if event["severity"] == "warning" else "#10b981"
        events_html += f"""
        <div style="padding: 10px; background: rgba(255,255,255,0.05); margin: 5px 0; border-left: 3px solid {sev_color};">
            <strong>{event['type']}</strong>: {event['count']} times
            <span style="float: right; color: {sev_color};">{event['severity'].upper()}</span>
        </div>
        """

    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>🛡️ Security Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #1e3a8a, #1e1b4b); color: white; padding: 20px; margin: 0; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 32px; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
        .card {{ background: rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; backdrop-filter: blur(10px); }}
        .score-circle {{
            width: 120px; height: 120px; border-radius: 50%;
            background: conic-gradient({score_color} {score * 3.6}deg, rgba(255,255,255,0.1) 0);
            display: flex; align-items: center; justify-content: center;
            margin: 20px auto;
            font-size: 32px; font-weight: bold;
        }}
        ul {{ list-style: none; padding: 0; }}
        ul li {{ padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        h2 {{ font-size: 18px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
        .stat-value {{ font-size: 36px; font-weight: bold; color: {score_color}; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Security Dashboard</h1>
        <div class="grid">
            <div class="card">
                <h2>Security Score</h2>
                <div class="score-circle">{score:.0f}</div>
                <p style="text-align: center;">{status.get('score_label', '')}</p>
            </div>

            <div class="card">
                <h2>Blocked IPs (24h)</h2>
                <div class="stat-value">{blocked_ips_count}</div>
                <p>Currently blocked</p>
            </div>

            <div class="card">
                <h2>Critical Events (7d)</h2>
                <div class="stat-value">{len(status.get('critical_events_7d', []))}</div>
                <p>High severity events</p>
            </div>

            <div class="card" style="grid-column: span 2;">
                <h2>Recent Events (24h)</h2>
                {events_html if events_html else '<p>No security events</p>'}
            </div>

            <div class="card">
                <h2>Recommendations</h2>
                <ul>
                    {recommendations_html if recommendations_html else '<li>✅ All security recommendations followed!</li>'}
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
""")
