"""
Security Middleware for FastAPI
=================================

Centralized security middleware that:
1. Rate limits all requests
2. Detects injection attempts
3. Logs security events
4. Adds security headers
5. Validates request integrity
6. Blocks suspicious IPs
"""

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from security_manager import (
    _rate_limiter, _brute_force,
    log_security_event, SecurityEventType,
    inspect_request, detect_injection_attempt,
)

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware.

    Applied to all requests automatically.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.public_endpoints = {
            "/ping",
            "/health",
            "/",
            "/dashboard/advanced",
            "/health/dashboard",
            "/admin/plans",  # Public pricing page
        }

    async def dispatch(self, request: Request, call_next):
        """Process request through security checks."""
        start_time = time.time()

        # Get client IP
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        # ── Localhost bypass — trust ONLY the real TCP peer, never a header ──
        # Internal keep-alive pings and health checks come from localhost, and we
        # trust them. BUT base that on request.client.host (the actual TCP peer),
        # NOT client_ip — client_ip comes from _get_client_ip(), which honours the
        # X-Forwarded-For header. Trusting that header would let a remote attacker
        # send "X-Forwarded-For: 127.0.0.1" to bypass EVERY check below (rate limit,
        # injection detection, IP blocking). The TCP peer cannot be spoofed.
        _TRUSTED_LOCAL = {"127.0.0.1", "::1", "localhost"}
        _real_peer = request.client.host if request.client else ""
        if _real_peer in _TRUSTED_LOCAL:
            return await call_next(request)

        try:
            # ── 0. Progressive blocking (enhanced) ───────────────────────
            try:
                from security_enhanced import is_ip_blocked as _enh_blocked
                if _enh_blocked(client_ip):
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Temporarily blocked"},
                        headers=self._get_security_headers(),
                    )
            except Exception:
                pass

            # ── 0.5 Suspicious pattern detection ─────────────────────────
            try:
                from security_enhanced import detect_suspicious_request, record_violation
                ua = request.headers.get("user-agent", "")
                qs = str(request.query_params)
                pattern = detect_suspicious_request(endpoint, ua, qs, dict(request.headers))
                if pattern:
                    log_security_event(
                        event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                        severity="warning",
                        source_ip=client_ip,
                        endpoint=endpoint,
                        description=f"Suspicious pattern: {pattern}",
                    )
                    result = record_violation(client_ip, "scanner")
                    if result.get("blocked"):
                        return JSONResponse(
                            status_code=403,
                            content={"error": "Forbidden"},
                            headers=self._get_security_headers(),
                        )
            except Exception:
                pass

            # ── 1. Check if IP is blocked ─────────────────────────────────
            if _rate_limiter.is_blocked(client_ip):
                log_security_event(
                    event_type=SecurityEventType.UNAUTHORIZED,
                    severity="warning",
                    source_ip=client_ip,
                    endpoint=endpoint,
                    description="Blocked IP attempted access",
                )
                return JSONResponse(
                    status_code=429,
                    content={"error": "IP temporarily blocked", "retry_after": 600},
                    headers=self._get_security_headers(),
                )

            # ── 2. Rate limiting ──────────────────────────────────────────
            rate_check = _rate_limiter.check_rate_limit(
                ip=client_ip,
                endpoint=endpoint,
            )

            if not rate_check["allowed"]:
                # Record violation for progressive blocking
                try:
                    from security_enhanced import record_violation as _rec_v
                    _rec_v(client_ip, "rate_limit")
                except Exception:
                    pass
                log_security_event(
                    event_type=SecurityEventType.RATE_LIMIT_HIT,
                    severity="warning",
                    source_ip=client_ip,
                    endpoint=endpoint,
                    description=rate_check["reason"],
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": rate_check["reason"],
                        "retry_after": rate_check["retry_after"],
                    },
                    headers={
                        **self._get_security_headers(),
                        "Retry-After": str(rate_check["retry_after"]),
                    },
                )

            # ── 3. Inspect request for attacks ────────────────────────────
            body_data = None
            if request.method in ("POST", "PUT", "PATCH") and endpoint not in self.public_endpoints:
                try:
                    body_bytes = await request.body()
                    if body_bytes:
                        body_data = json.loads(body_bytes.decode())

                    async def receive():
                        return {"type": "http.request", "body": body_bytes}
                    request._receive = receive

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Body parsing failed: {e}")

            # Check headers — skip safe/standard headers to avoid false positives
            # (e.g. Accept: */*  contains */ which matches SQL comment pattern)
            try:
                from security_manager import _SAFE_HEADERS as _sh
                _skip_headers = _sh
            except ImportError:
                _skip_headers = frozenset()

            for header_name, header_value in request.headers.items():
                if header_name.lower() in _skip_headers:
                    continue
                attack = detect_injection_attempt(header_value)
                if attack:
                    log_security_event(
                        event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                        severity="critical",
                        source_ip=client_ip,
                        endpoint=endpoint,
                        description=f"{attack} in header {header_name}",
                    )
                    try:
                        from security_enhanced import record_violation as _rv
                        _rv(client_ip, "injection")
                    except Exception:
                        pass
                    _rate_limiter.block_ip(client_ip, 3600)
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid request"},
                        headers=self._get_security_headers(),
                    )

            # Check URL params for injection
            for param_name, param_value in request.query_params.items():
                attack = detect_injection_attempt(param_value)
                if attack:
                    log_security_event(
                        event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                        severity="critical",
                        source_ip=client_ip,
                        endpoint=endpoint,
                        description=f"{attack} in query param {param_name}",
                    )
                    _rate_limiter.block_ip(client_ip, 3600)
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid request parameters"},
                        headers=self._get_security_headers(),
                    )

            # ── 4. Process request ───────────────────────────────────────
            response = await call_next(request)

            # ── 5. Add security headers ──────────────────────────────────
            for key, value in self._get_security_headers().items():
                response.headers[key] = value

            # ── 6. Log slow requests (potential DoS) ─────────────────────
            duration_ms = (time.time() - start_time) * 1000
            if duration_ms > 5000:  # > 5 seconds
                logger.warning(
                    f"[SECURITY] Slow request from {client_ip}: {endpoint} took {duration_ms:.0f}ms"
                )

            return response

        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
                headers=self._get_security_headers(),
            )

    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP, considering proxies."""
        # Render uses X-Forwarded-For
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Take first IP (original client)
            return forwarded.split(",")[0].strip()

        # Cloudflare
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip

        # Direct connection
        if request.client:
            return request.client.host

        return "unknown"

    def _get_security_headers(self) -> dict:
        """Get security headers to add to responses."""
        return {
            # Prevent XSS
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",

            # HSTS (HTTPS Strict Transport Security)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",

            # CSP - Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),

            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",

            # Permissions policy (replaces Feature-Policy)
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",

            # Remove server identification
            "Server": "Bot",
        }
