"""
Rental Admin Endpoints
======================

FastAPI endpoints for managing rental subscribers.
Provides admin interface for the rental/SaaS system.

Endpoints:
- POST /admin/subscriber/create - Create new subscriber
- GET /admin/subscriber/{id} - Get subscriber details
- GET /admin/subscribers - List all subscribers
- POST /admin/subscriber/{id}/renew - Renew subscription
- POST /admin/subscriber/{id}/upgrade - Upgrade tier
- POST /admin/subscriber/{id}/cancel - Cancel subscription
- GET /admin/revenue - Revenue report
- POST /admin/validate-license - Validate license key
- GET /admin/expirations - Check expirations

All endpoints require ADMIN_API_KEY in X-Admin-Key header.
"""

import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse

from rental_system import (
    SubscriptionTier, SUBSCRIPTION_PLANS,
    create_subscriber, get_subscriber, list_subscribers,
    renew_subscription, upgrade_subscription, cancel_subscription,
    validate_license_key, get_revenue_report, check_expirations,
    check_usage_limits, get_rental_status, is_rental_mode_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["rental"])


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def _verify_admin_key(provided_key: Optional[str]) -> bool:
    """Verify the admin API key using timing-safe comparison."""
    if not provided_key:
        return False

    expected_key = os.getenv("ADMIN_API_KEY", "")
    if not expected_key:
        logger.warning("ADMIN_API_KEY not configured")
        return False

    return hmac.compare_digest(provided_key, expected_key)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_status(x_admin_key: Optional[str] = Header(None)):
    """Get current rental system status."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return get_rental_status()


# ─────────────────────────────────────────────────────────────────────────────
# SUBSCRIBER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/subscriber/create")
async def create_new_subscriber(request: Request, x_admin_key: Optional[str] = Header(None)):
    """
    Create a new subscriber.

    Body:
        {
            "email": "user@example.com",
            "full_name": "John Doe",
            "tier": "basic" | "pro" | "enterprise" | "trial",
            "duration_days": 30,
            "payment_method": "stripe" | "paypal" | "manual",
            "telegram_chat_id": "12345"
        }
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()

        email = data.get("email", "").strip().lower()
        full_name = data.get("full_name", "")
        tier_str = data.get("tier", "trial").lower()
        duration_days = data.get("duration_days", 30)
        payment_method = data.get("payment_method", "manual")
        telegram_chat_id = data.get("telegram_chat_id")

        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Valid email required")

        try:
            tier = SubscriptionTier(tier_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier_str}")

        subscriber = create_subscriber(
            email=email,
            full_name=full_name,
            tier=tier,
            duration_days=duration_days,
            payment_method=payment_method,
            telegram_chat_id=telegram_chat_id,
        )

        plan = SUBSCRIPTION_PLANS[tier]

        return JSONResponse({
            "success": True,
            "subscriber": {
                "subscriber_id": subscriber.subscriber_id,
                "email": subscriber.email,
                "full_name": subscriber.full_name,
                "tier": subscriber.tier.value,
                "license_key": subscriber.license_key,
                "expires_at": subscriber.expires_at,
                "status": subscriber.status,
            },
            "plan": {
                "name": plan.name,
                "monthly_price": plan.monthly_price_usd,
                "max_budget": plan.max_budget_usd,
                "features": plan.features,
            },
            "next_steps": [
                "1. Send license key to subscriber",
                "2. Subscriber adds key to their .env: BOT_LICENSE_KEY=" + subscriber.license_key,
                "3. Bot will validate on startup",
            ],
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create subscriber: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscriber/{subscriber_id}")
async def get_subscriber_info(subscriber_id: str, x_admin_key: Optional[str] = Header(None)):
    """Get subscriber details by ID."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    sub = get_subscriber(subscriber_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscriber not found")

    plan = SUBSCRIPTION_PLANS[sub.tier]

    return {
        "subscriber_id": sub.subscriber_id,
        "email": sub.email,
        "full_name": sub.full_name,
        "tier": sub.tier.value,
        "license_key": sub.license_key,
        "created_at": sub.created_at,
        "activated_at": sub.activated_at,
        "expires_at": sub.expires_at,
        "status": sub.status,
        "payment_method": sub.payment_method,
        "total_revenue_usd": sub.total_revenue_usd,
        "auto_renew": sub.auto_renew,
        "plan": {
            "name": plan.name,
            "price": plan.monthly_price_usd,
            "max_budget": plan.max_budget_usd,
            "features": plan.features,
        },
    }


@router.get("/subscribers")
async def list_all_subscribers(
    status: Optional[str] = None,
    x_admin_key: Optional[str] = Header(None)
):
    """
    List all subscribers, optionally filtered by status.

    Query params:
        status: 'active' | 'trial' | 'expired' | 'cancelled'
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    subscribers = list_subscribers(status=status)

    return {
        "total": len(subscribers),
        "filter": status or "all",
        "subscribers": [
            {
                "subscriber_id": s.subscriber_id,
                "email": s.email,
                "full_name": s.full_name,
                "tier": s.tier.value,
                "status": s.status,
                "expires_at": s.expires_at,
                "total_revenue_usd": s.total_revenue_usd,
            }
            for s in subscribers
        ],
    }


@router.post("/subscriber/{subscriber_id}/renew")
async def renew_subscriber(
    subscriber_id: str,
    request: Request,
    x_admin_key: Optional[str] = Header(None)
):
    """
    Renew a subscriber's subscription.

    Body: {"duration_days": 30}
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
        duration_days = data.get("duration_days", 30)

        if renew_subscription(subscriber_id, duration_days):
            sub = get_subscriber(subscriber_id)
            return {
                "success": True,
                "subscriber_id": subscriber_id,
                "new_expiry": sub.expires_at if sub else None,
                "duration_days": duration_days,
            }
        else:
            raise HTTPException(status_code=400, detail="Renewal failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Renewal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscriber/{subscriber_id}/upgrade")
async def upgrade_subscriber(
    subscriber_id: str,
    request: Request,
    x_admin_key: Optional[str] = Header(None)
):
    """
    Upgrade subscriber to a higher tier.

    Body: {"tier": "pro" | "enterprise"}
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
        tier_str = data.get("tier", "").lower()

        try:
            new_tier = SubscriptionTier(tier_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier_str}")

        if upgrade_subscription(subscriber_id, new_tier):
            sub = get_subscriber(subscriber_id)
            return {
                "success": True,
                "subscriber_id": subscriber_id,
                "new_tier": new_tier.value,
                "new_license_key": sub.license_key if sub else None,
            }
        else:
            raise HTTPException(status_code=400, detail="Upgrade failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upgrade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscriber/{subscriber_id}/cancel")
async def cancel_subscriber(
    subscriber_id: str,
    request: Request,
    x_admin_key: Optional[str] = Header(None)
):
    """
    Cancel a subscriber.

    Body: {"reason": "user_request"}
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        try:
            data = await request.json()
            reason = data.get("reason", "admin_action")
        except Exception:
            reason = "admin_action"

        if cancel_subscription(subscriber_id, reason):
            return {
                "success": True,
                "subscriber_id": subscriber_id,
                "status": "cancelled",
                "reason": reason,
            }
        else:
            raise HTTPException(status_code=400, detail="Cancellation failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# REVENUE & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/revenue")
async def revenue_report(x_admin_key: Optional[str] = Header(None)):
    """Get comprehensive revenue and analytics report."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return get_revenue_report()


@router.get("/expirations")
async def check_subscription_expirations(x_admin_key: Optional[str] = Header(None)):
    """Run expiration check on all subscribers."""
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return await check_expirations()


# ─────────────────────────────────────────────────────────────────────────────
# LICENSE VALIDATION (PUBLIC - used by deployed bots)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/validate-license")
async def validate_license(request: Request):
    """
    Public endpoint - deployed bots call this to validate their license.

    Body:
        {
            "license_key": "PRO-X-...",
            "subscriber_id": "..."
        }
    """
    try:
        data = await request.json()
        license_key = data.get("license_key", "")
        subscriber_id = data.get("subscriber_id", "")

        if not license_key or not subscriber_id:
            raise HTTPException(status_code=400, detail="Missing license_key or subscriber_id")

        result = validate_license_key(license_key, subscriber_id)

        if result["valid"]:
            # Get plan details
            sub = get_subscriber(subscriber_id)
            if sub:
                plan = SUBSCRIPTION_PLANS[sub.tier]
                result["plan"] = {
                    "name": plan.name,
                    "max_budget": plan.max_budget_usd,
                    "max_positions": plan.max_positions,
                    "max_daily_trades": plan.max_daily_trades,
                    "features": {
                        "sentiment_analysis": plan.sentiment_analysis,
                        "ml_predictions": plan.ml_predictions,
                        "advanced_analytics": plan.advanced_analytics,
                        "priority_support": plan.priority_support,
                        "custom_strategies": plan.custom_strategies,
                    },
                }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"License validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PRICING/PLANS (PUBLIC)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/plans")
async def get_subscription_plans():
    """Public endpoint - shows available subscription plans (for marketing/website)."""
    return {
        "plans": [
            {
                "tier": tier.value,
                "name": plan.name,
                "monthly_price_usd": plan.monthly_price_usd,
                "annual_price_usd": plan.monthly_price_usd * 10,  # 2 months free
                "limits": {
                    "max_budget_usd": plan.max_budget_usd,
                    "max_positions": plan.max_positions,
                    "max_daily_trades": plan.max_daily_trades,
                },
                "features": plan.features,
                "includes": {
                    "sentiment_analysis": plan.sentiment_analysis,
                    "ml_predictions": plan.ml_predictions,
                    "advanced_analytics": plan.advanced_analytics,
                    "priority_support": plan.priority_support,
                    "custom_strategies": plan.custom_strategies,
                },
                "popular": tier == SubscriptionTier.PRO,
                "best_value": tier == SubscriptionTier.ENTERPRISE,
            }
            for tier, plan in SUBSCRIPTION_PLANS.items()
        ],
        "currency": "USD",
        "trial": {
            "available": True,
            "duration_days": 7,
            "credit_card_required": False,
        },
    }
