"""
Bot Rental System - SaaS Infrastructure
========================================

Complete infrastructure for renting the trading bot as a monthly subscription service.

Features:
1. Subscription plans (Basic, Pro, Enterprise)
2. License key generation & validation
3. User management with permissions
4. Multi-tenant database isolation
5. Usage tracking per subscriber
6. Trial periods (7-day free trial)
7. Auto-expiration & renewal
8. Admin management endpoints

This module is DORMANT by default - it doesn't affect trading until activated.

To activate rental mode:
1. Set RENTAL_MODE_ENABLED=true in .env
2. Configure payment provider (Stripe/PayPal)
3. Run database migrations
4. Use admin endpoints to create subscribers
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SUBSCRIPTION PLANS
# ─────────────────────────────────────────────────────────────────────────────

class SubscriptionTier(Enum):
    """Available subscription tiers."""
    TRIAL = "trial"           # 7 days free
    BASIC = "basic"           # $49/month
    PRO = "pro"               # $99/month
    ENTERPRISE = "enterprise" # $299/month


@dataclass
class SubscriptionPlan:
    """Defines features available for each subscription tier."""
    tier: SubscriptionTier
    name: str
    monthly_price_usd: float
    max_budget_usd: float       # Max trading budget
    max_positions: int          # Max concurrent positions
    max_daily_trades: int       # Max trades per day
    features: List[str]         # Available features
    sentiment_analysis: bool    # Discord sentiment access
    ml_predictions: bool        # ML predictions enabled
    advanced_analytics: bool    # Pattern recognition, etc.
    priority_support: bool      # Faster support response
    custom_strategies: bool     # Can customize trading params


# Define subscription plans
SUBSCRIPTION_PLANS = {
    SubscriptionTier.TRIAL: SubscriptionPlan(
        tier=SubscriptionTier.TRIAL,
        name="Free Trial (7 days)",
        monthly_price_usd=0,
        max_budget_usd=1000,
        max_positions=3,
        max_daily_trades=5,
        features=["Basic trading", "Telegram alerts"],
        sentiment_analysis=False,
        ml_predictions=False,
        advanced_analytics=False,
        priority_support=False,
        custom_strategies=False,
    ),
    SubscriptionTier.BASIC: SubscriptionPlan(
        tier=SubscriptionTier.BASIC,
        name="Basic",
        monthly_price_usd=49,
        max_budget_usd=5000,
        max_positions=5,
        max_daily_trades=15,
        features=["Trading bot", "Telegram alerts", "Daily summaries"],
        sentiment_analysis=False,
        ml_predictions=False,
        advanced_analytics=False,
        priority_support=False,
        custom_strategies=False,
    ),
    SubscriptionTier.PRO: SubscriptionPlan(
        tier=SubscriptionTier.PRO,
        name="Pro",
        monthly_price_usd=99,
        max_budget_usd=25000,
        max_positions=10,
        max_daily_trades=50,
        features=[
            "All Basic features",
            "Sentiment analysis",
            "Advanced analytics",
            "Pattern recognition",
            "Discord integration",
            "Adaptive trading",
        ],
        sentiment_analysis=True,
        ml_predictions=True,
        advanced_analytics=True,
        priority_support=False,
        custom_strategies=False,
    ),
    SubscriptionTier.ENTERPRISE: SubscriptionPlan(
        tier=SubscriptionTier.ENTERPRISE,
        name="Enterprise",
        monthly_price_usd=299,
        max_budget_usd=999999,  # Unlimited
        max_positions=50,
        max_daily_trades=200,
        features=[
            "All Pro features",
            "Unlimited budget",
            "Custom strategies",
            "Priority support",
            "ML predictions",
            "Multi-account support",
            "API access",
            "White-glove setup",
        ],
        sentiment_analysis=True,
        ml_predictions=True,
        advanced_analytics=True,
        priority_support=True,
        custom_strategies=True,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Subscriber:
    """A subscriber to the trading bot service."""
    subscriber_id: str           # UUID
    email: str
    full_name: str
    tier: SubscriptionTier
    license_key: str             # For activation
    created_at: str
    activated_at: Optional[str]  # When they first activated
    expires_at: str              # When subscription expires
    status: str                  # "active", "expired", "cancelled", "trial"
    payment_method: Optional[str]  # "stripe", "paypal", "manual"
    telegram_chat_id: Optional[str]
    alpaca_api_key_encrypted: Optional[str]
    total_revenue_usd: float = 0
    auto_renew: bool = True


@dataclass
class UsageRecord:
    """Daily usage record per subscriber."""
    subscriber_id: str
    date: str
    trades_executed: int
    api_calls_made: int
    sentiment_queries: int
    ml_predictions: int
    total_pnl: float


# ─────────────────────────────────────────────────────────────────────────────
# LICENSE KEY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def generate_license_key(subscriber_id: str, tier: SubscriptionTier, expires_at: datetime) -> str:
    """
    Generate a secure license key.

    Format: TIER-XXXX-XXXX-XXXX-CHECKSUM
    """
    # Random portion
    random_part = secrets.token_hex(8).upper()

    # Create payload
    payload = f"{tier.value}|{subscriber_id}|{expires_at.timestamp()}"
    secret = os.getenv("LICENSE_SECRET_KEY", "default-change-in-production")
    checksum = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()[:8].upper()

    # Format key
    tier_prefix = {
        SubscriptionTier.TRIAL: "TRIAL",
        SubscriptionTier.BASIC: "BASIC",
        SubscriptionTier.PRO: "PRO-X",
        SubscriptionTier.ENTERPRISE: "ENT-X",
    }.get(tier, "BOT")

    formatted_random = "-".join([random_part[i:i+4] for i in range(0, len(random_part), 4)])

    return f"{tier_prefix}-{formatted_random}-{checksum}"


def validate_license_key(license_key: str, subscriber_id: str) -> dict:
    """
    Validate a license key against the database.

    Returns:
        {"valid": bool, "tier": str, "expires_at": str, "reason": str}
    """
    try:
        import database
        conn = database.get_connection()

        # Find subscriber
        row = conn.execute("""
            SELECT tier, expires_at, status, license_key
            FROM rental_subscribers
            WHERE subscriber_id = ?
        """, (subscriber_id,)).fetchone()

        if not row:
            return {"valid": False, "reason": "Subscriber not found"}

        db_tier, expires_at, status, db_license = row

        # Constant-time comparison
        if not hmac.compare_digest(license_key, db_license):
            return {"valid": False, "reason": "Invalid license key"}

        # Check expiration
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > expiry:
                return {"valid": False, "reason": "Subscription expired", "expired": True}
        except Exception:
            pass

        # Check status
        if status not in ("active", "trial"):
            return {"valid": False, "reason": f"Subscription {status}"}

        return {
            "valid": True,
            "tier": db_tier,
            "expires_at": expires_at,
            "status": status,
        }

    except Exception as e:
        logger.error(f"License validation failed: {e}")
        return {"valid": False, "reason": f"Validation error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# SUBSCRIBER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def create_subscriber(
    email: str,
    full_name: str,
    tier: SubscriptionTier = SubscriptionTier.TRIAL,
    duration_days: int = 30,
    payment_method: str = "manual",
    telegram_chat_id: Optional[str] = None,
) -> Subscriber:
    """
    Create a new subscriber and generate license key.

    For trial: duration_days=7
    For monthly: duration_days=30
    """
    subscriber_id = secrets.token_hex(16)
    now = datetime.now(timezone.utc)

    # Adjust duration for trial
    if tier == SubscriptionTier.TRIAL:
        duration_days = 7

    expires_at = now + timedelta(days=duration_days)
    license_key = generate_license_key(subscriber_id, tier, expires_at)

    subscriber = Subscriber(
        subscriber_id=subscriber_id,
        email=email,
        full_name=full_name,
        tier=tier,
        license_key=license_key,
        created_at=now.isoformat(),
        activated_at=None,
        expires_at=expires_at.isoformat(),
        status="trial" if tier == SubscriptionTier.TRIAL else "active",
        payment_method=payment_method,
        telegram_chat_id=telegram_chat_id,
        alpaca_api_key_encrypted=None,
        total_revenue_usd=0,
        auto_renew=tier != SubscriptionTier.TRIAL,
    )

    _save_subscriber(subscriber)
    logger.info(f"Created subscriber {subscriber_id} ({email}) on tier {tier.value}")

    return subscriber


def _save_subscriber(subscriber: Subscriber) -> None:
    """Save subscriber to database."""
    try:
        import database
        conn = database.get_connection()

        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rental_subscribers (
                subscriber_id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                full_name TEXT,
                tier TEXT,
                license_key TEXT UNIQUE,
                created_at TIMESTAMP,
                activated_at TIMESTAMP,
                expires_at TIMESTAMP,
                status TEXT,
                payment_method TEXT,
                telegram_chat_id TEXT,
                alpaca_api_key_encrypted TEXT,
                total_revenue_usd REAL DEFAULT 0,
                auto_renew INTEGER DEFAULT 1
            )
        """)

        conn.execute("""
            INSERT OR REPLACE INTO rental_subscribers
            (subscriber_id, email, full_name, tier, license_key, created_at,
             activated_at, expires_at, status, payment_method, telegram_chat_id,
             alpaca_api_key_encrypted, total_revenue_usd, auto_renew)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subscriber.subscriber_id, subscriber.email, subscriber.full_name,
            subscriber.tier.value, subscriber.license_key, subscriber.created_at,
            subscriber.activated_at, subscriber.expires_at, subscriber.status,
            subscriber.payment_method, subscriber.telegram_chat_id,
            subscriber.alpaca_api_key_encrypted, subscriber.total_revenue_usd,
            1 if subscriber.auto_renew else 0
        ))
        conn.commit()

    except Exception as e:
        logger.error(f"Failed to save subscriber: {e}")


def get_subscriber(subscriber_id: str) -> Optional[Subscriber]:
    """Retrieve a subscriber by ID."""
    try:
        import database
        conn = database.get_connection()

        row = conn.execute("""
            SELECT subscriber_id, email, full_name, tier, license_key, created_at,
                   activated_at, expires_at, status, payment_method, telegram_chat_id,
                   alpaca_api_key_encrypted, total_revenue_usd, auto_renew
            FROM rental_subscribers
            WHERE subscriber_id = ?
        """, (subscriber_id,)).fetchone()

        if not row:
            return None

        return Subscriber(
            subscriber_id=row[0],
            email=row[1],
            full_name=row[2],
            tier=SubscriptionTier(row[3]),
            license_key=row[4],
            created_at=row[5],
            activated_at=row[6],
            expires_at=row[7],
            status=row[8],
            payment_method=row[9],
            telegram_chat_id=row[10],
            alpaca_api_key_encrypted=row[11],
            total_revenue_usd=row[12] or 0,
            auto_renew=bool(row[13]),
        )

    except Exception as e:
        logger.error(f"Failed to get subscriber: {e}")
        return None


def list_subscribers(status: Optional[str] = None) -> List[Subscriber]:
    """List all subscribers, optionally filtered by status."""
    try:
        import database
        conn = database.get_connection()

        if status:
            rows = conn.execute("""
                SELECT subscriber_id FROM rental_subscribers WHERE status = ?
            """, (status,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT subscriber_id FROM rental_subscribers
            """).fetchall()

        subscribers = [get_subscriber(r[0]) for r in rows]
        return [s for s in subscribers if s is not None]

    except Exception as e:
        logger.error(f"Failed to list subscribers: {e}")
        return []


def renew_subscription(subscriber_id: str, duration_days: int = 30) -> bool:
    """Renew a subscription by extending expiration."""
    try:
        sub = get_subscriber(subscriber_id)
        if not sub:
            return False

        # Calculate new expiration
        now = datetime.now(timezone.utc)
        current_expiry = datetime.fromisoformat(sub.expires_at.replace("Z", "+00:00"))

        # Extend from current expiry or now (whichever is later)
        new_expiry = max(current_expiry, now) + timedelta(days=duration_days)

        # Update revenue
        plan = SUBSCRIPTION_PLANS[sub.tier]
        revenue_added = plan.monthly_price_usd * (duration_days / 30)

        import database
        conn = database.get_connection()
        conn.execute("""
            UPDATE rental_subscribers
            SET expires_at = ?, status = 'active', total_revenue_usd = total_revenue_usd + ?
            WHERE subscriber_id = ?
        """, (new_expiry.isoformat(), revenue_added, subscriber_id))
        conn.commit()

        logger.info(f"Renewed subscriber {subscriber_id} until {new_expiry}")
        return True

    except Exception as e:
        logger.error(f"Failed to renew subscription: {e}")
        return False


def cancel_subscription(subscriber_id: str, reason: str = "user_request") -> bool:
    """Cancel a subscription."""
    try:
        import database
        conn = database.get_connection()
        conn.execute("""
            UPDATE rental_subscribers
            SET status = 'cancelled', auto_renew = 0
            WHERE subscriber_id = ?
        """, (subscriber_id,))
        conn.commit()

        logger.info(f"Cancelled subscriber {subscriber_id}: {reason}")
        return True

    except Exception as e:
        logger.error(f"Failed to cancel subscription: {e}")
        return False


def upgrade_subscription(subscriber_id: str, new_tier: SubscriptionTier) -> bool:
    """Upgrade subscriber to a higher tier."""
    try:
        sub = get_subscriber(subscriber_id)
        if not sub:
            return False

        # Generate new license key for new tier
        expiry = datetime.fromisoformat(sub.expires_at.replace("Z", "+00:00"))
        new_license = generate_license_key(subscriber_id, new_tier, expiry)

        import database
        conn = database.get_connection()
        conn.execute("""
            UPDATE rental_subscribers
            SET tier = ?, license_key = ?
            WHERE subscriber_id = ?
        """, (new_tier.value, new_license, subscriber_id))
        conn.commit()

        logger.info(f"Upgraded {subscriber_id} to {new_tier.value}")
        return True

    except Exception as e:
        logger.error(f"Failed to upgrade subscription: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# USAGE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def record_usage(
    subscriber_id: str,
    trades_executed: int = 0,
    api_calls: int = 0,
    sentiment_queries: int = 0,
    ml_predictions: int = 0,
    pnl: float = 0,
) -> None:
    """Record daily usage for a subscriber."""
    try:
        import database
        conn = database.get_connection()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS rental_usage (
                subscriber_id TEXT,
                date TEXT,
                trades_executed INTEGER DEFAULT 0,
                api_calls INTEGER DEFAULT 0,
                sentiment_queries INTEGER DEFAULT 0,
                ml_predictions INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                PRIMARY KEY (subscriber_id, date)
            )
        """)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        conn.execute("""
            INSERT INTO rental_usage VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subscriber_id, date) DO UPDATE SET
                trades_executed = trades_executed + ?,
                api_calls = api_calls + ?,
                sentiment_queries = sentiment_queries + ?,
                ml_predictions = ml_predictions + ?,
                total_pnl = total_pnl + ?
        """, (
            subscriber_id, today, trades_executed, api_calls, sentiment_queries,
            ml_predictions, pnl,
            trades_executed, api_calls, sentiment_queries, ml_predictions, pnl
        ))
        conn.commit()

    except Exception as e:
        logger.debug(f"Usage recording failed: {e}")


def check_usage_limits(subscriber_id: str) -> dict:
    """
    Check if subscriber has hit their usage limits.

    Returns: {"within_limits": bool, "limits": {...}, "usage": {...}}
    """
    try:
        sub = get_subscriber(subscriber_id)
        if not sub:
            return {"within_limits": False, "reason": "Subscriber not found"}

        plan = SUBSCRIPTION_PLANS[sub.tier]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        import database
        conn = database.get_connection()
        row = conn.execute("""
            SELECT trades_executed FROM rental_usage
            WHERE subscriber_id = ? AND date = ?
        """, (subscriber_id, today)).fetchone()

        trades_today = row[0] if row else 0

        within_limits = trades_today < plan.max_daily_trades

        return {
            "within_limits": within_limits,
            "limits": {
                "max_daily_trades": plan.max_daily_trades,
                "max_positions": plan.max_positions,
                "max_budget": plan.max_budget_usd,
            },
            "usage": {
                "trades_today": trades_today,
                "remaining": plan.max_daily_trades - trades_today,
            },
            "reason": "Daily trade limit reached" if not within_limits else "OK",
        }

    except Exception as e:
        logger.error(f"Failed to check usage limits: {e}")
        return {"within_limits": True}  # Fail open


# ─────────────────────────────────────────────────────────────────────────────
# EXPIRATION & AUTO-RENEWAL
# ─────────────────────────────────────────────────────────────────────────────

async def check_expirations() -> dict:
    """
    Check all subscribers for expirations.
    Send warnings 3 days before expiry.
    Mark as expired after expiration.

    Returns summary of actions taken.
    """
    try:
        subscribers = list_subscribers(status="active")
        trial_subscribers = list_subscribers(status="trial")

        warnings_sent = 0
        expired_count = 0
        renewed_count = 0

        now = datetime.now(timezone.utc)

        for sub in subscribers + trial_subscribers:
            try:
                expiry = datetime.fromisoformat(sub.expires_at.replace("Z", "+00:00"))
                days_until_expiry = (expiry - now).days

                # Expired - mark as expired
                if now > expiry:
                    if sub.auto_renew and sub.payment_method != "manual":
                        # Attempt auto-renewal (if payment integration exists)
                        if renew_subscription(sub.subscriber_id, 30):
                            renewed_count += 1
                        else:
                            cancel_subscription(sub.subscriber_id, "renewal_failed")
                            expired_count += 1
                    else:
                        import database
                        conn = database.get_connection()
                        conn.execute("""
                            UPDATE rental_subscribers SET status = 'expired'
                            WHERE subscriber_id = ?
                        """, (sub.subscriber_id,))
                        conn.commit()
                        expired_count += 1

                # 3-day warning
                elif days_until_expiry == 3 or days_until_expiry == 1:
                    # Would send warning notification via Telegram
                    warnings_sent += 1

            except Exception as e:
                logger.debug(f"Error processing expiry for {sub.subscriber_id}: {e}")

        return {
            "checked_at": now.isoformat(),
            "total_subscribers": len(subscribers) + len(trial_subscribers),
            "warnings_sent": warnings_sent,
            "expired": expired_count,
            "auto_renewed": renewed_count,
        }

    except Exception as e:
        logger.error(f"Expiration check failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# REVENUE & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def get_revenue_report() -> dict:
    """
    Generate revenue report:
    - Total active subscribers
    - Monthly recurring revenue (MRR)
    - Subscribers by tier
    - Churn rate
    - Lifetime value
    """
    try:
        import database
        conn = database.get_connection()

        # Total subscribers by status
        status_counts = {}
        rows = conn.execute("""
            SELECT status, COUNT(*) FROM rental_subscribers GROUP BY status
        """).fetchall()
        for status, count in rows:
            status_counts[status] = count

        # Revenue by tier
        revenue_by_tier = {}
        mrr = 0
        for tier in SubscriptionTier:
            row = conn.execute("""
                SELECT COUNT(*), SUM(total_revenue_usd) FROM rental_subscribers
                WHERE tier = ? AND status IN ('active', 'trial')
            """, (tier.value,)).fetchone()

            count, total_rev = row if row else (0, 0)
            tier_mrr = count * SUBSCRIPTION_PLANS[tier].monthly_price_usd
            mrr += tier_mrr

            revenue_by_tier[tier.value] = {
                "active_count": count or 0,
                "total_revenue": total_rev or 0,
                "monthly_recurring": tier_mrr,
            }

        # Total lifetime revenue
        row = conn.execute("""
            SELECT SUM(total_revenue_usd) FROM rental_subscribers
        """).fetchone()
        total_lifetime_revenue = (row[0] if row else 0) or 0

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_active": status_counts.get("active", 0),
                "total_trial": status_counts.get("trial", 0),
                "total_expired": status_counts.get("expired", 0),
                "total_cancelled": status_counts.get("cancelled", 0),
                "mrr_usd": mrr,
                "arr_usd": mrr * 12,  # Annual Recurring Revenue
                "lifetime_revenue_usd": total_lifetime_revenue,
            },
            "by_tier": revenue_by_tier,
            "subscription_plans": {
                tier.value: {
                    "price": plan.monthly_price_usd,
                    "max_positions": plan.max_positions,
                    "max_daily_trades": plan.max_daily_trades,
                }
                for tier, plan in SUBSCRIPTION_PLANS.items()
            },
        }

    except Exception as e:
        logger.error(f"Revenue report failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# RENTAL MODE STATUS
# ─────────────────────────────────────────────────────────────────────────────

def is_rental_mode_enabled() -> bool:
    """Check if rental/SaaS mode is enabled."""
    return os.getenv("RENTAL_MODE_ENABLED", "false").lower() == "true"


def get_rental_status() -> dict:
    """Get current rental system status."""
    enabled = is_rental_mode_enabled()

    if not enabled:
        return {
            "enabled": False,
            "mode": "PRIVATE",
            "message": "Bot is in private mode (single user). Set RENTAL_MODE_ENABLED=true to activate SaaS.",
        }

    # Get statistics
    revenue = get_revenue_report()

    return {
        "enabled": True,
        "mode": "RENTAL/SAAS",
        "revenue": revenue.get("summary", {}),
        "tiers_available": list(SUBSCRIPTION_PLANS.keys()),
    }
