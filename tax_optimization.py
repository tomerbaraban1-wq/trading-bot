"""
Tax Optimization Module
========================

Advanced tax optimization for trading:
- Wash sale detection
- Tax loss harvesting opportunities
- Short vs long-term gain analysis
- YTD tax summary
- Tax efficiency scoring
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WASH SALE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

async def detect_wash_sales(days_lookback: int = 30) -> list[dict]:
    """Detect potential wash sales - cannot claim loss if buy back within 30 days."""
    try:
        import database
        conn = database.get_connection()

        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_lookback)).isoformat()

        loss_sales = conn.execute("""
            SELECT ticker, exit_time, pnl_gross
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND pnl_gross < 0
            AND exit_time >= ?
        """, (cutoff_date,)).fetchall()

        wash_sales = []
        for ticker, sale_date, loss_pnl in loss_sales:
            try:
                sale_dt = datetime.fromisoformat(sale_date.replace("Z", "+00:00"))
                wash_end = sale_dt + timedelta(days=30)

                row = conn.execute("""
                    SELECT COUNT(*) FROM trade_log
                    WHERE ticker = ? AND created_at > ? AND created_at <= ?
                """, (ticker, sale_date, wash_end.isoformat())).fetchone()

                if row and row[0] > 0:
                    wash_sales.append({
                        "ticker": ticker,
                        "loss_amount": abs(loss_pnl),
                        "sale_date": sale_date,
                        "warning": "⚠️ Wash sale - loss may be disallowed",
                    })
            except Exception as e:
                logger.debug(f"Wash sale check error: {e}")

        return wash_sales
    except Exception as e:
        logger.error(f"Wash sale detection failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# TAX LOSS HARVESTING
# ─────────────────────────────────────────────────────────────────────────────

async def find_tax_loss_harvest_opportunities() -> list[dict]:
    """Find positions with losses worth selling for tax benefits."""
    try:
        import broker
        positions = await asyncio.to_thread(broker.get_positions)
        opportunities = []

        for p in positions:
            try:
                unrealized_pnl = float(p.get('unrealized_pl', 0))
                cost_basis = float(p.cost_basis)

                if unrealized_pnl < 0:
                    loss_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0

                    if loss_pct < -5:
                        opportunities.append({
                            "ticker": p.get('ticker'),
                            "unrealized_loss": unrealized_pnl,
                            "loss_pct": loss_pct,
                            "tax_benefit_estimate": abs(unrealized_pnl) * 0.20,
                            "recommendation": (
                                "💰 Significant loss - consider harvesting" if loss_pct < -10 else
                                "💡 Moderate loss - could harvest near year-end"
                            ),
                        })
            except Exception:
                continue

        return sorted(opportunities, key=lambda x: x["unrealized_loss"])
    except Exception as e:
        logger.error(f"Tax loss harvesting failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# YEAR-TO-DATE TAX SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

async def get_ytd_tax_summary() -> dict:
    """Generate year-to-date tax summary."""
    try:
        import database
        conn = database.get_connection()

        current_year = datetime.now(timezone.utc).year
        year_start = f"{current_year}-01-01"

        rows = conn.execute("""
            SELECT created_at, exit_time, pnl_gross
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= ?
        """, (year_start,)).fetchall()

        short_term_gains = 0
        short_term_losses = 0
        long_term_gains = 0
        long_term_losses = 0

        for created_at, exit_time, pnl in rows:
            try:
                purchase_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                sale_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                holding_days = (sale_dt - purchase_dt).days
                is_long_term = holding_days > 365

                if pnl > 0:
                    if is_long_term:
                        long_term_gains += pnl
                    else:
                        short_term_gains += pnl
                else:
                    if is_long_term:
                        long_term_losses += abs(pnl)
                    else:
                        short_term_losses += abs(pnl)
            except Exception:
                continue

        net_short_term = short_term_gains - short_term_losses
        net_long_term = long_term_gains - long_term_losses
        total_net = net_short_term + net_long_term

        estimated_short_term_tax = max(0, net_short_term * 0.32)
        estimated_long_term_tax = max(0, net_long_term * 0.20)
        total_estimated_tax = estimated_short_term_tax + estimated_long_term_tax

        return {
            "year": current_year,
            "total_trades": len(rows),
            "short_term": {
                "gains": short_term_gains,
                "losses": short_term_losses,
                "net": net_short_term,
                "estimated_tax": estimated_short_term_tax,
            },
            "long_term": {
                "gains": long_term_gains,
                "losses": long_term_losses,
                "net": net_long_term,
                "estimated_tax": estimated_long_term_tax,
            },
            "total": {
                "net_gain_loss": total_net,
                "estimated_total_tax": total_estimated_tax,
                "after_tax_profit": total_net - total_estimated_tax,
            },
        }
    except Exception as e:
        logger.error(f"YTD tax summary failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TAX EFFICIENCY SCORING
# ─────────────────────────────────────────────────────────────────────────────

async def calculate_tax_efficiency() -> dict:
    """Calculate tax efficiency score (0-100, higher = better)."""
    try:
        import database
        conn = database.get_connection()

        year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

        rows = conn.execute("""
            SELECT pnl_gross,
                   (julianday(exit_time) - julianday(created_at)) as holding_days
            FROM trade_log
            WHERE status IN ('stopped', 'sold')
            AND exit_time >= ?
        """, (year_ago,)).fetchall()

        if not rows:
            return {"error": "No trade history"}

        short_term_pnl = sum(pnl for pnl, days in rows if days <= 365)
        long_term_pnl = sum(pnl for pnl, days in rows if days > 365)

        total_pnl = short_term_pnl + long_term_pnl
        long_term_pct = (long_term_pnl / total_pnl * 100) if total_pnl else 0

        short_tax = max(0, short_term_pnl) * 0.32
        long_tax = max(0, long_term_pnl) * 0.20
        total_tax = short_tax + long_tax
        effective_rate = (total_tax / total_pnl * 100) if total_pnl > 0 else 0

        if total_pnl <= 0:
            efficiency_score = 50
        else:
            efficiency_score = max(0, min(100, 100 - ((effective_rate - 20) / 12 * 100)))

        return {
            "long_term_pct": long_term_pct,
            "effective_tax_rate": effective_rate,
            "efficiency_score": efficiency_score,
            "interpretation": (
                "🟢 Excellent tax efficiency" if efficiency_score > 80 else
                "🟡 Moderate - consider longer holds" if efficiency_score > 50 else
                "🔴 Poor - mostly short-term trades"
            ),
        }
    except Exception as e:
        logger.error(f"Tax efficiency calc failed: {e}")
        return {"error": str(e)}
