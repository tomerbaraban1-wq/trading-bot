"""
Telegram Interactive Commands
==============================

Advanced interactive commands for the Telegram bot.

Commands:
- /health - System health status
- /performance - Performance attribution report
- /news - Portfolio news summary
- /ai_decision <ticker> - Get AI decision for a ticker
- /backtest <ticker> - Run backtest
- /optimize - Strategy parameter optimization
- /risk - Portfolio risk analysis
- /confluence - Multi-timeframe opportunities
- /forecast - Market forecast
- /settings - Bot settings
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def handle_health_command() -> str:
    """Handle /health command."""
    try:
        from health_monitor import run_health_check
        report = await run_health_check()

        emoji_map = {
            "healthy": "🟢",
            "degraded": "🟡",
            "critical": "🔴",
        }

        lines = [
            f"{emoji_map.get(report.overall_status, '⚪')} <b>System Health: {report.overall_status.upper()}</b>",
            "━━━━━━━━━━━━━━━━━━",
            "",
        ]

        # Show key metrics
        for name, metric in list(report.metrics.items())[:8]:
            status_emoji = emoji_map.get(metric["status"], "⚪")
            lines.append(f"{status_emoji} {name}: {metric['value']:.2f} {metric['unit']}")

        if report.issues:
            lines.extend(["", "<b>⚠️ Issues:</b>"])
            for issue in report.issues[:5]:
                lines.append(f"  • {issue}")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Health check error: {e}"


async def handle_performance_command() -> str:
    """Handle /performance command."""
    try:
        from performance_attribution import generate_attribution_report

        report = await generate_attribution_report(days=30)

        if "error" in report:
            return f"❌ Performance error: {report['error']}"

        lines = [
            "📊 <b>Performance Report (30 days)</b>",
            "━━━━━━━━━━━━━━━━━━",
            "",
        ]

        # Top winners
        winners = report.get("by_ticker", {}).get("top_winners", [])
        if winners:
            lines.append("<b>🏆 Top Winners:</b>")
            for w in winners[:3]:
                lines.append(f"  • {w['ticker']}: ${w['pnl']:+.2f} ({w['win_rate']:.0f}% win, {w['trades']} trades)")

        # Top losers
        losers = report.get("by_ticker", {}).get("top_losers", [])
        if losers:
            lines.extend(["", "<b>💔 Top Losers:</b>"])
            for l in losers[:3]:
                lines.append(f"  • {l['ticker']}: ${l['pnl']:.2f} ({l['win_rate']:.0f}% win)")

        # Insights
        insights = report.get("insights", [])
        if insights:
            lines.extend(["", "<b>💡 Key Insights:</b>"])
            for insight in insights[:5]:
                lines.append(f"  {insight}")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Performance error: {e}"


async def handle_news_command() -> str:
    """Handle /news command - portfolio news summary."""
    try:
        from news_intelligence import get_portfolio_news

        news = await get_portfolio_news()

        if "error" in news:
            return f"❌ News fetch error: {news['error']}"

        lines = [
            "📰 <b>Portfolio News Summary</b>",
            "━━━━━━━━━━━━━━━━",
            f"📰 Total articles: {news.get('total_articles', 0)}",
            "",
        ]

        # Sentiment breakdown
        sentiment = news.get("sentiment_breakdown", {})
        if sentiment:
            bullish = sentiment.get("bullish", 0)
            bearish = sentiment.get("bearish", 0)
            neutral = sentiment.get("neutral", 0)
            lines.append(f"🟢 Bullish: {bullish} | 🔴 Bearish: {bearish} | ⚪ Neutral: {neutral}")
            lines.append("")

        # Catalysts
        catalysts = news.get("catalysts", [])
        if catalysts:
            lines.append("<b>🎯 Catalysts:</b>")
            for cat in catalysts[:3]:
                tickers = ", ".join(cat.get("tickers", []))
                lines.append(f"  • {tickers}: {cat['title'][:60]}...")
                lines.append(f"    Impact: {cat.get('impact', 0):.1f}/10")

        # Breaking
        breaking = news.get("breaking_news", [])
        if breaking:
            lines.extend(["", "<b>🚨 Breaking:</b>"])
            for art in breaking[:3]:
                lines.append(f"  • {art['title'][:80]}")

        return "\n".join(lines) if len(lines) > 4 else "📰 No significant news"

    except Exception as e:
        return f"❌ News error: {e}"


async def handle_ai_decision_command(ticker: str) -> str:
    """Handle /ai_decision <ticker> command."""
    if not ticker:
        return "Usage: /ai_decision TICKER (e.g., /ai_decision AAPL)"

    try:
        ticker = ticker.upper().strip()

        # Get current price
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        from ai_decision_engine import make_trading_decision
        decision = await make_trading_decision(
            ticker=ticker,
            current_price=current_price,
        )

        # Format response
        action_emoji = {
            "STRONG_BUY": "🟢🟢",
            "BUY": "🟢",
            "HOLD": "🟡",
            "SELL": "🔴",
            "STRONG_SELL": "🔴🔴",
        }.get(decision.action, "⚪")

        lines = [
            f"{action_emoji} <b>AI Decision: {ticker}</b>",
            "━━━━━━━━━━━━━━━━",
            f"Action: {decision.action}",
            f"Confidence: {decision.confidence:.0%}",
            f"Expected Return: {decision.expected_return:+.2f}%",
            f"Risk Score: {decision.risk_score:.0f}/100",
            f"Position Size: {decision.position_size_pct:.1f}% of capital",
            "",
        ]

        if decision.stop_loss_price:
            lines.append(f"🛑 Stop Loss: ${decision.stop_loss_price:.2f}")
        if decision.take_profit_price:
            lines.append(f"🎯 Take Profit: ${decision.take_profit_price:.2f}")
        if decision.holding_period_days:
            lines.append(f"⏱️ Suggested Hold: {decision.holding_period_days:.0f} days")

        if decision.reasoning:
            lines.extend(["", "<b>📋 Reasoning:</b>"])
            for r in decision.reasoning[:5]:
                lines.append(f"  • {r}")

        if decision.warnings:
            lines.extend(["", "<b>⚠️ Warnings:</b>"])
            for w in decision.warnings[:3]:
                lines.append(f"  • {w}")

        lines.extend(["", f"<i>{decision.final_explanation}</i>"])

        return "\n".join(lines)

    except Exception as e:
        return f"❌ AI decision error: {e}"


async def handle_backtest_command(ticker: str) -> str:
    """Handle /backtest <ticker> command - quick backtest."""
    if not ticker:
        return "Usage: /backtest TICKER (e.g., /backtest AAPL)"

    try:
        ticker = ticker.upper().strip()

        from backtesting_engine import StrategyConfig, run_backtest

        config = StrategyConfig(name="default")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=180)  # 6 months

        result = await run_backtest(ticker, config, start_date, end_date)

        lines = [
            f"📊 <b>Backtest: {ticker}</b>",
            "━━━━━━━━━━━━━━━━",
            f"📅 Period: 6 months",
            f"💰 Initial: $10,000 → Final: ${result.final_capital:,.2f}",
            f"📈 Return: {result.total_return_pct:+.2f}%",
            "",
            f"🎯 Win Rate: {result.win_rate:.1f}%",
            f"📊 Total Trades: {result.total_trades}",
            f"📊 Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"📉 Max Drawdown: {result.max_drawdown_pct:.1f}%",
            f"💹 Profit Factor: {result.profit_factor:.2f}",
        ]

        if result.total_trades > 0:
            lines.extend([
                "",
                f"🏆 Avg Win: ${result.avg_win:+.2f}",
                f"💔 Avg Loss: ${result.avg_loss:.2f}",
                f"⏱️ Avg Hold: {result.avg_holding_days:.1f} days",
                f"🔥 Best Streak: {result.longest_winning_streak} wins",
            ])

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Backtest error: {e}"


async def handle_risk_command() -> str:
    """Handle /risk command - portfolio risk analysis."""
    try:
        from risk_engine import analyze_portfolio_risk

        risk = await analyze_portfolio_risk()

        if "error" in risk:
            return f"❌ Risk analysis error: {risk['error']}"

        metrics = risk.get("risk_metrics", {})
        kelly = risk.get("kelly_criterion", {})

        lines = [
            "⚖️ <b>Portfolio Risk Analysis</b>",
            "━━━━━━━━━━━━━━━━━",
            "",
            f"<b>📊 Risk Score:</b> {metrics.get('risk_score', 0):.0f}/100",
            f"<b>📊 Level:</b> {metrics.get('risk_level', 'unknown')}",
            "",
            "<b>📉 Value at Risk:</b>",
            f"  • 95% VaR: ${metrics.get('var_95', 0):.2f}",
            f"  • 99% VaR: ${metrics.get('var_99', 0):.2f}",
            "",
            "<b>📈 Performance Metrics:</b>",
            f"  • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"  • Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}",
            f"  • Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%",
            f"  • Win Rate: {metrics.get('win_rate', 0):.1f}%",
            f"  • Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            "",
            "<b>💎 Kelly Criterion:</b>",
            f"  • Optimal Risk: {kelly.get('optimal_position_size', 0)*100:.1f}% per trade",
            f"  • {kelly.get('interpretation', '')}",
        ]

        recommendations = risk.get("recommendations", [])
        if recommendations:
            lines.extend(["", "<b>💡 Recommendations:</b>"])
            for rec in recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Risk error: {e}"


async def handle_confluence_command() -> str:
    """Handle /confluence command - multi-timeframe opportunities."""
    try:
        import broker
        from multi_timeframe import find_confluence_opportunities

        # Get watchlist
        positions = await asyncio.to_thread(broker.get_positions)
        watchlist = [p.symbol for p in positions[:8]]

        if not watchlist:
            return "📊 No positions to analyze. Add positions first."

        opportunities = await find_confluence_opportunities(watchlist)

        lines = [
            "🎯 <b>Multi-Timeframe Confluence</b>",
            "━━━━━━━━━━━━━━━━━━",
            "",
        ]

        if opportunities:
            for opp in opportunities[:5]:
                lines.append(f"🟢 <b>{opp['ticker']}</b>")
                lines.append(f"  Alignment: {opp['alignment_score']:.0%}")
                lines.append(f"  Trend: {opp['trend']}")
                if opp.get("setup"):
                    lines.append(f"  Setup: {opp['setup']}")
                lines.append(f"  {opp['recommendation']}")
                lines.append("")
        else:
            lines.append("⚪ No high-confluence opportunities found right now")
            lines.append("Bot is waiting for better setups...")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Confluence error: {e}"


async def handle_forecast_command() -> str:
    """Handle /forecast command - market forecast."""
    try:
        from market_intelligence import (
            detect_volatility_regime, get_market_breadth, analyze_sector_rotation
        )

        vol = await detect_volatility_regime()
        breadth = await asyncio.to_thread(get_market_breadth)
        sectors = await asyncio.to_thread(analyze_sector_rotation)

        lines = [
            "🔮 <b>Market Forecast</b>",
            "━━━━━━━━━━━━━━",
            "",
            f"<b>📊 Volatility Regime:</b>",
            f"  {vol.regime}",
            f"  {vol.recommendation}",
            "",
            f"<b>📈 Market Breadth:</b>",
            f"  {breadth.strength_indicator}",
            f"  Advances/Declines: {breadth.advances}/{breadth.declines}",
            "",
            "<b>🏆 Top Sectors (30d):</b>",
        ]

        for sector in sectors[:3]:
            lines.append(f"  {sector.rank}. {sector.sector}: {sector.performance_pct:+.2f}%")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Forecast error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND ROUTER
# ─────────────────────────────────────────────────────────────────────────────

async def handle_doctor_command() -> str:
    """
    Full system diagnostic - checks ALL bot subsystems.
    """
    try:
        lines = [
            "🩺 <b>בדיקה מקיפה של הבוט</b>",
            "━━━━━━━━━━━━━━━━━━",
        ]

        # 1. System health
        try:
            from health_monitor import run_health_check
            health = await run_health_check()

            emoji = {"healthy": "🟢", "degraded": "🟡", "critical": "🔴"}.get(health.overall_status, "⚪")
            lines.append(f"\n{emoji} <b>System:</b> {health.overall_status}")

            if health.issues:
                for issue in health.issues[:3]:
                    lines.append(f"  • {issue}")
        except Exception as e:
            lines.append(f"\n❌ <b>System:</b> Error - {e}")

        # 2. Portfolio risk
        try:
            from risk_engine import analyze_portfolio_risk
            risk = await analyze_portfolio_risk()
            if "error" not in risk:
                metrics = risk.get("risk_metrics", {})
                lines.append(f"\n⚖️ <b>Risk Score:</b> {metrics.get('risk_score', 0):.0f}/100")
                lines.append(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                lines.append(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        except Exception as e:
            lines.append(f"\n❌ <b>Risk:</b> Error - {e}")

        # 3. Today's performance
        try:
            from continuous_learner import track_live_performance
            import asyncio
            perf = await asyncio.to_thread(track_live_performance)
            lines.append(f"\n📊 <b>Today:</b> {perf.total_trades_today} trades, {perf.win_rate_today:.0f}% win rate")
        except Exception as e:
            lines.append(f"\n❌ <b>Today:</b> Error")

        # 4. Anomalies
        try:
            from anomaly_detector import scan_portfolio_anomalies
            anomalies = await scan_portfolio_anomalies()
            critical = anomalies.get("critical_count", 0)
            high = anomalies.get("high_count", 0)

            if critical > 0:
                lines.append(f"\n🚨 <b>Anomalies:</b> {critical} critical, {high} high")
            elif high > 0:
                lines.append(f"\n⚠️ <b>Anomalies:</b> {high} high severity")
            else:
                lines.append(f"\n✅ <b>Anomalies:</b> None detected")
        except Exception as e:
            lines.append(f"\n❌ <b>Anomalies:</b> Error")

        # 5. Translation cache
        try:
            from translation_service import get_translation_stats
            stats = get_translation_stats()
            cache = stats.get("cache_stats", {})
            lines.append(f"\n🌐 <b>Translation:</b> {'ON' if stats.get('enabled') else 'OFF'}")
            lines.append(f"  Cache: {cache.get('db_entries', 0)} entries, {cache.get('total_hits', 0)} hits")
        except Exception as e:
            lines.append(f"\n❌ <b>Translation:</b> Error")

        # 6. Security status
        try:
            from security_manager import get_security_status
            sec = get_security_status()
            score = sec.get("security_score", 0)
            lines.append(f"\n🛡️ <b>Security:</b> {score:.0f}/100")
            critical_events = len(sec.get("critical_events_7d", []))
            if critical_events > 0:
                lines.append(f"  ⚠️ {critical_events} critical events (7d)")
        except Exception as e:
            lines.append(f"\n❌ <b>Security:</b> Error")

        lines.append("\n━━━━━━━━━━━━━━━━━━")
        lines.append("✅ סריקה מלאה הושלמה")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Doctor command error: {e}"


async def handle_anomalies_command() -> str:
    """Handle /anomalies command."""
    try:
        from anomaly_detector import scan_portfolio_anomalies
        result = await scan_portfolio_anomalies()

        lines = [
            "🚨 <b>Anomaly Detection</b>",
            "━━━━━━━━━━━━━━━━",
            f"Scanned: {result.get('total_scanned', 0)} positions",
            f"Found: {result.get('count', 0)} anomalies",
            "",
        ]

        anomalies = result.get("anomalies", [])
        if not anomalies:
            lines.append("✅ No anomalies detected")
            return "\n".join(lines)

        for a in anomalies[:5]:
            sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(a["severity"], "⚪")
            lines.append(f"{sev_emoji} <b>{a['ticker']}</b> ({a['type']})")
            lines.append(f"  {a['description']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Anomalies error: {e}"


async def handle_positions_command() -> str:
    """Handle /positions — live portfolio snapshot."""
    try:
        import broker, database
        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return "📊 <b>אין פוזיציות פתוחות</b>"

        total_pnl = sum(float(p.unrealized_pl) for p in positions)
        total_val = sum(float(p.market_value) for p in positions)

        lines = ["📍 <b>פוזיציות פתוחות</b>", "━━━━━━━━━━━━━━━━"]
        for p in positions:
            pl   = float(p.unrealized_pl)
            plpc = float(p.unrealized_plpc) * 100
            cur  = float(p.current_price)
            qty  = float(p.qty)
            em   = "🟢" if pl >= 0 else "🔴"
            lines.append(f"{em} <b>{p.symbol}</b> {qty:.0f}sh @ ${cur:.2f} → <b>${pl:+.2f}</b> ({plpc:+.1f}%)")

        lines += ["━━━━━━━━━━━━━━━━",
                  f"💼 ${total_val:,.2f} | {'🟢' if total_pnl>=0 else '🔴'} <b>${total_pnl:+,.2f}</b>"]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Positions error: {e}"


async def handle_top_command() -> str:
    """Handle /top — best performing positions today."""
    try:
        import broker
        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return "📊 אין פוזיציות"

        sorted_pos = sorted(positions, key=lambda p: float(p.unrealized_plpc), reverse=True)
        lines = ["🏆 <b>ביצועים היום</b>", "━━━━━━━━━━━━"]
        for p in sorted_pos[:8]:
            plpc = float(p.unrealized_plpc) * 100
            pl   = float(p.unrealized_pl)
            em   = "🟢" if pl >= 0 else "🔴"
            lines.append(f"{em} {p.symbol}: {plpc:+.1f}% (${pl:+.2f})")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


COMMAND_HANDLERS = {
    "health": handle_health_command,
    "performance": handle_performance_command,
    "news": handle_news_command,
    "risk": handle_risk_command,
    "confluence": handle_confluence_command,
    "forecast": handle_forecast_command,
    "doctor": handle_doctor_command,
    "anomalies": handle_anomalies_command,
    "positions": handle_positions_command,
    "top": handle_top_command,
}

COMMAND_HANDLERS_WITH_ARG = {
    "ai_decision": handle_ai_decision_command,
    "ai": handle_ai_decision_command,
    "backtest": handle_backtest_command,
    "bt": handle_backtest_command,
}


async def route_command(command: str, args: str = "") -> str:
    """
    Route a command to the appropriate handler.

    Args:
        command: command name (without leading /)
        args: arguments string
    """
    command = command.lower().strip()

    if command in COMMAND_HANDLERS:
        return await COMMAND_HANDLERS[command]()

    if command in COMMAND_HANDLERS_WITH_ARG:
        return await COMMAND_HANDLERS_WITH_ARG[command](args.strip())

    return None  # Not handled by this module


def get_command_list() -> str:
    """Get formatted list of available commands."""
    return """
📋 <b>Advanced Commands</b>
━━━━━━━━━━━━━━━━

<b>📊 Analysis:</b>
/ai_decision TICKER - AI trading decision
/backtest TICKER - Run 6-month backtest
/confluence - Multi-TF opportunities
/forecast - Market forecast

<b>📈 Performance:</b>
/performance - Performance report (30 days)
/risk - Portfolio risk analysis

<b>📰 Information:</b>
/news - Portfolio news summary
/health - System health status

<i>Pro tip: Combine commands for full analysis!</i>
"""
