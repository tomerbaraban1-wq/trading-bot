"""
Advanced Visual Dashboard
==========================

Beautiful HTML dashboard with real-time updates of:
- Portfolio overview
- AI decisions
- Performance metrics
- Risk dashboard
- Recent trades
- Market intelligence
"""

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def generate_advanced_dashboard_html() -> str:
    """Generate a beautiful comprehensive dashboard."""

    try:
        # Gather all data
        from performance_attribution import generate_attribution_report
        from risk_engine import analyze_portfolio_risk
        from market_intelligence import get_market_intelligence_report
        from health_monitor import run_health_check
        from continuous_learner import track_live_performance

        results = await asyncio.gather(
            generate_attribution_report(days=30),
            analyze_portfolio_risk(),
            get_market_intelligence_report(),
            run_health_check(),
            asyncio.to_thread(track_live_performance),
            return_exceptions=True,
        )

        perf_data, risk_data, market_data, health_data, today_perf = results

        # Extract metrics with safe defaults
        def safe_get(data, *keys, default=0):
            try:
                for k in keys:
                    data = data[k]
                return data
            except:
                return default

        win_rate = safe_get(perf_data, "by_ticker", "top_winners", default=[])
        risk_score = safe_get(risk_data, "risk_metrics", "risk_score")
        risk_level = safe_get(risk_data, "risk_metrics", "risk_level", default="Unknown")
        sharpe = safe_get(risk_data, "risk_metrics", "sharpe_ratio")

        market_strength = safe_get(market_data, "market_breadth", "strength", default="Unknown")
        volatility_regime = safe_get(market_data, "volatility_regime", "regime", default="Unknown")

        health_status = health_data.overall_status if not isinstance(health_data, Exception) else "unknown"
        today_trades = today_perf.total_trades_today if not isinstance(today_perf, Exception) else 0
        today_win_rate = today_perf.win_rate_today if not isinstance(today_perf, Exception) else 0
        today_pnl = today_perf.avg_return_today if not isinstance(today_perf, Exception) else 0

        # Status colors
        status_color = {"healthy": "#10b981", "degraded": "#f59e0b", "critical": "#ef4444"}.get(health_status, "#6b7280")

        # Build top winners HTML
        winners_html = ""
        for w in (safe_get(perf_data, "by_ticker", "top_winners", default=[])[:5]):
            try:
                pnl = w.get("pnl", 0)
                pct_color = "#10b981" if pnl >= 0 else "#ef4444"
                winners_html += f"""
                <div class="trade-row">
                    <span class="ticker">{w.get('ticker', '?')}</span>
                    <span style="color: {pct_color}; font-weight: bold;">${pnl:+.2f}</span>
                    <span class="meta">{w.get('win_rate', 0):.0f}% win | {w.get('trades', 0)} trades</span>
                </div>
                """
            except:
                pass

        # Build insights HTML
        insights_html = ""
        for insight in (safe_get(perf_data, "insights", default=[])):
            insights_html += f'<li>{insight}</li>'

        # Sector rotation
        sector_html = ""
        for s in (safe_get(market_data, "sector_rotation", default=[])[:3]):
            try:
                perf = s.get("performance", 0)
                perf_color = "#10b981" if perf >= 0 else "#ef4444"
                sector_html += f"""
                <div class="sector-row">
                    <span>#{s.get('rank', 0)} {s.get('sector', '')}</span>
                    <span style="color: {perf_color};">{perf:+.2f}%</span>
                </div>
                """
            except:
                pass

        # Build the dashboard HTML
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Trading Bot Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e1b4b 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            font-size: 32px;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: #94a3b8;
            margin-bottom: 30px;
        }}
        .status-bar {{
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .status-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: {status_color};
            box-shadow: 0 0 20px {status_color};
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); }}
        .card-title {{
            font-size: 13px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 600;
        }}
        .card-value {{
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .card-subtitle {{
            font-size: 14px;
            color: #94a3b8;
        }}
        .trade-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 14px;
        }}
        .trade-row:last-child {{ border-bottom: none; }}
        .ticker {{ font-weight: 600; color: #60a5fa; }}
        .meta {{ font-size: 12px; color: #94a3b8; }}
        .sector-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .sector-row:last-child {{ border-bottom: none; }}
        ul {{ list-style: none; padding: 0; }}
        ul li {{
            padding: 8px 0;
            color: #cbd5e1;
            font-size: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        ul li:before {{
            content: "→";
            margin-right: 10px;
            color: #60a5fa;
        }}
        .refresh-btn {{
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            margin-left: auto;
        }}
        .refresh-btn:hover {{ opacity: 0.9; }}
        .green {{ color: #10b981; }}
        .red {{ color: #ef4444; }}
        .yellow {{ color: #f59e0b; }}
        @media (max-width: 768px) {{
            h1 {{ font-size: 24px; }}
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Trading Bot</h1>
        <div class="subtitle">Real-time Intelligence Dashboard</div>

        <div class="status-bar">
            <div class="status-dot"></div>
            <strong>System Status:</strong> {health_status.upper()}
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        </div>

        <div class="grid">
            <!-- Today's Performance -->
            <div class="card">
                <div class="card-title">📊 Today's Performance</div>
                <div class="card-value">{today_trades}</div>
                <div class="card-subtitle">trades today</div>
                <div style="margin-top: 12px;">
                    <div class="card-subtitle">Win rate: <span class="{('green' if today_win_rate > 50 else 'red')}">{today_win_rate:.1f}%</span></div>
                    <div class="card-subtitle">Avg P&L: <span class="{('green' if today_pnl > 0 else 'red')}">${today_pnl:+.2f}</span></div>
                </div>
            </div>

            <!-- Risk Score -->
            <div class="card">
                <div class="card-title">⚖️ Risk Score</div>
                <div class="card-value">{risk_score:.0f}<span style="font-size: 16px; color: #94a3b8;">/100</span></div>
                <div class="card-subtitle">{risk_level}</div>
                <div style="margin-top: 12px;">
                    <div class="card-subtitle">Sharpe: <span class="{('green' if sharpe > 1 else 'yellow')}">{sharpe:.2f}</span></div>
                </div>
            </div>

            <!-- Market Status -->
            <div class="card">
                <div class="card-title">🌍 Market Status</div>
                <div style="font-size: 16px; margin-bottom: 12px;">{market_strength}</div>
                <div class="card-subtitle">{volatility_regime}</div>
            </div>

            <!-- Top Performers -->
            <div class="card">
                <div class="card-title">🏆 Top Performers (30d)</div>
                {winners_html if winners_html else '<p class="card-subtitle">No trades yet</p>'}
            </div>

            <!-- Sector Leaders -->
            <div class="card">
                <div class="card-title">📈 Top Sectors</div>
                {sector_html if sector_html else '<p class="card-subtitle">Loading...</p>'}
            </div>

            <!-- Key Insights -->
            <div class="card">
                <div class="card-title">💡 Key Insights</div>
                <ul>
                    {insights_html if insights_html else '<li>Gathering insights...</li>'}
                </ul>
            </div>
        </div>

        <div style="margin-top: 30px; text-align: center; color: #94a3b8; font-size: 12px;">
            <p>Last updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>
            <p style="margin-top: 8px;">Auto-refresh every 60s | Powered by AI 🚀</p>
        </div>
    </div>

    <script>
        // Auto-refresh every 60 seconds
        setTimeout(() => location.reload(), 60000);
    </script>
</body>
</html>"""

    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        return f"""
        <html><body style="background: #1e1b4b; color: white; padding: 40px;">
            <h1>Dashboard Error</h1>
            <p>{e}</p>
            <p><a href="javascript:location.reload()" style="color: #60a5fa;">Retry</a></p>
        </body></html>
        """
