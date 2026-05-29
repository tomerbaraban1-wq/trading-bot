# This file shows the UPDATED lifespan() function for main.py
# Copy the content between the markers and paste it into your main.py

# ═══════════════════════════════════════════════════════════════════════════
# START: Updated async lifespan function (replace the old one completely)
# ═══════════════════════════════════════════════════════════════════════════

async def lifespan(app: FastAPI):
    # Install global asyncio exception handler immediately
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_asyncio_exception_handler)

    # Prevent Windows from sleeping while bot is running
    # Sleep = port conflicts + crash loops when system wakes up
    try:
        import ctypes
        ES_CONTINUOUS      = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        logger.info("Sleep prevention active — Windows will not suspend this process")
    except Exception:
        pass  # not on Windows, no problem

    # Startup
    settings.validate()

    # ── RUN COMPREHENSIVE STARTUP CHECKLIST ──────────────────────────────────
    try:
        from startup_checklist import run_startup_checklist
        is_safe, checks = await run_startup_checklist()

        if not is_safe:
            logger.critical("⛔ STARTUP CHECKLIST FAILED - BOT CANNOT START")
            logger.critical("Please fix the critical issues listed above before restarting.")
            logger.critical("")
            logger.critical("Critical issues must be resolved:")
            for check in checks:
                if check.status == "fail" and check.severity == "critical":
                    logger.critical(f"  {check.message}")
            logger.critical("")
            logger.critical("For detailed diagnostics, visit: /health")
            raise RuntimeError("Startup checklist failed - critical configuration issues detected")
        else:
            logger.info("✅ Startup checklist PASSED - proceeding with initialization")
    except ImportError:
        logger.warning("⚠️  startup_checklist module not found - skipping comprehensive checks")
    except RuntimeError as e:
        # Re-raise startup failures - don't let bot start with critical issues
        raise
    except Exception as e:
        logger.warning(f"⚠️  Startup checklist error (non-blocking): {e}")

    init_db()

    # Check database integrity
    db_ok = check_database_integrity()
    if not db_ok:
        logger.warning("Database integrity check failed but continuing...")

    # Log durability mode
    durability_mode = "HARDENED" if settings.HARDENED_DURABILITY else "NORMAL"
    logger.info("=== Trading Bot Started ===")
    _broker_info = settings.ACTIVE_BROKER if settings.ACTIVE_BROKER else settings.ALPACA_BASE_URL
    logger.info(f"Budget: ${settings.MAX_BUDGET:,.2f} | Broker: {_broker_info} | DB Mode: {durability_mode}")
    logger.info(f"Config: MIN_BUY_SCORE={settings.MIN_BUY_SCORE} | MAX_POSITIONS={settings.MAX_OPEN_POSITIONS} | MAX_HOLD={settings.MAX_HOLD_HOURS}h")

    # Send startup configuration report to Telegram
    try:
        from startup_validator import send_startup_report as _startup_report
        import asyncio as _asyncio_sr
        _asyncio_sr.create_task(_startup_report())
    except Exception:
        pass

    # ── Pre-warm Telegram context cache (parallel with other startup) ─────────
    # First user message after restart hits a 5-6s cold cache. Pre-build now
    # in background so by the time the user sends a command, cache is hot.
    try:
        import asyncio as _asyncio_pw
        async def _prewarm():
            try:
                import telegram_chat as _tc_pw
                await _asyncio_pw.to_thread(_tc_pw._build_context)
                logger.info("[STARTUP] Telegram context pre-warmed")
            except Exception as _pw_err:
                logger.debug(f"[STARTUP] Context prewarm failed: {_pw_err}")
        _asyncio_pw.create_task(_prewarm())
    except Exception:
        pass

    # ── Start polling loop when running locally (no RENDER_EXTERNAL_URL) ──────
    # On Render: webhook handles incoming messages (registered below).
    # Locally:   no public URL → use getUpdates polling instead.
    _polling_task = None
    try:
        from telegram_polling import is_local_mode, polling_loop as _polling_loop
        if is_local_mode():
            logger.info("[POLLING] Local mode — starting Telegram polling loop")
            _polling_task = asyncio.create_task(_polling_loop())
        else:
            logger.info("[POLLING] Cloud mode — using webhook")
    except Exception as _poll_err:
        logger.warning(f"Polling setup failed (non-critical): {_poll_err}")

    # ── Auto-register Telegram webhook + command menu ─────────────────────────
    try:
        _render_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        _tg_token   = settings.TELEGRAM_BOT_TOKEN
        if _render_url and _tg_token:
            import aiohttp as _aiohttp
            _webhook_url = f"{_render_url}/telegram/webhook"
            _secret      = settings.WEBHOOK_SECRET
            async with _aiohttp.ClientSession() as _sess:
                # Register webhook WITH secret_token — Telegram will send it in
                # X-Telegram-Bot-Api-Secret-Token header. Otherwise any client
                # can POST forged updates to our webhook.
                async with _sess.post(
                    f"https://api.telegram.org/bot{_tg_token}/setWebhook",
                    json={
                        "url": _webhook_url,
                        "secret_token": _secret,
                        "drop_pending_updates": False,
                    },
                    timeout=_aiohttp.ClientTimeout(total=10),
                ) as _resp:
                    _data = await _resp.json()
                    if _data.get("ok"):
                        logger.info(f"Telegram webhook registered: {_webhook_url}")

                # Set command menu (shows as clickable buttons in Telegram)
                _commands = [
                    {"command": "status",    "description": "📊 מצב התיק המלא"},
                    {"command": "manioth",   "description": "📂 איזה מניות יש לי"},
                    {"command": "revach",    "description": "💰 מה הרווח שלי"},
                    {"command": "shovi",     "description": "💼 מה שווי התיק"},
                    {"command": "mazon",     "description": "💵 כמה מזומן יש לי"},
                    {"command": "biztsuim",  "description": "🏆 ביצועים ואחוז הצלחה"},
                    {"command": "count",     "description": "🔢 כמה עסקאות (סך כל הזמן)"},
                    {"command": "market",    "description": "🌍 מצב השוק (SPY/QQQ/DIA)"},
                    {"command": "trending",   "description": "🔥 מניות בתנופה חזקה"},
                    {"command": "gainers",   "description": "🚀 מניות מובילות היום"},
                    {"command": "exposure",  "description": "🏢 חשיפת תיק לסקטורים"},
                    {"command": "volatility","description": "📐 תנודתיות ו-Beta מניה"},
                    {"command": "morning",   "description": "☀️ תדרוך בוקר ידני"},
                    {"command": "signals",   "description": "📡 הזדמנויות קנייה עכשיו"},
                    {"command": "quick",     "description": "⚡ סקירה מהירה של מניה"},
                    {"command": "position",  "description": "📂 פרטי פוזיציה מלאים"},
                    {"command": "monthly",   "description": "📅 סיכום 30 ימים"},
                    {"command": "watchadd",  "description": "➕ הוסף מניה לרשימה"},
                    {"command": "watchremove","description": "➖ הסר מניה מהרשימה"},
                    {"command": "macro",     "description": "📅 אירועים כלכליים קרובים"},
                    {"command": "sectors",   "description": "📈 דירוג סקטורים"},
                    {"command": "pause",     "description": "⏸️ עצור קניות חדשות"},
                    {"command": "resume",    "description": "▶️ חדש קניות"},
                    {"command": "next",       "description": "🕐 מתי השוק נפתח"},
                    {"command": "portfolio",  "description": "📊 הקצאת תיק"},
                    {"command": "summary",    "description": "📅 סיכום 7 ימים"},
                    {"command": "best",       "description": "🏆 העסקה הטובה ביותר"},
                    {"command": "worst",      "description": "📉 העסקה הגרועה ביותר"},
                    {"command": "uptime",     "description": "🤖 זמן פעילות הבוט"},
                    {"command": "taxes",      "description": "🧾 סיכום מס"},
                    {"command": "risk",       "description": "⚠️ ניתוח סיכון"},
                    {"command": "correlation","description": "📊 קורלציה בין פוזיציות"},
                    {"command": "health",    "description": "🩺 בריאות כל הפוזיציות"},
                    {"command": "pnl",       "description": "💰 רווח/הפסד מהיר"},
                    {"command": "volume",    "description": "📊 נפח מסחר מניה"},
                    {"command": "watchlist",  "description": "👁️ רשימת מניות לסריקה"},
                    {"command": "top",        "description": "🏆 מניות עם ציון גבוה"},
                    {"command": "winners",   "description": "🟢 פוזיציות ברווח"},
                    {"command": "losers",    "description": "🔴 פוזיציות בהפסד"},
                    {"command": "today",     "description": "📅 מה קרה היום"},
                    {"command": "vix",       "description": "🌡️ מדד הפחד VIX"},
                    {"command": "budget",    "description": "⚙️ הגדרות הבוט"},
                    {"command": "history",   "description": "📋 עסקאות אחרונות"},
                    {"command": "fear",      "description": "😨 Fear and Greed Index"},
                    {"command": "newscheck", "description": "📰 בדוק חדשות לכל הפוזיציות"},
                    {"command": "price",     "description": "💲 מחיר מניה"},
                    {"command": "alerts",    "description": "🔔 התראות מחיר פעילות"},
                    {"command": "scan",       "description": "🔍 הפעל סריקה מיידית"},
                    {"command": "chart",      "description": "📊 גרף מחיר 30 ימים"},
                    {"command": "fundamental","description": "📈 נתוני יסוד (P/E, הכנסות)"},
                    {"command": "dividend",  "description": "💰 דיבידנד ותשואה"},
                    {"command": "review",    "description": "🤖 AI סוקר את הפוזיציות"},
                    {"command": "journal",   "description": "📓 יומן עסקאות אישי"},
                    {"command": "whatsnew",  "description": "📋 5 הפעולות האחרונות"},
                    {"command": "levels",    "description": "📐 רמות תמיכה/תנגדות"},
                    {"command": "remind",    "description": "⏰ הגדר תזכורת"},
                    {"command": "quiet",     "description": "🔕 מצב שקט (פחות התראות)"},
                    {"command": "loud",      "description": "🔔 כל ההתראות"},
                    {"command": "ask",       "description": "🤖 שאל שאלה חופשית ל-AI"},
                    {"command": "advice",    "description": "🤖 ייעוץ AI על התיק"},
                    {"command": "explain",   "description": "📚 הסבר מונח פיננסי"},
                    {"command": "streak",    "description": "🔥 רצף ניצחונות"},
                    {"command": "diagnose",  "description": "🔍 למה הבוט לא קונה"},
                    {"command": "backtest",  "description": "🧠 למידה היסטורית"},
                    {"command": "help",      "description": "❓ כל הפקודות"},
                ]
                async with _sess.post(
                    f"https://api.telegram.org/bot{_tg_token}/setMyCommands",
                    json={"commands": _commands},
                    timeout=_aiohttp.ClientTimeout(total=10),
                ) as _resp2:
                    _data2 = await _resp2.json()
                    if _data2.get("ok"):
                        logger.info("Telegram command menu registered")
    except Exception as _e:
        logger.warning(f"Telegram setup failed (non-critical): {_e}")

    # ── Startup state restore + reconciliation ───────────────────────────────
    # Detects and fixes the case where broker has positions but SQLite is empty.
    # This happens when Render redeploys and wipes the ephemeral SQLite file,
    # but the Postgres-backed broker state (TVPaperBroker) still holds positions.
    # Without reconciliation those positions have no stop-loss protection.
    try:
        from database import get_open_trades, save_trade
        import broker as _broker
        open_trades = get_open_trades()

        acct = await asyncio.wait_for(
            asyncio.to_thread(_broker.get_account), timeout=20
        )
        cash = float(acct.get("cash", 0))
        equity = float(acct.get("equity", 0))
        logger.info(f"BROKER: cash=${cash:,.2f} | equity=${equity:,.2f}")

        if open_trades:
            tickers = [t["ticker"] for t in open_trades]
            logger.info(f"RESTORED {len(open_trades)} open position(s): {tickers}")

            # Cross-check: close SQLite records that no longer exist in the broker
            # (prevents stop-loss monitor from trying to sell non-existent positions)
            try:
                from database import close_trade as _close_trade
                broker_positions = await asyncio.wait_for(asyncio.to_thread(_broker.get_positions), timeout=20)
                # Guard: if broker returns empty list (API error / transient failure),
                # skip cross-check entirely to avoid closing ALL valid positions
                if not broker_positions:
                    logger.warning(
                        "RECONCILE: broker returned 0 positions — skipping cross-check "
                        "(could be API error; not closing valid SQLite records)"
                    )
                else:
                    broker_tickers = {p.get("ticker", "").upper() for p in broker_positions}
                    for t in open_trades:
                        if t["ticker"].upper() not in broker_tickers:
                            logger.warning(
                                f"RECONCILE: {t['ticker']} is open in SQLite but NOT in broker — "
                                f"closing as stale_restart"
                            )
                            _close_trade(t["id"], t["entry_price"], 0.0, 0.0, 0.0, 0.0, "stale_restart")
            except Exception as _ce:
                logger.warning(f"Cross-check reconciliation failed (non-critical): {_ce}")
        else:
            # SQLite is empty — check if broker has positions we need to recover
            broker_positions = await asyncio.to_thread(_broker.get_positions)
            if broker_positions:
                logger.warning(
                    f"RECONCILE: SQLite is empty but broker has {len(broker_positions)} position(s) — re-creating records"
                )
                from models import WebhookPayload, TradeAction
                for pos in broker_positions:
                    ticker    = pos.get("ticker", "").upper()
                    qty       = float(pos.get("qty", 0))
                    entry     = float(pos.get("avg_entry_price", 0))
                    if not ticker or qty <= 0 or entry <= 0:
                        continue
                    try:
                        trade = {
                            "ticker":         ticker,
                            "action":         "buy",
                            "qty":            qty,
                            "entry_price":    entry,
                            "trailing_stop_pct": None,
                            "rsi": None, "macd": None, "macd_signal": None,
                            "bb_position": None, "volume_ratio": None,
                            "sentiment_score": 5,
                            "sentiment_reasoning": "Recovered from broker on restart",
                        }
                        trade_id = save_trade(trade)
                        # Set ATR stop immediately
                        from atr_stop import compute_initial_stop
                        from database import update_trade_stop
                        stop_price, stop_meta = await asyncio.wait_for(asyncio.to_thread(compute_initial_stop, ticker, entry), timeout=20)
                        update_trade_stop(trade_id, stop_price, entry)
                        logger.info(
                            f"RECONCILE: restored {ticker} x{qty} @ ${entry:.2f} "
                            f"(trade_id={trade_id}, stop=${stop_price:.2f})"
                        )
                    except Exception as _re:
                        logger.warning(f"RECONCILE: failed to restore {ticker}: {_re}")
            else:
                logger.info("RESTORED: no open positions — clean slate")
    except Exception as _e:
        logger.warning(f"Startup reconciliation failed (non-critical): {_e}")
    # ─────────────────────────────────────────────────────────────────────────

    # Store the running event loop so worker threads can use it (e.g. Discord sentiment)
    try:
        import asyncio as _asyncio
        from discord_bot import set_event_loop as _set_loop
        _set_loop(_asyncio.get_running_loop())
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # CRASH PREVENTION SYSTEM INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    # Import and initialize TaskMonitor for crash prevention
    from task_monitor import init_monitor, get_monitor

    monitor = await init_monitor()
    logger.info("[CRASH_PREVENTION] TaskMonitor initialized — monitoring 50+ background tasks")

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPORT ALL HEARTBEAT LOOPS
    # ═══════════════════════════════════════════════════════════════════════════

    from heartbeat import (heartbeat_loop, heartbeat_cleanup_loop, sentiment_monitor, stop_loss_monitor,
                           auto_invest_loop, keep_alive_loop, daily_summary_loop,
                           weekly_report_loop, shadow_monitor_loop, portfolio_update_loop,
                           news_refresh_loop, news_monitor_loop, morning_briefing_loop,
                           position_alert_loop, backtest_learning_loop, eod_sweep_loop,
                           price_alert_loop, market_closed_training_loop,
                           telegram_context_warmup_loop, earnings_monitor_loop,
                           market_pulse_loop, webhook_keeper_loop,
                           golden_opportunity_loop, smart_reentry_loop,
                           weekend_research_loop, daily_ai_insights_loop,
                           self_improvement_loop, rapid_move_alert_loop,
                           drawdown_protection_loop, idle_cash_alert_loop,
                           adaptive_threshold_loop, daily_goal_progress_loop,
                           continuous_learning_loop, adaptive_parameters_monitor_loop,
                           correlation_monitor_loop, market_intelligence_loop,
                           detailed_analytics_loop, ai_decision_loop,
                           attribution_loop, notification_digest_loop,
                           multi_timeframe_loop, health_monitoring_loop,
                           news_catalyst_loop, pairs_trading_loop,
                           benchmark_comparison_loop, trade_journal_loop,
                           anomaly_detection_loop, stale_position_guard_loop,
                           fast_track_progress_loop, volume_surge_loop)

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK CREATION WITH TASKMONITOR (REPLACES asyncio.create_task)
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Core tasks (always run) ───────────────────────────────────────
    heartbeat_task         = await monitor.create_task(heartbeat_loop(), "heartbeat_loop")
    heartbeat_cleanup_task = await monitor.create_task(heartbeat_cleanup_loop(), "heartbeat_cleanup_loop")
    stop_loss_task         = await monitor.create_task(stop_loss_monitor(), "stop_loss_monitor")
    auto_invest_task       = await monitor.create_task(auto_invest_loop(), "auto_invest_loop")
    keep_alive_task        = await monitor.create_task(keep_alive_loop(), "keep_alive_loop")
    daily_summary_task     = await monitor.create_task(daily_summary_loop(), "daily_summary_loop")
    weekly_report_task     = await monitor.create_task(weekly_report_loop(), "weekly_report_loop")
    backtest_task          = await monitor.create_task(backtest_learning_loop(), "backtest_learning_loop")
    training_task          = await monitor.create_task(market_closed_training_loop(), "market_closed_training_loop")
    tg_warmup_task         = await monitor.create_task(telegram_context_warmup_loop(), "telegram_context_warmup_loop")
    eod_sweep_task         = await monitor.create_task(eod_sweep_loop(), "eod_sweep_loop")
    price_alert_task       = await monitor.create_task(price_alert_loop(), "price_alert_loop")
    morning_briefing_task  = await monitor.create_task(morning_briefing_loop(), "morning_briefing_loop")
    news_refresh_task      = await monitor.create_task(news_refresh_loop(), "news_refresh_loop")
    news_monitor_task      = await monitor.create_task(news_monitor_loop(), "news_monitor_loop")
    earnings_monitor_task  = await monitor.create_task(earnings_monitor_loop(), "earnings_monitor_loop")
    market_pulse_task      = await monitor.create_task(market_pulse_loop(), "market_pulse_loop")
    goal_progress_task     = await monitor.create_task(daily_goal_progress_loop(), "daily_goal_progress_loop")
    learning_task          = await monitor.create_task(continuous_learning_loop(), "continuous_learning_loop")
    adaptive_params_task   = await monitor.create_task(adaptive_parameters_monitor_loop(), "adaptive_parameters_monitor_loop")
    correlation_task       = await monitor.create_task(correlation_monitor_loop(), "correlation_monitor_loop")
    market_intel_task      = await monitor.create_task(market_intelligence_loop(), "market_intelligence_loop")
    analytics_task         = await monitor.create_task(detailed_analytics_loop(), "detailed_analytics_loop")
    ai_decision_task       = await monitor.create_task(ai_decision_loop(), "ai_decision_loop")
    attribution_task       = await monitor.create_task(attribution_loop(), "attribution_loop")
    digest_task            = await monitor.create_task(notification_digest_loop(), "notification_digest_loop")
    mtf_task               = await monitor.create_task(multi_timeframe_loop(), "multi_timeframe_loop")
    health_task            = await monitor.create_task(health_monitoring_loop(), "health_monitoring_loop")
    news_catalyst_task     = await monitor.create_task(news_catalyst_loop(), "news_catalyst_loop")
    pairs_task             = await monitor.create_task(pairs_trading_loop(), "pairs_trading_loop")
    benchmark_task         = await monitor.create_task(benchmark_comparison_loop(), "benchmark_comparison_loop")
    journal_task           = await monitor.create_task(trade_journal_loop(), "trade_journal_loop")
    anomaly_task           = await monitor.create_task(anomaly_detection_loop(), "anomaly_detection_loop")
    stale_guard_task       = await monitor.create_task(stale_position_guard_loop(), "stale_position_guard_loop")
    fast_track_task        = await monitor.create_task(fast_track_progress_loop(), "fast_track_progress_loop")
    webhook_keeper_task    = await monitor.create_task(webhook_keeper_loop(), "webhook_keeper_loop")
    golden_opp_task        = await monitor.create_task(golden_opportunity_loop(), "golden_opportunity_loop")
    reentry_task           = await monitor.create_task(smart_reentry_loop(), "smart_reentry_loop")
    weekend_task           = await monitor.create_task(weekend_research_loop(), "weekend_research_loop")
    ai_insights_task       = await monitor.create_task(daily_ai_insights_loop(), "daily_ai_insights_loop")
    self_improve_task      = await monitor.create_task(self_improvement_loop(), "self_improvement_loop")
    rapid_move_task        = await monitor.create_task(rapid_move_alert_loop(), "rapid_move_alert_loop")
    drawdown_task          = await monitor.create_task(drawdown_protection_loop(), "drawdown_protection_loop")
    idle_cash_task         = await monitor.create_task(idle_cash_alert_loop(), "idle_cash_alert_loop")
    adaptive_task          = await monitor.create_task(adaptive_threshold_loop(), "adaptive_threshold_loop")
    volume_surge_task      = await monitor.create_task(volume_surge_loop(), "volume_surge_loop")

    # ── Resource monitor: alerts on high CPU/memory ───────────────────
    try:
        from resource_monitor import resource_monitor_loop
        resource_monitor_task = await monitor.create_task(resource_monitor_loop(), "resource_monitor_loop")
    except ImportError:
        resource_monitor_task = None

    # ── Optional tasks (disabled on free tier to save memory) ────────
    import os as _os
    _full_mode = _os.getenv("FULL_MODE", "false").lower() == "true"
    sentiment_task      = await monitor.create_task(sentiment_monitor(), "sentiment_monitor") if _full_mode else None
    shadow_monitor_task = await monitor.create_task(shadow_monitor_loop(), "shadow_monitor_loop") if _full_mode else None
    portfolio_update_task = await monitor.create_task(portfolio_update_loop(), "portfolio_update_loop") if _full_mode else None
    position_alert_task = await monitor.create_task(position_alert_loop(), "position_alert_loop") if _full_mode else None

    if not _full_mode:
        logger.info("Memory-saving mode: shadow, portfolio_update, position_alert, sentiment disabled. Set FULL_MODE=true to enable.")

    yield

    # ═══════════════════════════════════════════════════════════════════════════
    # GRACEFUL SHUTDOWN
    # ═══════════════════════════════════════════════════════════════════════════

    logger.info("Initiating graceful shutdown...")

    # Shut down TaskMonitor first (stops health check, allows tasks to complete)
    try:
        monitor = get_monitor()
        if monitor:
            await monitor.shutdown()
            logger.info("[CRASH_PREVENTION] TaskMonitor shut down gracefully")
    except Exception as e:
        logger.warning(f"TaskMonitor shutdown failed: {e}")

    # Shutdown — Gracefully cancel and await all background tasks with timeout
    all_tasks = [t for t in [
        heartbeat_task, heartbeat_cleanup_task, sentiment_task, stop_loss_task, auto_invest_task,
        keep_alive_task, daily_summary_task, weekly_report_task, shadow_monitor_task,
        portfolio_update_task, news_refresh_task, news_monitor_task, morning_briefing_task,
        position_alert_task, backtest_task, training_task, eod_sweep_task, price_alert_task,
        earnings_monitor_task, market_pulse_task, goal_progress_task, learning_task, adaptive_params_task,
        correlation_task, market_intel_task, analytics_task, ai_decision_task,
        attribution_task, digest_task, mtf_task, health_task, news_catalyst_task,
        pairs_task, benchmark_task, journal_task, anomaly_task, stale_guard_task,
        fast_track_task, webhook_keeper_task,
        golden_opp_task, reentry_task, weekend_task,
        ai_insights_task, self_improve_task, rapid_move_task,
        drawdown_task, idle_cash_task, adaptive_task, tg_warmup_task, _polling_task,
        resource_monitor_task,
    ] if t is not None]

    # Cancel all background tasks
    for task in all_tasks:
        if not task.done():
            task.cancel()

    # Wait for tasks to complete with 10-second timeout
    try:
        await asyncio.wait_for(asyncio.gather(*all_tasks, return_exceptions=True), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Background tasks did not complete within 10s timeout, forcing shutdown...")
    except Exception as e:
        logger.warning(f"Exception during task shutdown: {e}")

    # Ensure database is flushed and properly closed
    flush_database()
    close_connections()
    logger.info("=== Trading Bot Stopped ===")

# ═══════════════════════════════════════════════════════════════════════════
# END: Updated lifespan function
# ═══════════════════════════════════════════════════════════════════════════
