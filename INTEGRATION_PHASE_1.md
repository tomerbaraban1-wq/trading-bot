# Phase 1: TaskMonitor Integration into main.py

## Overview
This document guides the integration of the TaskMonitor system into main.py's lifespan context. This will activate automatic monitoring and recovery of all 50+ background tasks.

## Critical Changes Required

### 1. Import TaskMonitor (Line ~422)

**BEFORE:**
```python
from heartbeat import (heartbeat_loop, heartbeat_cleanup_loop, sentiment_monitor, stop_loss_monitor,
                       auto_invest_loop, keep_alive_loop, daily_summary_loop, ...
```

**AFTER:**
```python
# Import crash prevention system
from task_monitor import init_monitor, get_monitor

from heartbeat import (heartbeat_loop, heartbeat_cleanup_loop, sentiment_monitor, stop_loss_monitor,
                       auto_invest_loop, keep_alive_loop, daily_summary_loop, ...
```

### 2. Initialize Monitor (Insert after line 420, before heartbeat imports)

```python
    # ── Crash Prevention System: TaskMonitor Initialization ────────────────────
    monitor = await init_monitor()
    logger.info("[CRASH_PREVENTION] TaskMonitor initialized - all background tasks will be monitored")
```

### 3. Replace ALL asyncio.create_task() calls (Lines 445-490)

The pattern is:
```python
# OLD:
task_name = asyncio.create_task(function_name())

# NEW:
task_name = await monitor.create_task(function_name(), "function_name")
```

**Complete replacement (Copy-paste this section):**

```python
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
```

### 4. Add TaskMonitor Status Endpoint (After line 548, before routes)

```python
@app.get("/monitor/health")
async def monitor_health():
    """Real-time task health status"""
    monitor = get_monitor()
    if monitor is None:
        return {"error": "Monitor not initialized"}
    return await monitor.get_status()
```

### 5. Shutdown Modifications (Around line 510, after yield)

**BEFORE:**
```python
    yield

    # Shutdown — Gracefully cancel and await all background tasks with timeout
    logger.info("Initiating graceful shutdown...")
```

**AFTER:**
```python
    yield

    # ── Shutdown: TaskMonitor cleanup ────────────────────────────────
    logger.info("Initiating graceful shutdown...")
    
    # Shut down monitor first (stops health check, allows tasks to complete)
    try:
        monitor = get_monitor()
        if monitor:
            await monitor.shutdown()
            logger.info("[CRASH_PREVENTION] TaskMonitor shut down gracefully")
    except Exception as e:
        logger.warning(f"TaskMonitor shutdown failed: {e}")
```

---

## What TaskMonitor Does

1. **Monitors all tasks** - Every 10 seconds, checks if any task has crashed
2. **Auto-restarts crashed tasks** - If a task dies, it restarts with exponential backoff
3. **Limits restarts** - Max 5 restart attempts to prevent restart loops
4. **Sends alerts** - After 3+ consecutive failures, sends Telegram alert
5. **Tracks health** - `/monitor/health` endpoint shows which tasks are alive/dead

## Testing Integration

After applying changes, run:

```bash
python startup_safety_check.py
```

This will verify:
- [OK] All crash prevention modules exist
- [OK] TaskMonitor can be imported
- [OK] Database is healthy
- [OK] Port 8000 is available

## Verification After Startup

In Telegram, you should see:
```
✅ Startup checklist PASSED
🤖 TaskMonitor initialized - monitoring 50+ tasks
```

Then check the health endpoint:
```bash
curl http://localhost:8000/monitor/health
```

Expected response:
```json
{
  "alive": 50,
  "dead": 0,
  "tasks": {
    "heartbeat_loop": {"alive": true, "restarts": 0},
    "sentiment_monitor": {"alive": true, "restarts": 0},
    ...
  }
}
```

## If a Task Crashes (Testing)

You'll get a Telegram alert like:
```
⚠️ TASK ISSUE
sentiment_monitor crashed 1x
Will restart after 5 failures
```

And it will automatically restart with exponential backoff (0.5s → 1s → 2s → 4s → 8s...).

---

## Line-by-Line Summary

| Line | Change | Type |
|------|--------|------|
| ~420 | Add `from task_monitor import init_monitor, get_monitor` | Import |
| ~422 | Add `monitor = await init_monitor()` | Init |
| 445-490 | Replace all `asyncio.create_task()` with `await monitor.create_task()` | Core |
| ~510 | Add `await monitor.shutdown()` | Shutdown |
| After 548 | Add `/monitor/health` endpoint | Health |

---

## Timeline

- **Minutes 1-5:** Apply changes to main.py
- **Minutes 5-10:** Run startup_safety_check.py to verify
- **Minutes 10+:** Start bot and monitor Telegram alerts
- **Hours 1-24:** Observe task health via `/monitor/health` endpoint

If any task crashes, TaskMonitor will:
1. Detect it within 10 seconds
2. Log the error
3. Restart it automatically
4. Send Telegram alert if repeated failures
5. Continue monitoring for 24+ hours

---

## Troubleshooting

**Q: Getting "Task Monitor not initialized" error?**
A: Ensure `await init_monitor()` is called before any tasks are created.

**Q: Task keeps restarting (restart loop)?**
A: After 5 restarts, it gives up and sends critical alert. Check Telegram for the error message, then investigate the root cause in the task's code.

**Q: `/monitor/health` returns empty?**
A: Tasks might still be starting. Check logs for `[MONITOR] Task created` entries.

**Q: Memory usage increasing?**
A: Check `/monitor/health` for "dead" tasks (memory leak). Look at restart counts — if a task has 5+ restarts, it's crashing repeatedly.

