# Phase 1 Integration Checklist - TaskMonitor

**Status:** Ready to integrate  
**Estimated Time:** 15-30 minutes  
**Risk Level:** Low (backward compatible, no breaking changes)

---

## Pre-Integration (5 minutes)

- [ ] **Backup main.py** — Save a copy just in case
  ```bash
  cp main.py main.py.backup
  ```

- [ ] **Verify files exist** — Check that all crash prevention modules are in place
  ```bash
  ls -la task_monitor.py api_resilience.py safe_loop.py
  ```

- [ ] **Run pre-flight checks**
  ```bash
  python startup_safety_check.py
  ```
  Expected output: All [OK] for crash prevention modules

---

## Integration Steps (10-15 minutes)

Choose ONE of these approaches:

### Option A: Auto-Replace (Recommended)

If you're comfortable with the changes:

1. **Copy the updated lifespan function**
   - Open `main_lifespan_UPDATED.py`
   - Copy the entire function between the markers (lines with `═══`)

2. **Replace in main.py**
   - Open `main.py`
   - Find the `async def lifespan(app: FastAPI):` function (around line 123)
   - Delete the entire function (from `async def lifespan` to the `yield` and cleanup)
   - Paste the new function from `main_lifespan_UPDATED.py`
   - Save the file

3. **Validate changes**
   ```bash
   python validate_integration.py
   ```
   Expected: All checks pass

### Option B: Manual Integration (Step-by-Step)

If you prefer to see exactly what changes:

1. **Add import** (around line 420)
   ```python
   from task_monitor import init_monitor, get_monitor
   ```

2. **Initialize monitor** (after line 420, before heartbeat imports)
   ```python
   monitor = await init_monitor()
   logger.info("[CRASH_PREVENTION] TaskMonitor initialized...")
   ```

3. **Replace all task creations** (lines 445-490)
   - Change: `asyncio.create_task(function_name())`
   - To: `await monitor.create_task(function_name(), "function_name")`
   - Repeat for all 50+ tasks
   - Tip: Use Find & Replace with regex if your editor supports it

4. **Add shutdown cleanup** (around line 510, after `yield`)
   ```python
   # Shut down TaskMonitor first
   monitor = get_monitor()
   if monitor:
       await monitor.shutdown()
       logger.info("[CRASH_PREVENTION] TaskMonitor shut down gracefully")
   ```

5. **Add health endpoints** (around line 595)
   ```python
   @app.get("/monitor/health")
   async def monitor_health():
       monitor = get_monitor()
       if monitor is None:
           return {"error": "Monitor not initialized"}
       return await monitor.get_status()
   ```

---

## Validation (2 minutes)

After integration, run:

```bash
python validate_integration.py
```

Expected output:
```
[1/8] Import TaskMonitor... [OK]
[2/8] Initialize monitor... [OK]
[3/8] Replace asyncio.create_task... [OK]
[4/8] Replace asyncio.create_task (stop_loss)... [OK]
[5/8] Replace asyncio.create_task (30+ tasks)... [OK] Found 50 instances
[6/8] TaskMonitor shutdown... [OK]
[7/8] Log TaskMonitor initialization... [OK]
[8/8] Health check endpoints... [SKIP] - Optional

[SUCCESS] All required TaskMonitor integrations are in place!
```

---

## Pre-Startup Check (5 minutes)

Before starting the bot:

```bash
python startup_safety_check.py
```

Expected output:
```
[STARTUP] Safety Pre-Check

[1] Checking required crash prevention modules... [OK]
[2] Checking .env configuration... [OK]
[3] Checking database integrity... [OK]
[4] Checking port 8000... [OK]
[5] Checking log file sizes... [OK]
[6] Checking watchdog configuration... [OK]

[RESULT] Startup Safety Check
[OK] Pre-startup checks PASSED
Bot is safe to start. The following protections are active:
  1. Task Monitor: Will watch 50+ background tasks
  2. API Resilience: All external calls have retry + timeout
  3. Safe Loop: All infinite loops protected from crash
  4. Watchdog: Will restart on any crash (30s recovery)

Expected uptime: 24+ hours without manual restart
```

---

## Startup Test (5 minutes)

1. **Start the bot**
   ```bash
   python -m uvicorn main:app --reload
   ```

2. **Check Telegram for startup message**
   - Should see: "[CRASH_PREVENTION] TaskMonitor initialized"
   - Should see: "Startup checklist PASSED"

3. **Test health endpoint**
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
           "stop_loss_monitor": {"alive": true, "restarts": 0},
           ...
       }
   }
   ```

4. **Monitor logs**
   - Watch `trading_bot.log` for any task crashes
   - All tasks should show "Task created" in logs

---

## Verification Checklist

- [ ] `python validate_integration.py` passes all checks
- [ ] `python startup_safety_check.py` shows all [OK]
- [ ] Bot starts without errors
- [ ] Telegram shows "TaskMonitor initialized"
- [ ] `/monitor/health` returns JSON with 50 alive tasks
- [ ] No task crashes in logs (for first hour)

---

## What to Expect

### After 1 hour
- All tasks running smoothly
- TaskMonitor shows 0 dead tasks
- No Telegram alerts about failures

### After 24 hours
- Expected behavior: Bot continues running without restart
- TaskMonitor restart count should be 0-1 for most tasks
- If any task has 3+ restarts, investigate the error in logs

### If a task crashes
1. Within 10 seconds: TaskMonitor detects it
2. Immediately: Task is restarted with exponential backoff
3. After 3 failures: Telegram alert sent
4. After 5 failures: Task is given up, critical alert sent

---

## Rollback Plan

If anything goes wrong:

```bash
# Restore from backup
cp main.py.backup main.py

# Verify
git diff main.py

# Restart bot
python -m uvicorn main:app --reload
```

---

## Next Steps

After successful integration:

1. **Monitor for 24+ hours** — Ensure stability
   - Check `/monitor/health` every few hours
   - Watch for any Telegram alerts
   - Review `trading_bot.log` for errors

2. **Phase 2: Wrap heartbeat loops with safe_loop()**
   - Coming next: Will add exception handling inside each loop
   - This prevents database deadlock crashes

3. **Phase 3: Add API retry decorators**
   - Coming after: Will add @ALPACA_RETRY, @GROQ_RETRY, etc
   - This prevents API timeout hangs

---

## Troubleshooting

### Bot won't start
- Check `/monitor/health` — may show database error
- Run `startup_safety_check.py` — will show what's wrong
- Check `trading_bot.log` — look for import errors

### Tasks keep crashing
- Check `trading_bot.log` for error message
- Look for pattern: "[{task_name}] CRASH #X"
- Fix the root cause in the task's code

### High memory usage
- Check `/monitor/health` — shows uptime per task
- If a task has very long uptime, may have memory leak
- Check if restart count is increasing (indicates repeated crashes)

### Monitor not responding
- Ensure `from task_monitor import init_monitor, get_monitor` is at top of lifespan
- Ensure `monitor = await init_monitor()` is called early
- Ensure monitor variable is passed to `await monitor.create_task(...)`

---

## Questions?

Refer to:
- `INTEGRATION_PHASE_1.md` — Detailed integration guide
- `CRASH_PREVENTION_README.md` — What each module does
- `main_lifespan_UPDATED.py` — Reference implementation
- `startup_safety_check.py` — Pre-flight diagnostics

---

## Success Criteria ✅

After 24 hours of operation:
1. ✅ Never have silent task deaths
2. ✅ All 50+ tasks monitored and alive
3. ✅ Zero automatic restarts (or <1 per task)
4. ✅ No Telegram critical alerts
5. ✅ Bot continues running without manual restart
6. ✅ `/monitor/health` shows 50 alive, 0 dead

If all criteria met → Phase 1 **COMPLETE** ✅

Next phase will add safe_loop() wrappers around individual loops.

