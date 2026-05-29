# Quick Reference: TaskMonitor Integration

## The 4 Key Changes

### Change 1: Add Import (Line ~420)
```python
from task_monitor import init_monitor, get_monitor
```

### Change 2: Initialize Monitor (Line ~422)
```python
monitor = await init_monitor()
logger.info("[CRASH_PREVENTION] TaskMonitor initialized")
```

### Change 3: Replace Task Creation (Lines 445-490)

**EVERY instance of:**
```python
task_name = asyncio.create_task(function_name())
```

**Becomes:**
```python
task_name = await monitor.create_task(function_name(), "function_name")
```

**Example:**
```python
# BEFORE:
heartbeat_task = asyncio.create_task(heartbeat_loop())

# AFTER:
heartbeat_task = await monitor.create_task(heartbeat_loop(), "heartbeat_loop")
```

Count: **50 tasks × 1 line each**

### Change 4: Add Shutdown (Line ~510, after `yield`)
```python
# Shut down TaskMonitor
monitor = get_monitor()
if monitor:
    await monitor.shutdown()
    logger.info("[CRASH_PREVENTION] TaskMonitor shut down gracefully")
```

---

## Quick Validation

After integration, run:
```bash
python validate_integration.py
```

Should show: `[SUCCESS] All required TaskMonitor integrations are in place!`

---

## Key Files

| File | Purpose | Action |
|------|---------|--------|
| `INTEGRATION_PHASE_1.md` | Detailed guide | Read first |
| `main_lifespan_UPDATED.py` | Reference code | Copy if doing auto-replace |
| `validate_integration.py` | Validation | Run after making changes |
| `startup_safety_check.py` | Pre-flight check | Run before starting bot |
| `health_endpoints.py` | Health endpoints | (Optional) Add to main.py |

---

## Common Find & Replace Patterns

### VS Code Find & Replace
- Find: `= asyncio\.create_task\((\w+)\(\)`
- Replace: `= await monitor.create_task($1(), "$1"`

(Then verify each replacement is correct)

---

## What TaskMonitor Does

```
Every 10 seconds:
├─ Check if task.done() == True
├─ If crashed: task.exception() shows error
├─ Restart with exponential backoff
├─ Alert Telegram after 3 failures
└─ Give up after 5 failures
```

---

## Health Endpoint

After integration, check:
```bash
curl http://localhost:8000/monitor/health
```

Expected:
```json
{
  "alive": 50,
  "dead": 0,
  "tasks": {
    "heartbeat_loop": {"alive": true, "restarts": 0},
    ...
  }
}
```

---

## If It Fails

1. **Validation fails?**
   - Run `python validate_integration.py` to see what's missing
   - Fix the specific lines it mentions

2. **Bot crashes?**
   - Restore: `cp main.py.backup main.py`
   - Check logs for import error
   - Try auto-replace option instead

3. **Task keeps crashing?**
   - That's TaskMonitor detecting a REAL crash
   - It's doing its job!
   - Check the error message in logs

---

## Timeline

| Step | Minutes | Milestone |
|------|---------|-----------|
| 1 | 2 | Read guide |
| 2 | 5 | Apply changes |
| 3 | 2 | Validate |
| 4 | 2 | Pre-flight check |
| 5 | 3 | Start bot |
| 6 | 1440 | Monitor 24+ hours |

**Total to integration:** 14 minutes  
**Total to verify:** 24 hours

---

## Checklist Before Starting Bot

- [ ] Backup: `cp main.py main.py.backup`
- [ ] Import added: `from task_monitor import init_monitor, get_monitor`
- [ ] Monitor initialized: `monitor = await init_monitor()`
- [ ] All 50 tasks use: `await monitor.create_task(...)`
- [ ] Shutdown added: `await monitor.shutdown()`
- [ ] Validation passes: `python validate_integration.py`
- [ ] Pre-flight passes: `python startup_safety_check.py`

---

## Success = No Silent Deaths

### Before TaskMonitor:
- Task crashes silently
- Bot continues but missing 1 function
- Within 2-4 hours: dependent failures cascade
- **Result:** Bot crashes

### After TaskMonitor:
- Task crashes detected in <10 seconds
- Auto-restarted with backoff
- Monitored continuously
- Telegram alerts if repeated
- **Result:** 24+ hour uptime

---

## Remember

✅ TaskMonitor = Task Health Monitor  
✅ Detects crashes automatically  
✅ Restarts failed tasks  
✅ Sends alerts on failures  
✅ Enables 24+ hour uptime  

🚀 **Let's do this!**

