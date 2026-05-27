# 🚨 Troubleshooting: "Bot Doesn't Respond to Telegram Commands"

## Problem: I Send `/status` in Telegram But The Bot Doesn't Reply

This guide will help you diagnose and fix the issue step-by-step.

---

## 🔍 Quick Diagnosis (Start Here)

### Step 1: Check if bot process is running

**Linux/Mac:**
```bash
ps aux | grep "python main.py"
```

**Windows (PowerShell):**
```powershell
Get-Process | Where-Object {$_.ProcessName -like "*python*"}
```

**Expected Output:**
```
user 12345  0.5 10.2 ... python main.py
```

❌ **If no output:** Bot process is NOT running → [Skip to "Starting The Bot"](#starting-the-bot)
✅ **If you see process:** Continue to Step 2

---

### Step 2: Check if port 8000 is listening

**Linux/Mac:**
```bash
netstat -tuln | grep 8000
# or
lsof -i :8000
```

**Windows (CMD):**
```cmd
netstat -ano | findstr :8000
```

**Expected Output:**
```
tcp  0  0  127.0.0.1:8000  0.0.0.0:*  LISTEN  12345
```

❌ **No output:** Port 8000 is NOT listening → [Check bot logs](#checking-logs)
✅ **If port is listening:** Continue to Step 3

---

### Step 3: Test if bot is responding

```bash
# Test health endpoint
curl http://localhost:8000/ping

# Expected response:
# {"ok": true, "uptime": 123}
```

❌ **Connection refused:** Port isn't listening properly → [Restart bot](#restarting-the-bot)
❌ **Timeout:** Bot is slow or hung → [Check logs](#checking-logs) then [restart](#restarting-the-bot)
✅ **OK response:** Bot is responding → [Check Telegram configuration](#checking-telegram-configuration)

---

## 🔧 Diagnosis Steps (Detailed)

### Checking Logs

**View recent logs:**
```bash
# Last 50 lines
tail -n 50 trading_bot.log

# Last 100 lines with timestamps
tail -n 100 -f trading_bot.log

# Search for errors
grep -i "error\|exception\|critical" trading_bot.log | tail -20

# Search for Telegram-specific errors
grep -i "telegram" trading_bot.log | tail -20
```

**What to look for:**
- ❌ `ERROR` - Something went wrong
- ⚠️ `WARNING` - Something might be wrong
- 🔴 `CRITICAL` - System issue preventing operation
- 🔐 `Telegram webhook` - Should show webhook registered
- ✅ `Startup checklist PASSED` - Bot initialized successfully

**Example good logs:**
```
2026-05-27 14:30:22 | main | INFO | === Trading Bot Started ===
2026-05-27 14:30:23 | main | INFO | ✅ Startup checklist PASSED
2026-05-27 14:30:24 | main | INFO | Telegram webhook registered
```

---

### Checking Telegram Configuration

#### Issue: Bot token is invalid

```bash
# Check if token is set
echo $TELEGRAM_BOT_TOKEN

# If empty or shows placeholder:
# ❌ Token is not configured in .env
```

**Fix:**
1. Get valid token from @BotFather in Telegram
2. Add to `.env`: `TELEGRAM_BOT_TOKEN=123456:ABC...`
3. Restart bot: `python main.py`

#### Issue: Chat ID is wrong

```bash
# Check if chat ID is set
echo $TELEGRAM_CHAT_ID

# If empty or shows placeholder:
# ❌ Chat ID is not configured in .env
```

**Fix:**
1. Get your chat ID: In Telegram, send any message to bot
2. Check bot logs: `grep "from_user_id\|chat_id" trading_bot.log`
3. Add to `.env`: `TELEGRAM_CHAT_ID=1234567890`
4. Restart bot

#### Issue: Webhook not registered

```bash
# Check if webhook is registered with Telegram
TELEGRAM_TOKEN="your-token-here"
curl "https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo" | jq

# Expected output:
{
  "ok": true,
  "result": {
    "url": "https://your-service.onrender.com/telegram/webhook",
    "has_custom_certificate": false,
    "pending_update_count": 0,
    "max_connections": 40,
    "allowed_updates": ["message", "callback_query"]
  }
}
```

❌ **If `url` is empty:** Webhook not registered
- For Render: Set `RENDER_EXTERNAL_URL` in environment
- For local: Use ngrok or localhost tunnel
- Bot auto-registers webhook on startup (check logs)

---

### Checking Network Connectivity

Can the bot reach Telegram?

```bash
# Test Telegram API
curl -I https://api.telegram.org/bot

# Test Alpaca API
curl -I https://api.alpaca.markets

# Test Groq API
curl -I https://api.groq.com

# Expected: HTTP 200, 400, 401, or 404 (NOT connection refused)
```

❌ **Connection refused or timeout:** Network issue
- Check internet connection: `ping 8.8.8.8`
- Check firewall rules
- Check if behind proxy (set HTTP_PROXY env var)

---

### Checking Database

Is the database working?

```bash
# Check if database file exists
ls -la trading_bot.db

# If not found:
# ❌ Database file missing or wrong path
```

**Check database integrity:**
```bash
sqlite3 trading_bot.db "SELECT COUNT(*) FROM trade_log;"

# Expected: Some number like 42
# ❌ If error: Database is corrupted
```

---

### Checking Configuration

Are all critical env vars set?

```bash
# Run startup checklist
python -c "
import asyncio
from startup_checklist import run_startup_checklist
success, checks = asyncio.run(run_startup_checklist())
print('\\n✅ SAFE TO START' if success else '\\n❌ CRITICAL ISSUES')
"
```

This will show:
- ✅ All required env vars are set
- ✅ Database is accessible
- ✅ APIs are reachable
- ❌ Any critical failures

---

## ⚙️ Solutions

### Starting The Bot

**Local machine:**
```bash
cd /path/to/trading-bot
python main.py
```

**In background (nohup):**
```bash
nohup python main.py > bot.log 2>&1 &

# Check it's running:
jobs
tail -f bot.log
```

**In screen (detachable):**
```bash
screen -S trading-bot
python main.py
# Press Ctrl+A then D to detach
# To reattach: screen -r trading-bot
```

**In tmux:**
```bash
tmux new-session -d -s trading-bot "python main.py"
tmux list-sessions
tmux attach -t trading-bot
```

### Restarting The Bot

**Graceful restart:**
```bash
# Kill existing process
pkill -f "python main.py"

# Wait a moment
sleep 2

# Verify it's dead
ps aux | grep "python main.py"

# Start new instance
python main.py
```

**Render dashboard restart:**
1. Go to https://dashboard.render.com
2. Select your web service
3. Click "Manual Deploy" → "Deploy Latest Commit"
4. Or click the "Restart" button

---

## 🚨 Advanced Issues

### Port 8000 is already in use

**Find what's using it:**
```bash
# Linux/Mac
lsof -i :8000
netstat -tuln | grep 8000

# Windows
netstat -ano | findstr :8000
```

**Kill the process:**
```bash
# Linux/Mac
kill -9 <PID>

# Windows
taskkill /PID <PID> /F
```

**Then restart bot:**
```bash
python main.py
```

---

### Bot is running but slow/hanging

**Check resource usage:**
```bash
# CPU and memory
top -p $(pgrep -f "python main.py")

# Or use monitoring dashboard:
curl http://localhost:8000/health/dashboard
```

**If memory is high (>500MB):**
1. Restart bot: `pkill -f "python main.py"`
2. Wait 2 minutes
3. Restart: `python main.py`
4. If persists, set `FULL_MODE=false` in .env to save memory

---

### Bot crashes on startup

**Check error message:**
```bash
python main.py 2>&1 | head -50
```

**Common errors:**

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `DATABASE_URL not set` | Check .env has DATABASE_URL |
| `Startup checklist FAILED` | Run checklist and fix critical issues |
| `port 8000 in use` | Kill other process on port 8000 |
| `permission denied` | Check file permissions: `chmod 644 *.db` |

---

### Telegram receives no updates at all

**Check webhook status:**

```bash
TELEGRAM_TOKEN="your-token"
curl "https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo"
```

**If `url` is empty:**
- Bot didn't register webhook
- Check logs: `grep -i webhook trading_bot.log`

**If webhook is registered but getting errors:**
- Check `last_error_message` in webhook info
- Verify `WEBHOOK_SECRET` is set correctly

**Manual webhook registration:**

```bash
TELEGRAM_TOKEN="your-token"
WEBHOOK_URL="https://your-domain.com/telegram/webhook"
WEBHOOK_SECRET="your-secret"

curl -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"$WEBHOOK_URL\",
    \"secret_token\": \"$WEBHOOK_SECRET\",
    \"drop_pending_updates\": false
  }"

# Should respond: {"ok":true,"result":true,...}
```

---

## 📋 Complete Diagnostic Checklist

If bot isn't working, go through this checklist:

- [ ] Process is running: `ps aux | grep "python main.py"`
- [ ] Port 8000 is listening: `netstat -tuln | grep 8000`
- [ ] Health endpoint responds: `curl http://localhost:8000/ping`
- [ ] Logs show no errors: `tail -f trading_bot.log`
- [ ] TELEGRAM_BOT_TOKEN is set: `echo $TELEGRAM_BOT_TOKEN`
- [ ] TELEGRAM_CHAT_ID is set: `echo $TELEGRAM_CHAT_ID`
- [ ] Webhook is registered: `curl https://api.telegram.org/bot.../getWebhookInfo`
- [ ] Database file exists: `ls -la trading_bot.db`
- [ ] Internet connectivity works: `ping 8.8.8.8`
- [ ] All critical env vars set: Run startup checklist
- [ ] No other app using port 8000: `lsof -i :8000`

---

## 🆘 If All Else Fails

1. **Gather diagnostics:**
   ```bash
   echo "=== Process ===" && ps aux | grep python
   echo "=== Port ===" && netstat -tuln | grep 8000
   echo "=== Health ===" && curl http://localhost:8000/ping
   echo "=== Errors ===" && grep ERROR trading_bot.log | tail -20
   echo "=== Config ===" && grep -E "TELEGRAM|ALPACA" .env
   ```

2. **Check monitoring dashboard:**
   ```
   http://localhost:8000/health/dashboard
   ```

3. **Send diagnostic command in Telegram:**
   ```
   /doctor
   ```
   This sends full diagnostics to Telegram

4. **Review DEPLOYMENT_GUIDE.md** for detailed setup instructions

5. **Check logs for startup issues:**
   ```bash
   tail -n 200 trading_bot.log | head -100
   ```

---

## 🔄 Prevention: Keep It Running

To prevent this problem from happening again:

1. **Use monitoring script:** `python monitor_bot.py` (auto-restarts if down)
2. **Deploy to Render:** Keeps bot running 24/7 automatically
3. **Use systemd (Linux):** Bot restarts on reboot
4. **Set up alerts:** Get notified if bot goes down

See `DEPLOYMENT_GUIDE.md` for detailed setup.

---

**Last Updated:** 2026-05-27  
**For quick health check:** `/ping` or `/health` in Telegram
