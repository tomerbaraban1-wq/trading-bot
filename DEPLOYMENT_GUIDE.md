# 🚀 Bot Deployment & Monitoring Guide

## Overview

The trading bot is a **FastAPI web server** that must run continuously to receive Telegram webhook events. This guide explains how to keep it running reliably and detect when it goes down.

---

## ⚠️ Critical Understanding

### Why The Bot Goes Down

The bot requires:
1. **Python process running** the FastAPI server (`python main.py`)
2. **Port 8000 listening** for incoming HTTP requests
3. **Telegram webhook registered** pointing to your server
4. **All critical env vars set** (TELEGRAM_BOT_TOKEN, ALPACA_API_KEY, etc.)
5. **Network connectivity** to reach Telegram, Alpaca, and other APIs

If ANY of these fail, the bot stops responding to Telegram commands.

### What Happens When Bot Is Down

- 🔴 Commands in Telegram get **no response**
- 🔴 Telegram sends webhook events to bot URL - **500 Error** (connection refused)
- 🔴 Trades continue in broker but **bot doesn't monitor positions**
- 🔴 **No alerts** about price movements or risk

---

## 📋 Pre-Flight Checklist

Before starting the bot, verify:

```bash
# 1. Check if port 8000 is free
netstat -tuln | grep 8000    # Linux/Mac
netstat -ano | findstr :8000 # Windows

# 2. Verify .env file has all critical vars
grep -E "TELEGRAM_BOT_TOKEN|ALPACA_API_KEY|GROQ_API_KEY" .env

# 3. Check database file exists
ls -la trading_bot.db        # Linux/Mac
dir trading_bot.db           # Windows

# 4. Test network connectivity
ping api.telegram.org
ping api.alpaca.markets
```

---

## 🖥️ Option 1: Running Locally (Development)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Environment
```bash
# Copy template and fill in your credentials
cp .env.template .env
nano .env  # Edit with your actual API keys
```

### Step 3: Verify Configuration
```bash
# Run startup checklist (will show all issues)
python -c "
import asyncio
from startup_checklist import run_startup_checklist
asyncio.run(run_startup_checklist())
"
```

### Step 4: Start The Bot
```bash
python main.py
```

**Output should show:**
```
| ... | INFO | === Trading Bot Started ===
| ... | INFO | ✅ Startup checklist PASSED
| ... | INFO | Telegram webhook registered: ...
```

### Step 5: Verify Bot Is Running
```bash
# In another terminal, test the bot
curl http://localhost:8000/ping

# Expected response:
# {"ok": true, "uptime": 123}
```

### Step 6: Keep Bot Running (Development Options)

**Option A: Using nohup (background)**
```bash
nohup python main.py > bot.log 2>&1 &
# Check logs: tail -f bot.log
```

**Option B: Using screen (detachable session)**
```bash
screen -S trading-bot
python main.py
# Press Ctrl+A then D to detach
# Reattach with: screen -r trading-bot
```

**Option C: Using tmux (more powerful)**
```bash
tmux new-session -d -s trading-bot "python main.py"
# Check: tmux ls
# Reattach: tmux attach -t trading-bot
```

---

## ☁️ Option 2: Deploying to Render (Recommended)

Render automatically keeps your bot running 24/7.

### Step 1: Create Render Account
- Go to https://render.com
- Sign up with GitHub or email

### Step 2: Create Web Service

1. Click "New" → "Web Service"
2. Connect your GitHub repository (or manually upload)
3. Fill in settings:

| Setting | Value |
|---------|-------|
| **Name** | `trading-bot` |
| **Environment** | Python 3.11 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python main.py` |
| **Instance Type** | Free (or Starter+) |

### Step 3: Add Environment Variables

Copy all your `.env` variables into Render's Environment panel:
- `TELEGRAM_BOT_TOKEN=xxx`
- `ALPACA_API_KEY=xxx`
- `ALPACA_SECRET_KEY=xxx`
- etc.

**IMPORTANT:** Set these in Render's dashboard, don't commit `.env` to Git!

### Step 4: Add Render Webhook URL

Render automatically provides an external URL. Set this in `.env`:
```bash
RENDER_EXTERNAL_URL=https://your-service-name.onrender.com
```

The bot will auto-register the Telegram webhook on startup.

### Step 5: Deploy

Push to your connected GitHub branch:
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

Render will automatically build and start your bot.

### Step 6: Verify Deployment

```bash
# Check the bot is responding
curl https://your-service-name.onrender.com/ping

# Check health dashboard
https://your-service-name.onrender.com/health/dashboard

# Telegram should receive startup message within 1 minute
```

**Render keeps bot running 24/7 automatically!**

---

## 🔍 Monitoring & Health Checks

### Quick Health Check
```bash
# Local
curl http://localhost:8000/health | jq

# Render
curl https://your-service.onrender.com/health | jq
```

### Visual Health Dashboard
```
http://localhost:8000/health/dashboard
https://your-service.onrender.com/health/dashboard
```

Shows:
- ✅ Broker connectivity
- ✅ Telegram API status
- ✅ Database health
- ✅ CPU/Memory usage
- ⚠️ Error rates by component

### Telegram Commands for Health

```
/health        - Quick health status
/uptime        - How long bot has been running
/budget        - Current configuration
/doctor        - Full diagnostic report
```

---

## 🚨 Emergency Restart

If bot stops responding:

### Local Machine
```bash
# Find the process
ps aux | grep "python main.py"

# Kill it
kill -9 <PID>

# Restart
python main.py
```

### Render Dashboard
1. Go to https://dashboard.render.com
2. Click your web service
3. Click "Manual Deploy" → "Deploy Latest Commit"
   - OR press the "Restart" button

---

## 📊 Automated Monitoring Script

Create `monitor_bot.py` to periodically check bot health:

```python
#!/usr/bin/env python3
"""Monitor trading bot health and auto-restart if down."""

import requests
import subprocess
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

HEALTH_URL = "http://localhost:8000/ping"
CHECK_INTERVAL = 60  # seconds
MAX_RETRIES = 3

def is_bot_running():
    """Check if bot is responding to health endpoint."""
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False

def restart_bot():
    """Restart the bot process."""
    logger.warning("🔴 Bot not responding - attempting restart...")
    try:
        # Kill existing process
        subprocess.run("pkill -f 'python main.py'", shell=True)
        time.sleep(2)

        # Start new process
        subprocess.Popen(
            ["python", "main.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        logger.info("🟢 Bot restarted")
        time.sleep(5)  # Give it time to start

        if is_bot_running():
            logger.info("✅ Bot is responding after restart")
            return True
        else:
            logger.error("❌ Bot still not responding after restart")
            return False
    except Exception as e:
        logger.error(f"❌ Restart failed: {e}")
        return False

def main():
    """Main monitoring loop."""
    logger.info("🤖 Starting bot monitor...")
    logger.info(f"Checking {HEALTH_URL} every {CHECK_INTERVAL}s")

    consecutive_failures = 0

    while True:
        try:
            if is_bot_running():
                consecutive_failures = 0
                logger.debug("✅ Bot is healthy")
            else:
                consecutive_failures += 1
                logger.warning(
                    f"⚠️  Bot not responding ({consecutive_failures}/{MAX_RETRIES})"
                )

                if consecutive_failures >= MAX_RETRIES:
                    logger.error("❌ Bot failed max retries - restarting...")
                    restart_bot()
                    consecutive_failures = 0

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
```

### Usage
```bash
# Run monitor in background
nohup python monitor_bot.py > monitor.log 2>&1 &

# Or in screen/tmux
screen -S monitor -d python monitor_bot.py
tmux new-session -d -s monitor "python monitor_bot.py"
```

---

## 🔐 Security Best Practices

### 1. Never Commit `.env`
```bash
# Add to .gitignore
echo ".env" >> .gitignore
git rm --cached .env
```

### 2. Use Environment Variables
- **Local:** Store in `.env` file
- **Render:** Store in dashboard, never commit
- **Docker:** Use build args or secrets

### 3. Restrict Webhook Access
- Bot verifies `X-Telegram-Bot-Api-Secret-Token` header
- Only Telegram can POST to `/telegram/webhook`

### 4. Regular Updates
```bash
# Check for security updates
pip list --outdated
pip install --upgrade pip setuptools
pip install -r requirements.txt --upgrade
```

---

## 📈 Troubleshooting

### "Bot doesn't respond to Telegram commands"

**Checklist:**
```bash
# 1. Is the process running?
ps aux | grep python
netstat -tuln | grep 8000

# 2. Check logs for errors
tail -f trading_bot.log

# 3. Is port 8000 free?
lsof -i :8000  # or netstat on Windows

# 4. Can you reach the health endpoint?
curl http://localhost:8000/ping

# 5. Check .env has all critical vars
grep TELEGRAM_BOT_TOKEN .env

# 6. Is Telegram webhook registered?
curl https://api.telegram.org/bot{TOKEN}/getWebhookInfo
```

### "Port 8000 already in use"

```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill it
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### "Database locked / can't write"

```bash
# Check database file permissions
ls -la trading_bot.db

# Set correct permissions
chmod 644 trading_bot.db
chmod 755 $(dirname trading_bot.db)
```

### "API key rejected"

```bash
# Verify key format is correct (no extra spaces)
cat .env | grep API_KEY

# Re-check key validity in Alpaca dashboard
# Re-request key from Groq if expired
```

---

## 📞 Getting Help

If bot stops working:

1. **Check `/health` endpoint** - shows all diagnostics
2. **Read logs** - `tail -f trading_bot.log` or Render logs
3. **Run startup checklist** - identifies missing config
4. **Check Telegram** - send `/doctor` command for full report
5. **Check network** - verify internet connectivity

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Start locally | `python main.py` |
| Check health | `curl http://localhost:8000/ping` |
| View logs | `tail -f trading_bot.log` |
| Kill bot | `pkill -f 'python main.py'` |
| View health dashboard | `http://localhost:8000/health/dashboard` |
| Full diagnostics | Send `/doctor` to Telegram bot |
| Deploy to Render | Push to GitHub (auto-deploys) |
| Restart on Render | Click "Restart" in dashboard |

---

**Last Updated:** 2026-05-27  
**Bot Status:** Always check `/health` before assuming it's down.
