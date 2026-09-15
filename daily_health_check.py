#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily EXTERNAL health check for the trading bot.

Runs OUTSIDE the bot process (via Windows Task Scheduler) so it can catch even a
fully-crashed bot — something the in-process health monitor can never report.
Checks the HTTP endpoints, the background-task monitor, and the database, then
sends a Telegram digest: green if everything is fine, red (with details) if not.

Pure standard library (urllib + sqlite3) — no third-party dependencies, so it
keeps working regardless of the bot's virtualenv state.
"""
import json
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def http_get(port: int, path: str, timeout: int = 10):
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def main() -> int:
    env = load_env(BASE / ".env")
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")
    port = int(env.get("PORT", "8000"))
    db_path = BASE / "data" / "trading.db"

    issues: list[str] = []
    info: list[str] = []

    # 1) Is the bot answering at all? (detects a fully-down bot)
    bot_up = False
    try:
        st, body = http_get(port, "/health")
        if st == 200:
            bot_up = True
            d = json.loads(body)
            info.append(
                f"זמן פעילות {int(d.get('uptime_seconds', 0)) // 60} דק' | "
                f"{d.get('open_positions', '?')} פוזיציות"
            )
        else:
            issues.append(f"/health החזיר HTTP {st}")
    except Exception as e:
        issues.append(f"הבוט לא מגיב על פורט {port} — ייתכן שהוא קרוס ({type(e).__name__})")

    # 2) Background-task monitor — any dead loops?
    if bot_up:
        try:
            st, body = http_get(port, "/monitor/health")
            d = json.loads(body)
            total, alive, dead = d.get("total_tasks"), d.get("alive"), d.get("dead", 0)
            if dead:
                deadlist = [k for k, v in d.get("tasks", {}).items() if not v.get("alive")]
                issues.append(f"{dead} משימות רקע מתות: {', '.join(deadlist[:5])}")
            else:
                info.append(f"{alive}/{total} משימות רקע חיות")
        except Exception as e:
            issues.append(f"בדיקת משימות נכשלה ({type(e).__name__})")

    # 3) Database integrity + broker/SQLite open-position count
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        chk = con.execute("PRAGMA quick_check").fetchone()[0]
        if chk != "ok":
            issues.append(f"תקינות DB: {chk}")
        open_n = con.execute("SELECT COUNT(*) FROM trade_log WHERE status='open'").fetchone()[0]
        con.close()
        info.append(f"DB תקין | {open_n} עסקאות פתוחות")
    except Exception as e:
        issues.append(f"בדיקת DB נכשלה ({type(e).__name__})")

    # Build the Telegram message
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if issues:
        lines = ["🔴 <b>בדיקת בריאות יומית — נמצאו בעיות</b>", f"🕐 {now}", "━━━━━━━━━━━━"]
        lines += [f"• {x}" for x in issues]
        if info:
            lines += ["", "<i>מה שכן תקין:</i>"] + [f"• {x}" for x in info]
    else:
        lines = ["✅ <b>בדיקת בריאות יומית — הכל תקין</b>", f"🕐 {now}", "━━━━━━━━━━━━"]
        lines += [f"• {x}" for x in info]
    msg = "\n".join(lines)

    # Send to Telegram
    sent = False
    if token and chat_id:
        try:
            data = urllib.parse.urlencode(
                {"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}
            ).encode()
            with urllib.request.urlopen(
                f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15
            ) as r:
                sent = json.loads(r.read().decode()).get("ok", False)
        except Exception as e:
            print(f"Telegram send failed: {e}")
    else:
        print("No TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env")

    # Local log line for debuggability
    try:
        with open(BASE / "daily_health_check.log", "a", encoding="utf-8") as f:
            f.write(f"{now} | issues={len(issues)} | telegram_sent={sent}\n")
    except Exception:
        pass

    print(msg)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
