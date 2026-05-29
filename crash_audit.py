"""
בדיקה מעמיקה לשורות קריסה אפשריות בבוט
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).parent

print("=" * 70)
print("🔍 בדיקת קריסות אפשריות - ניתוח מעמיק")
print("=" * 70)
print()

# 1. בדיקת heartbeat.py
print("1️⃣ בדיקת קריסות ב-heartbeat loops...")
heartbeat_file = BASE / "heartbeat.py"
if heartbeat_file.exists():
    content = heartbeat_file.read_text(encoding='utf-8')
    
    # חפש async functions שרצות בלולאה אינסופית
    issues = []
    
    if "while True:" in content:
        import re
        loops = len(re.findall(r'while True:', content))
        issues.append(f"  ⚠️  {loops} לולאות while True - כל אחת צריכה try/except")
    
    if "asyncio.create_task" not in content:
        issues.append(f"  ⚠️  לא נראה שיש monitring של crashed tasks")
    
    # בדוק אם יש exception handling
    if content.count("except Exception") < content.count("while True"):
        issues.append(f"  ⚠️  יש while loops ללא proper exception handling")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ heartbeat.py נראה בטוח")
else:
    print(f"  ❌ heartbeat.py לא קיים!")

print()

# 2. בדיקת database.py
print("2️⃣ בדיקת database connections...")
db_file = BASE / "database.py"
if db_file.exists():
    content = db_file.read_text(encoding='utf-8')
    
    issues = []
    
    # בדוק SQLite usage
    if "sqlite3.connect" in content:
        import re
        connects = len(re.findall(r'sqlite3\.connect', content))
        issues.append(f"  ⚠️  {connects} sqlite3.connect calls - כל אחד צריך connection pool")
    
    # בדוק transaction management
    if content.count("begin") == 0 and "transaction" not in content.lower():
        issues.append(f"  ⚠️  אין explicit transaction management - עלול להיות לוקה (deadlock)")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ database.py נראה בטוח")
else:
    print(f"  ❌ database.py לא קיים!")

print()

# 3. בדיקת broker.py
print("3️⃣ בדיקת broker connections...")
broker_file = BASE / "broker.py"
if broker_file.exists():
    content = broker_file.read_text(encoding='utf-8')
    
    issues = []
    
    # בדוק retry logic
    if "retry" not in content.lower():
        issues.append(f"  ⚠️  אין retry logic - API failures יגרמו לקריסה")
    
    # בדוק timeout
    if "timeout" not in content.lower():
        issues.append(f"  ⚠️  אין timeouts - API hangs יתקעו את הבוט")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ broker.py נראה בטוח")
else:
    print(f"  ❌ broker.py לא קיים!")

print()

# 4. בדיקת config.py settings
print("4️⃣ בדיקת settings...")
config_file = BASE / "config.py"
if config_file.exists():
    content = config_file.read_text(encoding='utf-8')
    
    issues = []
    
    # בדוק required settings
    if "TELEGRAM_BOT_TOKEN" not in content or "ALPACA_API_KEY" not in content:
        issues.append(f"  ⚠️  חסרים settings ודפים זה יגרום לקריסה בstartup")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ config.py נראה בטוח")
else:
    print(f"  ❌ config.py לא קיים!")

print()

# 5. בדיקת watchdog.py
print("5️⃣ בדיקת watchdog protection...")
watchdog_file = BASE / "watchdog.py"
if watchdog_file.exists():
    content = watchdog_file.read_text(encoding='utf-8')
    
    issues = []
    
    # בדוק hang detection
    if "HANG_TIMEOUT" not in content:
        issues.append(f"  ⚠️  אין hang detection - תקוע בלופ יקרוס")
    
    # בדוק port cleanup
    if "kill_process_on_port" not in content:
        issues.append(f"  ⚠️  אין port cleanup - 'port already in use' יקרוס")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ watchdog.py בנוי נכון")
else:
    print(f"  ⚠️  watchdog.py לא קיים!")

print()

# 6. בדיקת memory leaks
print("6️⃣ בדיקת potential memory leaks...")
issues = []

# בדוק אם יש caching בלי limits
if heartbeat_file.exists():
    content = heartbeat_file.read_text(encoding='utf-8')
    if "cache" in content.lower() and ("maxsize" not in content.lower() and "lru_cache" not in content):
        issues.append(f"  ⚠️  יש caching בלי size limit - יגדל לאינסוף")

# בדוק logfile rotation
if BASE.exists():
    logfiles = list(BASE.glob("*.log"))
    for lf in logfiles:
        size_mb = lf.stat().st_size / 1024 / 1024
        if size_mb > 100:
            issues.append(f"  ⚠️  {lf.name} בגודל {size_mb:.1f}MB - צריך rotation")

if issues:
    for issue in issues:
        print(issue)
else:
    print("  ✅ לא נראה memory leaks חמורים")

print()
print("=" * 70)
print("📊 סיכום - נקודות קריטיות:")
print("=" * 70)
print()
print("1. לול אינסופי בלי exception handling = קריסה שקטה (task מת בשתיקה)")
print("2. SQLite deadlock מתחרות = lockup מלא")  
print("3. API timeout בלי handling = hang בלי restart")
print("4. Port conflict = 'Address already in use' exit=1")
print("5. Memory leak מ-cache/logs = Out of Memory crash")
print()
print("✅ עדכן את הבוט עם הפיקסים למטה")
print()

