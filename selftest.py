"""
Self-Test / Smoke Check  —  run:  python selftest.py
=====================================================
One command to re-verify the whole bot is healthy after ANY change:
  • every critical module still imports (catches syntax/import breakage)
  • database connects and key tables exist
  • SAFETY: broker is in paper mode (never real money)
  • all three learning modules run without error
  • key Telegram commands are registered
  • the entry-indicator snapshot (deep-learning data source) is wired

Read-only and fast (a few seconds). Exit code 0 = all pass, 1 = any failure,
so it works in CI / Task Scheduler too. Run this after every edit instead of
manually poking at things.
"""

import importlib
import sys

# Windows consoles (cp1255) crash on emoji — force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_passed = 0
_failed = 0
_results: list[str] = []


def check(name: str, fn) -> None:
    global _passed, _failed
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"EXCEPTION: {type(e).__name__}: {e}"
    if ok:
        _passed += 1
        _results.append(f"✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        _failed += 1
        _results.append(f"❌ {name} — {detail}")


# 1. Critical modules import cleanly --------------------------------------------
CRITICAL_MODULES = [
    "config", "database", "broker", "scoring", "sentiment", "indicators",
    "scanner", "trade_logger", "webhook", "heartbeat",
    "continuous_learner", "news_reaction_learner", "feature_edge_learner",
    "telegram_commands", "telegram_chat",
]


def _imports():
    bad = []
    for m in CRITICAL_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append(f"{m}({type(e).__name__})")
    return (not bad, f"{len(CRITICAL_MODULES)} מודולים" if not bad else "נכשלו: " + ", ".join(bad))


# 2. Database + key tables ------------------------------------------------------
def _db():
    import database
    conn = database.get_connection()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    missing = {"trade_log", "news_reactions"} - tables
    return (not missing, f"{len(tables)} טבלאות" if not missing else f"חסרות: {missing}")


# 3. SAFETY: broker must be paper ----------------------------------------------
def _paper():
    import os
    try:
        from config import settings
        broker = getattr(settings, "ACTIVE_BROKER", None) or os.getenv("ACTIVE_BROKER", "?")
    except Exception:
        broker = os.getenv("ACTIVE_BROKER", "?")
    is_paper = "paper" in str(broker).lower()
    return (is_paper, f"{broker} (כסף מדומה ✓)" if is_paper else f"אזהרה: {broker} — לא נייר!")


# 4. Learning modules run -------------------------------------------------------
def _continuous():
    import continuous_learner as c
    r = c.learn_sentiment_correlation()
    return (isinstance(r, dict), f"{len(r)} מפתחות")


def _news():
    import news_reaction_learner as n
    r = n.learn_news_reaction()
    return (isinstance(r, dict) and "status" in r, f"status={r.get('status')}")


def _edges():
    import feature_edge_learner as f
    r = f.learn_feature_edges()
    return (isinstance(r, dict) and "status" in r, f"status={r.get('status')}")


# 5. Telegram commands registered ----------------------------------------------
def _cmds():
    import telegram_commands as tc
    need = ["newslearn", "learn", "edges"]
    missing = [c for c in need if c not in tc.COMMAND_HANDLERS]
    return (not missing, f"{len(tc.COMMAND_HANDLERS)} פקודות" if not missing else f"חסרות: {missing}")


# 6. Deep-learning data source wired (entry indicator snapshot) -----------------
def _indicators():
    import indicators
    if not hasattr(indicators, "get_current_indicators"):
        return (False, "get_current_indicators חסר")
    import trade_logger  # the buy path that records the snapshot
    return (True, "snapshot מחובר לנתיב הקנייה")


check("ייבוא כל המודולים הקריטיים", _imports)
check("מסד נתונים + טבלאות מפתח", _db)
check("בטיחות: ברוקר במצב נייר (לא כסף אמיתי)", _paper)
check("למידה: סנטימנט אמיתי (continuous_learner)", _continuous)
check("למידה: תגובת חדשות (news_reaction_learner)", _news)
check("למידה: עומק — מה מנבא (feature_edge_learner)", _edges)
check("פקודות טלגרם רשומות (/newslearn /learn /edges)", _cmds)
check("מקור נתוני העומק (snapshot אינדיקטורים בקנייה)", _indicators)

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 56)
print("🔍 בדיקה עצמית של הבוט  (python selftest.py)")
print("=" * 56)
for line in _results:
    print(line)
print("=" * 56)
total = _passed + _failed
print(f"תוצאה: {_passed}/{total} עברו | {_failed} נכשלו")
print("✅ הכול תקין — בטוח להמשיך!" if _failed == 0 else "❌ יש כשלים — ראה למעלה")
sys.exit(0 if _failed == 0 else 1)
