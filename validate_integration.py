"""
Validation script for TaskMonitor integration
Checks that all required changes have been made to main.py
"""

import re
import sys
from pathlib import Path

print("\n" + "="*80)
print("[VALIDATION] TaskMonitor Integration Check")
print("="*80 + "\n")

main_py_path = Path(__file__).parent / "main.py"

if not main_py_path.exists():
    print(f"[ERROR] main.py not found at {main_py_path}")
    sys.exit(1)

main_py_content = main_py_path.read_text(encoding='utf-8')

checks = [
    {
        "name": "Import TaskMonitor",
        "pattern": r"from task_monitor import init_monitor, get_monitor",
        "required": True
    },
    {
        "name": "Initialize monitor",
        "pattern": r"monitor\s*=\s*await init_monitor\(\)",
        "required": True
    },
    {
        "name": "Replace asyncio.create_task with monitor.create_task",
        "pattern": r"await monitor\.create_task\(heartbeat_loop\(\), \"heartbeat_loop\"\)",
        "required": True
    },
    {
        "name": "Replace asyncio.create_task with monitor.create_task (stop_loss)",
        "pattern": r"await monitor\.create_task\(stop_loss_monitor\(\), \"stop_loss_monitor\"\)",
        "required": True
    },
    {
        "name": "Replace asyncio.create_task with monitor.create_task (at least 30 tasks)",
        "pattern": r"await monitor\.create_task",
        "min_count": 30,
        "required": True
    },
    {
        "name": "TaskMonitor shutdown in cleanup",
        "pattern": r"await monitor\.shutdown\(\)",
        "required": True
    },
    {
        "name": "Log TaskMonitor initialization",
        "pattern": r"\[CRASH_PREVENTION\].*TaskMonitor",
        "required": True
    },
    {
        "name": "Health check endpoints defined or imported",
        "pattern": r"(@app\.get.*monitor.*health|from health_endpoints import|setup_health_endpoints)",
        "required": False  # Optional but recommended
    }
]

print("[1] Checking main.py for required changes...\n")

results = {
    "passed": 0,
    "failed": 0,
    "optional_missing": 0,
    "errors": []
}

for i, check in enumerate(checks, 1):
    print(f"[{i}/{len(checks)}] {check['name']}...", end=" ")

    if "min_count" in check:
        # Check for minimum number of matches
        matches = re.findall(check["pattern"], main_py_content, re.MULTILINE)
        count = len(matches)
        min_required = check["min_count"]

        if count >= min_required:
            print(f"[OK] Found {count} instances")
            results["passed"] += 1
        else:
            print(f"[FAIL] Found {count}, need {min_required}")
            results["failed"] += 1
            results["errors"].append(
                f"  {check['name']}: Found {count}, expected at least {min_required}"
            )
    else:
        # Check for pattern existence
        if re.search(check["pattern"], main_py_content, re.MULTILINE | re.DOTALL):
            print("[OK]")
            results["passed"] += 1
        else:
            if check["required"]:
                print("[FAIL] - REQUIRED")
                results["failed"] += 1
                results["errors"].append(f"  Missing: {check['name']}")
            else:
                print("[SKIP] - Optional")
                results["optional_missing"] += 1

print("\n" + "="*80)
print("[SUMMARY]")
print("="*80)
print(f"\nPassed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Optional missing: {results['optional_missing']}")

if results["errors"]:
    print("\n[ERRORS] Items that need fixing:")
    for error in results["errors"]:
        print(error)

print("\n" + "="*80)

if results["failed"] == 0:
    print("[SUCCESS] All required TaskMonitor integrations are in place!")
    print("\nNext steps:")
    print("  1. Run: python startup_safety_check.py")
    print("  2. Start the bot: python -m uvicorn main:app --reload")
    print("  3. Check health: curl http://localhost:8000/monitor/health")
    print("\n" + "="*80 + "\n")
    sys.exit(0)
else:
    print("[FAILURE] Some required changes are missing!")
    print("\nHow to fix:")
    print("  1. Open main.py")
    print("  2. Follow the steps in INTEGRATION_PHASE_1.md")
    print("  3. Or copy the lifespan function from main_lifespan_UPDATED.py")
    print("  4. Re-run this validation script")
    print("\n" + "="*80 + "\n")
    sys.exit(1)
