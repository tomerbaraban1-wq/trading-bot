#!/usr/bin/env python3
"""
Update Telegram Webhook URL
============================

Use this script when:
1. Running bot locally with ngrok
2. Moving from Render to another host
3. Render service is suspended

Usage:
    python update_webhook.py <new-url>
    # OR run interactively:
    python update_webhook.py
"""

import sys
import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()


def get_current_webhook():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        return None

    url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
        result = data.get("result", {})
        print("=== Current Webhook ===")
        print(f"  URL:     {result.get('url', 'NONE')}")
        print(f"  Pending: {result.get('pending_update_count', 0)} messages")
        print(f"  Error:   {result.get('last_error_message', 'none')}")
        print()
        return result


def set_webhook(new_url: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    secret = os.getenv("WEBHOOK_SECRET", "")

    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        return False

    webhook_url = f"{new_url.rstrip('/')}/telegram/webhook"

    payload = {
        "url": webhook_url,
        "drop_pending_updates": False,
    }
    if secret:
        payload["secret_token"] = secret

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/setWebhook",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
        if data.get("ok"):
            print(f"✅ Webhook updated to: {webhook_url}")
            return True
        else:
            print(f"❌ Failed: {data.get('description')}")
            return False


def delete_webhook():
    """Remove webhook — bot will use polling instead."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/deleteWebhook",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
        if data.get("ok"):
            print("✅ Webhook deleted — bot will use polling")
        else:
            print(f"❌ Failed: {data.get('description')}")


def main():
    print("━" * 50)
    print("🔔 Telegram Webhook Manager")
    print("━" * 50)
    print()

    # Show current state
    get_current_webhook()

    # Get new URL
    if len(sys.argv) > 1:
        new_url = sys.argv[1]
    else:
        print("Options:")
        print("  1. Set ngrok URL (running locally)")
        print("  2. Set Render URL (https://your-service.onrender.com)")
        print("  3. Delete webhook (use polling)")
        print("  q. Quit")
        print()

        while True:
            choice = input("Enter choice or full URL: ").strip()

            if choice == "q":
                return
            elif choice == "3":
                delete_webhook()
                return
            elif choice.startswith("https://"):
                new_url = choice
                break
            else:
                print("Enter a valid HTTPS URL or option number")
                continue

    # Update webhook
    print(f"Setting webhook to: {new_url}")
    success = set_webhook(new_url)

    if success:
        print()
        print("━" * 50)
        print("✅ Done! Test by sending /start to your bot in Telegram.")
        print()
        print("IMPORTANT: Your bot must be running and reachable at this URL.")
        print(f"  Local bot:  python main.py")
        print(f"  ngrok:      ngrok http 8000")
        print("━" * 50)
    else:
        print()
        print("❌ Failed to update webhook. Check your token in .env")


if __name__ == "__main__":
    main()
