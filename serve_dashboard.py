"""
Live dashboard server  —  run:  python serve_dashboard.py
=========================================================
Serves a LIVE, auto-updating version of the paper-bot dashboard at
http://localhost:8080 . It regenerates the dashboard from the database every
45 seconds in the background, and the page auto-refreshes itself, so what you
see is always current — no manual re-running.

Completely separate from the trading bot (its own process, its own port 8080).
Read-only: shows the simulated ($10k) activity, never touches money or the bot.
Stop it anytime with Ctrl+C.
"""

import http.server
import socketserver
import threading
import time

import generate_dashboard

PORT = 8080
_page = {"html": "<!doctype html><meta charset='utf-8'><h1>טוען נתונים…</h1>"}


def _regen_loop():
    while True:
        try:
            _page["html"] = generate_dashboard.build()
        except Exception as e:  # never let a transient DB hiccup kill the server
            print(f"[dashboard] regen error: {type(e).__name__}: {e}")
        time.sleep(45)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = _page["html"].encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request logging
        pass


def main():
    threading.Thread(target=_regen_loop, daemon=True).start()
    time.sleep(2)  # let the first generation finish before serving
    print(f"✅ דשבורד חי רץ:  http://localhost:{PORT}")
    print("   (מתעדכן לבד כל 45 שניות · עצור עם Ctrl+C)")
    with socketserver.TCPServer(("127.0.0.1", PORT), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nנעצר.")


if __name__ == "__main__":
    main()
