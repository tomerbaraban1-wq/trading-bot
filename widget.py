import customtkinter as ctk
import threading
import subprocess
import sys
import os
import json
import requests
from pathlib import Path

ctk.set_appearance_mode("dark")

API_URL = "http://localhost:8000"
BOT_PROCESS = None


class TradeBotWidget(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TradeBot")
        self.geometry("320x500+50+50")
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.95)
        self.configure(fg_color="#0d1117")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bot_running = False
        self.expanded = False

        self._build()
        self._auto_refresh()

    def _build(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#161b22", height=45, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="TradeBot", font=("Segoe UI", 16, "bold"),
                     text_color="#58a6ff").pack(side="left", padx=12)

        self.status_dot = ctk.CTkLabel(header, text="●", font=("Segoe UI", 18),
                                        text_color="#f87171")
        self.status_dot.pack(side="left", padx=5)

        self.status_text = ctk.CTkLabel(header, text="OFFLINE", font=("Segoe UI", 11, "bold"),
                                         text_color="#f87171")
        self.status_text.pack(side="left")

        # Start/Stop button
        self.toggle_btn = ctk.CTkButton(header, text="▶", width=35, height=30,
                                         fg_color="#15803d", hover_color="#16a34a",
                                         font=("Segoe UI", 14), command=self.toggle_bot)
        self.toggle_btn.pack(side="right", padx=8)

        # === Stats ===
        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.pack(fill="x", padx=10, pady=(10, 5))

        # Budget
        self._make_row(self.stats_frame, "Budget", "budget_val", "$0")
        self._make_row(self.stats_frame, "Used", "used_val", "0%")
        self.budget_bar = ctk.CTkProgressBar(self.stats_frame, height=6,
                                              progress_color="#2dd4bf", fg_color="#21262d")
        self.budget_bar.pack(fill="x", pady=(2, 8))
        self.budget_bar.set(0)

        # Separator
        ctk.CTkFrame(self, fg_color="#30363d", height=1).pack(fill="x", padx=10)

        # PnL
        pnl_frame = ctk.CTkFrame(self, fg_color="transparent")
        pnl_frame.pack(fill="x", padx=10, pady=8)

        self._make_row(pnl_frame, "Gross P&L", "pnl_gross_val", "$0.00")
        self._make_row(pnl_frame, "Net P&L", "pnl_net_val", "$0.00")
        self._make_row(pnl_frame, "Open P&L", "pnl_open_val", "$0.00")

        # Separator
        ctk.CTkFrame(self, fg_color="#30363d", height=1).pack(fill="x", padx=10)

        # Tax
        tax_frame = ctk.CTkFrame(self, fg_color="transparent")
        tax_frame.pack(fill="x", padx=10, pady=8)

        self._make_row(tax_frame, "Tax Reserved", "tax_res_val", "$0.00")
        self._make_row(tax_frame, "Tax Credit", "tax_cred_val", "$0.00")

        # Separator
        ctk.CTkFrame(self, fg_color="#30363d", height=1).pack(fill="x", padx=10)

        # Positions
        pos_frame = ctk.CTkFrame(self, fg_color="transparent")
        pos_frame.pack(fill="x", padx=10, pady=8)

        self._make_row(pos_frame, "Open Positions", "positions_count", "0")
        self.positions_list = ctk.CTkLabel(pos_frame, text="", font=("Consolas", 11),
                                            text_color="#8b949e", anchor="w", justify="left")
        self.positions_list.pack(fill="x")

        # Separator
        ctk.CTkFrame(self, fg_color="#30363d", height=1).pack(fill="x", padx=10)

        # Controls
        ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=10, pady=8)

        exit_row = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        exit_row.pack(fill="x")

        self.exit_entry = ctk.CTkEntry(exit_row, placeholder_text="AAPL", width=80,
                                        height=30, fg_color="#21262d", border_color="#30363d",
                                        font=("Segoe UI", 12))
        self.exit_entry.pack(side="left", padx=(0, 5))

        ctk.CTkButton(exit_row, text="Emergency Exit", fg_color="#b91c1c",
                       hover_color="#dc2626", height=30, font=("Segoe UI", 12),
                       command=self.emergency_exit).pack(side="left", fill="x", expand=True)

        # Dashboard button
        ctk.CTkButton(ctrl_frame, text="Open Full Dashboard", fg_color="#1d4ed8",
                       hover_color="#2563eb", height=30, font=("Segoe UI", 12),
                       command=self.open_dashboard).pack(fill="x", pady=(8, 0))

    def _make_row(self, parent, label, attr_name, default):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=1)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 12),
                     text_color="#8b949e").pack(side="left")
        val = ctk.CTkLabel(row, text=default, font=("Segoe UI", 12, "bold"),
                            text_color="#e6edf3")
        val.pack(side="right")
        setattr(self, attr_name, val)

    def toggle_bot(self):
        if self.bot_running:
            self.stop_bot()
        else:
            self.start_bot()

    def start_bot(self):
        global BOT_PROCESS
        if self.bot_running:
            return
        bot_dir = Path(__file__).parent
        BOT_PROCESS = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(bot_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        self.bot_running = True
        self.toggle_btn.configure(text="■", fg_color="#b91c1c", hover_color="#dc2626")
        threading.Thread(target=self._watch_process, daemon=True).start()

    def stop_bot(self):
        global BOT_PROCESS
        if BOT_PROCESS:
            BOT_PROCESS.terminate()
            BOT_PROCESS = None
        self.bot_running = False
        self.toggle_btn.configure(text="▶", fg_color="#15803d", hover_color="#16a34a")
        self.status_dot.configure(text_color="#f87171")
        self.status_text.configure(text="OFFLINE", text_color="#f87171")

    def _watch_process(self):
        global BOT_PROCESS
        if BOT_PROCESS:
            BOT_PROCESS.wait()
        BOT_PROCESS = None
        self.bot_running = False
        self.after(0, lambda: self.toggle_btn.configure(
            text="▶", fg_color="#15803d", hover_color="#16a34a"))
        self.after(0, lambda: self.status_dot.configure(text_color="#f87171"))
        self.after(0, lambda: self.status_text.configure(text="OFFLINE", text_color="#f87171"))

    def _auto_refresh(self):
        threading.Thread(target=self._fetch, daemon=True).start()
        self.after(5000, self._auto_refresh)

    def _fetch(self):
        try:
            status = requests.get(f"{API_URL}/status", timeout=3).json()
            tax = requests.get(f"{API_URL}/tax", timeout=3).json()
            health = requests.get(f"{API_URL}/health", timeout=3).json()
            self.after(0, lambda: self._update(status, tax, health))
        except Exception:
            self.after(0, self._set_offline)

    def _set_offline(self):
        self.status_dot.configure(text_color="#f87171")
        self.status_text.configure(text="OFFLINE", text_color="#f87171")

    def _update(self, status, tax, health):
        b = status.get("budget", {})

        # Status
        self.status_dot.configure(text_color="#2dd4bf")
        self.status_text.configure(text="RUNNING", text_color="#2dd4bf")

        # Budget
        total = b.get("total_budget", 0)
        used = b.get("budget_used_pct", 0)
        self.budget_val.configure(text=f"${total:,.0f}")
        self.used_val.configure(text=f"{used:.1f}%")
        self.budget_bar.set(min(used / 100, 1.0))
        color = "#f87171" if used > 80 else "#f0883e" if used > 50 else "#2dd4bf"
        self.budget_bar.configure(progress_color=color)

        # PnL
        gross = b.get("realized_pnl_gross", 0)
        net = b.get("realized_pnl_net", 0)
        opnl = b.get("open_pnl", 0)
        self.pnl_gross_val.configure(text=f"${gross:+,.2f}",
                                      text_color="#2dd4bf" if gross >= 0 else "#f87171")
        self.pnl_net_val.configure(text=f"${net:+,.2f}",
                                    text_color="#2dd4bf" if net >= 0 else "#f87171")
        self.pnl_open_val.configure(text=f"${opnl:+,.2f}",
                                     text_color="#2dd4bf" if opnl >= 0 else "#f87171")

        # Tax
        self.tax_res_val.configure(text=f"${tax.get('tax_reserved', 0):,.2f}")
        self.tax_cred_val.configure(text=f"${tax.get('tax_credit', 0):,.2f}")

        # Positions
        open_trades = status.get("open_trades", [])
        positions = status.get("positions", [])
        self.positions_count.configure(text=str(len(open_trades)))

        if open_trades:
            lines = []
            for t in open_trades:
                pos = next((p for p in positions if p["ticker"] == t["ticker"]), None)
                pnl = pos["unrealized_pl"] if pos else 0
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  {t['ticker']}  x{t['qty']}  ${sign}{pnl:.2f}")
            self.positions_list.configure(text="\n".join(lines))
        else:
            self.positions_list.configure(text="")

    def emergency_exit(self):
        ticker = self.exit_entry.get().strip().upper()
        if not ticker:
            return
        # Get secret from env (.env via dotenv) or local file fallback
        secret = os.environ.get("WEBHOOK_SECRET", "")
        if not secret:
            try:
                from config import settings as _s
                secret = _s.WEBHOOK_SECRET
            except Exception:
                pass
        if not secret:
            return  # no secret configured — can't authenticate
        try:
            requests.post(
                f"{API_URL}/emergency-exit/{ticker}",
                params={"secret": secret},
                timeout=5,
            )
        except Exception:
            pass

    def open_dashboard(self):
        os.startfile("http://localhost:8000")

    def on_close(self):
        self.stop_bot()
        self.destroy()


if __name__ == "__main__":
    app = TradeBotWidget()
    app.mainloop()
