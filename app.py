import customtkinter as ctk
import threading
import subprocess
import time
import json
import sys
import os
import requests
from pathlib import Path

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

API_URL = "http://localhost:8000"
BOT_PROCESS = None


class TradeBotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TradeBot")
        self.geometry("1000x700")
        self.minsize(900, 600)
        self.configure(fg_color="#0a0e17")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bot_running = False

        self._build_ui()
        self._start_auto_refresh()

    def _build_ui(self):
        # === TOP BAR ===
        top = ctk.CTkFrame(self, fg_color="#161b22", height=60, corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="TradeBot", font=("Segoe UI", 24, "bold"),
                     text_color="#58a6ff").pack(side="left", padx=20)

        self.status_label = ctk.CTkLabel(top, text="  OFFLINE  ", font=("Segoe UI", 13, "bold"),
                                          fg_color="#3b1c1c", text_color="#f87171",
                                          corner_radius=12, width=100)
        self.status_label.pack(side="left", padx=10)

        self.uptime_label = ctk.CTkLabel(top, text="", font=("Segoe UI", 12),
                                          text_color="#8b949e")
        self.uptime_label.pack(side="left", padx=5)

        # Buttons
        self.stop_btn = ctk.CTkButton(top, text="Stop", fg_color="#b91c1c",
                                       hover_color="#dc2626", width=80,
                                       command=self.stop_bot)
        self.stop_btn.pack(side="right", padx=5, pady=10)

        self.start_btn = ctk.CTkButton(top, text="Start", fg_color="#15803d",
                                        hover_color="#16a34a", width=80,
                                        command=self.start_bot)
        self.start_btn.pack(side="right", padx=5, pady=10)

        # === MAIN AREA ===
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=10)

        # Top cards row
        cards_row = ctk.CTkFrame(main, fg_color="transparent")
        cards_row.pack(fill="x", pady=(0, 10))
        cards_row.columnconfigure((0, 1, 2, 3), weight=1)

        # Card 1: Budget
        self.budget_card = self._make_card(cards_row, "BUDGET", 0)
        self.budget_val = ctk.CTkLabel(self.budget_card, text="$0", font=("Segoe UI", 32, "bold"),
                                        text_color="#e6edf3")
        self.budget_val.pack(pady=(5, 0))
        self.budget_bar = ctk.CTkProgressBar(self.budget_card, width=200, height=8,
                                              progress_color="#2dd4bf", fg_color="#21262d")
        self.budget_bar.pack(pady=5)
        self.budget_bar.set(0)
        self.budget_sub = ctk.CTkLabel(self.budget_card, text="0% in use",
                                        font=("Segoe UI", 11), text_color="#8b949e")
        self.budget_sub.pack()

        info_row = ctk.CTkFrame(self.budget_card, fg_color="transparent")
        info_row.pack(fill="x", pady=(8, 0))
        self.cash_label = self._make_stat(info_row, "$0", "Cash", "left")
        self.positions_label = self._make_stat(info_row, "$0", "Positions", "right")

        # Card 2: P&L
        self.pnl_card = self._make_card(cards_row, "PROFIT & LOSS", 1)
        self.pnl_gross = ctk.CTkLabel(self.pnl_card, text="$0.00", font=("Segoe UI", 32, "bold"),
                                       text_color="#2dd4bf")
        self.pnl_gross.pack(pady=(5, 2))
        ctk.CTkLabel(self.pnl_card, text="Gross P&L", font=("Segoe UI", 11),
                     text_color="#8b949e").pack()

        info_row2 = ctk.CTkFrame(self.pnl_card, fg_color="transparent")
        info_row2.pack(fill="x", pady=(8, 0))
        self.pnl_net_label = self._make_stat(info_row2, "$0", "Net", "left")
        self.pnl_open_label = self._make_stat(info_row2, "$0", "Open P&L", "right")

        # Card 3: Tax
        self.tax_card = self._make_card(cards_row, "TAX VAULT", 2)
        self.tax_reserved = ctk.CTkLabel(self.tax_card, text="$0.00", font=("Segoe UI", 32, "bold"),
                                          text_color="#f0883e")
        self.tax_reserved.pack(pady=(5, 2))
        ctk.CTkLabel(self.tax_card, text="Reserved (25%)", font=("Segoe UI", 11),
                     text_color="#8b949e").pack()

        info_row3 = ctk.CTkFrame(self.tax_card, fg_color="transparent")
        info_row3.pack(fill="x", pady=(8, 0))
        self.tax_credit_label = self._make_stat(info_row3, "$0", "Tax Credit", "left")
        self.tax_eff_label = self._make_stat(info_row3, "$0", "Effective", "right")

        # Card 4: Controls
        self.ctrl_card = self._make_card(cards_row, "CONTROLS", 3)

        exit_frame = ctk.CTkFrame(self.ctrl_card, fg_color="transparent")
        exit_frame.pack(pady=(5, 0))
        self.exit_entry = ctk.CTkEntry(exit_frame, placeholder_text="AAPL", width=100,
                                        fg_color="#21262d", border_color="#30363d")
        self.exit_entry.pack(side="left", padx=(0, 5))
        ctk.CTkButton(exit_frame, text="Emergency Exit", fg_color="#b91c1c",
                       hover_color="#dc2626", width=120,
                       command=self.emergency_exit).pack(side="left")

        self.webhook_label = ctk.CTkLabel(self.ctrl_card, text="Webhook: /webhook",
                                           font=("Segoe UI", 11), text_color="#8b949e")
        self.webhook_label.pack(pady=(10, 0))

        self.open_dash_btn = ctk.CTkButton(self.ctrl_card, text="Open Dashboard",
                                            fg_color="#1d4ed8", hover_color="#2563eb",
                                            width=160, command=self.open_dashboard)
        self.open_dash_btn.pack(pady=(8, 0))

        # === TABS ===
        self.tabview = ctk.CTkTabview(main, fg_color="#161b22", segmented_button_fg_color="#21262d",
                                       segmented_button_selected_color="#30363d",
                                       segmented_button_unselected_color="#21262d")
        self.tabview.pack(fill="both", expand=True)

        self.tab_positions = self.tabview.add("Open Positions")
        self.tab_history = self.tabview.add("Trade History")
        self.tab_learning = self.tabview.add("Learning Log")
        self.tab_log = self.tabview.add("Live Log")

        # Positions tab
        self.positions_text = ctk.CTkTextbox(self.tab_positions, fg_color="#0d1117",
                                              text_color="#e6edf3", font=("Consolas", 13))
        self.positions_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.positions_text.insert("end", "No open positions")
        self.positions_text.configure(state="disabled")

        # History tab
        self.history_text = ctk.CTkTextbox(self.tab_history, fg_color="#0d1117",
                                            text_color="#e6edf3", font=("Consolas", 13))
        self.history_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.history_text.insert("end", "No trades yet")
        self.history_text.configure(state="disabled")

        # Learning tab
        self.learning_text = ctk.CTkTextbox(self.tab_learning, fg_color="#0d1117",
                                             text_color="#e6edf3", font=("Consolas", 13))
        self.learning_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.learning_text.insert("end", "No patterns detected yet")
        self.learning_text.configure(state="disabled")

        # Log tab
        self.log_text = ctk.CTkTextbox(self.tab_log, fg_color="#0d1117",
                                        text_color="#e6edf3", font=("Consolas", 12))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text.insert("end", "Bot log will appear here...\n")
        self.log_text.configure(state="disabled")

    def _make_card(self, parent, title, col):
        card = ctk.CTkFrame(parent, fg_color="#161b22", border_color="#30363d",
                             border_width=1, corner_radius=10)
        card.grid(row=0, column=col, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(card, text=title, font=("Segoe UI", 11, "bold"),
                     text_color="#8b949e").pack(pady=(10, 0))
        return card

    def _make_stat(self, parent, value, label, side):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(side=side, expand=True)
        val = ctk.CTkLabel(frame, text=value, font=("Segoe UI", 14, "bold"),
                            text_color="#e6edf3")
        val.pack()
        ctk.CTkLabel(frame, text=label, font=("Segoe UI", 10),
                     text_color="#8b949e").pack()
        return val

    def start_bot(self):
        global BOT_PROCESS
        if self.bot_running:
            return
        self._log("Starting bot...")
        bot_dir = Path(__file__).parent
        BOT_PROCESS = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(bot_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.bot_running = True
        threading.Thread(target=self._read_log, daemon=True).start()
        self._log("Bot starting...")

    def stop_bot(self):
        global BOT_PROCESS
        if BOT_PROCESS:
            BOT_PROCESS.terminate()
            BOT_PROCESS = None
        self.bot_running = False
        self.status_label.configure(text="  OFFLINE  ", fg_color="#3b1c1c", text_color="#f87171")
        self._log("Bot stopped.")

    def _read_log(self):
        global BOT_PROCESS
        if not BOT_PROCESS:
            return
        for line in BOT_PROCESS.stdout:
            self._log(line.strip())
        BOT_PROCESS = None
        self.bot_running = False
        self.after(0, lambda: self.status_label.configure(
            text="  OFFLINE  ", fg_color="#3b1c1c", text_color="#f87171"))

    def _log(self, msg):
        def _update():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(0, _update)

    def _start_auto_refresh(self):
        self._refresh_data()
        self.after(8000, self._start_auto_refresh)

    def _refresh_data(self):
        threading.Thread(target=self._fetch_all, daemon=True).start()

    def _fetch_all(self):
        try:
            status = requests.get(f"{API_URL}/status", timeout=3).json()
            trades = requests.get(f"{API_URL}/trades?limit=20", timeout=3).json()
            tax = requests.get(f"{API_URL}/tax", timeout=3).json()
            health = requests.get(f"{API_URL}/health", timeout=3).json()
            learning = requests.get(f"{API_URL}/learning?limit=10", timeout=3).json()
            self.after(0, lambda: self._update_ui(status, trades, tax, health, learning))
        except Exception:
            self.after(0, lambda: self.status_label.configure(
                text="  OFFLINE  ", fg_color="#3b1c1c", text_color="#f87171"))

    def _update_ui(self, status, trades, tax, health, learning):
        b = status.get("budget", {})

        # Status
        self.status_label.configure(text="  RUNNING  ", fg_color="#1b4332", text_color="#2dd4bf")
        self.bot_running = True
        secs = health.get("uptime_seconds", 0)
        h, m = int(secs // 3600), int((secs % 3600) // 60)
        self.uptime_label.configure(text=f"Uptime: {h}h {m}m" if h > 0 else f"Uptime: {m}m")

        # Budget
        total = b.get("total_budget", 0)
        used_pct = b.get("budget_used_pct", 0)
        self.budget_val.configure(text=f"${total:,.2f}")
        self.budget_bar.set(min(used_pct / 100, 1.0))
        color = "#f87171" if used_pct > 80 else "#f0883e" if used_pct > 50 else "#2dd4bf"
        self.budget_bar.configure(progress_color=color)
        self.budget_sub.configure(text=f"{used_pct:.1f}% in use")
        self.cash_label.configure(text=f"${b.get('cash_available', 0):,.2f}")
        self.positions_label.configure(text=f"${b.get('positions_value', 0):,.2f}")

        # PnL
        gross = b.get("realized_pnl_gross", 0)
        net = b.get("realized_pnl_net", 0)
        opnl = b.get("open_pnl", 0)
        self.pnl_gross.configure(text=f"{'+'if gross > 0 else ''}${gross:,.2f}",
                                  text_color="#2dd4bf" if gross >= 0 else "#f87171")
        self.pnl_net_label.configure(text=f"${net:,.2f}",
                                      text_color="#2dd4bf" if net >= 0 else "#f87171")
        self.pnl_open_label.configure(text=f"${opnl:,.2f}",
                                       text_color="#2dd4bf" if opnl >= 0 else "#f87171")

        # Tax
        self.tax_reserved.configure(text=f"${tax.get('tax_reserved', 0):,.2f}")
        self.tax_credit_label.configure(text=f"${tax.get('tax_credit', 0):,.2f}")
        eff = tax.get("tax_reserved", 0) - tax.get("tax_credit", 0)
        self.tax_eff_label.configure(text=f"${eff:,.2f}")

        # Positions
        open_trades = status.get("open_trades", [])
        positions = status.get("positions", [])
        self.positions_text.configure(state="normal")
        self.positions_text.delete("1.0", "end")
        if not open_trades:
            self.positions_text.insert("end", "No open positions")
        else:
            header = f"{'Ticker':<10}{'Qty':<8}{'Entry':<12}{'Current':<12}{'P&L':<12}{'Sentiment':<10}{'Status'}\n"
            self.positions_text.insert("end", header)
            self.positions_text.insert("end", "-" * 75 + "\n")
            for t in open_trades:
                pos = next((p for p in positions if p["ticker"] == t["ticker"]), None)
                curr = pos["current_price"] if pos else t["entry_price"]
                pnl = pos["unrealized_pl"] if pos else 0
                sent = str(t.get("sentiment_score", "-"))
                line = f"{t['ticker']:<10}{t['qty']:<8}${t['entry_price']:<11.2f}${curr:<11.2f}${pnl:<+11.2f}{sent:<10}{t['status']}\n"
                self.positions_text.insert("end", line)
        self.positions_text.configure(state="disabled")

        # History
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        if not trades:
            self.history_text.insert("end", "No trades yet")
        else:
            header = f"{'#':<5}{'Ticker':<8}{'Action':<7}{'Qty':<6}{'Entry':<10}{'Exit':<10}{'P&L':<12}{'Tax':<10}{'Sent':<6}{'Status':<12}{'Time'}\n"
            self.history_text.insert("end", header)
            self.history_text.insert("end", "-" * 100 + "\n")
            for t in trades:
                pnl = t.get("pnl_gross")
                pnl_str = f"${pnl:+.2f}" if pnl is not None else "-"
                exit_str = f"${t['exit_price']:.2f}" if t.get("exit_price") else "-"
                tax_str = f"${t['tax_reserved']:.2f}" if t.get("tax_reserved") else "-"
                sent = str(t.get("sentiment_score", "-"))
                time_str = (t.get("entry_time") or "")[:16]
                line = f"{t['id']:<5}{t['ticker']:<8}{t['action']:<7}{t['qty']:<6}${t['entry_price']:<9.2f}{exit_str:<10}{pnl_str:<12}{tax_str:<10}{sent:<6}{t['status']:<12}{time_str}\n"
                self.history_text.insert("end", line)
        self.history_text.configure(state="disabled")

        # Learning
        self.learning_text.configure(state="normal")
        self.learning_text.delete("1.0", "end")
        if not learning:
            self.learning_text.insert("end", "No patterns detected yet - the bot needs more trades to learn")
        else:
            for l in learning:
                self.learning_text.insert("end",
                    f"[{l.get('pattern_type', '')}] {l['description']}\n"
                    f"  Outcome: {l.get('outcome', '-')} | P&L: ${l.get('pnl', 0):+.2f}\n\n")
        self.learning_text.configure(state="disabled")

    def emergency_exit(self):
        ticker = self.exit_entry.get().strip().upper()
        if not ticker:
            return
        # Read secret from env or settings
        secret = os.environ.get("WEBHOOK_SECRET", "")
        if not secret:
            try:
                from config import settings as _s
                secret = _s.WEBHOOK_SECRET
            except Exception:
                pass
        if not secret:
            self._log("Emergency exit failed: WEBHOOK_SECRET not configured")
            return
        try:
            res = requests.post(
                f"{API_URL}/emergency-exit/{ticker}",
                params={"secret": secret},
                timeout=5,
            )
            data = res.json()
            self._log(f"Emergency exit {ticker}: {json.dumps(data)}")
        except Exception as e:
            self._log(f"Emergency exit failed: {e}")

    def open_dashboard(self):
        os.startfile("http://localhost:8000")

    def on_close(self):
        self.stop_bot()
        self.destroy()


if __name__ == "__main__":
    app = TradeBotApp()
    app.mainloop()
