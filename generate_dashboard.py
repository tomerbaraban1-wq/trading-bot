"""
Dashboard generator  —  run:  python generate_dashboard.py
==========================================================
Builds a clean, visual HTML dashboard for the PAPER trading bot from the
local database. Read-only: it shows the bot's simulated ($10k) activity and
NEVER touches money, the broker, or anything live. Re-run anytime to refresh.

Sections: headline stats · cumulative-P&L chart · vs S&P 500 · learning
insights · open positions · recent trades · broker-fee comparison (read from
the Excel file in the parent folder — edit the Excel and re-run to update).

Output: dashboard.html  (open it in any browser / double-click it).
"""

import datetime
import html

import database


def _fmt(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def _verdict(c):
    a = abs(c)
    if a >= 0.30:
        return "✅ קשר חזק"
    if a >= 0.15:
        return "🟡 קשר חלש"
    return "⚪ רעש — אין ניבוי"


def _spark(series, w=920, h=190):
    if len(series) < 2:
        return "<div class='muted' style='padding:30px;text-align:center'>אין מספיק נתונים לגרף</div>"
    lo, hi = min(series), max(series)
    rng = (hi - lo) or 1
    pts = []
    for i, v in enumerate(series):
        x = i / (len(series) - 1) * w
        y = h - (v - lo) / rng * h
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    area = f"M0,{h} L" + " L".join(pts) + f" L{w},{h} Z"
    zero_y = (h - (0 - lo) / rng * h) if lo <= 0 <= hi else None
    zline = (f"<line x1='0' y1='{zero_y:.1f}' x2='{w}' y2='{zero_y:.1f}' "
             f"stroke='#8b949e' stroke-dasharray='5' opacity='.4'/>") if zero_y is not None else ""
    color = "#3fb950" if series[-1] >= 0 else "#f85149"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" style="display:block">'
            f'<path d="{area}" fill="{color}" opacity="0.13"/>{zline}'
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/></svg>')


def _fees_section() -> str:
    """Broker-fee comparison table, read from the Excel file in the parent
    folder. Returns "" (section hidden) if the file or openpyxl is missing,
    so the dashboard never breaks because of it. All cell values are
    HTML-escaped — the Excel is data, never markup."""
    import os
    xlsx = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "השוואת_עמלות_מסחר_בורסה_2026.xlsx")
    try:
        import openpyxl
        ws = openpyxl.load_workbook(xlsx, data_only=True, read_only=True).active
        rows = [[("" if c is None else str(c).strip()) for c in r]
                for r in ws.iter_rows(values_only=True)]
    except Exception:
        return ""

    title, notes, header, data = "", [], None, []
    for r in rows:
        filled = [c for c in r if c]
        if not filled:
            continue
        if header is None and len(filled) >= 5:
            header = r
        elif header is None and not title:
            title = filled[0]
        elif header is None:
            notes.append(filled[0])
        else:
            data.append(r)
    if header is None or not data:
        return ""

    head = "".join(f"<th>{html.escape(c)}</th>" for c in header if c)
    ncols = sum(1 for c in header if c)
    body = "".join(
        "<tr>" + "".join(
            f"<td{' class=tk' if i == 0 else ''}>{html.escape(c) or '—'}</td>"
            for i, c in enumerate(r[:ncols])) + "</tr>"
        for r in data)
    notes_html = "".join(f"<div class='sub'>{html.escape(t)}</div>" for t in notes)
    return f"""
  <h2>💰 {html.escape(title) or 'השוואת עמלות מסחר'}</h2>
  <div class="panel" style="padding:12px">
    <div style="overflow-x:auto">
      <table style="border:none;min-width:900px"><tr>{head}</tr>{body}</table>
    </div>
    <div style="margin-top:10px">{notes_html}</div>
  </div>"""


def build() -> str:
    conn = database.get_connection()
    closed = conn.execute(
        "SELECT ticker, action, entry_price, exit_price, pnl_gross, "
        "COALESCE(exit_reason,status) AS reason, entry_time, exit_time, "
        "sentiment_score, rsi "
        "FROM trade_log WHERE status!='open' AND pnl_gross IS NOT NULL "
        "ORDER BY COALESCE(exit_time, entry_time) DESC"
    ).fetchall()
    open_rows = conn.execute(
        "SELECT ticker, entry_price, COALESCE(created_at, entry_time) AS t "
        "FROM trade_log WHERE status='open' ORDER BY t DESC"
    ).fetchall()

    n = len(closed)
    pnls = [float(r["pnl_gross"]) for r in closed]
    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = (len(wins) / n * 100) if n else 0
    avg_win = (sum(wins) / len(wins)) if wins else 0
    avg_loss = (sum(losses) / len(losses)) if losses else 0
    rr = abs(avg_win / avg_loss) if avg_loss else 0
    start_budget = 10000.0
    ret_pct = total / start_budget * 100
    pnl_cls = "pos" if total >= 0 else "neg"

    # ── headline: the ACTUAL account equity, not the DB trade-sum ────────────
    # The trade-log sum includes +$656 of pre-migration "legacy" wins that were
    # wiped from the account by the old Friday-reset bug (fixed long ago) and
    # -$162 of stale_restart entries that never debited the account — so the
    # DB sum overstates the real bottom line. Equity is the honest number.
    _headline = ""
    try:
        import broker
        _eq = float(broker.get_account().get("equity", 0) or 0)
        if _eq > 0:
            _tr = (_eq - start_budget) / start_budget * 100
            _hc = "pos" if _tr >= 0 else "neg"
            _headline = f"""
      <div class="card"><div class="lbl">💼 שווי התיק — השורה האמיתית</div>
        <div class="val {_hc}">${_fmt(_eq)}</div><div class="sub">{_tr:+.2f}% מאז ההתחלה ($10,000)</div></div>"""
    except Exception:
        pass  # price feed down — dashboard still renders without the headline

    cards = _headline + f"""
      <div class="card"><div class="lbl">רווח/הפסד ממומש (יומן)</div>
        <div class="val {pnl_cls}">${_fmt(total)}</div><div class="sub">{ret_pct:+.2f}% על $10,000 · כולל עסקאות legacy</div></div>
      <div class="card"><div class="lbl">אחוז הצלחה</div>
        <div class="val">{win_rate:.0f}%</div><div class="sub">{len(wins)} רווח · {len(losses)} הפסד</div></div>
      <div class="card"><div class="lbl">יחס סיכון/סיכוי</div>
        <div class="val">{_fmt(rr)}</div><div class="sub">מנצח ${_fmt(avg_win,0)} · מפסיד ${_fmt(avg_loss,0)}</div></div>
      <div class="card"><div class="lbl">עסקאות סגורות</div>
        <div class="val">{n}</div><div class="sub">{len(open_rows)} פתוחות כעת</div></div>
    """

    # ── cumulative P&L chart ──
    asc = sorted(closed, key=lambda r: (r["exit_time"] or r["entry_time"] or ""))
    cum, run = [], 0.0
    for r in asc:
        run += float(r["pnl_gross"])
        cum.append(run)
    chart = _spark(cum)

    # ── vs S&P 500 ──
    try:
        import pandas as pd
        from yfinance_cache import get_ohlcv
        entries = [r["entry_time"][:10] for r in closed if r["entry_time"]]
        exits = [r["exit_time"][:10] for r in closed if r["exit_time"]]
        first = min(entries)
        last = max(exits) if exits else max(entries)
        df = get_ohlcv("SPY", days=420)
        df = df[(df.index >= pd.Timestamp(first)) & (df.index <= pd.Timestamp(last))]
        spy0, spy1 = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
        spy_ret = (spy1 - spy0) / spy0 * 100
        diff = ret_pct - spy_ret
        dcls = "pos" if diff >= 0 else "neg"
        sp_html = f"""
          <div class="bars">
            <div class="bar"><span class="bn">🤖 הבוט</span>
              <div class="track"><div class="fill bot" style="width:{min(abs(ret_pct)/max(abs(ret_pct),abs(spy_ret),1)*100,100):.0f}%"></div></div>
              <span class="bv {('pos' if ret_pct>=0 else 'neg')}">{ret_pct:+.2f}%</span></div>
            <div class="bar"><span class="bn">📈 S&P 500</span>
              <div class="track"><div class="fill spy" style="width:{min(abs(spy_ret)/max(abs(ret_pct),abs(spy_ret),1)*100,100):.0f}%"></div></div>
              <span class="bv {('pos' if spy_ret>=0 else 'neg')}">{spy_ret:+.2f}%</span></div>
          </div>
          <div class="sub" style="margin-top:10px">
            הפרש: <b class="{dcls}">{diff:+.2f}%</b> · תקופה {first} → {last}<br>
            <span style="opacity:.7">⚠️ נייר + שוק עולה ≠ יתרון מוכח. מדגם קטן.</span></div>"""
    except Exception as e:
        sp_html = f"<div class='muted'>השוואת S&P לא זמינה כרגע ({type(e).__name__})</div>"

    # ── learning insights (computed directly from the trade DB — honest) ──
    li = []
    srows = [(float(r["sentiment_score"]), float(r["pnl_gross"])) for r in closed
             if r["sentiment_score"] is not None]
    if len(srows) >= 3:
        c = _pearson([s for s, _ in srows], [p for _, p in srows])
        if c is not None:
            li.append(("סנטימנט ← תוצאה", f"מתאם {c:+.2f}", _verdict(c)))
    rrows = [(float(r["rsi"]), float(r["pnl_gross"])) for r in closed if r["rsi"] is not None]
    if rrows:
        rwin = [x for x, p in rrows if p > 0]
        rloss = [x for x, p in rrows if p <= 0]
        if rwin and rloss:
            li.append(("RSI ממוצע בכניסה", f"מנצחים {sum(rwin)/len(rwin):.0f} · מפסידים {sum(rloss)/len(rloss):.0f}",
                       "ככל שגבוה יותר במנצחים — סימן טוב"))
    # overall honest verdict
    li.append(("מסקנת הבוט", "אין יתרון ניבוי מובהק",
               "⚪ ההפסדים אקראיים ברובם — וזו האמת"))
    li_html = "".join(
        f"<div class='card'><div class='lbl'>{html.escape(t)}</div>"
        f"<div class='val' style='font-size:20px'>{html.escape(v)}</div>"
        f"<div class='sub'>{html.escape(s)}</div></div>"
        for t, v, s in li
    )

    # ── best / worst + P&L by exit reason ──
    extra = ""
    if closed:
        best = max(closed, key=lambda r: float(r["pnl_gross"]))
        worst = min(closed, key=lambda r: float(r["pnl_gross"]))
        from collections import defaultdict
        byr = defaultdict(lambda: [0, 0.0])
        for r in closed:
            g = byr[str(r["reason"])]
            g[0] += 1
            g[1] += float(r["pnl_gross"])
        reason_rows = "".join(
            f"<tr><td class='tk' style='font-weight:600'>{html.escape(k)}</td>"
            f"<td>{v[0]}</td>"
            f"<td class='{'pos' if v[1] >= 0 else 'neg'}'>${_fmt(v[1])}</td></tr>"
            for k, v in sorted(byr.items(), key=lambda x: -x[1][1])
        )
        extra = f"""
  <h2>📌 שיא העסקאות</h2>
  <div class="grid">
    <div class="card"><div class="lbl">🏆 העסקה הטובה ביותר</div>
      <div class="val pos" style="font-size:24px">{html.escape(str(best['ticker']))} &nbsp;${_fmt(best['pnl_gross'])}</div></div>
    <div class="card"><div class="lbl">📉 העסקה הגרועה ביותר</div>
      <div class="val neg" style="font-size:24px">{html.escape(str(worst['ticker']))} &nbsp;${_fmt(worst['pnl_gross'])}</div></div>
  </div>
  <h2>🎯 רווח לפי סוג יציאה</h2>
  <table><tr><th>סוג יציאה</th><th>עסקאות</th><th>רווח/הפסד</th></tr>{reason_rows}</table>"""

    # ── tables ──
    rows_html = ""
    for r in closed[:20]:
        p = float(r["pnl_gross"])
        cls = "pos" if p > 0 else "neg"
        date = (r["exit_time"] or r["entry_time"] or "")[:10]
        rows_html += (f"<tr><td class='tk'>{html.escape(str(r['ticker']))}</td>"
                      f"<td>${_fmt(r['entry_price'])}</td><td>${_fmt(r['exit_price'])}</td>"
                      f"<td class='{cls}'>${_fmt(p)}</td>"
                      f"<td class='reason'>{html.escape(str(r['reason']))}</td>"
                      f"<td class='date'>{date}</td></tr>")
    if not rows_html:
        rows_html = "<tr><td colspan='6' style='text-align:center;opacity:.6'>אין עדיין עסקאות סגורות</td></tr>"

    open_html = "".join(f"<span class='pill'>{html.escape(str(r['ticker']))} @ ${_fmt(r['entry_price'])}</span>"
                        for r in open_rows) or "<span class='muted'>אין פוזיציות פתוחות כרגע</span>"

    fees = _fees_section()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html><html lang="he" dir="rtl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="45">
<title>מנהל ההשקעות שלך — דשבורד</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:24px;max-width:1000px;margin:auto}}
  .hdr{{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}}
  h1{{font-size:26px}}
  .badge{{background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb55;padding:6px 14px;border-radius:20px;font-size:14px;font-weight:600}}
  .paper{{background:#d2992222;color:#e3b341;border-color:#d2992255}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px}}
  .lbl{{font-size:13px;opacity:.7;margin-bottom:8px}}
  .val{{font-size:32px;font-weight:700}}
  .sub{{font-size:13px;opacity:.65;margin-top:6px}}
  .pos{{color:#3fb950}} .neg{{color:#f85149}}
  h2{{font-size:18px;margin:26px 0 12px}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px}}
  .bars{{display:flex;flex-direction:column;gap:14px}}
  .bar{{display:flex;align-items:center;gap:12px}}
  .bn{{width:90px;font-size:14px}} .bv{{width:80px;font-weight:700;text-align:left}}
  .track{{flex:1;background:#0d1117;border-radius:8px;height:22px;overflow:hidden}}
  .fill{{height:100%;border-radius:8px}} .fill.bot{{background:#3fb950}} .fill.spy{{background:#58a6ff}}
  .pills{{display:flex;gap:8px;flex-wrap:wrap}}
  .pill{{background:#161b22;border:1px solid #30363d;border-radius:20px;padding:6px 14px;font-size:14px}}
  .muted{{opacity:.5}}
  table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:14px;overflow:hidden}}
  th,td{{padding:11px 14px;text-align:right;font-size:14px;border-bottom:1px solid #21262d}}
  th{{background:#1c2128;opacity:.8;font-weight:600}}
  tr:last-child td{{border-bottom:none}}
  .tk{{font-weight:700}} .reason{{opacity:.7;font-size:12px}} .date{{opacity:.5;font-size:12px}}
  .foot{{margin-top:28px;text-align:center;opacity:.5;font-size:13px;line-height:1.7}}
</style></head><body>
  <div class="hdr"><h1>🤖 מנהל ההשקעות שלך</h1>
    <div><span class="badge paper">💵 כסף מדומה — נייר</span>
    <span class="badge">עודכן: {now}</span></div></div>

  <div class="grid">{cards}</div>

  <h2>📈 רווח מצטבר לאורך זמן</h2>
  <div class="panel">{chart}</div>

  <h2>📊 מול S&P 500</h2>
  <div class="panel">{sp_html}</div>

  <h2>🧠 תובנות למידה (אמת מדודה)</h2>
  <div class="grid">{li_html}</div>
{extra}
  <h2>📍 פוזיציות פתוחות</h2>
  <div class="pills">{open_html}</div>

  <h2>📋 עסקאות אחרונות (20)</h2>
  <table><tr><th>מניה</th><th>כניסה</th><th>יציאה</th><th>רווח/הפסד</th><th>סיבה</th><th>תאריך</th></tr>{rows_html}</table>
{fees}
  <div class="foot">⚠️ דשבורד תצוגה בלבד של בוט על <b>כסף מדומה ($10,000)</b> — לא כסף אמיתי.<br>
    לרענון: הרץ <code>python generate_dashboard.py</code> שוב.</div>
</body></html>"""


if __name__ == "__main__":
    import os
    out = build()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"OK dashboard.html נוצר: {path}")
