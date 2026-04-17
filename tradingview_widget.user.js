// ==UserScript==
// @name         TradeBot Panel
// @namespace    https://tradingview.com
// @version      1.0
// @description  TradeBot control panel inside TradingView
// @author       TradeBot
// @match        https://www.tradingview.com/*
// @match        https://tradingview.com/*
// @grant        GM_xmlhttpRequest
// @grant        GM_addStyle
// @connect      localhost
// @run-at       document-end
// ==/UserScript==

(function () {
    'use strict';

    const BOT_URL = 'http://localhost:8000';
    const SECRET  = 'tradebot_wh_2026_secure';
    const REFRESH = 8000;

    // ── Styles ──────────────────────────────────────────────────────────────
    GM_addStyle(`
        #tb-panel {
            position: fixed;
            top: 60px;
            right: 12px;
            width: 280px;
            background: #131722;
            border: 1px solid #2a2e39;
            border-radius: 10px;
            font-family: -apple-system, 'Segoe UI', sans-serif;
            font-size: 12px;
            color: #d1d4dc;
            z-index: 9999;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            user-select: none;
        }
        #tb-panel.collapsed #tb-body { display: none; }

        #tb-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            background: #1e222d;
            border-radius: 10px 10px 0 0;
            cursor: move;
            border-bottom: 1px solid #2a2e39;
        }
        #tb-title {
            font-weight: 700;
            font-size: 13px;
            color: #2962ff;
            letter-spacing: -0.3px;
        }
        #tb-title span { color: #f0883e; }
        #tb-status-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: #ef5350;
            margin-left: 6px;
            animation: tb-pulse 2s infinite;
        }
        #tb-status-dot.online { background: #26a69a; }
        @keyframes tb-pulse {
            0%,100% { opacity: 1; }
            50%      { opacity: 0.4; }
        }
        #tb-toggle {
            background: none; border: none;
            color: #787b86; font-size: 16px;
            cursor: pointer; padding: 0 4px;
            line-height: 1;
        }
        #tb-toggle:hover { color: #d1d4dc; }

        #tb-body { padding: 10px 12px; }

        #tb-ticker-row {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 10px;
        }
        #tb-ticker-badge {
            background: #2a2e39;
            border: 1px solid #363a45;
            border-radius: 5px;
            padding: 4px 10px;
            font-weight: 700;
            font-size: 14px;
            color: #d1d4dc;
            flex: 1;
            text-align: center;
            cursor: pointer;
        }
        #tb-ticker-badge:hover { border-color: #2962ff; }

        .tb-btn-row { display: flex; gap: 6px; margin-bottom: 10px; }
        .tb-btn {
            flex: 1;
            border: none;
            border-radius: 6px;
            padding: 8px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s;
            letter-spacing: 0.3px;
        }
        .tb-btn:active { transform: scale(0.96); }
        .tb-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        #tb-buy  { background: #26a69a; color: #fff; }
        #tb-buy:hover:not(:disabled)  { background: #2bbbad; }
        #tb-sell { background: #ef5350; color: #fff; }
        #tb-sell:hover:not(:disabled) { background: #f44336; }

        .tb-stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-bottom: 10px;
        }
        .tb-stat {
            background: #1e222d;
            border-radius: 6px;
            padding: 7px 9px;
        }
        .tb-stat-label {
            font-size: 10px;
            color: #787b86;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }
        .tb-stat-val {
            font-size: 13px;
            font-weight: 600;
            color: #d1d4dc;
        }
        .tb-stat-val.green { color: #26a69a; }
        .tb-stat-val.red   { color: #ef5350; }
        .tb-stat-val.amber { color: #f0883e; }

        #tb-positions-title {
            font-size: 10px;
            color: #787b86;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        #tb-positions-list .tb-pos-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid #2a2e39;
            font-size: 12px;
        }
        #tb-positions-list .tb-pos-row:last-child { border: none; }
        .tb-pos-ticker { font-weight: 700; color: #d1d4dc; }
        .tb-pos-pnl { font-weight: 600; }
        .tb-exit-btn {
            background: #2a2e39;
            border: none;
            border-radius: 4px;
            color: #ef5350;
            font-size: 10px;
            font-weight: 700;
            padding: 2px 7px;
            cursor: pointer;
        }
        .tb-exit-btn:hover { background: #363a45; }

        #tb-empty { color: #363a45; font-size: 11px; text-align: center; padding: 8px 0; }

        #tb-msg {
            font-size: 11px;
            text-align: center;
            margin-top: 8px;
            min-height: 16px;
            color: #787b86;
        }
        #tb-msg.ok  { color: #26a69a; }
        #tb-msg.err { color: #ef5350; }

        #tb-footer {
            font-size: 10px;
            color: #363a45;
            text-align: center;
            margin-top: 8px;
            word-break: break-all;
        }
    `);

    // ── Build HTML ───────────────────────────────────────────────────────────
    const panel = document.createElement('div');
    panel.id = 'tb-panel';
    panel.innerHTML = `
        <div id="tb-header">
            <div style="display:flex;align-items:center">
                <span id="tb-title">Trade<span>Bot</span></span>
                <div id="tb-status-dot"></div>
            </div>
            <button id="tb-toggle">−</button>
        </div>
        <div id="tb-body">
            <div id="tb-ticker-row">
                <div id="tb-ticker-badge" title="Click to edit">AAPL</div>
            </div>

            <div class="tb-btn-row">
                <button class="tb-btn" id="tb-buy">▲ BUY</button>
                <button class="tb-btn" id="tb-sell">▼ SELL</button>
            </div>

            <div class="tb-stat-grid">
                <div class="tb-stat">
                    <div class="tb-stat-label">Budget</div>
                    <div class="tb-stat-val" id="tb-budget">—</div>
                </div>
                <div class="tb-stat">
                    <div class="tb-stat-label">Allocated</div>
                    <div class="tb-stat-val" id="tb-alloc">—</div>
                </div>
                <div class="tb-stat">
                    <div class="tb-stat-label">P&L</div>
                    <div class="tb-stat-val" id="tb-pnl">—</div>
                </div>
                <div class="tb-stat">
                    <div class="tb-stat-label">Tax Vault</div>
                    <div class="tb-stat-val amber" id="tb-tax">—</div>
                </div>
            </div>

            <div id="tb-positions-title">Open Positions</div>
            <div id="tb-positions-list">
                <div id="tb-empty">No open positions</div>
            </div>

            <div id="tb-msg"></div>
            <div id="tb-footer">localhost:8000</div>
        </div>
    `;
    document.body.appendChild(panel);

    // ── Collapse toggle ──────────────────────────────────────────────────────
    const toggleBtn = document.getElementById('tb-toggle');
    toggleBtn.addEventListener('click', () => {
        const collapsed = panel.classList.toggle('collapsed');
        toggleBtn.textContent = collapsed ? '+' : '−';
    });

    // ── Drag ────────────────────────────────────────────────────────────────
    let dragX = 0, dragY = 0;
    const header = document.getElementById('tb-header');
    header.addEventListener('mousedown', e => {
        dragX = e.clientX - panel.getBoundingClientRect().left;
        dragY = e.clientY - panel.getBoundingClientRect().top;
        const onMove = e => {
            panel.style.left  = (e.clientX - dragX) + 'px';
            panel.style.top   = (e.clientY - dragY) + 'px';
            panel.style.right = 'auto';
        };
        const onUp = () => {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    // ── Ticker detection ─────────────────────────────────────────────────────
    function getTicker() {
        // Try URL param
        const sym = new URLSearchParams(location.search).get('symbol');
        if (sym) return sym.replace(/^.*:/, '');
        // Try page title "AAPL • 150.00 — TradingView"
        const m = document.title.match(/^([A-Z]{1,5})\s*[•·]/);
        if (m) return m[1];
        // Try DOM (TradingView header)
        const el = document.querySelector('[class*="tickerDescription"]') ||
                   document.querySelector('[class*="symbol-"]') ||
                   document.querySelector('[data-symbol]');
        if (el) return (el.textContent || el.getAttribute('data-symbol') || '').trim().toUpperCase().slice(0,5);
        return 'AAPL';
    }

    const tickerBadge = document.getElementById('tb-ticker-badge');

    function updateTicker() {
        const t = getTicker();
        if (t) tickerBadge.textContent = t;
    }

    // Click to manually edit ticker
    tickerBadge.addEventListener('click', () => {
        const val = prompt('Enter ticker symbol:', tickerBadge.textContent);
        if (val) tickerBadge.textContent = val.trim().toUpperCase().slice(0,5);
    });

    // Watch URL changes (SPA navigation)
    let lastUrl = location.href;
    setInterval(() => {
        if (location.href !== lastUrl) { lastUrl = location.href; updateTicker(); }
    }, 1000);
    updateTicker();

    // ── API helpers ──────────────────────────────────────────────────────────
    function apiFetch(path) {
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method: 'GET',
                url: BOT_URL + path,
                timeout: 5000,
                onload: r => {
                    try { resolve(JSON.parse(r.responseText)); }
                    catch { reject(new Error('Parse error')); }
                },
                onerror:   () => reject(new Error('Network error')),
                ontimeout: () => reject(new Error('Timeout')),
            });
        });
    }

    function apiPost(path, data) {
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method: 'POST',
                url: BOT_URL + path,
                headers: { 'Content-Type': 'application/json' },
                data: JSON.stringify(data),
                timeout: 8000,
                onload: r => {
                    try { resolve(JSON.parse(r.responseText)); }
                    catch { reject(new Error('Parse error')); }
                },
                onerror:   () => reject(new Error('Network error')),
                ontimeout: () => reject(new Error('Timeout')),
            });
        });
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    function money(n) {
        if (n == null) return '—';
        return (n >= 0 ? '+$' : '-$') + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    function setMsg(txt, type = '') {
        const el = document.getElementById('tb-msg');
        el.textContent = txt;
        el.className = type;
        if (type) setTimeout(() => { el.textContent = ''; el.className = ''; }, 4000);
    }

    // ── Render ───────────────────────────────────────────────────────────────
    function render(status, health) {
        const b = status.budget || {};
        const pnl = b.realized_pnl_gross || 0;

        document.getElementById('tb-budget').textContent = '$' + (b.total_budget || 0).toLocaleString();
        document.getElementById('tb-alloc').textContent  = (b.budget_used_pct || 0).toFixed(1) + '%';

        const pnlEl = document.getElementById('tb-pnl');
        pnlEl.textContent = money(pnl);
        pnlEl.className = 'tb-stat-val ' + (pnl >= 0 ? 'green' : 'red');

        document.getElementById('tb-tax').textContent = '$' + ((b.tax_reserved || 0)).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

        // Status dot
        const dot = document.getElementById('tb-status-dot');
        dot.className = 'online';

        // Positions
        const trades = status.open_trades || [];
        const positions = status.positions || [];
        const list = document.getElementById('tb-positions-list');
        if (!trades.length) {
            list.innerHTML = '<div id="tb-empty">No open positions</div>';
        } else {
            list.innerHTML = trades.map(t => {
                const p = positions.find(p => p.ticker === t.ticker);
                const pl = p ? (p.unrealized_pl || 0) : 0;
                const cls = pl >= 0 ? 'green' : 'red';
                return `<div class="tb-pos-row">
                    <span class="tb-pos-ticker">${t.ticker}</span>
                    <span class="tb-pos-pnl ${cls}">${money(pl)}</span>
                    <button class="tb-exit-btn" onclick="tbExit('${t.ticker}')">EXIT</button>
                </div>`;
            }).join('');
        }
    }

    // ── Refresh ──────────────────────────────────────────────────────────────
    async function refresh() {
        try {
            const [status, health] = await Promise.all([
                apiFetch('/status'),
                apiFetch('/health'),
            ]);
            render(status, health);
        } catch (e) {
            document.getElementById('tb-status-dot').className = '';
        }
    }

    refresh();
    setInterval(refresh, REFRESH);

    // ── BUY button ───────────────────────────────────────────────────────────
    document.getElementById('tb-buy').addEventListener('click', async () => {
        const ticker = tickerBadge.textContent;
        if (!ticker) return;
        setMsg(`Sending BUY ${ticker}…`);
        try {
            const price = getCurrentPrice();
            const r = await apiPost('/webhook', {
                secret: SECRET,
                ticker,
                action: 'buy',
                price,
            });
            if (r.status === 'executed') {
                setMsg(`✅ BUY ${ticker} executed!`, 'ok');
            } else {
                setMsg(r.reason || r.message || 'Blocked by bot', 'err');
            }
            refresh();
        } catch (e) {
            setMsg('❌ ' + e.message, 'err');
        }
    });

    // ── SELL button ──────────────────────────────────────────────────────────
    document.getElementById('tb-sell').addEventListener('click', async () => {
        const ticker = tickerBadge.textContent;
        if (!ticker) return;
        setMsg(`Sending SELL ${ticker}…`);
        try {
            const price = getCurrentPrice();
            const r = await apiPost('/webhook', {
                secret: SECRET,
                ticker,
                action: 'sell',
                price,
            });
            if (r.status === 'executed') {
                setMsg(`✅ SELL ${ticker} executed!`, 'ok');
            } else {
                setMsg(r.reason || r.message || JSON.stringify(r), 'err');
            }
            refresh();
        } catch (e) {
            setMsg('❌ ' + e.message, 'err');
        }
    });

    // ── Emergency exit from positions list ───────────────────────────────────
    window.tbExit = async function (ticker) {
        if (!confirm(`Emergency exit ${ticker}?`)) return;
        try {
            const r = await apiPost(`/emergency-exit/${ticker}`, {});
            setMsg(`✅ ${ticker} closed`, 'ok');
            refresh();
        } catch (e) {
            setMsg('❌ ' + e.message, 'err');
        }
    };

    // ── Get current price from TradingView DOM ────────────────────────────────
    function getCurrentPrice() {
        // Try to read last price from TradingView header
        const el = document.querySelector('[class*="lastPrice"]') ||
                   document.querySelector('[class*="priceValue"]') ||
                   document.querySelector('[data-field="last_price"]');
        if (el) {
            const n = parseFloat(el.textContent.replace(/[^0-9.]/g, ''));
            if (n > 0) return n;
        }
        return 0;
    }

})();
