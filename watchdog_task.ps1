# ═══════════════════════════════════════════════════════════════════════════
# TradingBot Watchdog — runs from Windows Task Scheduler every 2 minutes.
# Survives sleep/resume and session death (unlike the cmd-window :loop).
#
# Logic:
#   - If bot_supervisor.py is already running  -> do nothing (it manages the bot)
#   - Otherwise                                -> launch the supervisor (detached, hidden)
#
# Launches python DIRECTLY (not via start_bot.bat) to avoid the `timeout`
# console bug that hangs the BAT when started without an interactive console.
# ═══════════════════════════════════════════════════════════════════════════
$ErrorActionPreference = 'SilentlyContinue'

$dir = $PSScriptRoot
$py  = "C:\Users\תומר\AppData\Local\Programs\Python\Python313\python.exe"
$log = "$dir\watchdog_task.log"

function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg" | Add-Content -Path $log -Encoding UTF8
}

# Is a supervisor already running?
$sup = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -match 'bot_supervisor\.py' }

if ($sup) {
    # Supervisor alive — nothing to do. (Quiet: no log spam every 2 min.)
    exit 0
}

# No supervisor — verify the bot port too, then (re)start.
$listening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    # Port is up but no supervisor — a bare/orphan bot. Leave it; log once.
    Write-Log "Port 8000 up but no supervisor process found (orphan bot?)."
    exit 0
}

Write-Log "Supervisor NOT running and port 8000 down — starting supervisor."
Start-Process -FilePath $py -ArgumentList 'bot_supervisor.py' -WorkingDirectory $dir -WindowStyle Hidden
Write-Log "Supervisor launch issued."
