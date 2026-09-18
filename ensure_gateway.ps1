# ensure_gateway.ps1 - keep IB Gateway up so the bot can trade at the open.
#
# Gateway drops its session every few hours; when that happens the bot reads
# $0.00 cash and silently stops buying. This runs before the US open.
# It cannot log in (password + 2FA are the user's alone) - if credentials are
# not saved it sends a Telegram alert asking for a manual login.
$ErrorActionPreference = 'Stop'

$GatewayExe = Join-Path $env:USERPROFILE 'Jts-Gateway\ibgateway.exe'
$ApiPort    = 4002   # 4002 = paper, 4001 = live

function Test-ApiPort {
    $null -ne (Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue)
}

function Send-Telegram([string]$Text) {
    $envPath = Join-Path $PSScriptRoot '.env'
    if (-not (Test-Path $envPath)) { return }
    $token = $null; $chat = $null
    foreach ($line in Get-Content $envPath -Encoding UTF8) {
        if ($line -match '^TELEGRAM_BOT_TOKEN=(.+)$') { $token = $Matches[1].Trim() }
        if ($line -match '^TELEGRAM_CHAT_ID=(.+)$')   { $chat  = $Matches[1].Trim() }
    }
    if (-not $token -or -not $chat) { return }
    try {
        $json = @{ chat_id = $chat; text = $Text; parse_mode = 'HTML' } | ConvertTo-Json
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
        Invoke-RestMethod -Method Post -TimeoutSec 15 -ContentType 'application/json; charset=utf-8' -Uri "https://api.telegram.org/bot$token/sendMessage" -Body $bytes | Out-Null
    } catch { }
}

# ── Bot deadman check ───────────────────────────────────────────────────────
# The bot's own alerts (disk, CPU, shorts) run INSIDE the bot process — so if
# the whole bot dies, nothing can warn about it from the inside. This runs on
# its own schedule (every 30 min, independent of TradingBotWatchdog) so a
# single point of failure can't silence both. 2026-08-31: TradingBotWatchdog
# was found disabled and the bot had been dead for 3 days, unnoticed.
$BotPort = 8000
function Test-BotPort {
    $null -ne (Get-NetTCPConnection -LocalPort $BotPort -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-BotPort)) {
    $watchdog = Get-ScheduledTask -TaskName "TradingBotWatchdog" -ErrorAction SilentlyContinue
    if ($watchdog -and $watchdog.State -eq 'Disabled') {
        Enable-ScheduledTask -TaskName "TradingBotWatchdog" | Out-Null
        Send-Telegram "🚨 <b>הבוט לא רץ!</b>`n`nמשימת TradingBotWatchdog הייתה מושבתת — הפעלתי אותה מחדש ומנסה להקים את הבוט."
    } elseif ($watchdog) {
        Send-Telegram "🚨 <b>הבוט לא רץ!</b>`n`nTradingBotWatchdog פעילה אבל הבוט עדיין לא עלה. בודק שוב בעוד 30 דקות."
    } else {
        Send-Telegram "🚨 <b>הבוט לא רץ, ומשימת TradingBotWatchdog לא נמצאה בכלל!</b>`n`nצריך בדיקה ידנית."
    }
    if ($watchdog) {
        Start-ScheduledTask -TaskName "TradingBotWatchdog" -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 20
    if (Test-BotPort) {
        Send-Telegram "✅ הבוט קם בהצלחה אחרי ההתערבות."
    }
}

# ── Gateway is only needed when the bot trades through IBKR ────────────────
# On tv_paper (internal demo) Gateway is irrelevant, and launching it every
# 30 minutes just pops a login window at the user (reported 2026-09-18).
# The deadman check above still runs for every broker.
$activeBroker = ''
$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        if ($line -match '^\s*ACTIVE_BROKER\s*=\s*(\S+)') { $activeBroker = $Matches[1].Trim().ToLower() }
    }
}
if ($activeBroker -ne 'ibkr') {
    Write-Output "ACTIVE_BROKER='$activeBroker' - Gateway not needed, skipping launch."
    exit 0
}

if (Test-ApiPort) {
    Write-Output "Gateway already connected - nothing to do."
    exit 0
}

Write-Output "Gateway not connected. Launching..."
if (-not (Test-Path $GatewayExe)) {
    Send-Telegram "Gateway not found - bot cannot trade."
    throw "Gateway executable not found at $GatewayExe"
}

if (-not (Get-Process | Where-Object { $_.Path -like '*Jts-Gateway*' })) {
    Start-Process $GatewayExe
}

# 5 minutes: a cold start plus auto-login has been observed taking well over
# the 2 minutes this waited before, so it reported "manual login required" and
# sent a false alarm while Gateway was still coming up on its own.
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    if (Test-ApiPort) { break }
}

if (Test-ApiPort) {
    Write-Output "Gateway connected automatically."
    Send-Telegram "✅ Gateway התחבר אוטומטית — הבוט מוכן לפתיחת השוק"
} else {
    Write-Output "Gateway did NOT auto-connect - manual login required."
    Send-Telegram "⚠️ Gateway לא מחובר!`n`nהבוט לא יסחר בפתיחת השוק (16:30).`n`nפתח את IB Gateway והתחבר ידנית."
}
exit 0
