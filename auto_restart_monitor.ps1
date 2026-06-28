# ═══════════════════════════════════════════════════════════════════════════
# Auto-Restart Monitor — runs forever, restarts bot if it crashes
# This is the PRACTICAL solution given that yfinance leak can't be fully fixed
# ═══════════════════════════════════════════════════════════════════════════

$BotPath = $PSScriptRoot
$LogPath = "$BotPath\auto_restart.log"

function Write-MonLog([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Add-Content -Path $LogPath -Value $line
    Write-Host $line
}

function Get-BotPid {
    $conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object {$_.State -eq 'Listen'}
    if ($conn) { return $conn.OwningProcess }
    return $null
}

function Start-Bot {
    $proc = Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $BotPath `
        -RedirectStandardOutput "$BotPath\bot_background.log" `
        -RedirectStandardError "$BotPath\bot_background_err.log" `
        -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 15
    return $proc.Id
}

function Restart-IfHighCloseWait([int]$botPid) {
    # PROACTIVE: if CloseWait grows past 1000, restart preemptively before crash
    $conns = Get-NetTCPConnection -OwningProcess $botPid -ErrorAction SilentlyContinue
    $cw = ($conns | Where-Object {$_.State -eq 'CloseWait'}).Count
    if ($cw -gt 1000) {
        Write-MonLog "⚠️ CloseWait=$cw too high — preemptive restart"
        Stop-Process -Id $botPid -Force
        Start-Sleep -Seconds 10
        return $true
    }
    return $false
}

Write-MonLog "============================================================"
Write-MonLog "Auto-Restart Monitor STARTED"
Write-MonLog "============================================================"

# Find or start bot
$botPid = Get-BotPid
if (-not $botPid) {
    Write-MonLog "No bot running — starting new..."
    $botPid = Start-Bot
}
Write-MonLog "Monitoring PID $botPid"

$restartCount = 0
$lastRestart = $null

while ($true) {
    Start-Sleep -Seconds 30

    $p = Get-Process -Id $botPid -ErrorAction SilentlyContinue
    if (-not $p) {
        $restartCount++
        Write-MonLog "❌ Bot DIED (#$restartCount) — restarting..."

        # Crash loop protection
        if ($lastRestart -and ((Get-Date) - $lastRestart).TotalMinutes -lt 2) {
            Write-MonLog "🚨 Crash loop detected — pausing 5 min"
            Start-Sleep -Seconds 300
        }
        $lastRestart = Get-Date

        # Wait for port
        $waitFor = 0
        while ($waitFor -lt 60) {
            if (-not (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Seconds 2
            $waitFor += 2
        }

        $botPid = Start-Bot
        Write-MonLog "Restarted as PID $botPid"
    } else {
        # Preemptive restart if CloseWait too high
        $restartedPreemptively = Restart-IfHighCloseWait $botPid
        if ($restartedPreemptively) {
            $botPid = Start-Bot
            Write-MonLog "Preemptive restart — new PID $botPid"
        }
    }
}
