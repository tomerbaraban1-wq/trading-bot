# ═══════════════════════════════════════════════════════════════════════════
# Bot Health Monitor — בדיקה אוטומטית כל דקה
# מעקב מתמשך - אם הבוט נופל, מפעיל מחדש אוטומטית
# ═══════════════════════════════════════════════════════════════════════════

$BotPath = "C:\Users\תומר\Pictures\קלוד קוד\trading-bot"
$LogPath = "$BotPath\health_monitor.log"
$RestartCount = 0
$LastRestartTime = $null

function Write-Log {
    param([string]$msg)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line
}

function Start-Bot {
    Write-Log "Starting bot..."
    $proc = Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $BotPath `
        -RedirectStandardOutput "$BotPath\bot_background.log" `
        -RedirectStandardError "$BotPath\bot_background_err.log" `
        -WindowStyle Hidden `
        -PassThru
    Start-Sleep -Seconds 10
    return $proc.Id
}

Write-Log "================================================================"
Write-Log "  Bot Health Monitor STARTED"
Write-Log "================================================================"

# Find existing bot or start new one
$existingBot = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
if ($existingBot) {
    $botPid = $existingBot.OwningProcess
    Write-Log "Existing bot detected — PID $botPid"
} else {
    $botPid = Start-Bot
    Write-Log "Started new bot — PID $botPid"
}

# Monitor loop
while ($true) {
    Start-Sleep -Seconds 60

    $p = Get-Process -Id $botPid -ErrorAction SilentlyContinue
    if ($p) {
        $memMB = [math]::Round($p.WorkingSet64/1MB, 1)
        $uptime = [math]::Round(((Get-Date) - $p.StartTime).TotalMinutes, 1)
        Write-Log "Bot ALIVE — PID $botPid | ${memMB}MB | ${uptime}min uptime"

        # Memory warning
        if ($memMB -gt 800) {
            Write-Log "WARN: High memory usage ($memMB MB)"
        }
    } else {
        $RestartCount++
        Write-Log "Bot CRASHED! Restarting (#$RestartCount)..."

        # Check for crash loop
        if ($LastRestartTime) {
            $sinceLastRestart = ((Get-Date) - $LastRestartTime).TotalMinutes
            if ($sinceLastRestart -lt 5) {
                Write-Log "CRITICAL: Restart loop detected — pausing 5 min before retry"
                Start-Sleep -Seconds 300
            }
        }
        $LastRestartTime = Get-Date

        # Wait for port to free
        $portWait = 0
        while ($portWait -lt 30) {
            $port = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
            if (-not $port) { break }
            Start-Sleep -Seconds 2
            $portWait += 2
        }

        $botPid = Start-Bot
        Write-Log "Restarted bot — new PID $botPid"
    }
}
