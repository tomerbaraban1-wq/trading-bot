# ═══════════════════════════════════════════════════════════════════════════
# Setup auto-restart — creates Windows Scheduled Tasks for bot stability
# Run this ONCE to create both:
#   1. Daily restart at 04:00 (prevents socket leak buildup)
#   2. Health check every 5 min (restarts if dead)
# ═══════════════════════════════════════════════════════════════════════════

$BotPath = $PSScriptRoot
$ErrorActionPreference = "Stop"

Write-Host "=== Setting up auto-restart tasks ===" -ForegroundColor Cyan

# ─── Task 1: Daily restart at 04:00 ─────────────────────────────────
$task1Name = "TradingBotDailyRestart"
$task1Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$BotPath\start_background.ps1`"" `
    -WorkingDirectory $BotPath
$task1Trigger = New-ScheduledTaskTrigger -Daily -At "04:00"
$task1Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Hours 1)

try {
    Unregister-ScheduledTask -TaskName $task1Name -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $task1Name `
        -Action $task1Action -Trigger $task1Trigger -Settings $task1Settings `
        -Description "Restart trading bot daily to prevent socket leak buildup" | Out-Null
    Write-Host "[OK] Task 1: Daily restart at 04:00" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Task 1: $_" -ForegroundColor Red
}

# ─── Task 2: Health check every 5 minutes ───────────────────────────
$task2Name = "TradingBotHealthCheck"

# Create the health check script
$healthScript = @'
$BotPath = $PSScriptRoot
$port = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object {$_.State -eq 'Listen'}
if (-not $port) {
    # Bot is down — start it
    Add-Content -Path "$BotPath\health_check.log" -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Bot dead — restarting"
    Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $BotPath `
        -RedirectStandardOutput "$BotPath\bot_background.log" `
        -RedirectStandardError "$BotPath\bot_background_err.log" `
        -WindowStyle Hidden
}
'@
$healthScriptPath = "$BotPath\health_check.ps1"
$healthScript | Out-File -FilePath $healthScriptPath -Encoding UTF8 -Force

$task2Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$healthScriptPath`""
$task2Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
$task2Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 1)

try {
    Unregister-ScheduledTask -TaskName $task2Name -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $task2Name `
        -Action $task2Action -Trigger $task2Trigger -Settings $task2Settings `
        -Description "Check bot every 5 min and restart if dead" | Out-Null
    Write-Host "[OK] Task 2: Health check every 5 min" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Task 2: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Cyan
Get-ScheduledTask -TaskName "TradingBot*" | Select-Object TaskName, State | Format-Table

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "The bot will:"
Write-Host "  - Restart every day at 04:00 (clears socket leak)"
Write-Host "  - Auto-restart within 5 min if it crashes"
Write-Host ""
Write-Host "To check status:    Get-ScheduledTask -TaskName 'TradingBot*'"
Write-Host "To remove:          Unregister-ScheduledTask -TaskName 'TradingBotDailyRestart'"
Write-Host "                    Unregister-ScheduledTask -TaskName 'TradingBotHealthCheck'"
