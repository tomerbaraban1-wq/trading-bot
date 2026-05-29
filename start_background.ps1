# ═══════════════════════════════════════════════════════════════════════════
# הפעלת הבוט ברקע — לא נסגר עם סגירת הטרמינל
# ═══════════════════════════════════════════════════════════════════════════

$BotPath = "C:\Users\תומר\Pictures\קלוד קוד\trading-bot"
$LogPath = "$BotPath\bot_background.log"

# הסר תהליך קודם של הבוט (אם יש)
Get-Process | Where-Object {
    $_.ProcessName -eq "python" -and $_.Path -like "*python*"
} | ForEach-Object {
    try {
        $cmd = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
        if ($cmd -match "uvicorn|main:app") {
            Write-Host "Killing existing bot process (PID $($_.Id))..."
            Stop-Process -Id $_.Id -Force
            Start-Sleep -Seconds 2
        }
    } catch {}
}

# המתן ש-port 8000 יתפנה
Write-Host "Waiting for port 8000 to be free..."
$attempts = 0
while ($attempts -lt 30) {
    $port = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    if (-not $port) { break }
    Start-Sleep -Seconds 2
    $attempts++
}

# הפעל את הבוט ברקע — מנותק מהטרמינל
Write-Host "Starting bot in background..."
$proc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory $BotPath `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError "$BotPath\bot_background_err.log" `
    -WindowStyle Hidden `
    -PassThru

Write-Host ""
Write-Host "================================================================"
Write-Host " Bot started as detached process!"
Write-Host "================================================================"
Write-Host "  PID:    $($proc.Id)"
Write-Host "  Log:    $LogPath"
Write-Host "  Err:    $BotPath\bot_background_err.log"
Write-Host ""
Write-Host "  The bot will NOT die when you close this terminal."
Write-Host "  To stop the bot:"
Write-Host "    Stop-Process -Id $($proc.Id) -Force"
Write-Host ""
Write-Host "  To check status:"
Write-Host "    Get-Process -Id $($proc.Id)"
Write-Host "================================================================"
