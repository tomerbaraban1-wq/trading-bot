# ═══════════════════════════════════════════════════════════════════════════
# Scheduled Restart — restart bot every 8 hours to prevent socket exhaustion
# This is the pragmatic solution given yfinance 1.4.0 leaks via curl_cffi
# ═══════════════════════════════════════════════════════════════════════════

$BotPath = $PSScriptRoot
$RestartIntervalHours = 8

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

Write-Host "Scheduled restart monitor — restart every $RestartIntervalHours hours"
Write-Host "Press Ctrl+C to stop monitoring (bot will continue)"

while ($true) {
    # Wait 8 hours
    Start-Sleep -Seconds ($RestartIntervalHours * 3600)

    Write-Host "$(Get-Date -Format 'HH:mm:ss') — Scheduled restart..."

    $oldPid = Get-BotPid
    if ($oldPid) {
        $oldConns = Get-NetTCPConnection -OwningProcess $oldPid -ErrorAction SilentlyContinue
        $cw = ($oldConns | Where-Object {$_.State -eq 'CloseWait'}).Count
        Write-Host "  Old bot PID $oldPid had $cw CloseWait sockets"
        Stop-Process -Id $oldPid -Force
        Start-Sleep -Seconds 10
    }

    # Wait for port to free
    $wait = 0
    while ($wait -lt 60) {
        if (-not (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Seconds 2
        $wait += 2
    }

    $newPid = Start-Bot
    Write-Host "  Restarted as PID $newPid"
}
