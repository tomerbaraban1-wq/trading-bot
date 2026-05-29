# ═══════════════════════════════════════════════════════════════════════════
# Stability Monitor — 30 minute deep test
# Tracks: CloseWait, Memory, Handles, Threads + auto-restart if dead
# ═══════════════════════════════════════════════════════════════════════════

param([int]$DurationMinutes = 30, [int]$CheckInterval = 180)

$BotPath = "C:\Users\תומר\Pictures\קלוד קוד\trading-bot"
$ReportFile = "$BotPath\stability_report.txt"
$ExpectedPid = 20420  # Current bot PID

function Get-BotPid {
    $conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object {$_.State -eq 'Listen'}
    if ($conn) { return $conn.OwningProcess }
    return $null
}

function Start-NewBot {
    $proc = Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $BotPath `
        -RedirectStandardOutput "$BotPath\bot_background.log" `
        -RedirectStandardError "$BotPath\bot_background_err.log" `
        -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 10
    return $proc.Id
}

"=== STABILITY MONITOR STARTED $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $ReportFile
"Duration: $DurationMinutes min, Check every: $CheckInterval sec" | Out-File $ReportFile -Append
"Initial PID: $ExpectedPid" | Out-File $ReportFile -Append
"" | Out-File $ReportFile -Append

$checks = [Math]::Floor(($DurationMinutes * 60) / $CheckInterval)
$crashes = 0
$currentPid = $ExpectedPid

for ($i = 1; $i -le $checks; $i++) {
    $p = Get-Process -Id $currentPid -ErrorAction SilentlyContinue
    $now = Get-Date -Format "HH:mm:ss"

    if ($p) {
        $uptime = [math]::Round(((Get-Date) - $p.StartTime).TotalMinutes, 1)
        $mem = [math]::Round($p.WorkingSet64/1MB, 1)
        $threads = $p.Threads.Count
        $handles = $p.HandleCount
        $conns = Get-NetTCPConnection -OwningProcess $currentPid -ErrorAction SilentlyContinue
        $closeWait = ($conns | Where-Object {$_.State -eq 'CloseWait'}).Count
        $total = $conns.Count

        $msg = "[$now] Check $i/$checks | PID $currentPid | uptime=${uptime}min mem=${mem}MB threads=${threads} handles=${handles} CW=${closeWait} conns=${total}"
        Write-Host $msg
        $msg | Out-File $ReportFile -Append
    } else {
        $crashes++
        $msg = "[$now] Check $i/$checks | ❌ BOT DIED (crash #$crashes) — auto-restarting..."
        Write-Host $msg -ForegroundColor Red
        $msg | Out-File $ReportFile -Append

        # Wait for port to free
        $portWait = 0
        while ($portWait -lt 30) {
            if (-not (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Seconds 2
            $portWait += 2
        }

        $currentPid = Start-NewBot
        "[$now] Restarted as PID $currentPid" | Out-File $ReportFile -Append
    }

    if ($i -lt $checks) { Start-Sleep -Seconds $CheckInterval }
}

"" | Out-File $ReportFile -Append
"=== TEST COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $ReportFile -Append
"Total crashes during $DurationMinutes min test: $crashes" | Out-File $ReportFile -Append
if ($crashes -eq 0) {
    "✅ STABILITY VERIFIED — bot survived $DurationMinutes minutes" | Out-File $ReportFile -Append
} else {
    "⚠️ $crashes crashes detected — needs further investigation" | Out-File $ReportFile -Append
}
