# ═══════════════════════════════════════════════════════════════════════════
# אבחון מי הורג את הבוט (אנטי-וירוס? Windows? משתמש?)
# ═══════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "==================================================================="
Write-Host " Bot Crash Diagnostic Tool"
Write-Host "==================================================================="
Write-Host ""

# 1. בדיקת Windows Defender
Write-Host "[1/4] Checking Windows Defender exclusions..."
$BotPath = "C:\Users\תומר\Pictures\קלוד קוד\trading-bot"
try {
    $exclusions = Get-MpPreference | Select-Object -ExpandProperty ExclusionPath
    if ($exclusions -contains $BotPath) {
        Write-Host "  [OK] Bot path is excluded from Defender" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Bot path NOT excluded from Defender!" -ForegroundColor Yellow
        Write-Host "  Fix: Add-MpPreference -ExclusionPath '$BotPath'"
    }
} catch {
    Write-Host "  [SKIP] Could not check Defender (admin needed)" -ForegroundColor Yellow
}

# 2. בדיקה ב-Event Log אם יש process termination
Write-Host ""
Write-Host "[2/4] Checking Event Log for process kills (last 24h)..."
try {
    $events = Get-WinEvent -FilterHashtable @{
        LogName='System'
        StartTime=(Get-Date).AddHours(-24)
    } -ErrorAction SilentlyContinue | Where-Object {
        $_.Message -match "python|uvicorn" -or
        ($_.Id -eq 1074) -or  # System shutdown
        ($_.Id -eq 41)        # Critical kernel power
    }
    if ($events) {
        Write-Host "  [FOUND] $($events.Count) related events" -ForegroundColor Yellow
        $events | Select-Object -First 5 | ForEach-Object {
            Write-Host "    $($_.TimeCreated): $($_.Id) - $($_.Message.Substring(0, [Math]::Min(80, $_.Message.Length)))..."
        }
    } else {
        Write-Host "  [OK] No process kill events in event log" -ForegroundColor Green
    }
} catch {
    Write-Host "  [SKIP] Event log check failed" -ForegroundColor Yellow
}

# 3. בדיקה אם NSSM מותקן
Write-Host ""
Write-Host "[3/4] Checking for NSSM..."
$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if ($nssm) {
    Write-Host "  [OK] NSSM is installed at $($nssm.Source)" -ForegroundColor Green
    Write-Host "  Recommendation: Run install_as_service.bat"
} else {
    Write-Host "  [MISSING] NSSM not installed" -ForegroundColor Yellow
    Write-Host "  Download: https://nssm.cc/download"
}

# 4. בדיקת זמני קריסה
Write-Host ""
Write-Host "[4/4] Analyzing crash patterns..."
$LogFile = "$BotPath\watchdog.log"
if (Test-Path $LogFile) {
    $crashes = Select-String -Path $LogFile -Pattern "קרס" -SimpleMatch
    Write-Host "  Total crashes in watchdog.log: $($crashes.Count)"

    # Count by type
    $ctrlc = ($crashes | Where-Object { $_.Line -match "Ctrl\+C|terminal" }).Count
    $sigkill = ($crashes | Where-Object { $_.Line -match "4294967295" }).Count
    $portbusy = ($crashes | Where-Object { $_.Line -match "port busy" }).Count

    Write-Host "    Ctrl+C / Terminal Closed: $ctrlc  (you closing terminal)"
    Write-Host "    SIGKILL (external):       $sigkill (antivirus/task manager?)"
    Write-Host "    Port busy:                $portbusy (restart artifacts)"

    if ($sigkill -gt 3) {
        Write-Host ""
        Write-Host "  ⚠️  High SIGKILL count detected!" -ForegroundColor Red
        Write-Host "  This usually means antivirus or Task Manager is killing it."
        Write-Host "  Solution: Add Windows Defender exclusion + install as Service"
    }
    if ($ctrlc -gt 5) {
        Write-Host ""
        Write-Host "  💡 You're closing the terminal often." -ForegroundColor Cyan
        Write-Host "  Solution: Run as background process (start_background.ps1)"
        Write-Host "       OR: Install as Windows Service (install_as_service.bat)"
    }
}

Write-Host ""
Write-Host "==================================================================="
Write-Host " Diagnostic Complete"
Write-Host "==================================================================="
