# backup_bot.ps1 - safe daily backup of the trading bot to OneDrive.
# DB-safe: copies the live SQLite DB with shared-read, so it works even while
# the bot is running (a plain Compress-Archive fails on the locked trading.db).
# Keeps the 7 most recent backups. Scheduled to run daily via Task Scheduler.
$ErrorActionPreference = 'Stop'

$src        = $PSScriptRoot
$backupRoot = Join-Path $env:USERPROFILE 'OneDrive\TradingBot-Backups'
$stamp      = Get-Date -Format 'yyyy-MM-dd_HHmm'
$staging    = Join-Path $env:TEMP "botbackup_$stamp"

New-Item -ItemType Directory -Path $staging    -Force | Out-Null
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

# 1) mirror everything except regenerable junk, logs, and the live DB (copied separately)
robocopy $src $staging /E `
    /XD (Join-Path $src '__pycache__') (Join-Path $src 'node_modules') `
    /XF *.log trading.db trading.db-wal trading.db-shm `
    /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
# robocopy uses exit codes 0-7 for success (>=8 = real failure)
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }

# 2) consistent online backup of the live SQLite DB via Python's sqlite3 .backup()
#    (SQLite byte-range write locks make a raw file copy fail mid-write; .backup()
#    is the official, safe way to snapshot a running DB into a single clean file).
$dbSrc = Join-Path $src 'data\trading.db'
if (Test-Path $dbSrc) {
    $dbDst = Join-Path $staging 'data\trading.db'
    New-Item -ItemType Directory -Path (Split-Path $dbDst) -Force | Out-Null
    python (Join-Path $src 'db_backup.py') $dbSrc $dbDst
    if ($LASTEXITCODE -ne 0) { throw "sqlite online backup failed (exit $LASTEXITCODE)" }
}

# 3) zip the staging folder into OneDrive
$zip = Join-Path $backupRoot "bot-backup-$stamp.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
$items = Get-ChildItem -LiteralPath $staging | Select-Object -ExpandProperty FullName
Compress-Archive -LiteralPath $items -DestinationPath $zip -CompressionLevel Optimal
# cmd's rd handles the 8.3 short-path tilde in %TEMP% (C:\Users\3501~1\...) and
# read-only files, where PowerShell's Remove-Item/.NET Delete fail. A cleanup
# failure must never fail the backup itself (the zip is already written).
cmd /c rd /s /q "$staging" 2>$null

# 4) retention: keep only the 7 most recent backups
Get-ChildItem $backupRoot -Filter 'bot-backup-*.zip' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 7 |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Output "Backup OK: $zip"
exit 0
