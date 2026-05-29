' Trading Bot Launcher  (LEGACY ALIAS — kept for backward compatibility)
' =====================================================================
' This used to start watchdog.py directly, which competed with the current
' supervisor (bot_supervisor.py) over port 8000 and caused conflicts.
'
' It now simply delegates to the CURRENT launcher, start_supervisor.vbs, which
' starts the single detached supervisor (idempotent — safe to run repeatedly).

Option Explicit

Dim oShell, oFSO, sDir, sTarget
Set oFSO   = CreateObject("Scripting.FileSystemObject")
sDir       = oFSO.GetParentFolderName(WScript.ScriptFullName)
sTarget    = sDir & "\start_supervisor.vbs"

If oFSO.FileExists(sTarget) Then
    Set oShell = CreateObject("WScript.Shell")
    oShell.Run "wscript.exe """ & sTarget & """", 0, False
End If
