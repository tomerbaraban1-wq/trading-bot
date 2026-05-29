' Trading Bot — Detached Supervisor Launcher
' ============================================
' Starts bot_supervisor.py as a COMPLETELY DETACHED process with no console
' window and no parent. Such a process:
'   - cannot be killed when a terminal/cmd window closes
'   - survives Windows sleep/resume (it is suspended, not terminated)
'   - keeps running independently until reboot/logoff/its own exit
'
' Replaces the fragile start_bot.bat (whose cmd-window :loop dies when the
' session is torn down — the root cause of the 2026-05-29 morning outage).
'
' NOTE: contains NO non-ASCII literals on purpose. The username folder is in
' Hebrew, so paths are resolved at runtime (env vars / ScriptFullName) to stay
' encoding-safe regardless of how this .vbs file is saved.
'
' Idempotent: if a supervisor is already running it does nothing, so it is safe
' to run from the Startup folder on every login.

Option Explicit

Dim sDir, sPython, sSupervisor, sCmd
Dim oShell, oFSO, oWMI, colProcs, oProc, bRunning

Set oShell = CreateObject("WScript.Shell")
Set oFSO   = CreateObject("Scripting.FileSystemObject")

' Folder of this script (runtime value -> not affected by file encoding)
sDir        = oFSO.GetParentFolderName(WScript.ScriptFullName)
sSupervisor = sDir & "\bot_supervisor.py"

' Python path via env var (avoids hardcoding the Hebrew username)
sPython = oShell.ExpandEnvironmentStrings("%LOCALAPPDATA%\Programs\Python\Python313\python.exe")
If Not oFSO.FileExists(sPython) Then
    ' Fallback: rely on PATH
    sPython = "python.exe"
End If
If Not oFSO.FileExists(sSupervisor) Then WScript.Quit 2

' Idempotency: is bot_supervisor.py already running?
bRunning = False
On Error Resume Next
Set oWMI = GetObject("winmgmts:\\.\root\cimv2")
Set colProcs = oWMI.ExecQuery("SELECT CommandLine FROM Win32_Process WHERE Name='python.exe'")
For Each oProc In colProcs
    If Not IsNull(oProc.CommandLine) Then
        If InStr(LCase(oProc.CommandLine), "bot_supervisor.py") > 0 Then bRunning = True
    End If
Next
On Error GoTo 0
If bRunning Then WScript.Quit 0

' Launch detached: window style 0 (hidden), False (do not wait)
oShell.CurrentDirectory = sDir
sCmd = """" & sPython & """ """ & sSupervisor & """"
oShell.Run sCmd, 0, False
