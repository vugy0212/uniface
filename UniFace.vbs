' ==============================================================================
' UniFace Studio - Tiho pokretanje aplikacije (100% bez crnog terminala)
' ==============================================================================
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
strPath = fso.GetParentFolderName(WScript.ScriptFullName)

WshShell.CurrentDirectory = strPath

' 1. Oslobodi port 7860 ako je zaostao stari proces
WshShell.Run "cmd /c for /f ""tokens=5"" %a in ('netstat -aon ^| findstr "":7860"" ^| findstr ""LISTENING""') do taskkill /f /pid %a >nul 2>&1", 0, True

' 2. Pronadi pythonw.exe (Windowed Python bez konzole)
pyw = "pythonw.exe"
If fso.FileExists(strPath & "\python\pythonw.exe") Then
    pyw = """" & strPath & "\python\pythonw.exe"""
ElseIf fso.FileExists(strPath & "\.venv\Scripts\pythonw.exe") Then
    pyw = """" & strPath & "\.venv\Scripts\pythonw.exe"""
End If

' 3. Pokreni aplikaciju potpuno tiho (prozor = 0 -> skriveno)
WshShell.Run pyw & " """ & strPath & "\run_studio.py""", 0, False
