@echo off
title UniFace - Live Prepoznavanje Lica (Logitech C270)
echo ===================================================
echo     UniFace Live Camera - Logitech C270
echo ===================================================
echo.

cd /d "%~dp0"

:: Pronadi Python: ugradeni runtime, .venv ili sistemski
if exist "python\python.exe" (
    set "PY_CMD=python\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

echo Pokrecem video prozor za prepoznavanje uzivo...
echo.
echo Tipkovnicke kratice u prozoru:
echo   [Q] ili [ESC] - Zatvaranje prozora
echo   [S]           - Spremi trenutni kadar
echo   [+] / [-]     - Promjena praga osjetljivosti
echo   [R]           - Osvjezi bazu osoba
echo   [SPACE]       - Zamrzni / nastavi sliku
echo.

"%PY_CMD%" "src\live_cam.py"

if errorlevel 1 (
    echo.
    echo Doslo je do greske prilikom pokretanja kamere.
    pause
)
