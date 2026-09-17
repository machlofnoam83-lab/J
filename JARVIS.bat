@echo off
REM Windows consoles default to an OEM codepage (cp862/cp437, cp1255 on a Hebrew
REM system). JARVIS prints Hebrew, and Python would encode it in that codepage
REM while a parent process decoded UTF-8 - which aborted test runs with
REM UnicodeDecodeError. Switch the console to UTF-8 and pin Python to match.
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM ═══════════════════════════════════════════════════════════════════
REM  J.A.R.V.I.S. — start the desktop HUD (Windows)
REM  First run? Execute tools\setup_windows.bat once.
REM ═══════════════════════════════════════════════════════════════════
setlocal ENABLEEXTENSIONS
cd /d "%~dp0"

set ELECTRON=node_modules\electron\dist\electron.exe
if exist "%ELECTRON%" goto desktop

where npm >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [*] Electron shell not installed yet - installing now ^(one time^)...
    call npm install
    if exist "%ELECTRON%" goto desktop
)
goto browser

:desktop
echo [*] booting J.A.R.V.I.S. desktop shell
start "" "%ELECTRON%" .
goto end

:browser
echo [!] Electron is unavailable - falling back to browser mode.
where py >nul 2>nul
if %ERRORLEVEL%==0 (set PY=py -3) else (set PY=python)
start "JARVIS brain" cmd /k "%PY% -m core.server"
echo [*] waiting for the brain to come up...
timeout /t 8 /nobreak >nul
start "" "http://127.0.0.1:8756/"

:end
endlocal
exit /b 0
