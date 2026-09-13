@echo off
REM Windows consoles default to an OEM codepage (cp862/cp437, cp1255 on a Hebrew
REM system). JARVIS prints Hebrew, and Python would encode it in that codepage
REM while a parent process decoded UTF-8 - which aborted test runs with
REM UnicodeDecodeError. Switch the console to UTF-8 and pin Python to match.
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM ═══════════════════════════════════════════════════════════════════
REM  J.A.R.V.I.S. — one-time Windows setup
REM  Installs the Python brain dependencies and the Electron shell.
REM  Everything stays local: no API keys, no cloud services.
REM ═══════════════════════════════════════════════════════════════════
setlocal ENABLEEXTENSIONS
cd /d "%~dp0\.."
echo.
echo  ================================================
echo   J.A.R.V.I.S.  -  SETUP  (Mark VII)
echo  ================================================
echo.

REM ---- 1. python ---------------------------------------------------
where py >nul 2>nul
if %ERRORLEVEL%==0 (set PY=py -3) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (set PY=python) else (
    echo [x] Python 3.10+ was not found on PATH.
    echo     Install it from https://www.python.org/downloads/windows/
    echo     and tick "Add python.exe to PATH", then run this file again.
    pause & exit /b 1
  )
)
echo [*] using: %PY%
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if %ERRORLEVEL%==1 (
  echo [x] Python 3.10 or newer is required.
  pause & exit /b 1
)

REM ---- 2. brain dependencies ---------------------------------------
echo.
echo [*] installing brain dependencies (torch, numpy, scipy, soundfile, aiohttp, psutil)
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
if %ERRORLEVEL%==1 (
  echo [x] pip install failed. Check your network or run with --user.
  pause & exit /b 1
)

REM ---- 3. optional extras that make the desktop skills work ---------
echo.
echo [*] installing optional desktop integrations
%PY% -m pip install pyperclip mss pyautogui pillow pynput
echo     (if any of these fail JARVIS still runs - the matching skills
echo      simply report that the component is unavailable)

REM ---- 4. node + electron ------------------------------------------
echo.
where npm >nul 2>nul
if %ERRORLEVEL%==0 (
  echo [*] installing the Electron shell
  call npm install
  if %ERRORLEVEL%==1 echo [!] npm install reported a problem - the HUD can still run in browser mode.
) else (
  echo [!] Node.js was not found. Install LTS from https://nodejs.org to get the
  echo     desktop shell, or run: %PY% -m core.server  and open the HUD in a browser.
)

REM ---- 5. verify ---------------------------------------------------
echo.
echo [*] self-test
%PY% tests\run_all.py
echo.
echo  ================================================
echo   Setup complete.  Start JARVIS with:  JARVIS.bat
echo  ================================================
echo.
pause
endlocal
