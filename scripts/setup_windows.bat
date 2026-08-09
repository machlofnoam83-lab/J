@echo off
REM Adiel Junior - Windows Setup Script FIXED v1.1
REM מתקין את כל התלויות ל-Windows - גרסה מתוקנת

setlocal enabledelayedexpansion

echo ============================================
echo   Adiel Junior - Setup for Windows
echo   אדיאל ג'וניור - התקנה
echo ============================================
echo.

REM Get script dir and project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
echo [INFO] Script dir: %SCRIPT_DIR%
echo [INFO] Project root: %PROJECT_ROOT%
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+ from python.org
    echo Make sure to check "Add to PATH"
    echo https://python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Python found
python --version

REM Create venv in project root
echo.
echo [2/5] Creating virtual environment...
cd /d "%PROJECT_ROOT%"
if not exist venv (
    echo Creating venv...
    python -m venv venv
) else (
    echo venv already exists
)

echo Activating venv...
call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [WARN] Could not activate venv, trying without...
)

echo.
echo [3/5] Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed
    echo Trying with --break-system-packages...
    pip install --break-system-packages -r backend\requirements.txt
)

echo.
echo [4/5] Checking Tesseract OCR (optional, for screen text)...
where tesseract >nul 2>&1
if errorlevel 1 (
    echo [WARN] Tesseract OCR not found - screen text recognition will be limited
    echo [INFO] Optional - install from: https://github.com/UB-Mannheim/tesseract/wiki
    echo And add to PATH, with Hebrew language pack
) else (
    echo [OK] Tesseract found
    tesseract --version
)

echo.
echo [5/5] Installing Node dependencies for HUD...
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Install from https://nodejs.org
    echo Need Node.js 18+ (recommend 20 LTS, not 24)
    pause
    exit /b 1
)

echo [INFO] npm: 
call npm --version
echo [INFO] node:
call node --version

echo.
echo [INFO] Installing frontend deps in: %PROJECT_ROOT%\frontend
cd /d "%PROJECT_ROOT%\frontend"
if not exist package.json (
    echo [ERROR] package.json not found in frontend!
    echo Current dir: %CD%
    dir
    pause
    exit /b 1
)

echo [INFO] Running npm install in %CD%
call npm install
if errorlevel 1 (
    echo [ERROR] npm install failed!
    echo Trying with --legacy-peer-deps...
    call npm install --legacy-peer-deps
)

cd /d "%PROJECT_ROOT%"

echo.
echo ============================================
echo   Setup Complete! התקנה הושלמה - הכל תקין
echo ============================================
echo.
echo להרצה:
echo   cd /d "%PROJECT_ROOT%"
echo   scripts\run.bat              - הרצת Dev mode
echo   build\build.bat              - בניית EXE
echo   python launcher.py           - משגר מאוחד
echo.
echo אם יש בעיה עם npm בגרסת Node 24:
echo   ממליץ להתקין Node 20 LTS מ- nodejs.org
echo   או הרץ: cd frontend ^&^& npm install --legacy-peer-deps
echo.
pause
