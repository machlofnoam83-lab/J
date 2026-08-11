@echo off
chcp 65001 >nul
REM Adiel Junior - Run with Auto-Fix
REM מריץ עם מערכת תיקון אוטומטית - כמו פעם קודמת רק חכם יותר

setlocal

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d "%PROJECT_ROOT%"

echo ============================================
echo   Adiel Junior - Auto Fix Runner
echo   מריץ עם תיקון שגיאות אוטומטי
echo ============================================
echo.

REM Check venv
if not exist venv\Scripts\python.exe (
    echo [INFO] venv לא קיים, יוצר...
    python -m venv venv
)

echo [1/4] מפעיל מערכת תיקון אוטומטית...
call venv\Scripts\activate.bat
python scripts\auto_installer.py
if errorlevel 1 (
    echo [WARN] Auto installer נכשל חלקית, ממשיך עם fallback...
)

echo.
echo [2/4] בודק Backend...
python -c "import fastapi, uvicorn; print('Backend OK')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Backend תלויות חסרות, מריץ תיקון אגרסיבי...
    python -m pip install --upgrade pip
    python -m pip install -r backend\requirements.txt --no-cache-dir
    python -m pip install pygame-ce webrtcvad-wheels Pillow --upgrade --break-system-packages 2>nul
)

echo.
echo [3/4] בודק Frontend...
if not exist frontend\node_modules (
    echo [INFO] node_modules חסר, מתקין...
    cd /d "%PROJECT_ROOT%\frontend"
    call npm install --legacy-peer-deps
    cd /d "%PROJECT_ROOT%"
)

echo.
echo [4/4] מריץ את אדיאל ג'וניור...
echo Backend: http://localhost:8765/status
echo HUD ייפתח אוטומטית...
echo.

REM Start backend in background
start "Adiel Backend" cmd /k "cd /d %PROJECT_ROOT%\backend && python main.py"

timeout /t 4 /nobreak >nul

REM Try Electron, fallback to Python GUI if fails
cd /d "%PROJECT_ROOT%\frontend"
if exist node_modules\electron (
    echo [INFO] מריץ Electron HUD...
    call npx electron . --dev
) else (
    echo [WARN] Electron לא מותקן, מריץ Fallback GUI...
    cd /d "%PROJECT_ROOT%"
    python backend\gui_fallback.py
)

pause
