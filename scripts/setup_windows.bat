@echo off
REM Adiel Junior - Windows Setup Script v2.0 AUTO-FIX
REM מתקין את כל התלויות עם מערכת תיקון אוטומטית

setlocal enabledelayedexpansion

echo ============================================
echo   Adiel Junior - Setup v2.0 AUTO-FIX
echo   אדיאל ג'וניור - התקנה חכמה
echo   עם מערכת תיקון שגיאות אוטומטית
echo ============================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
echo [INFO] Project: %PROJECT_ROOT%
echo [INFO] Python:
python --version
echo [INFO] Node:
node --version 2>nul || echo Node not found
echo [INFO] npm:
npm --version 2>nul || echo npm not found
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python לא נמצא! התקן מ-https://python.org + סמן Add to PATH
    pause
    exit /b 1
)

echo [1/4] יוצר סביבה וירטואלית...
cd /d "%PROJECT_ROOT%"
if not exist venv (
    python -m venv venv
    echo [OK] venv נוצר
) else (
    echo [OK] venv כבר קיים
)

call "%PROJECT_ROOT%\venv\Scripts\activate.bat"

echo.
echo [2/4] מריץ מערכת תיקון אוטומטית להתקנת Python...
echo זה יתקן לבד: pygame->pygame-ce, Pillow, webrtcvad, וכו'
python scripts\auto_installer.py
if errorlevel 1 (
    echo [WARN] חלק מההתקנות נכשלו, מנסה תיקון ידני אגרסיבי...
    echo [FIX] מתקין חבילות קריטיות ישירות...
    pip install --upgrade pip
    pip install pygame-ce>=2.4.1 webrtcvad-wheels>=0.2.13 Pillow>=10.4.0
    pip install -r backend\requirements.txt --no-cache-dir --only-binary=:all: 2>nul
    pip install -r backend\requirements.txt --no-cache-dir
)

echo.
echo [3/4] מתקין Frontend (HUD) עם תיקון אוטומטי...
where npm >nul 2>&1
if errorlevel 1 (
    echo [WARN] npm לא נמצא, מדלג - אפשר להריץ רק Backend
    echo הורד Node.js מ-https://nodejs.org (מומלץ 20 LTS)
    goto :skip_npm
)

cd /d "%PROJECT_ROOT%\frontend"
if not exist package.json (
    echo [ERROR] package.json לא נמצא! תבדוק שחילצת נכון
    pause
    exit /b 1
)

echo [INFO] מנסה npm install...
call npm install
if errorlevel 1 (
    echo [FIX] נכשל, מנסה עם --legacy-peer-deps (ל-Node 24)...
    call npm install --legacy-peer-deps
    if errorlevel 1 (
        echo [WARN] npm עדיין נכשל - אפשר להמשיך עם GUI חלופי
        echo [INFO] תריץ: python backend\gui_fallback.py במקום Electron
    )
)

:skip_npm
cd /d "%PROJECT_ROOT%"

echo.
echo [4/4] בודק שהכל עובד...
python -c "import fastapi; print('fastapi OK')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] fastapi לא מותקן
) else (
    echo [OK] Backend deps OK
)

if exist frontend\node_modules (
    echo [OK] Frontend deps OK
) else (
    echo [WARN] Frontend לא הותקן - יש fallback
)

echo.
echo ============================================
echo   Setup הושלם! עם Auto-Fix
echo ============================================
echo.
echo איך להריץ:
echo   1. scripts\run_with_autofix.bat  - הכי מומלץ, עם תיקון אוטומטי
echo   2. scripts\run.bat               - רגיל
echo   3. python launcher.py            - משגר מאוחד
echo.
echo אם עדיין יש שגיאה, המערכת תתקן לבד בהרצה הבאה!
echo.
echo לוג שגיאות נשמר ב: backend\data\error_recovery_log.json
echo.
pause
