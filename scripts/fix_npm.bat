@echo off
REM Quick fix for ENOENT package.json error
REM תיקון מהיר לשגיאה שקיבלת

echo ============================================
echo   FIX - תיקון שגיאת npm package.json
echo ============================================
echo.
echo השגיאה שקיבלת:
echo "Could not read package.json in ...\scripts\package.json"
echo הסיבה: הרצת npm install בתיקייה הלא נכונה (scripts במקום frontend)
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set FRONTEND_DIR=%PROJECT_ROOT%\frontend

echo [1/3] בודק איפה package.json באמת נמצא...
if exist "%FRONTEND_DIR%\package.json" (
    echo [OK] נמצא ב: %FRONTEND_DIR%\package.json
) else (
    echo [ERROR] לא נמצא! תבדוק שחילצת את ה-ZIP נכון
    echo מחפש...
    dir /s "%PROJECT_ROOT%\package.json"
    pause
    exit /b 1
)

echo.
echo [2/3] עובר לתיקיית frontend...
cd /d "%FRONTEND_DIR%"
echo נמצא ב: %CD%
echo.

echo [3/3] מריץ npm install כמו שצריך...
echo מריץ: npm install
call npm install

if errorlevel 1 (
    echo.
    echo [WARN] נכשל, מנסה עם --legacy-peer-deps (בגלל Node 24 שאתה משתמש)
    call npm install --legacy-peer-deps
)

if errorlevel 1 (
    echo.
    echo [ERROR] עדיין נכשל - אולי Node 24 חדש מדי
    echo פתרונות:
    echo 1. התקן Node 20 LTS מ- https://nodejs.org
    echo 2. או מחק node_modules ונסה שוב
    echo 3. או הרץ רק Backend בלי Electron: python launcher.py --backend-only
    pause
    exit /b 1
)

echo.
echo ============================================
echo   תוקן! Fixed!
echo ============================================
echo עכשיו תריץ:
echo   cd /d "%PROJECT_ROOT%"
echo   scripts\run.bat
echo.
pause
