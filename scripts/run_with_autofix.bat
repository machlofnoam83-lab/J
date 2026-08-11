@echo off
chcp 65001 >nul
title Adiel Junior - Auto Runner v2.2
REM ============================================================
REM  Adiel Junior - Auto Runner (v2.2)
REM  עדכון אוטומטי מגיט + תיקון תלויות + הרצה נקייה בלי כפילויות
REM ============================================================

REM --- הגנה: git pull באמצע ריצה של הקובץ עצמו עלול לשבור אותו.
REM --- לכן מריצים עותק מתיקיית TEMP; העותק כבר לא משתנה תחתינו.
if /i "%~f0"=="%TEMP%\adiel_auto_runner.bat" goto :main
copy /y "%~f0" "%TEMP%\adiel_auto_runner.bat" >nul 2>&1
call "%TEMP%\adiel_auto_runner.bat" %*
exit /b %ERRORLEVEL%

:main
setlocal EnableExtensions EnableDelayedExpansion
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d "%PROJECT_ROOT%"

REM UTF-8 לכל תהליכי הפייתון - מונע UnicodeEncodeError באימוג'י (הבעיה הגדולה של גרסאות ישנות)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   Adiel Junior - Auto Runner v2.2
echo   עדכון אוטומטי + תיקון שגיאות + הרצה בלי Backend כפול
echo ============================================================
echo.

REM =============== [0/5] עדכון קוד אוטומטי מגיט ===============
set DID_UPDATE=
echo [0/5] בודק עדכונים מ-GitHub...
where git >nul 2>&1
if errorlevel 1 (
    echo   [WARN] git לא מותקן - ממשיך עם הגירסה המקומית
) else (
    for /f "tokens=*" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set CUR_BRANCH=%%b
    echo   ענף נוכחי: !CUR_BRANCH!
    git diff --quiet 2>nul
    if errorlevel 1 (
        echo   [INFO] יש שינויים מקומיים - שומר בצד לפני העדכון ^(git stash^)
        git stash push -u -m "auto-stash before update" >nul 2>&1
    )
    git pull --ff-only > "%TEMP%\adiel_pull.txt" 2>&1
    if errorlevel 1 (
        echo   [WARN] העדכון לא הצליח - ממשיך עם הגירסה המקומית
    ) else (
        set /p PULL_OUT=<"%TEMP%\adiel_pull.txt"
        echo   !PULL_OUT! | findstr /i "Already" >nul
        if errorlevel 1 (
            set DID_UPDATE=1
            echo   [OK] ירד שדרוג חדש מהשרת! התקנתי אותו עכשיו.
        ) else (
            echo   [OK] הקוד כבר בגירסה האחרונה
        )
    )
)
echo.

REM =============== [1/5] סביבת Python ===============
echo [1/5] בודק סביבת Python...
if not exist venv\Scripts\python.exe (
    echo   [INFO] venv לא קיים, יוצר...
    python -m venv venv
    if errorlevel 1 (
        echo   [ERROR] לא הצלחתי ליצור venv - ודא ש-Python מותקן
        pause
        exit /b 1
    )
)
echo   [OK] venv קיים
echo.

REM =============== [2/5] תלויות מתוקנות אוטומטית ===============
echo [2/5] מפעיל תיקון תלויות אוטומטי...
venv\Scripts\python.exe scripts\auto_installer.py
if errorlevel 1 (
    echo   [WARN] Auto installer נכשל חלקית, מנסה התקנה ישירה...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r backend\requirements.txt --no-cache-dir
)
echo.

REM =============== [3/5] Frontend ===============
echo [3/5] בודק Frontend...
if not exist frontend\node_modules (
    echo   [INFO] node_modules חסר, מתקין npm...
    cd /d "%PROJECT_ROOT%\frontend"
    call npm install --legacy-peer-deps
    cd /d "%PROJECT_ROOT%"
)
echo   [OK] Frontend מוכן
echo.

REM =============== [4/5] Backend - בלי כפילויות! ===============
echo [4/5] מפעיל Backend...
set BACKEND_ALIVE=
curl -s -o nul --max-time 2 http://127.0.0.1:8765/health >nul 2>&1
if not errorlevel 1 set BACKEND_ALIVE=1

if defined BACKEND_ALIVE (
    echo   [OK] Backend כבר רץ על פורט 8765 - משתמש בו, לא פותח עותק שני!
    if defined DID_UPDATE (
        echo   [חשוב!] ירד שדרוג חדש אבל ה-Backend הישן עוד רץ עם קוד ישן.
        echo           סגור את חלון "Adiel Backend" הישן והרץ את הקובץ הזה שוב.
    )
) else (
    echo   מפעיל Backend בחלון נפרד ^(עם venv - קידוד UTF-8 תקין^)...
    start "Adiel Backend" cmd /k "cd /d %PROJECT_ROOT%\backend && set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& %PROJECT_ROOT%\venv\Scripts\python.exe main.py"
    timeout /t 5 /nobreak >nul
)
echo.

REM =============== [5/5] HUD ===============
echo [5/5] מפעיל HUD...
cd /d "%PROJECT_ROOT%\frontend"
if exist node_modules\electron (
    echo   [INFO] מריץ Electron HUD ^(בלי DevTools^)...
    call npx electron .
) else (
    echo   [WARN] Electron לא מותקן - פותח HUD בדפדפן...
    start "" "%PROJECT_ROOT%\frontend\src\index.html"
)

cd /d "%PROJECT_ROOT%"
echo.
echo   אדיאל סיימה את הריצה. אם משהו השתבש - צלם את החלון הזה ושלח לי.
pause
