@echo off
chcp 65001 >nul
title Adiel Junior - Auto Runner v2.3.0
REM ============================================================
REM  Adiel Junior - Auto Runner (v2.3.0)
REM  עדכון אוטומטי (git או ZIP) + תיקון תלויות + הרצה נקייה
REM ============================================================

REM --- הגנה: לרוץ מעותק ב-TEMP כדי שעדכון הקובץ עצמו לא ישבור את הריצה ---
REM --- חשוב: העותק ב-TEMP מקבל את נתיב הפרויקט האמיתי ב-ADIEL_ROOT ---
if /i "%~f0"=="%TEMP%\adiel_auto_runner.bat" goto :main
for %%I in ("%~dp0..") do set ADIEL_ROOT=%%~fI
copy /y "%~f0" "%TEMP%\adiel_auto_runner.bat" >nul 2>&1
call "%TEMP%\adiel_auto_runner.bat" %*
exit /b %ERRORLEVEL%

:main
setlocal EnableExtensions EnableDelayedExpansion
REM --- Clean broken git env vars (GIT_DIR can point to a foreign path) ---
set "GIT_DIR="
set "GIT_WORK_TREE="
set "GIT_INDEX_FILE="
set "GIT_CEILING_DIRECTORIES="
if defined ADIEL_ROOT (
    set PROJECT_ROOT=%ADIEL_ROOT%
) else (
    for %%I in ("%~dp0..") do set PROJECT_ROOT=%%~fI
)
set SCRIPT_DIR=%PROJECT_ROOT%\scripts\
cd /d "%PROJECT_ROOT%"

REM בדיקת שפיות: אם הנתיב לא נראה כמו הפרויקט - לעצור מיד במקום להשחית נתיבים
if not exist "%PROJECT_ROOT%\backend\main.py" (
    echo ============================================================
    echo   [ERROR] לא מצאתי את backend\main.py בנתיב:
    echo   %PROJECT_ROOT%
    echo.
    echo   הרץ את הקובץ הזה מתוך תיקיית הפרויקט ^(scripts\run_with_autofix.bat^)
    echo ============================================================
    pause
    exit /b 1
)

REM UTF-8 לכל תהליכי הפייתון - מונע UnicodeEncodeError באימוג'י
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set REPO_ZIP_URL=https://codeload.github.com/machlofnoam83-lab/J/zip/refs/heads/arena/019ff012-j
set DID_UPDATE=

echo ============================================================
echo   Adiel Junior - Auto Runner v2.3.0
echo   עדכון אוטומטי + תיקון שגיאות + הרצה בלי Backend כפול
echo ============================================================
echo.

REM =============== [0/5] עדכון קוד אוטומטי ===============
echo [0/5] בודק עדכונים...
if exist ".git" goto :update_git
goto :update_zip

:update_git
where git >nul 2>&1
if errorlevel 1 (
    echo   [WARN] git לא מותקן - מנסה עדכון דרך ZIP...
    goto :update_zip
)
echo   מצב git - מושך מהענף arena/019ff012-j
git diff --quiet 2>nul
if errorlevel 1 (
    echo   [INFO] יש שינויים מקומיים - שומר בצד לפני העדכון ^(git stash^)
    git stash push -u -m "auto-stash before update" >nul 2>&1
)
git pull --ff-only origin arena/019ff012-j > "%TEMP%\adiel_pull.txt" 2>&1
if errorlevel 1 (
    echo   [WARN] pull נכשל - מנסה עדכון דרך ZIP במקום...
    goto :update_zip
) else (
    set /p PULL_OUT=<"%TEMP%\adiel_pull.txt"
    echo   !PULL_OUT! | findstr /i "Already" >nul
    if errorlevel 1 (
        set DID_UPDATE=1
        echo   [OK] ירד שדרוג חדש מהשרת! הקוד עודכן.
    ) else (
        echo   [OK] הקוד כבר בגירסה האחרונה
    )
)
goto :update_done

:update_zip
echo   אין git - מוריד גירסה חדשה מ-GitHub ^(ZIP, בלי למחוק כלום^)...
where curl >nul 2>&1
if errorlevel 1 (
    echo   [WARN] curl לא זמין - ממשיך עם הגירסה המקומית
    goto :update_done
)
where tar >nul 2>&1
if errorlevel 1 (
    echo   [WARN] tar לא זמין - ממשיך עם הגירסה המקומית
    goto :update_done
)
curl -sL --max-time 60 -o "%TEMP%\adiel_latest.zip" "%REPO_ZIP_URL%"
if errorlevel 1 (
    echo   [WARN] ההורדה נכשלה ^(אין אינטרנט?^) - ממשיך עם הגירסה המקומית
    goto :update_done
)
if exist "%TEMP%\adiel_update" rd /s /q "%TEMP%\adiel_update" >nul 2>&1
mkdir "%TEMP%\adiel_update" >nul 2>&1
tar -xf "%TEMP%\adiel_latest.zip" -C "%TEMP%\adiel_update" >nul 2>&1
if errorlevel 1 (
    echo   [WARN] החילוץ נכשל - ממשיך עם הגירסה המקומית
    goto :update_done
)
set EXTRACTED=
for /d %%D in ("%TEMP%\adiel_update\*") do if not defined EXTRACTED set EXTRACTED=%%D
if not defined EXTRACTED (
    echo   [WARN] לא נמצאה תיקיית עדכון - ממשיך עם הגירסה המקומית
    goto :update_done
)
echo   מעתיק קבצים חדשים... ^(venv / data / node_modules נשמרים, דבר לא נמחק^)
robocopy "!EXTRACTED!" "%PROJECT_ROOT%" /E /XD .git venv node_modules /R:2 /W:2 /NFL /NDL /NP /NJH /NJS >nul 2>&1
if errorlevel 8 (
    echo   [WARN] העתקה חלקית בלבד - ממשיך בכל זאת
) else (
    set DID_UPDATE=1
    echo   [OK] הקוד עודכן מה-ZIP האחרון!
)
rd /s /q "%TEMP%\adiel_update" >nul 2>&1
del /q "%TEMP%\adiel_latest.zip" >nul 2>&1
goto :update_done

:update_done
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

REM =============== [2/5] תלויות ===============
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

REM =============== [4/5] Backend - בלי כפילויות ===============
echo [4/5] מפעיל Backend...
set BACKEND_ALIVE=
curl -s -o nul --max-time 2 http://127.0.0.1:8765/health >nul 2>&1
if not errorlevel 1 set BACKEND_ALIVE=1

if defined BACKEND_ALIVE (
    echo   [OK] Backend כבר רץ על פורט 8765 - לא פותח עותק שני!
    if defined DID_UPDATE (
        echo   [חשוב!] ירד שדרוג חדש אבל ה-Backend הישן עוד רץ עם קוד ישן.
        echo           סגור את חלון "Adiel Backend" הישן והרץ את הקובץ הזה שוב.
    )
) else (
    echo   מפעיל Backend בחלון נפרד...
    start "Adiel Backend" cmd /k "cd /d %PROJECT_ROOT%\backend && set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& %PROJECT_ROOT%\venv\Scripts\python.exe main.py"
    timeout /t 5 /nobreak >nul
)
echo.

REM =============== [5/5] HUD ===============
echo [5/5] מפעיל HUD...
cd /d "%PROJECT_ROOT%\frontend"
if exist node_modules\electron (
    echo   [INFO] מריץ Electron HUD...
    call npx electron .
) else (
    echo   [WARN] Electron לא מותקן - פותח HUD בדפדפן...
    start "" "%PROJECT_ROOT%\frontend\src\index.html"
)

cd /d "%PROJECT_ROOT%"
echo.
echo   אדיאל סיימה את הריצה. אם משהו השתבש - צלם את החלון הזה ושלח לי.
pause
