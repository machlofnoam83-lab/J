@echo off
chcp 65001 >nul
REM Fix fastapi ModuleNotFoundError + audio

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d "%PROJECT_ROOT%"

echo ============================================
echo   תיקון מהיר - fastapi לא נמצא + קול
echo ============================================
echo.

echo [1/4] בודק venv...
if not exist venv\Scripts\python.exe (
    echo יוצר venv...
    python -m venv venv
)

echo [2/4] מפעיל venv ומתקין תלויות...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo מתקין fastapi + כל השאר...
pip install fastapi uvicorn websockets pydantic python-dotenv requests --upgrade
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo מנסה עם תיקון אוטומטי...
    python scripts\auto_installer.py
)

echo.
echo [3/4] מתקין תיקוני קול...
pip install pygame-ce edge-tts pyttsx3 sounddevice soundfile webrtcvad-wheels --upgrade

echo.
echo [4/4] בודק שהתיקון עבד...
python -c "import fastapi; print('✓ fastapi', fastapi.__version__)"
if errorlevel 1 (
    echo [ERROR] עדיין לא עובד, מנסה שוב...
    python -m pip install fastapi --break-system-packages
) else (
    echo [OK] fastapi תקין!
)

python -c "import pygame; print('✓ pygame-ce', pygame.__version__)" 2>nul || echo [WARN] pygame חסר

echo.
echo ============================================
echo   תוקן! עכשיו תריץ:
echo   scripts\run_with_autofix.bat
echo   או
echo   venv\Scripts\python.exe backend\main.py
echo ============================================
echo.
pause
