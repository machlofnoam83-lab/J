@echo off
REM Adiel Junior - Windows Setup Script
REM מתקין את כל התלויות ל-Windows

echo ============================================
echo   Adiel Junior - Setup for Windows
echo   אדיאל ג'וניור - התקנה
echo ============================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+ from python.org
    echo Make sure to check "Add to PATH"
    pause
    exit /b 1
)

echo [1/5] Python found

REM Create venv
echo [2/5] Creating virtual environment...
if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat

echo [3/5] Installing Python dependencies...
pip install --upgrade pip
pip install -r backend\requirements.txt

echo [4/5] Installing Tesseract OCR (for screen text)...
REM User needs to install manually if not present
where tesseract >nul 2>&1
if errorlevel 1 (
    echo [WARN] Tesseract OCR not found - screen text recognition will be limited
    echo Please install from: https://github.com/UB-Mannheim/tesseract/wiki
    echo And add to PATH, with Hebrew language pack
) else (
    echo Tesseract found
)

echo [5/5] Installing Node dependencies for HUD...
cd frontend
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Install from nodejs.org
    cd ..
    pause
    exit /b 1
)
call npm install
cd ..

echo.
echo ============================================
echo   Setup Complete! התקנה הושלמה
echo ============================================
echo.
echo להרצה:
echo   run.bat              - הרצת Dev mode
echo   build\build.bat      - בניית EXE
echo.
echo הערה: להפעלת זיהוי קולי בעברית:
echo  - אם יש GPU, המודל יטען מהר יותר
echo  - Edge-TTS דורש אינטרנט בפעם הראשונה
echo  - ל-Vosk עברי, הורד מודל מ: https://alphacephei.com/vosk/models
echo    ושם בתיקייה backend\data\vosk-model-small-he-0.22
echo.
pause
