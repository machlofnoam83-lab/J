@echo off
chcp 65001 >nul
echo ============================================
echo   Adiel Junior - ULTIMATE - ONE FILE!
echo   No million files - just this!
echo ============================================
echo.

REM Use English file to avoid Hebrew encoding issues
if exist venv\Scripts\python.exe (
    echo [OK] Found venv - running...
    venv\Scripts\python.exe Adiel.py
) else (
    echo [INFO] No venv, using global Python...
    python Adiel.py
    if errorlevel 1 (
        echo Trying py...
        py Adiel.py
    )
)

pause
