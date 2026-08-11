@echo off
chcp 65001 >nul
REM Adiel Junior - Run Dev Mode
echo Starting Adiel Junior in DEV mode...
echo מריץ במצב פיתוח...

REM Activate venv if exists
if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Start backend in new window
echo Starting Backend on port 8765...
start "Adiel Backend" cmd /k "cd /d %~dp0..\backend && python main.py"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting HUD Frontend...
cd /d %~dp0..\frontend
if exist node_modules (
    call npm start
) else (
    echo Installing frontend deps first...
    call npm install
    call npm start
)

pause
