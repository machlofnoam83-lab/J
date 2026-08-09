@echo off
echo Building Adiel Junior Executable...
echo בונה EXE...

cd /d %~dp0\..

REM Activate venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python build\build.py --all

echo.
echo Build done! Check dist\ folder
pause
