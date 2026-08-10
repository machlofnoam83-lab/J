@echo off
chcp 65001 >nul
echo Starting Adiel Junior ULTIMATE...
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe Adiel.py
) else (
    python Adiel.py
)
pause
