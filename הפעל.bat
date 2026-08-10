@echo off
echo ============================================
echo   אדיאל ג'וניור - ULTIMATE - קובץ אחד!
echo ============================================
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe "הפעל_את_אדיאל.py"
) else (
    python "הפעל_את_אדיאל.py"
)
pause
