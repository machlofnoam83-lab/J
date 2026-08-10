@echo off
chcp 65001 >nul
echo ============================================
echo   אדיאל ג'וניור - ULTIMATE - קובץ אחד!
echo   לא צריך מיליון קבצים - רק זה!
echo ============================================
echo.

REM Try English file first (no encoding issues), then Hebrew
if exist Adiel.py (
    echo מריץ Adiel.py (English - no encoding issues)...
    if exist venv\Scripts\python.exe (
        venv\Scripts\python.exe Adiel.py
    ) else (
        python Adiel.py
    )
) else if exist "הפעל_את_אדיאל.py" (
    echo מריץ הפעל_את_אדיאל.py...
    if exist venv\Scripts\python.exe (
        venv\Scripts\python.exe "הפעל_את_אדיאל.py"
    ) else (
        python "הפעל_את_אדיאל.py"
    )
) else (
    echo מחפש launcher...
    if exist venv\Scripts\python.exe (
        venv\Scripts\python.exe launcher.py
    ) else (
        python launcher.py
    )
)

pause
