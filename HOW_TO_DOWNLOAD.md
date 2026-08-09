# איך להוריד ולהריץ את אדיאל ג'וניור - מדריך מלא

## אופציה 1: הורדה ישירה מ-GitHub (הכי פשוט)

1.  כנס ללינק של ה-branch שלך:
    **https://github.com/machlofnoam83-lab/J/tree/arena/019fe46c-j**

2.  לחץ על הכפתור הירוק `<> Code` -> `Download ZIP`

3.  חלץ את ה-ZIP על שולחן העבודה -> תקבל תיקייה `J-arena-019fe46c-j`

4.  כנס לתיקייה.

## אופציה 2: הורדה דרך Git Clone (למפתחים)

פתח CMD / PowerShell והרץ:

```bat
git clone -b arena/019fe46c-j https://github.com/machlofnoam83-lab/J.git Adiel-Junior
cd Adiel-Junior
```

## אופציה 3: ה-ZIP שבניתי לך כאן

הקובץ `Adiel-Junior-Source.zip` נמצא בתיקיית הפרויקט שלך כאן. 
אתה יכול להוריד אותו מהקישור ב-Arena - לחץ עליו ב-file viewer.

---

## אחרי ההורדה - התקנה ב-3 צעדים (Windows)

### דרישות לפני:
- Python 3.10+ מ- https://python.org - **חובה לסמן V על Add to PATH בהתקנה**
- Node.js 18+ מ- https://nodejs.org
- מיקרופון + רמקולים

### צעד 1: התקנה אוטומטית

דאבל קליק על:
```
scripts\setup_windows.bat
```
זה מתקין הכל: venv, ספריות Python, ספריות Electron.

זה ייקח 3-5 דקות בפעם הראשונה.

### צעד 2: הורדת מודלים קוליים (עברית)

דאבל קליק או ב-CMD:
```bat
python scripts\install_models.py
```
זה מוריד Whisper small (~200MB) לזיהוי עברית.
Vosk עברי - אם האוטומטי נכשל, תוריד ידנית מ:
https://alphacephei.com/vosk/models -> `vosk-model-small-he-0.22.zip`
וחלץ לתיקייה `backend/data/vosk-model-small-he-0.22`

### צעד 3: הרצה!

דאבל קליק על:
```
scripts\run.bat
```
או:
```
python launcher.py
```

יפתחו 2 חלונות:
1. שחור - Backend ב- http://localhost:8765
2. שקוף עם Reactor כחול - זה ה-HUD!

---

## בניית EXE אמיתי (קובץ התקנה אחד)

אם אתה רוצה קובץ `Setup.exe` כמו תוכנה רגילה שאתה יכול לשלוח לחברים:

```bat
build\build.bat
```

אחרי 5-10 דקות תקבל ב:
```
frontend/dist/Adiel Junior Setup.exe        <- מתקין מלא
frontend/dist/Adiel-Junior-Portable.exe     <- גרסה ניידת, בלי התקנה - פשוט דאבל קליק ורצה
backend/dist/adiel_backend.exe              <- רק ה-backend
```

**ה-Portable EXE הוא הכי נוח - קובץ אחד שמריץ הכל.**

> הערה: בניית EXE חייבת להתבצע על מחשב Windows. אם אתה מנסה לבנות על Linux תקבל קובץ Linux, לא Windows.

---

## הרצה בלי התקנה (אם אתה ממהר)

אם לא בא לך להתקין Node.js:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
pip install PyQt6

python backend\gui_fallback.py
# או
python launcher.py --backend-only
# ובדפדפן תפתח: frontend/src/index.html
```

---

## איפה הקבצים אחרי ההתקנה?

- `Adiel Junior Setup.exe` - תתקין לתיקיית Program Files
- אחרי התקנה, תמצא קיצור דרך על שולחן העבודה ובתפריט התחל
- לחיצה עליו מריצה את אדיאל, ה-HUD יופיע במרכז
- לחץ `Ctrl+Shift+A` בכל מקום כדי להעיר אותה
- אמור "אדיאל ג'וניור" + פקודה בעברית

---

## בעיות נפוצות?

**לא שומעת אותי?**
- בדוק הרשאות מיקרופון ב-Windows Settings -> Privacy -> Microphone
- נסה כפתור 🎤 ב-HUD (התעוררות ידנית)

**אין קול?**
- Edge-TTS צריך אינטרנט בפעם הראשונה
- בדוק ווליום Windows

**Backend לא עולה?**
- פורט 8765 תפוס: `netstat -ano | findstr :8765` ואז `taskkill /PID XXXX /F`

**רוצה עזרה?** פתח Issue ב-GitHub!

תהנה בוס!
