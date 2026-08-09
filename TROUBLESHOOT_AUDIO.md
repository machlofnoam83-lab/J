# אדיאל לא מדברת / לא מקשיבה - פתרון

## הבעיה: לא מדברת

### בדיקה מהירה:
```bat
python scripts/test_audio.py
```

### סיבות ופתרונות:

**1. edge-tts לא מותקן / אין אינטרנט**
- אדיאל משתמשת ב-edge-tts לקול עברי טבעי, זה צריך אינטרנט בפעם הראשונה
- פתרון אוטומטי: היא תנסה pyttsx3 ו-SAPI אם edge נכשל
- הרץ:
```bat
pip install edge-tts pyttsx3 pygame-ce
python scripts/test_audio.py
```

**2. רמקולים / ווליום**
- בדוק ווליום Windows + שהרמקולים לא על Mute
- ב-HUD החדש, הקול יתנגן גם בדפדפן (Audio element) וגם דרך pygame ב-backend, אז גם אם אחד נכשל השני יעבוד

**3. pygame-ce לא מותקן**
- תוקן ב-requirements החדש
- הרץ:
```bat
pip install pygame-ce
```

**4. איך זה עובד עכשיו v2.1:**
Backend מסנתז קובץ MP3/WAV + שולח base64 ל-Frontend
Frontend מנגן עם `new Audio(base64)` - אז גם אם pygame נכשל ב-backend, תשמעי בדפדפן!

---

## הבעיה: לא מקשיבה / לא מזהה "אדיאל ג'וניור"

### בדיקה:
```bat
python scripts/test_audio.py
```

### סיבות:

**1. אין מיקרופון / הרשאות**
- Windows 10/11: Settings -> Privacy & Security -> Microphone -> Allow apps to access microphone = ON
- חבר מיקרופון, בדוק שהוא לא על Mute
- **פתרון: אפשר להשתמש במקלדת!** ה-HUD עובד מצוין עם טקסט בעברית, לא חייב מיקרופון

**2. sounddevice לא מותקן**
```bat
pip install sounddevice soundfile
```

**3. faster-whisper מודל לא ירד**
- בפעם הראשונה הוא מוריד ~200MB
- הרץ `python scripts/install_models.py`
- או פשוט תכתוב במקלדת, זה יעבוד

**4. Wake Word**
- אם המיקרופון עובד אבל לא מזהה "אדיאל ג'וניור", לחץ על כפתור 🎤 ב-HUD
- זה מדמה wake word ידנית ומתחיל להקשיב 8 שניות

---

## תיקון אוטומטי מלא:

הכי פשוט:
```bat
scripts\run_with_autofix.bat
```

זה:
1. מריץ auto_installer שמתקן pygame, pillow, webrtcvad וכו'
2. בודק רמקולים ומיקרופון
3. מתקין מה שחסר
4. מריץ backend + frontend

אם עדיין לא עובד:
```bat
python scripts/test_audio.py --fix
python scripts/auto_installer.py
```

---

## עבודה בלי קול בכלל (100% עובד):

אדיאל עובדת מצוין גם בלי מיקרופון ורמקולים:

1. פתח `scripts\run.bat`
2. ב-HUD, כתוב בעברית בתיבת הטקסט למטה
3. לחץ Enter
4. תראה תשובה + אם יש הצעות למידה, כרטיסים למטה

הזיכרון החכם, הלמידה, והעדכון העצמי עובדים גם בלי קול!

---

## לוגים לבדיקה:

- Backend לוג: חלון שחור שפתח `run.bat`
- חפש שורות כמו:
  - `[TTS] edge-tts success` = מדברת
  - `[WakeWord] listening for 'אדיאל ג'וניור'` = מקשיבה
  - `[STT] Recorded Xs` = הקליטה
  - `[Brain] Intent: ...` = הבינה

אם אתה רואה `sounddevice not available` - זה אומר שאין מיקרופון או שהספריה לא הותקנה, אבל עדיין אפשר לכתוב.

---

## פתרון קסם (אם כלום לא עובד):

```bat
# מחק venv ותתחיל מחדש עם auto-fix
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
python scripts\auto_installer.py
python scripts\test_audio.py
scripts\run_with_autofix.bat
```
