# אדיאל ג'וניור - Adiel Junior 🤖

> **עוזרת AI אישית למחשב בסגנון Jarvis / FRIDAY מאירון מן - בעברית מלאה, עם HUD עתידני שקוף, ראיית מסך חיה, ומוח פרטי שנבנה מאפס.**

![Version](https://img.shields.io/badge/version-2.0.0-00d2ff)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Electron](https://img.shields.io/badge/electron-43-00ffaa)
![FastAPI](https://img.shields.io/badge/fastapi-0.141-009688)
![Hebrew](https://img.shields.io/badge/language-Hebrew%20%2B%20English-ff8a00)

---

## 🧠 v2.1 - AdielMind: מודל שפה ביתי (אוגוסט 2026)

שדרוג המודל המרכזי — **בלי API keys, בלי ענן, בלי שירותים חיצוניים. הכול נבנה מאפס:**

- 🧠 **AdielMind LM** (`core/adiel_lm.py`) — מודל שפה גנרטיבי ביתי: n-grams עם interpolation, מותנה-כוונה, דגימת temperature + nucleus. **מתאמן לבד מכל שיחה** ונשמר לדיסק — אדיאל נהיית חכמה יותר ככל שמדברים איתה
- 🎯 **מסווג כוונות נלמד** (`core/intent_nb.py`) — Naive Bayes על char n-grams (pure Python) שמשלים את כללי האצבע: עמיד לשגיאות כתיב, סלנג וניסוחים חדשים
- 🔍 **זיכרון BM25 + recency** — שליפת זכרונות בשיטת הדירוג הסטנדרטית של מנועי חיפוש (שדרוג מ-TF-IDF)
- ✍️ **מלחין תשובות** — ייצור מבוסס-הקשר + grounding מהזיכרון + **בקרת חזרתיות** (לא חוזרת על אותם משפטים) + שאלות המשך
- 🏷️ **תג מודל ב-HUD** — מתחת לכל תשובה רואים איך היא נוצרה (AdielMind / JARVIS / כלל מקומי)
- 🏋️ **אימון מ-UI** — כפתור "אמן" מריץ אימון מלא (`POST /model/train`) על כל הידע והשיחות
- 🔧 **endpoints חדשים**: `GET /model/status`, `POST /model/train`, `GET /model/generate?text=...`

---

## 🚀 v2.0 - שדרוג מלא (אוגוסט 2026)

סיבוב שדרוג מלא של כל המערכת:

- 📦 **כל הספריות עודכנו** - Electron 43, FastAPI 0.141, Pydantic 2.13, edge-tts 7, NumPy 2, ועוד עשרות
- 🐛 **תוקנו באגים קריטיים** - שגיאת Syntax ששברה את סקריפט ה-HUD, אנדפוינט `/frames` שהיה חסר, base64 קטוע ב-`/screen`, וקריסות JS בגלל אלמנטים חסרים ב-DOM
- ⚡ **FastAPI מודרני** - מעבר מ-`on_event` (deprecated) ל-`lifespan`
- 🖼️ **`/frames` חי** - 24 פריימים עם נתונים אמיתיים ישר ל-HUD (+ נתיב `/frames/{id}`)
- 🎨 **HUD v2** - טאב "הצעות" חדש לאישורי למידה, אווטאר זוהר ב-Reactor (מקשיבה/מדברת), אינדיקטור "חושבת...", התחברות אוטומטית מחדש ל-WebSocket, ועיצוב מלוטש
- 🔊 **הודעות שגיאה ידידותיות** - כשחסרה ספרייה אופציונלית המערכת מסבירה וממשיכה לעבוד
- 🧪 **נבדק מקצה לקצה** - Backend + WebSocket + צ'אט + מילון (814 מילים) + פריימים

---

## ✨ מה זה?

**אדיאל ג'וניור** היא לא עוד צ'אטבוט. זו אפליקציית דסקטופ אמיתית שיושבת על מסך המחשב שלך:

- 🎤 **מבינה עברית שוטפת** - סלנג, פקודות טבעיות, לא רובוטית
- 👂 **מילת הפעלה**: אומרים **"אדיאל ג'וניור"** והיא מתעוררת - לא מקשיבה סתם ברקע
- 👁️ **רואה את המסך שלך LIVE** - יכולה לנתח קוד, שגיאות, דפדפן, הכל
- 🎨 **HUD עתידני** - חלון שקוף מרחף בסגנון Iron Man, עם אנימציית Reactor פועם
- 📍 **דינמית**: `שים בצד`, `חזור לאמצע`, `הסתר` - פקודות קוליות שמזיזות אותה
- 🧠 **מוח פרטי** - AI שנבנה מאפס, לא רק wrapper ל-ChatGPT. זוכר אותך, לומד אותך
- 🔊 **מדברת עברית טבעית** - Edge-TTS + קול נשי צעיר
- 📦 **EXE אחד** - מתקין אחד, לוחצים, עובדת

---

## 🏗️ ארכיטקטורה

```
┌─────────────────────────────────────────────────┐
│  ELECTRON HUD - Transparent Frameless Window    │
│  HTML5 + Canvas Reactor + Tailwind              │
│  Modes: CENTER (520x680) | SIDE (400x720)       │
│         ORB (140x140) - כדור פועם               │
└────────────────────┬────────────────────────────┘
                     │ WebSocket ws://localhost:8765
┌────────────────────▼────────────────────────────┐
│  PYTHON BACKEND - FastAPI + WebSockets          │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐     │
│  │ AUDIO    │ │ VISION   │ │ BRAIN        │     │
│  │ WakeWord │ │ mss      │ │ AdielBrain   │     │
│  │ Whisper  │ │ OCR      │ │ Memory TF-IDF│     │
│  │ EdgeTTS  │ │ GPT4V?   │ │ Intents      │     │
│  └──────────┘ └──────────┘ └──────────────┘     │
└─────────────────────────────────────────────────┘
```

**פירוט מלא**: קרא את `ARCHITECTURE.md`

---

## 🚀 התקנה מהירה - Windows

### דרישות:
- Windows 10/11 (גם עובד ב-Linux/Mac בפיתוח)
- Python 3.10+ - https://python.org (סמן ✓ Add to PATH)
- Node.js 18+ - https://nodejs.org (ל-HUD)
- מיקרופון 🎤
- (אופציונלי) Tesseract OCR - https://github.com/UB-Mannheim/tesseract/wiki (לזיהוי טקסט במסך)

### התקנה אוטומטית:

```bat
# 1. שכפל
git clone <repo>
cd J

# 2. התקן הכל
scripts\setup_windows.bat

# 3. הורד מודלים (whisper + vosk hebrew)
python scripts\install_models.py

# 4. הרץ!
scripts\run.bat
```

זה יפתח 2 חלונות:
- Backend (FastAPI) ב-`http://localhost:8765/status`
- HUD ה-Electron השקוף

### בדיקה שזה עובד:

1. תראה HUD עם reactor כחול פועם
2. אמור בקול: **"אדיאל ג'וניור"** (או לחץ 🎤)
3. היא תענה "כן בוס? אני כאן"
4. דבר פקודה: "מה את רואה במסך?" או "שים בצד"
5. היא תענה בקול + טקסט

---

## 💬 פקודות לדוגמה (עברית)

**HUD:**
- "אדיאל ג'וניור, שים בצד"
- "חזור לאמצע"
- "הסתר / תיעלמי"
- "תופיעי"

**מסך:**
- "מה את רואה במסך?"
- "יש לי שגיאה, מה הבעיה?"
- "תעזרי לי עם הקוד הזה"

**מערכת:**
- "תפתח את כרום / VS Code / Spotify"
- "תחפש בגוגל איך עושים..."
- "תגביר ווליום / תנמיך / השתק"

**שיחה:**
- "תזכור שקוראים לי דני"
- "מה השעה?"
- "תודה, את אלופה"

---

## 📦 בניית EXE (הפצה)

רוצה קובץ התקנה אחד ללקוחות?

```bat
# בניית Backend + Frontend
build\build.bat

# או ידנית:
python build\build.py --all
```

פלט:
- `backend/dist/adiel_backend.exe` (~150MB עם מודלים)
- `frontend/dist/Adiel Junior Setup.exe` - המתקין המלא
- `frontend/dist/Adiel-Junior-Portable.exe` - גרסה ניידת ללא התקנה

**המתקין המלא כולל backend בפנים - exe אחד שמריץ הכל!**

### PyInstaller ידני (Backend בלבד):

```bat
cd backend
pyinstaller --onefile --noconsole --name adiel_backend main.py
# הפלט ב-backend/dist/
```

---

## 🐍 הרצה ידנית למפתחים

### Backend בלבד:

```bash
cd backend
pip install -r requirements.txt
python main.py
# ירוץ ב- http://localhost:8765
# Docs ב- http://localhost:8765/docs
```

**בדיקות API:**

```bash
curl http://localhost:8765/status
curl -X POST http://localhost:8765/chat -H "Content-Type: application/json" -d "{\"text\": \"היי אדיאל, מה שלומך?\", \"with_screen\": false}"
curl -X POST http://localhost:8765/wake
```

### Frontend בלבד (Electron):

```bash
cd frontend
npm install
npm start        # dev
npm run build    # build exe
```

### Launcher מאוחד (הכי פשוט):

```bash
python launcher.py           # מנסה Electron, נופל ל-PyQt fallback
python launcher.py --dev     # מכריח python backend (לא exe)
python launcher.py --backend-only
```

### Fallback GUI ללא Electron:

```bash
pip install PyQt6
python backend/gui_fallback.py
```

---

## 🧠 המוח הפרטי - Built From Scratch

**זה לא רק `openai.ChatCompletion`!**

המוח של אדיאל מורכב מ-4 רכיבים שנבנו מאפס:

1. **personality.py** - פרסונה קבועה: שנונה, צינית קלות, קוראת "בוס"
2. **memory.py** - זיכרון עם:
   - טוקנייזר עברי
   - TF-IDF + Cosine similarity ממומש מאפס (ללא sklearn!)
   - זיכרון קצר (12 הודעות) וארוך (JSON)
3. **intents.py** - מסווג כוונות:
   - מילות מפתח בעברית
   - Centroid model מאפס עם Counter + cosine
   - Entities (איזו אפליקציה לפתוח, מה לחפש)
4. **brain.py** - אורקסטרטור:
   - קודם מנסה כלים מקומיים (ללא LLM בכלל!)
   - אם צריך - Ollama מקומי (llama3.1) - פרטי לגמרי
   - אם מותר - OpenAI/Claude כ-power-up
   - Fallback: תבניות NLG מקומיות

**כלומר: גם בלי אינטרנט, בלי API keys - היא עובדת!**

---

## 🔧 קונפיגורציה

צור קובץ `.env` ב-`backend/` (אופציונלי):

```ini
# LLM - אופציונלי, עובד גם בלי
ALLOW_CLOUD_LLM=false
ALLOW_CLOUD_VISION=false
OPENAI_API_KEY=sk-...

# Ollama מקומי (פרטי)
OLLAMA_MODEL=llama3.1:8b

# Whisper
WHISPER_MODEL=small  # tiny / small / medium / large-v3

# Port
PORT=8765

# TTS
TTS_VOICE=avigail  # avigail / hila / asaf
```

---

## 🗂️ מבנה פרויקט

```
J/
├── backend/
│   ├── main.py                 # FastAPI server + WebSocket
│   ├── requirements.txt
│   ├── core/
│   │   ├── brain.py           # המוח הראשי - פרטי מאפס
│   │   ├── memory.py          # זיכרון TF-IDF מאפס
│   │   ├── intents.py         # מסווג כוונות עברי מאפס
│   │   └── personality.py     # פרסונה + תבניות
│   ├── audio/
│   │   ├── wake_word.py       # זיהוי "אדיאל ג'וניור"
│   │   ├── stt.py             # Whisper עברית
│   │   └── tts.py             # Edge-TTS עברית
│   ├── vision/
│   │   └── screen.py          # צילום מסך + OCR
│   ├── tools/
│   │   ├── system_tools.py    # פתיחת אפליקציות, ווליום
│   │   └── hud_tools.py
│   └── gui_fallback.py        # HUD חלופי ב-PyQt
├── frontend/
│   ├── main.js                # Electron main - transparent window
│   ├── preload.js
│   ├── package.json
│   └── src/
│       ├── index.html         # HUD UI
│       ├── styles.css         # Tailwind + Iron Man styles
│       ├── animations.js      # Reactor Canvas animation
│       └── hud.js             # לוגיקת HUD + WS client
├── build/
│   ├── build.py               # בונה EXE
│   └── build.bat
├── scripts/
│   ├── setup_windows.bat
│   ├── run.bat
│   └── install_models.py
├── launcher.py                # משגר מאוחד Backend+Frontend
├── ARCHITECTURE.md            # תיעוד ארכיטקטורה
└── README.md                  # את כאן
```

---

## 🎯 Roadmap - מה הלאה?

- [ ] ראיית מסך LIVE streaming (כל 500ms עם diff)
- [ ] שליטה בעכבר/מקלדת via קול: "תלחצי שם, תכתבי..."
- [ ] חיבור ל-Home Assistant / IoT
- [ ] פלאגינים קהילתיים (לוח שנה, מיילים)
- [ ] Voice cloning - שתדבר בקול שלך
- [ ] אפליקציית מובייל שתתחבר למחשב
- [ ] תמיכה ב-GPU acceleration ל-Whisper

---

## 🛠️ פתרון בעיות נפוצות

**Backend לא עולה:**
```bat
# בדוק פורט תפוס
netstat -ano | findstr :8765
# הרוג תהליך אם צריך
taskkill /PID <pid> /F
```

**לא מזהה "אדיאל ג'וניור":**
- בדוק מיקרופון עובד ב-Windows Settings
- נסה כפתור 🎤 ידני ב-HUD
- התקן Vosk Hebrew model ל-wake מדויק
- אמור ברור, לא מהר מדי, "אַדִיאֵל ג'וּנְיוֹר"

**אין קול:**
- Edge-TTS צריך אינטרנט בפעם הראשונה
- בדוק רמקולים / ווליום
- נסה `python -m edge_tts --voice he-IL-AvigailNeural --text "שלום"`

**מסך שחור ב-capture:**
- Windows: צריך להריץ כרגיל, לא כ-admin בהכרח
- Linux ללא DISPLAY: זה mock, לא יעבוד בשרת

**Electron לא נפתח:**
- `cd frontend && npm install`
- נסה fallback: `python backend/gui_fallback.py`

---

## 📜 רישיון

MIT - חופשי לשימוש פרטי. המוח של אדיאל הוא שלך, פרטי, לא שולח דאטה לאף אחד כברירת מחדל.

---

## 👨‍💻 נוצר ע"י

ארכיטקט תוכנה + מהנדס AI - עבורך, בוס.

**אדיאל ג'וניור מחכה לך במסך. תגיד "אדיאל ג'וניור" ותראה איזה קסם. ✨**

---

> 💡 **טיפ**: לחץ `Ctrl+Shift+A` בכל מקום כדי להעיר אותה מהר!
