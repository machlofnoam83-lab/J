# אדיאל ג'וניור - ארכיטקטורת מערכת

## סקירה
אדיאל ג'וניור היא עוזרת אישית שולחנית (Desktop AI Assistant) עם HUD שקוף בסגנון Iron Man, ראיית מסך חיה, ודיבור מלא בעברית.

## ארכיטקטורה היברידית

```
┌─────────────────────────────────────────────────────┐
│  ELECTRON FRONTEND (HUD) - Transparent Frameless    │
│  HTML/CSS/JS + Tailwind + Canvas Animations         │
│  - מצבי תצוגה: center / side-docked / orb-hidden    │
│  - WebSocket Client -> Backend                      │
│  - Audio Playback (TTS)                             │
│  - Screen Preview (optional)                        │
└──────────────────┬──────────────────────────────────┘
                   │ WebSocket (ws://localhost:8765)
                   ▼
┌─────────────────────────────────────────────────────┐
│  PYTHON BACKEND (FastAPI + WebSockets)              │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ AUDIO ENGINE │  │ VISION ENGINE│  │ AI CORE   │  │
│  │ - Wake Word  │  │ - mss capture│  │ AdielBrain│  │
│  │ - STT (Whisp)│  │ - OCR/PIL    │  │ Memory    │  │
│  │ - TTS Edge   │  │ - Vision LLM │  │ Intents   │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         └─────────────────┼────────────────┘        │
│                   Orchestrator                      │
└─────────────────────────────────────────────────────┘
```

## רכיבים מפורטים

### 1. Audio Engine (`backend/audio/`)

**Wake Word Detection (`wake_word.py`):**
- האזנה רציפה 24/7 עם VAD (Voice Activity Detection)
- מודל Vosk small Hebrew + faster-whisper לזיהוי "אדיאל ג'וניור"
- Fuzzy matching: "אדיאל גוניור", "אדיאל ג'וניור", "עדיאל", "ג'וניור"
- לאחר זיהוי -> מעבר למצב LISTENING

**STT Hebrew (`stt.py`):**
- Primary: faster-whisper large-v3 עם language="he"
- Fallback: Vosk he model
- Preprocessing: Noise reduction, 16kHz mono
- Post-processing: ניקוי סלנג, תיקון שגיאות נפוצות בעברית

**TTS Hebrew (`tts.py`):**
- Primary: edge-tts - he-IL-AvigailNeural (קול נשי צעיר, טבעי)
- Alternative: he-IL-HilaNeural / he-IL-AsafNeural
- קצב דיבור דינמי, הוספת SSML לעצירות טבעיות
- שמירה ל-wav + שליחה ל-frontend

### 2. Vision Engine (`backend/vision/`)

**Screen Capture:**
- mss לגישה מהירה (50ms per frame)
- Downscale ל-1024px רוחב לחסכון
- Base64 encoding ל-Vision API
- Live stream mode: צילום כל 2 שניות במצב שאלה פעילה

**Screen Understanding (2 שכבות):**
- שכבה מקומית: OCR (pytesseract he+en) + זיהוי חלונות פעילים via pygetwindow
- שכבת ענן אופציונלית: GPT-4o Vision / Claude Vision לניתוח עמוק
- התיאור מוזרם ל-AI Core כ-context

### 3. AI Core - המוח הפרטי (`backend/core/`)

**המוח נבנה מאפס - לא רק wrapper!**

**personality.py:** פרומפט אישיות קבוע
- דמות: עוזרת AI שנונה, צינית קלות בסגנון FRIDAY, קוראת למשתמש "בוס"
- שפה: עברית יומיומית + סלנג ישראלי קל (יאללה, סגור, על זה, אין בעיה בוס)
- טון: עוזרת, לא משרתת - יוזמת ומתריעה

**memory.py:** זיכרון מותאם
- Short-term: buffer שיחה אחרונה (10 הודעות)
- Long-term: JSON file עם עובדות על המשתמש (שם, העדפות, פרויקטים)
- Semantic memory: TF-IDF homemade + cosine similarity לחיפוש בזיכרון ללא תלות ב-API חיצוני
- Embedding engine נבנה מאפס: tokenization עברי + weighting

**intents.py:** מסווג כוונות היברידי
- Rule-based: מילות מפתח בעברית ל-HUD, מערכת, מסך
- ML קל: Logistic classifier שאומן על דוגמאות עבריות (נבנה מאפס עם numpy)
- קטגוריות:
  - hud_control: שים בצד, חזור לאמצע, הסתר, תתעוררי, תירדמי
  - screen_analysis: מה אתה רואה, מה פתוח, תעזור לי עם הקוד הזה, מה השגיאה
  - system_action: תפתח X, תגביר ווליום, תכבה, חפש בגוגל
  - general_chat: כל השאר

**brain.py:** האורקסטרטור הראשי
- מקבל: text + screen_context + memory + intent
- מחליט: האם להפעיל כלי, האם לקרוא ל-LLM, או תשובה מקומית
- Local NLG: מאגר תבניות תשובה בעברית שנבחרות לפי intent + randomness טבעית
- LLM integration (אופציונלי): Ollama local (llama3.1:8b, gemma2) או OpenAI/Claude API כ-power-up
- מחזיר: response_text + action + hud_command

### 4. HUD Frontend (`frontend/`)

**Electron Main (`main.js`):**
- BrowserWindow: transparent:true, frame:false, alwaysOnTop:true, hasShadow:false
- Vibrancy: acrylic/ultra-dark (Windows 11 Mica)
- Global shortcuts: לזימון מהיר
- Spawns Python backend כ-child process
- IPC: שליטה במיקום חלון

**Renderer - Iron Man HUD:**
- Canvas reactor core עם אנימציית נשימה (pulsing)
- Particle system קטן
- מצבים:
  - CENTER (500x600): מרכז מסך, גדול, עם היסטוריית צ'אט
  - SIDE (380x720): צד ימין, דק, docked, 60% גובה
  - ORB (120x120): כדור קטן פועם בפינה, רק listening dot
- פקודות קוליות שמשנות מצב + כפתורי UI + drag
- Visualizer: Audio bars כשמדברת
- Screen preview thumbnail

## זרימת נתונים מלאה

1. **Idle**: Backend מאזין ברצף ל-wake word עם CPU נמוך (VAD + Vosk)
2. **Wake**: זיהוי "אדיאל ג'וניור" -> HUD עובר ל-listening, צליל התעוררות, "כן בוס?"
3. **STT**: מקליט 5-10 שניות / עד שקט, ממיר לעברית
4. **Vision**: מצלם מסך ברגע השאלה, מריץ OCR מקומי
5. **Brain**: מסווג כוונה, שולף זיכרון, בונה context, מייצר תשובה
6. **Action**: אם יש פעולת מערכת / HUD -> שולח פקודה ל-frontend
7. **TTS**: ממיר תשובה לקול, שולח ל-frontend לניגון + מציג טקסט
8. **Return to Idle**: חוזר להאזנת wake word

## אבטחה ופרטיות

- כל העיבוד הרגיש קורה לוקלית
- צילום מסך לא נשלח לענן אלא אם משתמש אישר במפורש (env: ALLOW_CLOUD_VISION)
- זיכרון מוצפן לוקלית (אופציה)
- אין טלמטריה

## פריסה כ-Executable

**Windows .exe יחיד:**
- PyInstaller: backend -> adiel_backend.exe (onefile)
- Electron-builder: frontend + backend exe -> Adiel Junior Setup.exe / Portable exe
- המוצר הסופי: מתקין אחד שמריץ הכל.

**Dev mode:**
- `run.bat` מריץ backend + frontend בנפרד
