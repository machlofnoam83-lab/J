# פרויקט J.A.R.V.I.S — תוכנית אב (Master Plan)

> **מטרה:** עוזר AI אישי, ריבוני לחלוטין. אפס API keys. אפס ענן. אפס משקולות שהורדו מבחוץ.
> **הכול** — הטוקנייזר, המודל העצבי, מנוע הדיבור, זיהוי הקול, הזיכרון, סוכני המשנה, ה־UI והשליטה במחשב — נבנה ומאומן כאן, מאפס, על ידינו.

---

## 0. מציאות הסביבה (נבדק בפועל, לא הנחות)

| משאב | מצב | השפעה על התכנון |
|---|---|---|
| HuggingFace / גוגל / ויקיפדיה | ❌ חסום | אין Llama, אין Whisper, אין Piper. **בונים הכול מאפס** |
| `download.pytorch.org` | ❌ חסום | — |
| PyPI (`pypi.org`) | ✅ פתוח | torch 2.14, numpy, scipy, numba, onnxruntime, psutil, soundfile — הותקנו |
| npm registry | ✅ פתוח | Electron כתלות קוד ✅ |
| בינארי Electron | ❌ חסום בסנדבוקס | הקוד מלא ומוכן; אצלך ב־Windows ירד אוטומטית. בסנדבוקס — LIVE PREVIEW של אותו renderer בדיוק + harness לאימות ה־main process |
| חומרת הסנדבוקס | 2 vCPU, 3.8GB RAM, **אין GPU**, אין X server | המודל העצבי חייב להיות קומפקטי; האימון כאן = הוכחת הצינור. מסלול אימון מלא על GPU אצלך מצורף |
| יעד ריצה אמיתי | **Windows** + Electron | מודול שליטה במחשב נכתב ל־Windows (PowerShell / Win32), עם שכבת abstraction ל־Linux/macOS |

**עיקרון מנחה:** מכיוון שאין משקולות חיצוניות ואין GPU — "סופר חכם" מושג **לא** על ידי מודל ענק, אלא על ידי **ארכיטקטורת מוח היברידית**: מודל עצבי קטן (שנבנה ומאומן מאפס) + קרנל סימבולי דטרמיניסטי + לולאת חשיבה־פעולה־אימות + זיכרון קבוע. זו הדרך היחידה הכנה לקבל סוכן שבאמת חושב, מתכנת ופועל — במקום צעצוע שמפברק.

---

## 1. ארכיטקטורת העל

```
        ┌────────────────────────────────────────────────────────────┐
        │  SHELL — Electron HUD (מסך מלא, שקוף, RTL, קיצורים גלובליים)│
        └───────────────▲────────────────────────────┬───────────────┘
                        │ WebSocket / IPC            │
        ┌───────────────┴────────────────────────────▼───────────────┐
        │              ORCHESTRATOR  — "JARVIS" (המנצח)               │
        │   תכנון משימות • ניתוב לסוכנים • ניהול הרשאות • שיחה        │
        └───▲──────────▲──────────▲──────────▲──────────▲────────────┘
            │          │          │          │          │
     ┌──────┴───┐ ┌────┴─────┐ ┌──┴──────┐ ┌─┴───────┐ ┌┴──────────┐
     │ NEURAL   │ │ CODER    │ │ VOICE   │ │ MEMORY  │ │ SYSTEM    │
     │ CORE     │ │ AGENT    │ │ ENGINE  │ │ PALACE  │ │ OPERATIONS│
     │ (LLM מאפס)│ │(מתכנת+   │ │(TTS/STT │ │(זיכרון  │ │(שליטה     │
     │          │ │ מאמת ע"י │ │ שלנו)   │ │ קבוע)   │ │ במחשב)    │
     │          │ │ הרצה)    │ │         │ │         │ │           │
     └──────────┘ └──────────┘ └─────────┘ └─────────┘ └───────────┘
```

### Multi-Agent (לפי הבקשה שלך: "שיהיה לו סוכן לצידו שיודע לתכנת")

| סוכן | תפקיד | מנגנון |
|---|---|---|
| **JARVIS** (מנצח) | שיחה, הבנת כוונה, תכנון, קבלת החלטות, ניהול כל השאר | Neural Core + Reasoning Loop |
| **HEPHAESTUS** (סוכן המתכנת) | כתיבת קוד, רפקטורינג, איתור באגים, הסבר קוד, בניית פרויקטים | Code-gen + **sandbox execution + self-repair loop** (מרץ, קורא stderr, מתקן, חוזר) |
| **MNEMOSYNE** (הזיכרון) | שינון עובדות, העדפות משתמש, הסטוריה, שליפה סמנטית | SQLite + vector store + embeddings משלנו |
| **ARGUS** (העיניים) | מצב המחשב, קבצים, תהליכים, צילום מסך | psutil + Win32 + OCR |
| **HERMES** (הידיים) | ביצוע פעולות במחשב, אוטומציה | מודולי שליטה + **Permission Firewall** |
| **VOICE** (הפה והאוזניים) | דיבור והאזנה | מנוע TTS/STT עצמי |

---

## 2. NEURAL CORE — המודל שנבנה מאפס

### 2.1 טוקנייזר משלנו (`brain/tokenizer.py`)
- **BPE מאפס** (אין `sentencepiece`/`tiktoken` — מימוש עצמי מלא של merge-לוג).
- תמיכה דו-לשונית **עברית + אנגלית**: ניקוד (niqqud) אופציונלי, normalizer עברי (סופיות ךםןץת → כמנפצ, הסרת מקפים/גרשיים), טיפול ב־RTL.
- vocab יעד: 16,384 טוקנים.
- פלט: `tokenizer.json` + `vocab.json` — **שלנו, נבנה כאן**.

### 2.2 קורפוס אימונים סינתטי (`tools/corpus_forge/`)
מכיוון שאי אפשר להוריד טקסט מהרשת — **מייצרים את הקורפוס בעצמנו**:
1. `knowledge_base.yaml` — עובדות, הגדרות, מדע, תאריכים, מתמטיקה (נכתב ידנית).
2. `dialogue_forge.py` — מחולל דיאלוגים עבריים (שאלה/תשובה, נימוס, הוראות, סמול-טוק) ע"י דקדוק טמפלטים + קומבינטוריקה → מיליוני וריאציות.
3. `code_forge.py` — קורפוס תכנות סינתטי: פונקציות, הסברים בעברית על קוד, צמדי (תיאור → קוד), שגיאות נפוצות → תיקון.
4. `reasoning_forge.py` — שרשראות חשיבה (Chain-of-Thought) בעברית: תוכנית → שלבים → תוצאה.
5. `persona_forge.py` — אישיות JARVIS: טון, נימוס, הומור יבש, פניות למשתמש.
6. **Self-distillation loop**: המודל מאומן, מייצר טקסט, המאמת הסימבולי מסנן רק את הנכון, והתוצאה חוזרת לקורפוס (איטרציות).

### 2.3 המודל (`brain/model.py`)
Transformer decoder-only, מימוש עצמי מלא ב־PyTorch:
- RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, gradient checkpointing.
- **שלושה גדלים** (אותה ארכיטקטורה, פרמטרים ב־config):
  - `JARVIS-NANO` — d=256, L=6, H=4 (~12M פרמטרים) → ריצה חלקה על CPU חלש, **אימון בסנדבוקס**.
  - `JARVIS-MICRO` — d=384, L=10, H=6 (~45M) → היעד ל־CPU שלך.
  - `JARVIS-CORE` — d=512, L=16, H=8 (~120M) → אם יש לך GPU, `tools/train.py --size core`.
- **Dual-path export**: PyTorch (אימון/פיתוח) → **ONNX** (ריצה מהירה ב־onnxruntime אצלך, שהותקן כבר) + מסלול **NumPy טהור** ל-inference (אפס תלויות, עובד גם בלי torch).

### 2.4 אימון (`tools/train.py`)
- AdamW + cosine schedule + warmup, mixed-precision אם יש GPU.
- checkpointing, resumption, live loss curves (נשלחים ל־HUD בזמן אמת!).
- eval set נפרד + **perplexity + מבחני יכולת אמיתיים** (לא רק loss).
- כאן בסנדבוקס: אימון הוכחת-צינור (millions of tokens). אצלך: `python tools/train.py --size core --epochs 3` על GPU.

### 2.5 השכבה הסימבולית — המקום שבו נמצאת החוכמה האמיתית
המודל הקטן לבדו יפברק. לכן מעליו **קרנל סימבולי דטרמיניסטי**:
- **Intent Router** — סיווג כוונה (היברידי: חוקים + המודל), מפה לכלי/סוכן.
- **Skill Registry** — סכמת JSON לכל כלי; המודל פולט tool-call JSON, הקרנל מאמת (jsonschema משלנו) ומריץ.
- **Math Engine** — parser + evaluator מתמטי מדויק משלנו (לא נותן למודל לחשב!).
- **Fact Store + Retriever** — שליפה וקטורית מתוך הידע המקומי; המודל עונה **על בסיס מה שנשלף**, לא מהזיכרון בלבד → ממזער הזיות.
- **Verifier / Hallucination Guard** — בודק טענות מול Fact Store, בודק קוד ע"י הרצה, בודק חישוב ע"י חישוב מחדש.
- **Reasoning Loop** — `חשוב → תכנן → פעל → צפה → אמת → ענה` (ReAct-style, מימוש עצמי), עם תקציב צעדים, reflection וחשיבה מחדש אם האימות נכשל.

### 2.6 Memory Palace (`brain/memory.py`)
- **Episodic** — יומן שיחות ואירועים (SQLite).
- **Semantic** — עובדות על העולם ועל המשתמש (העדפות, שמות, פרויקטים) עם confidence + timestamp.
- **Procedural** — "מיומנויות שנלמדו": אם JARVIS פתר משהו, הוא שומר את התבנית ומשתמש בה שוב.
- **Vector recall** — embeddings משלנו (מגדל המודל) + אינדקס kNN ב־numpy.
- **שכחה חכמה** — decay לפי זמן ורלוונטיות, כמו זיכרון אנושי.

---

## 3. VOICE ENGINE — דיבור בלי API, בלי ענן

### 3.1 TTS (`voice/tts/`) — שלושה מנועים, cascade אוטומטי
1. **מנוע קונקטיבי (הראשי)** — בנק קול **שלי** שמורכב מהקלטות אמיתיות שיופקו כאן בסביבה: פונמות, דיפתונגים, הברות, מילים נפוצות, ספרות. אחסון ב־`voicebank/` (מחוץ ל־git, עם סקריפט שחזור). Pipeline: טקסט עברי → **Grapheme-to-Phoneme עברי משלנו** (חוקי הגייה, שווא, דגש, בג"ד כפ"ת, מילים לועזיות) → בחירת יחידות → prosody (אינטונציה, קצב, הדגשה, הטעם במשפט) → crossfade + pitch-slicing ב־scipy/numpy → PCM.
2. **מנוע פורמנטי (גיבוי)** — סינתזה מודלית טהורה ב־numpy: מקור glottal pulse + מסנני formant + noise לעיצורים חוככים. עובד תמיד, גם אם בנק הקול חסר. קול "רובוטי־אלגנטי" בסגנון JARVIS.
3. **Voice Personality** — קצב דיבור מותאם מצב רוח/דחיפות, micro-pauses לפני תשובות כבדות, צליל "הפעלה" ממותג.

### 3.2 STT (`voice/stt/`) — כאן נהיה כנים: זו החלק הקשה ביותר
מדרג ריאלי (כל שלב עובד ועצמאי):
1. **שלב 0 (מיידי, עובד 100%):** קלט טקסט + Web Speech API בדפדפן/Electron (מקומי מבחינת המשתמש, בלי מפתח שלנו) + **Push-to-talk**.
2. **שלב 1 (נבנה כאן):** **Wake-word engine משלנו** — MFCC extractor (numpy/scipy, מימוש עצמי) + מסווג קל שמזהה "ג'רוויס". רץ תמיד ברקע, CPU אפסי.
3. **שלב 2 (נבנה כאן):** זיהוי פקודות מוגבל-אוצר־מילים (command grammar) — template matching + DTW על MFCC. מצוין לפקודות: "פתח את X", "כבה", "מה השעה", "כתוב קוד ל־Y".
4. **שלב 3 (מתקדם, R&D):** acoustic model קטן מאומן על **דיבור סינתטי שאנחנו מייצרים בעצמנו** (ה־TTS שלנו מייצר אוטומטית dataset מתויג! self-supervised). CTC/attention decoder. יסומן כניסיוני עם דיווח כנות על דיוק.

### 3.3 Audio I/O
- Windows: WASAPI דרך `sounddevice`/`pyaudio` (יוגדר אצלך), fallback ל־`<audio>` ב־Electron renderer.
- **VAD** (voice activity detection) משלנו — אנרגיה + zero-crossing + spectral flux.
- **Barge-in**: אפשר לקטוע את JARVIS באמצע דיבור.

---

## 4. SYSTEM OPERATIONS — גישה מלאה למחשב (Windows)

כל מודול = Python אמיתי + חוזה JSON + בדיקות:

| מודול | יכולות | מימוש |
|---|---|---|
| `fs_ops` | קריאה/כתיבה/חיפוש/עץ/ניטור שינויים | `pathlib` + watchdog-like polling |
| `proc_ops` | רשימת תהליכים, CPU/RAM, הריגה, פתיחה | `psutil` |
| `app_launcher` | הפעלת תוכניות, קיצורי דרך, UWP apps | Start menu scan + `subprocess` |
| `clipboard` | קריאה/כתיבה, היסטוריה | `pyperclip`/Win32 |
| `screen` | צילום מסך, זיהוי חלונות, **OCR משלנו** | mss/PIL + OCR pipeline |
| `input_ctrl` | הקלדה, עכבר, קיצורים | pyautogui/Win32 SendInput |
| `window_mgr` | פריסת חלונות, minimize/maximize, החלפה | Win32 API |
| `sysinfo` | חומרה, דיסק, רשת, סוללה, טמפרטורות | psutil + WMI |
| `media_ctrl` | play/pause/next, ווליום | media keys + PowerShell |
| `notify` | התראות Windows toast | PowerShell/WinRT |
| `shell_exec` | PowerShell/cmd עם **סינון ופיקוח** | subprocess + parser |
| `scheduler` | משימות מתוזמנות, תזכורות, cron | APScheduler-like משלנו |
| `web_local` | חיפוש/גלישה **רק אם המשתמש יגדיר proxy** | אופציונלי, כבוי כברירת מחדל |

### 🔒 Permission Firewall (חובה — לא מוסיפים "אחר כך")
- **3 רמות:** `SAFE` (קריאה בלבד) / `WRITE` (שינוי קבצים, פתיחת אפליקציות) / `CRITICAL` (מחיקה, registry, kill process, shell).
- כל פעולה עוברת דרך **policy engine** עם allowlist/denylist + אישור משתמש ב־HUD לרמת CRITICAL.
- **Dry-run mode** — JARVIS מציג מה הוא *היה* עושה לפני ביצוע.
- **Audit log** בלתי ניתן למחיקה — כל פעולה מתועדת.
- **Kill switch** — מקש אחד (Esc כפול / F12) עוצר הכול מיידית.

---

## 5. UI — ה־HUD ההולוגרמי (יותר ממה שיש לטוני סטארק)

**מחסנית:** Electron (fullscreen, frameless, transparent, click-through אופציונלי) + HTML/CSS/JS טהור (אין bundler — מהיר, שקוף, בלי תלויות) + Canvas 2D/WebGL + Web Audio API.

### רכיבים
1. **Arc Reactor Core** — ליבת אנרגיה מרכזית: שכבות מסתובבות, פלזמה, reactive לפי עוצמת הקול של JARVIS ומצב החשיבה (idle / listening / thinking / speaking / executing / alert). כל מצב = צבע + התנהגות + צליל משלו.
2. **Holographic Rings** — טבעות 3D עם perspective, gyro parallax לפי עכבר, נתוני טלמטריה על הטבעות.
3. **Audio Reactive Particles** — מערכת חלקיקים (10k+) שמגיבה בסנכרון מלא ל־spectrum של הקול בזמן אמת (FFT ב־Web Audio).
4. **Neural Web** — ויזואליזציה חיה של "המוח": צמתים = סוכנים/כלים פעילים, קווים = זרימת מידע, הבזקים = token שנוצר **בזמן אמת**. אתה רואה את JARVIS חושב.
5. **Waveform + Spectrum** — סרגל קול כפול (מיקרופון + פלט) עם oscilloscope.
6. **Command Stream** — זרם פקודות/tool-calls חי עם סטטוס (queued → running → verified → done).
7. **Telemetry HUD** — CPU, RAM, GPU, דיסק, רשת, תהליכים חיים — גרפים בזמן אמת.
8. **Memory Timeline** — ציר זמן של מה ש־JARVIS זוכר, עם חיפוש.
9. **Code Forge Panel** — כש־HEPHAESTUS עובד: עורך עם syntax highlighting משלנו, diff view, פלט הרצה חי, status של self-repair loop.
10. **Conversation** — בועות שיחה RTL עברית, typing effect, streaming token-by-token.
11. **Boot Sequence** — אנימציית אתחול קולנועית (POST-style: בדיקת מערכות, טעינת מודלים, כיול קול) — 8 שניות של "וואו" בכל הפעלה.
12. **Themes** — `Mark I` (כחול-זהב קלאסי), `Mark VII` (אדום-זהב), `Night Ops` (ירוק טרמינל), `Crimson Alert` (אדום חירום).
13. **Global UX** — Command palette (Ctrl+K), קיצורי מקשים גלובליים, always-on-top, מצב "הצמד לפינה" (mini orb), מצב "מסך מלא קולנועי", transparency + blur, drag-to-move.

**עיקרון עיצוב:** כל מסך = **שכבות זכוכית (glassmorphism) + זוהר neon + עומק תלת-ממדי + אנימציה 60fps + משוב קולי לכל פעולה**. שום דבר לא סטטי.

---

## 6. מבנה הפרויקט

```
J/
├── docs/                     PLAN.md, ARCHITECTURE.md, SECURITY.md, USER_GUIDE.md, TRAINING.md
├── brain/                    tokenizer, model, corpus, trainer, reasoning, knowledge, memory
├── voice/                    tts (g2p, voicebank, prosody, formant, concat), stt (mfcc, wake, cmd, am), audio
├── agents/                   orchestrator + jarvis, hephaestus (coder), argus, hermes, mnemosyne
├── skills/                   registry + fs_ops, proc_ops, launcher, clipboard, screen, input, sysinfo, media, shell, scheduler
├── security/                 permissions.py, audit.py, killswitch
├── core/                     server.py (FastAPI-like עצמי / WebSocket), bus.py (event bus), config.py, ipc.py
├── ui/                       index.html, css/, js/ (hud, reactor, particles, neural_web, telemetry, code_forge, boot), assets/sfx/
├── desktop/                  main.js, preload.js, package.json, windows/ (installer scripts), native/ (Win32 helpers)
├── tools/                    train.py, forge_corpus.py, build_voicebank.py, export_onnx.py, bench.py, selftest.py
├── tests/                    unit + integration + golden-dialogue tests
└── jarvis.py                 נקודת כניסה אחת: `python jarvis.py` (מרים מוח + שרת + UI)
```

---

## 7. שלבי ביצוע (Build Phases)

> נבחר: **Brain-first**. כל שלב מסתיים במשהו **שרץ וניתן לבדיקה**.

- **P0 — תשתית ✅** — אימות סביבה, התקנת stack, תוכנית, שלד הפרויקט, config, event bus, self-test.
- **P1 — NEURAL CORE ✅** — טוקנייזר BPE עברי מאפס → corpus forge → מודל transformer → אימון הוכחת-צינור → inference engine (torch/ONNX/numpy) → **JARVIS אומר את המשפט הראשון שלו**.
- **P2 — SYMBOLIC KERNEL ✅** — intent router, skill registry, math engine, fact store + retriever, reasoning loop, verifier. **JARVIS באמת עונה נכון ולא מפברק.**
- **P3 — MEMORY ✅** — SQLite + vectors + שכחה חכמה + procedural skills. **JARVIS זוכר אותך.**
- **P4 — VOICE ✅** — G2P עברי → בנק קול → מנוע קונקטיבי → מנוע פורמנטי → prosody → **JARVIS מדבר בקול אמיתי**. ואז STT: wake-word → פקודות → AM סינתטי.
- **P5 — CODER AGENT (HEPHAESTUS) ✅** — code-gen → sandbox → self-repair loop → code explainer → פרויקט שלם. **JARVIS כותב ומריץ קוד בעצמו.**
- **P6 — SYSTEM OPS ✅** — כל מודולי ה־Windows + Permission Firewall + audit + kill switch. **JARVIS שולט במחשב.**
- **P7 — UI HUD ✅** — Reactor, rings, particles, neural web, telemetry, code forge, boot sequence, themes. **המסך העתידני.**
- **P8 — ELECTRON SHELL ✅ (קוד נשלח; אימות סופי על Windows)** — main.js, preload, חלון fullscreen שקוף, קיצורים גלובליים, tray, auto-start. **אפליקציה אמיתית.**
- **P9 — אינטגרציה קצה-אל-קצה + קשיחות ✅** — צינור מלא: דיבור → חשיבה → פעולה → דיבור. stress tests, benchmarks, fallbacks.
- **P10 — אריזה והפצה ◐ (חלקי)** — `tools/train.py` מלא ל־GPU שלך, PyInstaller/Nuitka exe, Electron builder ל־Windows installer, סקריפט התקנה בלחיצה, מדריך משתמש.
### מה נמדד בפועל (נכון ל־P9)

| שכבה | תוצאה מדידה |
|---|---|
| טוקנייזר | 8,057 טוקנים · 7,791 מיזוגים · 2.999 תו/טוקן · 0/2000 כשלי שחזור |
| קורפוס | 100,910 אימון / 5,312 בדיקה · סדר ReAct תקין |
| ליבה עצבית | 6.20M פרמטרים · 6 שכבות · dev loss 0.188 · ppl **1.207** · 37 דק׳ על 2×CPU |
| קרנל סימבולי | מתמטיקה מדויקת · 11 כוונות · 20 עובדות / 52 QA · מאמת טענות |
| כלים | 54 skills (SAFE 41 / WRITE 6 / CRITICAL 7) |
| זיכרון | SQLite · אפיזודי/סמנטי/פרוצדורלי · שכחה בחצי-חיים 21 יום |
| דיבור (פלט) | concat מ־118 מילים + 381 פונמות אמיתיות · 26/29 פונמות · 24kHz · RTF 0.11–0.21 |
| דיבור (קלט) | 25 פקודות + wake · 50/50 זיהוי עצמי · 0 זיהויי שווא · 36ms לפקודה · כיול עצמי |
| מתכנת | 10 תבניות · 8 חוקי תיקון עצמי · הרצה מבודדת עם timeout · בדיקות מיוצרות |
| אבטחה | חומת הרשאות · אישור אנושי חי דרך ה־HUD · audit JSONL · dry-run · kill switch |
| HUD | Electron frameless · Arc Reactor canvas · זרם אירועים · טלמטריה · תורי הרשאות · שמע ב־WS |
| שרת | aiohttp · loopback בלבד · WS + 9 REST · CORS ל־renderer · אפס בלוקציות בלולאת הקריאה |
| בדיקות | **307 passed / 0 failed** בשבעה מודולים (101 שניות) |

- **P11 — R&D** — הרחבת מודל, self-distillation, STT מתקדם, vision (OCR + תיאור מסך), ריבוי סוכנים מקבילי.

---

## 8. סיכונים כנים + מענה

| סיכון | מענה |
|---|---|
| מודל קטן מאומן על קורפוס סינתטי = יכולת שפה מוגבלת | השכבה הסימבולית + retrieval + verification עושות את העבודה הכבדה. המודל אחראי על ניסוח וניתוב, לא על עובדות. בנוסף: מסלול אימון מלא על GPU אצלך. |
| STT מאפס הוא בעיה קשה | מדרג ריאלי: טקסט + wake-word + פקודות עובדים מיד; AM סינתטי מסומן כניסיוני עם מדידת דיוק אמיתית. לא נשקר לך על היכולות. |
| אין GPU כאן לאימון רציני | מוכיחים את הצינור על NANO כאן; המשקולות הסופיות מאומנות אצלך. כל הסקריפטים מוכנים מראש. |
| שליטה במחשב = סיכון בטיחותי | Permission Firewall + audit + dry-run + kill switch כברירת מחדל, לא כתוספת. |
| קול עברי קונקטיבי עלול להישמע מקוטע | מנוע פורמנטי כגיבוי/היברידי + prosody חכם + crossfade מבוסס-pitch. |
| Electron לא רץ בסנדבוקס | LIVE PREVIEW של אותו renderer + harness שמדמה את Electron API ומאמת את main.js לוגית. |

---

## 9. הגדרת "מוכן" (Definition of Done)

JARVIS נחשב מוכן כשהוא, **ללא אינטרנט וללא אף מפתח API**:
1. נפתח כאפליקציית מסך מלא עם boot sequence קולנועי.
2. מגיב לשם שלו, מקשיב, **מדבר בעברית בקול**.
3. מנהל שיחה רציפה עם אישיות עקבית וזיכרון של שיחות קודמות.
4. פותר מתמטיקה **בדיוק מושלם** (דרך המנוע הסימבולי).
5. **כותב קוד, מריץ אותו, מתקן שגיאות בעצמו ומציג את התוצאה.**
6. פותח אפליקציות, קורא/כותב קבצים, מציג מצב מערכת, מצלם מסך — עם פיקוח הרשאות.
7. כל המחשבה, הקול והפעולה מוצגים ב־HUD חי בזמן אמת.
8. `python tools/selftest.py` עובר 100%.
