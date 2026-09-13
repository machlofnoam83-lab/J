# J.A.R.V.I.S — Just A Rather Very Intelligent System

> עוזר AI ריבוני לחלוטין. **אפס API keys. אפס ענן. אפס משקולות שהורדו מבחוץ.**
> הטוקנייזר, המודל העצבי, מנוע הדיבור, זיהוי הקול, הזיכרון, סוכני המשנה, ה־UI והשליטה במחשב — **הכול נבנה ומאומן כאן, מאפס**.

📖 **[תוכנית האב המלאה → `docs/PLAN.md`](docs/PLAN.md)**

---

## מה זה

אפליקציית Desktop מלאה (Electron, מסך מלא, HUD הולוגרמי) עם מוח AI היברידי ש:

- 🧠 **חושב** — מודל Transformer שנבנה ואומן מאפס + קרנל סימבולי (תכנון, אימות, מניעת הזיות)
- 🗣️ **מדבר ומקשיב** — מנוע TTS/STT עצמי בעברית, בלי ענן
- 💻 **מתכנת** — סוכן `HEPHAESTUS` שכותב קוד, **מריץ אותו, קורא את השגיאות ומתקן את עצמו**
- 🖥️ **שולט במחשב** — קבצים, תהליכים, אפליקציות, קליפבורד, צילום מסך, אוטומציה (Windows)
- 🧬 **זוכר** — זיכרון קבוע: אפיזודי, סמנטי, פרוצדורלי + שכחה חכמה
- 🔒 **מפוקח** — Permission Firewall, audit log, dry-run, kill switch
- ✨ **נראה כמו מחר** — Arc Reactor, טבעות הולוגרפיות, 10k חלקיקים reactive לקול, ויזואליזציה חיה של המחשבה

## סוכנים

| סוכן | תפקיד |
|---|---|
| **JARVIS** | המנצח — שיחה, כוונה, תכנון, קבלת החלטות |
| **HEPHAESTUS** | סוכן המתכנת — כתיבה, הרצה, תיקון עצמי |
| **MNEMOSYNE** | הזיכרון — שינון ושליפה |
| **ARGUS** | העיניים — מצב המחשב, מסך |
| **HERMES** | הידיים — ביצוע פעולות |

## הרצה

```bash
pip install -r requirements.txt
python jarvis.py                 # מוח + שרת + HUD
python jarvis.py --mode desktop  # אפליקציית Electron (Windows)
python tools/selftest.py         # בדיקת כל המערכות
```

## בניית המוח מאפס

```bash
python tools/forge_corpus.py --size large   # יצירת קורפוס עברי+אנגלי+קוד
python tools/build_tokenizer.py             # BPE דו-לשוני משלנו
python tools/train.py --size nano           # אימון הוכחת-צינור (CPU)
python tools/train.py --size core --gpu     # אימון מלא (אם יש GPU)
python tools/export_onnx.py                 # ריצה מהירה ב-onnxruntime
```

## מצב הפרויקט

ראה **[ROADMAP](docs/PLAN.md#7-שלבי-ביצוע-build-phases)** — שלבי P0→P11.

---

*נבנה מאפס. שום דבר כאן לא הורד מוכן.*
