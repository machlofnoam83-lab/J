# JAI 3B — מודל AI אישי בעברית

מערכת מלאה לבניית עוזר AI כללי משלך על גבי מודל פתוח של כ־3 מיליארד פרמטרים. המערכת מותאמת ל־**RTX 4060 עם 8GB VRAM** ומשתמשת ב־QLoRA של 4 ביט, כך שבמקום לאמן מחדש את כל 3 מיליארד הפרמטרים היא מאמנת מתאם אישי קטן מעל `Qwen/Qwen2.5-3B-Instruct`.

> חשוב: אימון אמיתי ממשקולות אקראיות דורש מיליארדי טוקנים ואשכול GPU יקר. בכרטיס 4060 הדרך המעשית לקבל מודל חכם היא להתחיל ממודל 3B מאומן, ואז לבצע המשך אימון/התאמה על הנתונים שלך. זה בדיוק מה שהפרויקט עושה.

## מה כלול

- הכנת נתונים: אימות, הסרת כפילויות, ערבוב וחלוקת train/eval
- תמיכה בשיחות JSONL ובטקסט גולמי להמשך pretraining
- אימון QLoRA ב־4 ביט המותאם ל־8GB VRAM
- masking נכון: במצב שיחה מחושב loss רק על תשובות העוזר
- checkpoints, חידוש אימון ומדדי evaluation
- צ׳אט דרך הטרמינל
- ממשק ווב מקומי ו־REST API
- מיזוג ה־LoRA למודל מלא לצורך ייצוא
- בדיקות יחידה שאינן מורידות מודל

## דרישות

- Linux או Windows עם WSL2 (מומלץ Ubuntu 22.04/24.04)
- Python 3.10–3.12
- NVIDIA RTX 4060 ודרייבר עדכני
- כ־15GB מקום פנוי; 16GB RAM לפחות, 32GB מומלץ למיזוג

## התקנה

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# התקן wheel של PyTorch התואם לדרייבר/CUDA שלך. לדוגמה:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
pytest -q
```

אין צורך במפתח API. **בהפעלה הראשונה המערכת מורידה אוטומטית** את משקלי הבסיס מ־Hugging Face, מציגה התקדמות ושומרת אותם במטמון לשימוש הבא. האימון, הצ׳אט והשרת מפעילים את ההורדה בעצמם אם הקבצים עדיין חסרים.

אם רוצים להוריד מראש לפני האימון:

```bash
jai-download
```

לאחר שההורדה הושלמה אפשר לעבוד ללא אינטרנט באמצעות `JAI_OFFLINE=1`. הפקודה `make install` מתקינה את המערכת וגם מורידה את המודל אוטומטית.

## 1. הכנת הנתונים שלך

### פורמט מומלץ לשיחה

כל שורה היא JSON עצמאי:

```json
{"messages":[{"role":"system","content":"אתה JAI..."},{"role":"user","content":"שאלה"},{"role":"assistant","content":"תשובה איכותית"}]}
```

אפשר לכלול שיחות מרובות תורות. התפקידים המותרים הם `system`, `user`, `assistant`.

### טקסט גולמי

אפשר להעביר גם קובצי UTF-8 עם סיומת `.txt`. הם יחולקו למקטעים וישמשו להמשך pretraining. הכנת קבצי האימון:

```bash
jai-prepare my_data/*.jsonl my_documents/*.txt \
  --train-out data/train.jsonl \
  --eval-out data/eval.jsonl \
  --eval-ratio 0.05
```

לניסוי טכני בלבד עם דוגמאות הפרויקט:

```bash
jai-prepare data/sample_train.jsonl data/sample_eval.jsonl
```

שלוש דוגמאות אינן מספיקות כדי לשפר מודל. לתוצאה מורגשת מומלץ להתחיל ב־5,000–50,000 דוגמאות נקיות ומגוונות. איכות, דיוק ורישיונות הנתונים חשובים יותר מכמות עיוורת. אין להכניס סיסמאות, מידע אישי או תוכן שאין לך הרשאה להשתמש בו.

## 2. בחירת סוג האימון

ערוך את `configs/train_4060.yaml`:

- `training_mode: assistant_only` — מומלץ לעוזר שיחה; מאמן רק על תשובות העוזר.
- `training_mode: all_tokens` — המשך pretraining על כל הטקסט; מתאים למסמכי ידע או שפה, אך אינו מחליף instruction tuning.

לתוצאה כללית טובה: בצע תחילה ריצה על טקסט איכותי ב־`all_tokens`, ולאחריה ריצה נפרדת על שיחות ב־`assistant_only`. אל תנסה ללמד „הכול” ממאגר קטן — שמור תערובת של עברית, אנגלית, הסברים, קוד, חשיבה, סיכום, בטיחות ושאלות ידע.

## 3. אימון על RTX 4060

```bash
jai-train --config configs/train_4060.yaml
```

המתאם יישמר ב־`outputs/jai-3b-he`. ההגדרות הבטוחות הן batch בגודל 1, אורך 512 ו־gradient accumulation של 16. זמן האימון תלוי בכמות הנתונים; 500 צעדים עשויים לקחת מעשרות דקות ועד מספר שעות.

חידוש מה־checkpoint האחרון:

```bash
jai-train --config configs/train_4060.yaml --resume latest
```

אם מתקבלת שגיאת CUDA out of memory:

1. הקטן `max_length` מ־512 ל־384 או 256.
2. ודא שאין תוכנה אחרת שמשתמשת ב־GPU (`nvidia-smi`).
3. השאר batch בגודל 1 ו־`load_in_4bit: true`.
4. אל תפעיל `use_flash_attention` בלי להתקין `flash-attn` תואם.

## 4. הערכה

```bash
jai-evaluate \
  --model outputs/jai-3b-he \
  --data data/eval.jsonl \
  --output outputs/evaluation.json
```

הפקודה מחשבת loss/perplexity על טוקני תשובת העוזר ושומרת גם תשובות לדוגמה לבדיקה ידנית. השווה תמיד למודל הבסיס:

```bash
jai-evaluate --model Qwen/Qwen2.5-3B-Instruct --data data/eval.jsonl \
  --output outputs/base-evaluation.json
```

Loss נמוך יותר אינו מבטיח תשובה אמיתית או בטוחה. כדאי להכין סט מבחן שלא הופיע באימון ולדרג ידנית דיוק, עברית, ציות להוראות, הזיות ובטיחות.

## 5. שימוש במודל

צ׳אט בטרמינל:

```bash
jai-chat --model outputs/jai-3b-he
```

ממשק ווב מקומי ו־API:

```bash
jai-serve --model outputs/jai-3b-he --host 0.0.0.0 --port 8000
```

פתח `http://localhost:8000`. קריאת API לדוגמה:

```bash
curl http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"שלום, מי אתה?"}]}'
```

## 6. מיזוג וייצוא (אופציונלי)

המתאם הקטן הוא הדרך המומלצת לשימוש. אם תוכנה אחרת דורשת מודל מלא:

```bash
jai-merge --adapter outputs/jai-3b-he --output outputs/jai-3b-merged
```

המיזוג מתבצע על CPU ועשוי לדרוש 12–16GB RAM. לאחר מכן ניתן להמיר את התיקייה ל־GGUF באמצעות `llama.cpp` לצורך Ollama/LM Studio.

## מבנה הפרויקט

```text
configs/train_4060.yaml  הגדרות אימון ל־4060
data/                     דוגמאות ונתונים מקומיים
src/j_ai/prepare.py       הכנת מידע
src/j_ai/train.py         אימון QLoRA
src/j_ai/evaluate.py      הערכה והשוואה
src/j_ai/chat.py          צ׳אט CLI
src/j_ai/serve.py         אתר ו־API
src/j_ai/merge.py         מיזוג LoRA
```

## מה הופך את המודל ל״שלך״

ה־adapter שנוצר מכיל את ההתאמה לסגנון, למשימות ולמידע שאימנת. יש לשמור גם את שם/גרסת מודל הבסיס. לפני הפצה מסחרית, בדוק את רישיון מודל הבסיס ואת רישיונות כל מקורות הנתונים. מודל 3B יכול להיות עוזר מקומי מהיר וטוב, אך לא ישתווה בכל משימה למודלי ענן גדולים בהרבה ועלול להמציא עובדות — במידע חשוב יש לאמת את התשובות.
