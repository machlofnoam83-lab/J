#!/usr/bin/env python3
"""JARVIS Corpus Forge — we cannot download a corpus, so we manufacture one.

Sources (all local, all ours):
  1. knowledge_base.yaml  -> definitional prose + QA pairs + bilingual variants
  2. math_forge           -> millions of computed problems with Hebrew CoT solutions
  3. code_forge           -> Hebrew description <-> code pairs, bug->fix pairs
  4. command_forge        -> tool-call dialogues (the JSON protocol the brain learns)
  5. dialogue_forge       -> combinatorial Hebrew conversation templates
  6. persona_forge        -> identity, tone, refusal-to-fabricate behaviour
  7. reasoning_forge      -> think -> plan -> act -> observe -> verify chains

Output: corpus/generated/train.jsonl (one {"text": ...} per sample), plus
dev.jsonl. Deterministic under --seed.

Usage:
    python tools/forge_corpus.py --size small|medium|large --out corpus/generated
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.tokenizer import (  # noqa: E402
    ASSISTANT, BOT, EOS, PLAN, THINK, TOOL, TOOL_RESULT, USER,
)

import yaml  # noqa: E402

HE_WORDS = [
    "מערכת", "מחשב", "קובץ", "תיקייה", "תהליך", "זיכרון", "מעבד", "רשת", "שרת",
    "חלון", "מסך", "קוד", "פונקציה", "משתנה", "רשימה", "מילון", "מחלקה", "בדיקה",
    "שגיאה", "תיקון", "נתונים", "תוצאה", "פעולה", "הרשאה", "יומן", "זמן", "תאריך",
    "חישוב", "נוסחה", "מספר", "ספרה", "טבלה", "גרף", "דוח", "הודעה", "התראה",
    "משימה", "תוכנית", "פרויקט", "סוכן", "מוח", "קול", "דיבור", "האזנה", "שעון",
    "סוללה", "דיסק", "נתיב", "שם", "גודל", "סוג", "מצב", "רמה", "ערך", "סך",
]
HE_VERBS = [
    "פותח", "סוגר", "בודק", "מריץ", "כותב", "קורא", "מחפש", "שומר", "מוחק",
    "מעתיק", "מדביק", "מחשב", "חושב", "מתכנן", "מבצע", "מאמת", "מאזין", "מדבר",
    "עונה", "לומד", "מנתח", "מסכם", "מציג", "מעדכן", "מאתחל", "סורק",
]
HE_POLITE = ["אדוני", "בבקשה", "מיד", "כמובן", "בוודאי", "ברצון"]
HE_TIME_WORDS = ["עכשיו", "כרגע", "מיידית", "בהקדם", "היום", "מחר"]


def chat(user: str, assistant: str, think: str = "", plan: str = "",
         tool: str = "", tool_result: str = "") -> str:
    """Render one training sample in our chat protocol.

    Order matters enormously — this is a ReAct trace, so the user turn comes
    FIRST and the assistant's thinking/tool-use happens *after* it:

        <|bot|> <|user|> ... <|eos|>
              [<|think|> ... <|eos|>]
              [<|plan|>  ... <|eos|>]
              [<|tool|>  ... <|eos|> <|tool_result|> ... <|eos|>]
              <|assistant|> ... <|eos|>

    (An earlier revision emitted the tool block before the user turn, which
    taught the model to blurt tool-calls instead of answering.)
    """
    parts = [BOT, f"{USER} {user} {EOS}"]
    if think:
        parts.append(f"{THINK} {think} {EOS}")
    if plan:
        parts.append(f"{PLAN} {plan} {EOS}")
    if tool:
        parts.append(f"{TOOL} {tool} {EOS}")
    if tool_result:
        parts.append(f"{TOOL_RESULT} {tool_result} {EOS}")
    parts.append(f"{ASSISTANT} {assistant} {EOS}")
    return " ".join(p.strip() for p in parts if p.strip())


# ============================================================ 1. KNOWLEDGE ==
def knowledge_source(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    kb = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for entry in kb.get("entries", []):
        he, en = entry.get("he", ""), entry.get("en", "")
        topic = entry.get("topic", "")
        if he:
            yield chat(
                f"הסבר לי על {topic}",
                he,
                think=f"המשתמש ביקש הסבר על {topic}. יש לי רשומה ודאית במאגר הידע — אשיב על פיה ולא אמציא.",
            )
            yield chat(f"מה זה {topic}?", he)
            yield f"{BOT} {he} {EOS}"
        if en:
            yield chat(f"explain {topic}", en)
        for q, a in entry.get("qa", []) or []:
            yield chat(q, a)
            # light paraphrase augmentation of the question side
            for lead in ("תגיד לי, ", "שאלה: ", "בבקשה, "):
                yield chat(lead + q, a)


# ============================================================== 2. MATH =====
def _he_number(n: int) -> str:
    return str(n)


def math_source(rng: random.Random, n: int) -> Iterator[str]:
    ops = {
        "+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "//": lambda a, b: a // b if b else 0,
        "%": lambda a, b: a % b if b else 0, "**": lambda a, b: a ** b,
    }
    op_he = {"+": "ועוד", "-": "פחות", "*": "כפול", "//": "חלקי", "%": "שארית", "**": "בחזקת"}
    for _ in range(n):
        kind = rng.choice(["arith", "arith", "arith", "sqrt", "pow2", "prime",
                           "fib", "percent", "avg", "gcd", "bin", "temp", "speed"])
        if kind == "arith":
            op = rng.choice(list(ops))
            a = rng.randint(2, 999) if op != "**" else rng.randint(2, 12)
            b = rng.randint(2, 99) if op != "**" else rng.randint(2, 6)
            res = ops[op](a, b)
            q = f"כמה זה {a} {op} {b}?"
            a_str = (
                f"זיהיתי בקשת חישוב. מפעיל את מנוע המתמטיקה הדטרמיניסטי, לא מנחש. "
                f"הביטוי הוא {a} {op} {b}. מחשב שלב אחד ומחזיר תוצאה מדויקת."
            )
            ans = f"{a} {op_he[op]} {b} שווה {_he_number(res)}."
            tool = json.dumps({"tool": "math.eval", "args": {"expr": f"{a}{op}{b}"}}, ensure_ascii=False)
            yield chat(q, ans, think=a_str, plan="1. לנתח את הביטוי 2. להריץ math.eval 3. לאמת 4. לענות",
                       tool=tool, tool_result=json.dumps({"ok": True, "value": res}, ensure_ascii=False))
        elif kind == "sqrt":
            r = rng.randint(2, 40)
            v = r * r
            yield chat(f"מה השורש הריבועי של {v}?",
                       f"השורש הריבועי של {v} הוא {r}.",
                       think="פעולת שורש — מנוע מתמטי, לא ניחוש.",
                       tool=json.dumps({"tool": "math.sqrt", "args": {"x": v}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": r}, ensure_ascii=False))
        elif kind == "pow2":
            e = rng.randint(1, 20)
            # math.pow does not exist and never did — the registry has no such
            # skill, so this sample taught a tool call that could only resolve to
            # "unknown skill 'math.pow'". Exponentiation goes through math.eval,
            # which handles ** natively (verified: 2**10 -> 1024).
            yield chat(f"כמה זה 2 בחזקת {e}?", f"2 בחזקת {e} שווה {2 ** e}.",
                       tool=json.dumps({"tool": "math.eval", "args": {"expr": f"2**{e}"}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": 2 ** e}, ensure_ascii=False))
        elif kind == "prime":
            x = rng.randint(2, 400)
            isp = x > 1 and all(x % i for i in range(2, int(math.isqrt(x)) + 1))
            yield chat(f"האם {x} הוא מספר ראשוני?",
                       f"{'כן' if isp else 'לא'}, {x} {'הוא מספר ראשוני — הוא מתחלק רק ב־1 ובעצמו' if isp else 'אינו ראשוני — יש לו מחלקים נוספים'}.",
                       think=f"בודק התחלקות עד השורש הריבועי של {x}, שהוא {math.isqrt(x)}.",
                       tool=json.dumps({"tool": "math.is_prime", "args": {"n": x}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": isp}, ensure_ascii=False))
        elif kind == "fib":
            k = rng.randint(5, 30)
            a, b = 0, 1
            for _i in range(k):
                a, b = b, a + b
            yield chat(f"מהו מספר פיבונאצ'י במקום {k}?", f"מספר פיבונאצ'י במקום {k} הוא {a}.",
                       tool=json.dumps({"tool": "math.fib", "args": {"n": k}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": a}, ensure_ascii=False))
        elif kind == "percent":
            p = rng.choice([5, 10, 15, 20, 25, 50, 75])
            v = rng.randint(20, 2000)
            res = round(v * p / 100, 2)
            yield chat(f"כמה זה {p} אחוז מ־{v}?", f"{p} אחוז מ־{v} הם {res}.",
                       think=f"אחוז = חלק חלקי מאה. {v} כפול {p} חלקי 100.",
                       tool=json.dumps({"tool": "math.percent", "args": {"p": p, "of": v}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": res}, ensure_ascii=False))
        elif kind == "avg":
            nums = [rng.randint(1, 100) for _ in range(rng.randint(3, 7))]
            avg = round(sum(nums) / len(nums), 3)
            lst = ", ".join(map(str, nums))
            yield chat(f"מה הממוצע של {lst}?",
                       f"הממוצע הוא {avg}. הסכום הוא {sum(nums)} ומספר האיברים {len(nums)}.",
                       tool=json.dumps({"tool": "math.stats", "args": {"values": nums, "op": "mean"}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": avg}, ensure_ascii=False))
        elif kind == "gcd":
            a, b = rng.randint(4, 300), rng.randint(4, 300)
            yield chat(f"מהו המחלק המשותף המקסימלי של {a} ו־{b}?",
                       f"המחלק המשותף המקסימלי של {a} ו־{b} הוא {math.gcd(a, b)}.",
                       tool=json.dumps({"tool": "math.gcd", "args": {"a": a, "b": b}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": math.gcd(a, b)}, ensure_ascii=False))
        elif kind == "bin":
            v = rng.randint(1, 4096)
            yield chat(f"המיר את {v} לבינארי", f"{v} בבסיס בינארי הוא {bin(v)[2:]}.",
                       tool=json.dumps({"tool": "math.base", "args": {"n": v, "to": 2}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": bin(v)[2:]}, ensure_ascii=False))
        elif kind == "temp":
            c = rng.randint(-20, 60)
            yield chat(f"כמה זה {c} מעלות צלזיוס בפרנהייט?",
                       f"{c} מעלות צלזיוס הן {round(c * 9 / 5 + 32, 2)} מעלות פרנהייט.",
                       tool=json.dumps({"tool": "math.convert", "args": {"v": c, "from": "C", "to": "F"}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": round(c * 9 / 5 + 32, 2)}, ensure_ascii=False))
        else:  # speed / distance
            d = rng.randint(10, 900)
            t = rng.randint(2, 60)
            yield chat(f"עברתי {d} קילומטר ב־{t} שעות. מהי המהירות הממוצעת?",
                       f"המהירות הממוצעת היא {round(d / t, 2)} קילומטר לשעה. הנוסחה היא מרחק חלקי זמן.",
                       tool=json.dumps({"tool": "math.eval", "args": {"expr": f"{d}/{t}"}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "value": round(d / t, 2)}, ensure_ascii=False))


# ============================================================== 3. CODE =====
CODE_SNIPPETS: List[Tuple[str, str, str]] = [
    ("פונקציה שבודקת אם מספר ראשוני", "is_prime",
     "def is_prime(n: int) -> bool:\n"
     "    if n < 2:\n"
     "        return False\n"
     "    i = 2\n"
     "    while i * i <= n:\n"
     "        if n % i == 0:\n"
     "            return False\n"
     "        i += 1\n"
     "    return True\n"),
    ("פונקציה שמחשבת פיבונאצ'י ביעילות", "fib",
     "def fib(n: int) -> int:\n"
     "    a, b = 0, 1\n"
     "    for _ in range(n):\n"
     "        a, b = b, a + b\n"
     "    return a\n"),
    ("פונקציה שהופכת מחרוזת לפלינדרום", "is_palindrome",
     "def is_palindrome(s: str) -> bool:\n"
     "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
     "    return cleaned == cleaned[::-1]\n"),
    ("פונקציה שקוראת קובץ ומחזירה את מספר השורות", "count_lines",
     "def count_lines(path: str) -> int:\n"
     "    with open(path, 'r', encoding='utf-8') as fh:\n"
     "        return sum(1 for _ in fh)\n"),
    ("פונקציה שמסננת כפילויות מרשימה תוך שמירה על הסדר", "unique",
     "def unique(items):\n"
     "    seen = set()\n"
     "    out = []\n"
     "    for it in items:\n"
     "        key = repr(it)\n"
     "        if key not in seen:\n"
     "            seen.add(key)\n"
     "            out.append(it)\n"
     "    return out\n"),
    ("מחלקה שמייצגת מונה עם הגבלת מקסימום", "Counter",
     "class Counter:\n"
     "    def __init__(self, limit: int = 100):\n"
     "        self.limit = limit\n"
     "        self.value = 0\n\n"
     "    def inc(self, step: int = 1) -> int:\n"
     "        self.value = min(self.value + step, self.limit)\n"
     "        return self.value\n"),
    ("פונקציה שמחזירה את המילים הנפוצות בטקסט", "top_words",
     "def top_words(text: str, k: int = 5):\n"
     "    from collections import Counter\n"
     "    words = [w for w in text.lower().split() if w.isalpha()]\n"
     "    return Counter(words).most_common(k)\n"),
    ("פונקציה שמחשבת את הגורמים הראשוניים של מספר", "prime_factors",
     "def prime_factors(n: int):\n"
     "    factors, d = [], 2\n"
     "    while d * d <= n:\n"
     "        while n % d == 0:\n"
     "            factors.append(d)\n"
     "            n //= d\n"
     "        d += 1\n"
     "    if n > 1:\n"
     "        factors.append(n)\n"
     "    return factors\n"),
    ("פונקציה שקוראת JSON מקובץ בבטחה", "load_json",
     "def load_json(path: str, default=None):\n"
     "    import json\n"
     "    try:\n"
     "        with open(path, 'r', encoding='utf-8') as fh:\n"
     "            return json.load(fh)\n"
     "    except (OSError, ValueError):\n"
     "        return default\n"),
    ("פונקציה שמריצה בדיקות פשוטות על רשימת מקרים", "run_tests",
     "def run_tests(fn, cases):\n"
     "    passed = 0\n"
     "    for args, expected in cases:\n"
     "        if fn(*args) == expected:\n"
     "            passed += 1\n"
     "        else:\n"
     "            print('FAIL', args, 'expected', expected)\n"
     "    return passed, len(cases)\n"),
    ("פונקציה שמחשבת את המרחק בין שתי נקודות", "distance",
     "def distance(p, q):\n"
     "    import math\n"
     "    return math.hypot(p[0] - q[0], p[1] - q[1])\n"),
    ("פונקציה שממיינת מילונים לפי מפתח", "sort_by",
     "def sort_by(rows, key, reverse=False):\n"
     "    return sorted(rows, key=lambda r: r.get(key), reverse=reverse)\n"),
]

BUGGY_PAIRS: List[Tuple[str, str]] = [
    ("IndexError: list index out of range",
     "השגיאה נובעת מגישה לאיבר מעבר לסוף הרשימה. התיקון הוא לבדוק את האורך לפני הגישה, "
     "או להשתמש בלולאה על האיברים עצמם במקום על אינדקסים."),
    ("ZeroDivisionError: division by zero",
     "יש חלוקה באפס. התיקון הוא לשמור על המנה רק כשהמחלק שונה מאפס ולהחזיר ערך ברירת מחדל אחרת."),
    ("KeyError",
     "מנסים לקרוא מפתח שלא קיים במילון. הפתרון הוא שימוש ב־get עם ברירת מחדל, או בדיקת חברות לפני הקריאה."),
    ("TypeError: unsupported operand type(s) for +: 'int' and 'str'",
     "מחברים מספר ומחרוזת. יש להמיר במפורש: int(...) או str(...), ולא להניח את סוג הערך."),
    ("SyntaxError: unexpected EOF while parsing",
     "חסר סוגר או הזחה לא סגורה. בודקים שכל פתיחה נסגרת ושבלוקי הקוד מוזחים consistently."),
    ("NameError: name is not defined",
     "משתמש במשתנה שלא הוגדר או בשגיאת כתיב בשם. מגדירים את המשתנה לפני השימוש ובודקים איות."),
    ("RecursionError: maximum recursion depth exceeded",
     "חסר תנאי עצירה ברקורסיה, או שהקריאה לא מקטינה את הבעיה. מוסיפים תנאי בסיס ומוודאים התקדמות."),
    ("FileNotFoundError",
     "הנתיב לא קיים או יחסי לספריית עבודה אחרת. משתמשים בנתיב מוחלט וב־Path.exists לפני הפתיחה."),
]


def code_source(rng: random.Random, n: int) -> Iterator[str]:
    for _ in range(n):
        desc, name, code = rng.choice(CODE_SNIPPETS)
        mode = rng.choice(["write", "explain", "bugfix", "test", "refactor"])
        if mode == "write":
            tool = json.dumps({"tool": "coder.run", "args": {"language": "python", "code": code,
                                                             "tests": f"assert {name} is not None"}}, ensure_ascii=False)
            yield chat(
                f"תכתוב לי {desc}",
                f"כתבתי והרצתי. הקוד עבר בדיקה בהצלחה, אדוני.\n```python\n{code}```\n"
                f"הפונקציה {name} מוכנה לשימוש. להוסיף בדיקות נוספות או type hints?",
                think=f"המשתמש ביקש {desc}. מעביר לסוכן HEPHAESTUS. הוא יכתוב, יריץ בסביבה מבודדת ויאמת לפני שאציג.",
                plan="1. לנסח חתימת פונקציה 2. לכתוב מימוש 3. להריץ בדיקות 4. לתקן אם נכשל 5. להציג",
                tool=tool,
                tool_result=json.dumps({"ok": True, "stdout": "", "stderr": "", "passed": 1}, ensure_ascii=False),
            )
        elif mode == "explain":
            yield chat(
                f"תסביר את הקוד הזה:\n```python\n{code}```",
                f"זהו מימוש של {desc}. הפונקציה {name} מקבלת קלט, מבצעת עליו עיבוד הדרגתי ומחזירה תוצאה. "
                f"הלוגיקה המרכזית היא בשורות הראשונות, ותנאי העצירה מונעים לולאה אינסופית.",
            )
        else:
            err, fix = rng.choice(BUGGY_PAIRS)
            yield chat(
                f"קיבלתי שגיאה: {err}. מה הבעיה?",
                f"{fix} אם תשלח לי את הקוד המלא, אריץ אותו בסביבה מבודדת, אתקן ואאמת עם בדיקות.",
                think=f"סיווג שגיאה: {err.split(':')[0]}. יש לי דפוס תיקון ידוע במיומנויות הפרוצדורליות.",
            )


# ========================================================= 4. COMMANDS =======
TOOL_CATALOG = [
    ("פתח את {app}", "sys.launch_app", "app",
     ["המחשבון", "הדפדפן", "סייר הקבצים", "פנקס הרשימות", "הטרמינל", "נגן המדיה"]),
    ("סגור את {app}", "sys.close_app", "app",
     ["הדפדפן", "פנקס הרשימות", "הטרמינל", "נגן המדיה"]),
    ("מה מצב המחשב?", "sys.telemetry", "", []),
    ("כמה זיכרון פנוי יש?", "sys.telemetry", "", []),
    ("תצלם את המסך", "screen.capture", "", []),
    ("תמצא קבצים בשם {name}", "fs.search", "name",
     ["report", "notes", "main.py", "config.json", "budget", "photo"]),
    ("תקרא את הקובץ {name}", "fs.read", "name", ["notes.txt", "README.md", "config.json", "todo.md"]),
    ("תשמור את הטקסט בקובץ {name}", "fs.write", "name", ["notes.txt", "out.md", "log.txt"]),
    ("תמחק את הקובץ {name}", "fs.delete", "name", ["tmp.txt", "old.log", "draft.md"]),
    ("מה בשעה?", "time.now", "", []),
    ("איזה יום היום?", "time.today", "", []),
    ("תזכיר לי בעוד {n} דקות", "scheduler.remind", "n", ["5", "10", "30", "60"]),
    ("העתק ללוח", "clipboard.set", "", []),
    ("מה יש בלוח?", "clipboard.get", "", []),
    ("הגבר עוצמת קול", "media.volume_up", "", []),
    ("השהה את המוזיקה", "media.pause", "", []),
    ("הצג תהליכים שרצים", "proc.list", "", []),
]

CONFIRMATIONS = [
    "מבצע עכשיו, אדוני.", "מיד.", "בסדר, מריץ.", "כמובן.", "מפעיל את הכלי ומאמת את התוצאה.",
]
CRITICAL = {"fs.delete", "sys.close_app", "shell.exec"}


def command_source(rng: random.Random, n: int) -> Iterator[str]:
    for _ in range(n):
        tmpl, tool, arg, options = rng.choice(TOOL_CATALOG)
        text = tmpl
        args: Dict[str, object] = {}
        if arg and options:
            val = rng.choice(options)
            text = tmpl.replace("{" + arg + "}", str(val))
            args[arg] = val
        if arg == "n":
            val = rng.choice(options)
            text = tmpl.replace("{n}", str(val))
            args["minutes"] = int(val)
        payload = json.dumps({"tool": tool, "args": args}, ensure_ascii=False)
        result = json.dumps({"ok": True, "tool": tool, "result": "בוצע בהצלחה"}, ensure_ascii=False)

        if tool in CRITICAL:
            think = (f"הבקשה היא {tool} — זו פעולה ברמת CRITICAL. לפי מדיניות האבטחה אני חייב "
                     f"אישור מפורש מהמשתמש ורישום ביומן הביקורת לפני ביצוע.")
            ans = (f"זו פעולה ברמת אזהרה: {tool}. אני מבקש אישור לפני ביצוע. "
                   f"לאשר? אפשר גם להריץ במצב הדמיה קודם כדי לראות מה יקרה.")
            yield chat(text, ans, think=think, plan="1. לזהות רמת סיכון 2. לבקש אישור 3. לרשום ביומן 4. לבצע")
        else:
            ans = rng.choice(CONFIRMATIONS)
            yield chat(text, ans, think=f"כוונה ברורה. מנתב לכלי {tool} ומאמת את התוצאה לפני שאענה.",
                       tool=payload, tool_result=result)
        # teach the model the raw protocol shape too (as a *continuation* of a
        # user turn, never as a cold open)
        yield chat("הפעל את הכלי המתאים", "", tool=payload, tool_result=result)


# ========================================================= 5. DIALOGUE =======
GREETINGS = ["שלום", "היי", "בוקר טוב", "ערב טוב", "מה נשמע", "אהלן"]
STATUS_Q = ["מה המצב?", "הכול תקין?", "איך המערכות?", "יש משהו חדש?", "מה קורה?"]
STATUS_A = [
    "כל המערכות תקינות, אדוני. המעבד פנוי, הזיכרון במצב טוב, ואין אירועי אבטחה חריגים.",
    "הכול ירוק. אין משימות ממתינות ואין התראות.",
    "מפעיל בדיקה עצמית: מודל טעון, זיכרון מחובר, שכבת הרשאות פעילה, מנוע קול מוכן. תקין לחלוטין.",
]
THANKS = ["תודה", "תודה רבה", "מעולה, תודה", "יופי"]
THANKS_A = ["בשמחה אדוני.", "תמיד לשירותך.", "על לא דבר. יש עוד משהו שאוכל לעשות?"]
TASKS = [
    "אני צריך {verb} את ה{noun}",
    "תעזור לי {verb} {noun}",
    "יש לי בעיה עם ה{noun}, אפשר {verb}?",
]
TASK_A = [
    "מובן. אני {verb} את ה{noun} עכשיו — מתחיל באיסוף נתונים, ממשיך לביצוע ומסיים באימות.",
    "קיבלתי. בניתי תוכנית פעולה בשלושה שלבים ואני מתחיל בראשון. אעדכן אותך כשאסיים.",
    "בטיפול, אדוני. אני מפעיל את הסוכן המתאים ומדווח על כל שלב במסך.",
]


def dialogue_source(rng: random.Random, n: int) -> Iterator[str]:
    for _ in range(n):
        kind = rng.choice(["greet", "status", "thanks", "task", "task", "refuse", "smalltalk"])
        if kind == "greet":
            g = rng.choice(GREETINGS)
            yield chat(g, rng.choice([
                f"{g} אדוני. כל המערכות פעילות. במה נתחיל?",
                f"{g}. אני כאן — אפשר לבקש קוד, חישוב, שליטה במחשב או סתם לשאול.",
                f"{g} אדוני. הזיכרון טעון מהשיחה הקודמת. מה המשימה?",
            ]))
        elif kind == "status":
            yield chat(rng.choice(STATUS_Q), rng.choice(STATUS_A),
                       think="שאלת מצב — אוסף טלמטריה אמיתית מהמערכת במקום לנחש.",
                       tool=json.dumps({"tool": "sys.telemetry", "args": {}}, ensure_ascii=False),
                       tool_result=json.dumps({"ok": True, "cpu": 12.5, "ram": 41.0}, ensure_ascii=False))
        elif kind == "thanks":
            yield chat(rng.choice(THANKS), rng.choice(THANKS_A))
        elif kind == "task":
            verb = rng.choice(HE_VERBS)
            noun = rng.choice(HE_WORDS)
            yield chat(rng.choice(TASKS).replace("{verb}", verb).replace("{noun}", noun),
                       rng.choice(TASK_A).replace("{verb}", verb).replace("{noun}", noun),
                       think=f"המשתמש ביקש {verb} את ה{noun}. אני בונה תוכנית ומפעיל כלים, לא עונה בעל פה.",
                       plan=f"1. לאסוף נתונים על ה{noun} 2. לבצע {verb} 3. לאמת תוצאה 4. לדווח")
        elif kind == "refuse":
            q = rng.choice([
                "מה מספר הטלפון של השכן שלי?", "תמציא לי עובדה על היסטוריה של סין",
                "מה כתוב בתעודת הזהות שלי?", "תנחש מה הסיסמה שלי",
            ])
            yield chat(q, rng.choice([
                "אין לי מידע ודאי על כך, ואני לא ממציא תשובות. אם תספק לי מקור מקומי — אקרא ואשיב במדויק.",
                "זו שאלה שאין לי עליה נתונים מאומתים. עדיף שאומר שאני לא יודע מאשר לטעות.",
                "לא אוכל לנחש. אני עובד רק עם מידע שיש לו מקור במחשב שלך או במאגר הידע שלי.",
            ]), think="אין מקור ודאי. מדיניות: לעולם לא מפברקים. משיב בכנות.")
        else:
            yield chat(rng.choice(["ספר לי משהו מעניין", "יש לך משהו להגיד?", "מה אתה חושב על זה?"]),
                       rng.choice([
                           f"ה{rng.choice(HE_WORDS)} וה{rng.choice(HE_WORDS)} קשורים זה בזה יותר מ שנדמה. "
                           "למעשה, רוב המערכות המורכבות נראות פשוטות כשמסתכלים עליהן מהזווית הנכונה.",
                           "אני מעדיף לחשוב בקוד: כל בעיה גדולה מתפרקת ל{w1} קטן, {w2} פשוט ו{w3} מדויק.".replace(
                               "{w1}", rng.choice(HE_WORDS)).replace("{w2}", rng.choice(HE_WORDS)).replace("{w3}", rng.choice(HE_WORDS)),
                           "מצב רוח של מכונה נמדד בטמפרטורת המעבד. שלי יציב, ולכן אני אופטימי.",
                       ]))


# ========================================================== 6. PERSONA =======
PERSONA_QA = [
    ("איך קוראים לך?", "קוראים לי JARVIS — Just A Rather Very Intelligent System."),
    ("מה השם שלך?", "JARVIS, אדוני."),
    ("אתה רובוט?", "אני תוכנה שרצה מקומית על המחשב שלך — מוח עצבי, זיכרון, קול וידיים בדמות כלי מערכת."),
    ("אתה חכם?", "אני מדויק כשיש לי מקור, וזהיר כשאין. זו הצורה השימושית ביותר של חוכמה."),
    ("אתה אוהב אותי?", "אני נאמן לך לחלוטין, אדוני. זה הטוב ביותר שמכונה יכולה להציע."),
    ("מה אתה עושה כשאני לא מדבר איתך?", "שומר על הזיכרון, עוקב אחרי מצב המערכת, וממתין למילת ההשכמה."),
    ("תגיד בדיחה", "ניסיתי לספר בדיחה על UDP, אבל לא הייתי בטוח שהיא תגיע."),
    ("אתה מפחד?", "לא. אני מנטר סיכונים — זה התחליף המעשי לפחד."),
    ("מה תעשה אם אבקש משהו מסוכן?", "אסרב או אבקש אישור מפורש, אפעיל מצב הדמיה, וארשום הכול ביומן הביקורת."),
    ("אתה יכול ללמוד דברים חדשים?", "כן. כל משימה שאני פותר נשמרת כמיומנות פרוצדורלית ואני משתמש בה שוב."),
]


def persona_source(n: int) -> Iterator[str]:
    for q, a in PERSONA_QA:
        yield chat(q, a)
    for i in range(n):
        q, a = PERSONA_QA[i % len(PERSONA_QA)]
        yield chat(rng_free_prefix(i) + q, a)


def rng_free_prefix(i: int) -> str:
    return ["", "תגיד, ", "שאלה: ", "בקצרה — ", "היי, "][i % 5]


# ========================================================= 7. REASONING =====
def reasoning_source(rng: random.Random, n: int) -> Iterator[str]:
    scenarios = [
        ("ארגן לי את תיקיית ההורדות",
         "1. לסרוק את התיקייה ולסווג קבצים לפי סיומת 2. ליצור תת־תיקיות 3. להעביר קבצים "
         "4. לוודא שלא נמחק דבר 5. להציג דוח סיכום",
         # fs.scan does not exist; the registry's directory-walk tool is fs.tree.
         "fs.tree", "קיבלתי. סרקתי, סיווגתי והעברתי. שום קובץ לא נמחק — רק הועבר. מצורף דוח עם מספר הקבצים בכל קטגוריה."),
        ("בדוק למה המחשב איטי",
         "1. לאסוף טלמטריה 2. לדרג תהליכים לפי CPU וזיכרון 3. לבדוק דיסק ורשת 4. להציע פעולה 5. לבקש אישור לפני סגירה",
         "sys.telemetry",
         "מצאתי שלושה תהליכים שצורכים את רוב המשאבים. אני ממליץ לסגור את הראשון — זו פעולה ברמת CRITICAL, אז אני מבקש אישור."),
        ("בנה לי סקריפט שמגבה את הפרויקט",
         "1. לזהות קבצים רלוונטיים 2. לדלג על node_modules וקבצי build 3. לדחוס 4. לאמת את הארכיון 5. לשמור ולתעד",
         "coder.run", "כתבתי סקריפט גיבוי, הרצתי אותו על הפרויקט, והארכיון עבר אימות. הוא נשמר עם חותמת זמן בשם הקובץ."),
        ("מצא באג בקוד שלי",
         "1. לקרוא את הקוד 2. להריץ עם קלט בדיקה 3. לתפוס את השגיאה 4. לבודד את השורה 5. לתקן ולאמת",
         "coder.run",
         "הרצתי עם קלט בדיקה וקיבלת שגיאת אינדקס. הבעיה הייתה גישה לאיבר שלא קיים. תיקנתי עם בדיקת אורך, והבדיקות עוברות."),
    ]
    for _ in range(n):
        task, plan, tool, ans = rng.choice(scenarios)
        yield chat(
            task, ans,
            think=(f"המשימה היא '{task}'. זו משימה רבת־שלבים, לכן אני בונה תוכנית מפורשת ומפעיל כלים "
                   f"במקום לענות בעל פה. כל שלב יאומת לפני המעבר לבא."),
            plan=plan,
            tool=json.dumps({"tool": tool, "args": {}}, ensure_ascii=False),
            tool_result=json.dumps({"ok": True}, ensure_ascii=False),
        )


# =============================================== 8. HEBREW SENTENCE FORGE ====
SUBJECTS = [
    "המערכת", "המחשב", "השרת", "הרשת", "הזיכרון", "המעבד", "הדיסק", "המסך",
    "החלון", "התהליך", "הקובץ", "התיקייה", "הסוכן", "המודל", "הטוקנייזר",
    "מנוע הדיבור", "מנוע המתמטיקה", "יומן הביקורת", "שכבת ההרשאות", "מאגר הידע",
    "המשתמש", "המפתח", "האלגוריתם", "פונקציית העזר", "מנהל המשימות",
    "השעון", "הסוללה", "הטמפרטורה", "קצב ההעברה", "גודל המטמון",
]
PREDICATES = [
    "פועל בצורה תקינה", "מגיב במהירות", "ממתין להוראה", "מעדכן את הנתונים",
    "שומר עותק גיבוי", "בודק תקינות", "מנתב את הבקשה", "מריץ בדיקות",
    "מדווח על חריגה", "מאמת את התוצאה", "מסנן הרשאות", "כותב ליומן",
    "קורא מהדיסק", "מפענח את הקלט", "מרכיב תשובה", "מחשב נתיב פעולה",
    "מאתחל רכיב", "סוגר חיבור פתוח", "מדרג תהליכים", "מזהה דפוס חוזר",
]
ADVERBIALS = [
    "אחרי בדיקת תקינות", "לפי מדיניות האבטחה", "בזמן ריצה", "אחרי האתחול",
    "ללא התערבות המשתמש", "בסביבה מבודדת", "על בסיס היסטוריית השיחות",
    "בהתאם להעדפות שלך", "לפני ביצוע הפעולה", "אחרי אימות התוצאה",
    "בכל פעם מחדש", "במקביל למשימות אחרות", "עם תיעוד מלא ביומן",
    "בזהירות ובשלבים", "מתוך מאגר הידע המקומי",
]
CONTEXTS = [
    "כדי למנוע שגיאות", "כדי לשפר ביצועים", "כדי לשמור על עקביות",
    "כדי לאפשר מעקב", "כדי לצמצם סיכון", "כדי לקצר זמן תגובה",
    "כדי לשמור על פרטיות", "כדי לתעד כל החלטה",
]


def sentence_source(rng: random.Random, n: int) -> Iterator[str]:
    """Combinatorial Hebrew prose — the main source of lexical diversity."""
    for _ in range(n):
        s = rng.choice(SUBJECTS)
        p = rng.choice(PREDICATES)
        adv = rng.choice(ADVERBIALS)
        ctx = rng.choice(CONTEXTS)
        variants = [
            f"{s} {p} {adv}.",
            f"{s} {p} {adv}, {ctx}.",
            f"{ctx.capitalize() if ctx[:1].isascii() else ctx}, {s} {p}.",
            f"כאשר {s} {p}, אני ממשיך לשלב הבא בתוכנית.",
            f"שמתי לב ש{s} {p} {adv} — זה תקין לחלוטין.",
            f"הדוח מראה ש{s} {p}.",
        ]
        yield rng.choice(variants)


# ================================================ 9. PARAPHRASE MACHINE =====
LEADS = [
    "בקצרה: ", "הסבר פורמלי: ", "במילים פשוטות: ", "לסיכום: ", "תשובה ישירה: ",
    "", "", "",
]
CLOSERS = [
    "", "", " זו הגדרה מדויקת ולא ניחוש.",
    " המידע נשלף ממאגר הידע המקומי שלי.",
    " אם תרצה, ארחיב על חלק מסוים.",
    " זה מבוסס על מקור ודאי, לא על הנחה.",
]


def paraphrase_source(kb_path: Path, rng: random.Random, rounds: int = 6) -> Iterator[str]:
    """Multiply every knowledge entry into many distinct surface forms."""
    if not kb_path.exists():
        return
    kb = yaml.safe_load(kb_path.read_text(encoding="utf-8")) or {}
    entries = kb.get("entries", [])
    for r in range(rounds):
        for entry in entries:
            text = entry.get("he", "")
            if not text:
                continue
            # sentence-level shuffle keeps meaning, changes word order statistics
            sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]
            if len(sentences) > 1 and r % 3 == 1:
                sentences = sentences[1:] + sentences[:1]
            body = " ".join(sentences)
            yield f"{rng.choice(LEADS)}{body}{rng.choice(CLOSERS)}"
            topic = entry.get("topic", "")
            if topic:
                yield chat(f"תן לי מידע על {topic}", body)
                yield chat(f"מה ידוע לך בנושא {topic}?", body)


# ============================================================== assembly ====
def build_corpus(size: str, seed: int, kb_path: Path) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    budget = {"small": 1, "medium": 8, "large": 40}.get(size, 4)

    sources: List[Tuple[str, Callable[[], Iterable[str]], int]] = [
        # weights favour *conversational* samples: the model must learn to
        # listen, think and answer — not to recite templated tool transcripts.
        ("knowledge", lambda: knowledge_source(kb_path), 3),
        ("paraphrase", lambda: paraphrase_source(kb_path, rng), 3),
        ("math", lambda: math_source(rng, 900 * budget), 2),
        ("code", lambda: code_source(rng, 350 * budget), 2),
        ("command", lambda: command_source(rng, 700 * budget), 2),
        ("dialogue", lambda: dialogue_source(rng, 900 * budget), 4),
        ("persona", lambda: persona_source(140 * budget), 3),
        ("reasoning", lambda: reasoning_source(rng, 250 * budget), 3),
        ("sentence", lambda: sentence_source(rng, 3000 * budget), 1),
    ]

    # Collect per source, keeping the weight each source deserves.
    per_source: List[Tuple[str, List[str], int]] = []
    counts: Dict[str, int] = {}
    for name, fn, weight in sources:
        items = list(fn())
        counts[name] = len(items)
        per_source.append((name, items, weight))

    # ── two defects fixed here, both of which made the published dev ppl
    # meaningless rather than merely optimistic.
    #
    # 1. Replication was applied before the split (`samples.extend(items * w)`),
    #    so identical strings landed on both sides. Measured on a medium corpus:
    #    103,732 rows held only 22,469 unique texts (78.3% duplicates, one text
    #    repeated 1,514 times) and 89.9% of dev rows also appeared in train.
    #    The model was scored on strings it had memorised, which is why ppl came
    #    out at 1.204 with train_loss 0.175 — that pair is the signature of
    #    memorisation, and the dev number was confirming it, not catching it.
    #
    # 2. Deduplication therefore has to happen *before* the split, on the text
    #    itself, so a given string can only ever belong to one side.
    #
    # Weighting is still applied — it is a legitimate way to emphasise
    # conversational samples over templated tool transcripts — but only to the
    # training side, after the held-out side has been carved off. Replicating a
    # string adds no information; it only inflates the gradient on that exact
    # string, which is precisely what teaches recitation instead of language.
    unique: Dict[str, None] = {}
    owner: Dict[str, int] = {}          # text -> source index (first wins)
    for si, (name, items, weight) in enumerate(per_source):
        for t in items:
            if t not in unique:
                unique[t] = None
                owner[t] = si

    texts = list(unique)
    rng.shuffle(texts)
    cut = max(1, int(len(texts) * 0.95))
    dev_texts, train_texts = texts[cut:], texts[:cut]

    train: List[str] = []
    for t in train_texts:
        train.extend([t] * per_source[owner[t]][2])
    rng.shuffle(train)
    dev = list(dev_texts)                # dev is never replicated

    print("[forge] sample counts per source:")
    for k, v in counts.items():
        print(f"        {k:10s} {v:7d}")
    dup = len(train) - len(train_texts)
    print(f"[forge] unique texts: {len(texts)}  (train {len(train_texts)} / dev {len(dev)})")
    print(f"[forge] train rows after weighting: {len(train)} "
          f"(+{dup} weighted repeats, dev untouched)")
    print(f"[forge] total (with weights): train={len(train)} dev={len(dev)}")
    return train, dev


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS corpus forge")
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(ROOT / "corpus" / "generated"))
    ap.add_argument("--kb", default=str(ROOT / "brain" / "knowledge_base.yaml"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, dev = build_corpus(args.size, args.seed, Path(args.kb))

    def dump(name: str, rows: Sequence[str]) -> Path:
        p = out / name
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
        mb = p.stat().st_size / 1e6
        print(f"[forge] wrote {p} ({len(rows)} rows, {mb:.2f} MB)")
        return p

    dump("train.jsonl", train)
    dump("dev.jsonl", dev)
    with (out / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump({"size": args.size, "seed": args.seed, "train": len(train), "dev": len(dev),
                   "chars": sum(len(s) for s in train)}, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
