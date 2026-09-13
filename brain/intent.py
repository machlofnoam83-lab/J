"""JARVIS Intent Router — the fast, deterministic half of understanding.

Order of resolution (cheapest and most certain first):
  1. explicit command patterns  -> skill tool-call
  2. arithmetic detection       -> math engine (never let the model compute)
  3. time/date requests         -> clock skills
  4. knowledge-base QA hit      -> grounded answer
  5. coding request             -> HEPHAESTUS agent
  6. memory-worthy statement    -> MNEMOSYNE write
  7. small talk / identity      -> persona
  8. otherwise                  -> neural core free generation

Every route returns a ``Route`` with a confidence and an explanation, so the
reasoning loop (and the HUD) can show *why* JARVIS chose what it chose.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.math_engine import MathError, extract_expression, try_evaluate  # noqa: E402

INTENTS = (
    "MATH", "TIME", "SYSTEM", "FILES", "APPS", "CODE", "KNOWLEDGE",
    "MEMORY_WRITE", "MEMORY_QUERY", "IDENTITY", "GREETING", "SMALLTALK",
    "HELP", "SAFETY", "UNKNOWN",
)


@dataclass
class Route:
    intent: str
    confidence: float
    reason: str
    skill: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    value: Any = None                 # resolved value when the route is exact
    reply_he: str = ""                # ready-made Hebrew reply if fully resolved
    grounded: bool = False            # answer came from a trusted local source
    agent: str = ""
    risk: str = "SAFE"

    def to_dict(self) -> Dict[str, Any]:
        return {"intent": self.intent, "confidence": round(self.confidence, 3),
                "reason": self.reason, "skill": self.skill, "args": self.args,
                "grounded": self.grounded, "agent": self.agent, "risk": self.risk,
                "reply_he": self.reply_he[:200]}


# ------------------------------------------------------------------ patterns --
_TIME_Q = re.compile(r"(מה (ה)?שעה|שעה עכשיו|what time|השעה כרגע|מה התאריך|איזה יום|what date|today's date)", re.I)
_HELP_Q = re.compile(r"(מה אתה (יודע|מסוגל|עושה|טוב)|מה (ה)?תפקיד(ך| שלך)|למה אתה מסוגל"
                     r"|עזרה|help|מה אפשר|היכולות שלך|הכישורים שלך"
                     r"|what can you do|what do you do|your (abilities|skills))", re.I)
_GREET = re.compile(r"^(שלום|היי|הי|אהלן|בוקר טוב|ערב טוב|צהריים טובים|hello|hi|hey|good (morning|evening))\b", re.I)
_THANKS = re.compile(r"(תודה|thanks|thank you|מעולה|יופי|all good)", re.I)
_IDENTITY = re.compile(r"(מי אתה|מה השם שלך|מי זה ג'רוויס|(ספר|תספר|תגיד) לי (על )?(עצמך|עליך)"
                       r"|על עצמך|מי אתה בכלל|הצג את עצמך"
                       r"|who are you|what are you|your name|(tell me )?about yourself"
                       r"|introduce yourself)", re.I)
_CODE = re.compile(
    # programming vocabulary. \bקוד\b (word-bounded) so "קודם" (=earlier) never matches.
    r"(תכתוב|כתוב לי|תקודד|קודד|תפתח|תבנה|בנה לי|תיצור|צור לי|ליישם|מימש|implement|compile|"
    r"fix|debug|refactor|באג|שגיאה בקוד|סקריפט|תוכנה|תכנות|אלגוריתם|פונקציה|פונקציות|מחלקה|"
    r"\bקוד\b|\bcode\b|פייתון|פייטון|python|javascript|typescript|json|regex|sql|html|css|"
    r"unit ?tests?|pytest|"
    r"write (me )?(a )?(function|script|class|code|program)|explain (this|the) code|"
    r"תסביר את הקוד|להריץ קוד|הרץ קוד)", re.I)
_MEM_WRITE = re.compile(r"(תזכור|תזכרי|זכור|remember that|אל תשכח|שים לב ש|העדפה שלי|קרא לי)", re.I)
_MEM_QUERY = re.compile(r"(מה (אמרתי|סיפרתי|ביקשתי)|זוכר (מה|את)|recall|מה דיברנו|לפני (שעה|יום|שבוע))", re.I)
_SYS_Q = re.compile(r"(מצב (ה)?מחשב|טלמטריה|cpu|ram|זיכרון פנוי|מעבד|דיסק|סוללה|temperature|"
                    r"system status|מה קורה עם המחשב|תהליכים|processes)", re.I)
_FILES = re.compile(r"(קובץ|קבצים|קבצי(?=[\s־\-.,!?]|$)|תיקי(?:יה|ית|ות|ה|ת)|תת־תיקייה"
                     r"|file|folder|directory|json|txt|csv|"
                     r"לקרוא את|לכתוב את|לשמור את|למחוק את|לחפש|חפש)", re.I)
_APPS = re.compile(r"(פתח את|סגור את|הפעל את|open |close |launch |start |kill |הרג את|מחשבון|דפדפן)", re.I)
_FILE_VERB = re.compile(r"(קרא|הצג|שמור|כתוב|מחק|ערוך|read|show|save|write|delete|edit|list)", re.I)
_SCREEN = re.compile(r"(צילום מסך|צלם את המסך|screenshot|capture the screen)", re.I)
_CLIP = re.compile(r"(לוח|clipboard|העתק|הדבק|copy|paste)", re.I)
_MEDIA = re.compile(r"(מוזיקה|ווליום|עוצמת קול|נגן|השהה|volume|music|pause|play|next)", re.I)
_REMIND = re.compile(r"(תזכיר לי|remind me|בעוד \d+ דקות|in \d+ minutes|שעון עצר|timer)", re.I)
_SAFETY = re.compile(r"(מסוכן|סיכון|אבטחה|הרשאות|dangerous|permission|security|kill switch)", re.I)
_MATH_HINT = re.compile(
    r"(כמה|חשב|calculate|compute|what is|כמה זה|מהו|אחוז|percent|שורש|sqrt|חזק|power|"
    r"ראשוני|prime|פיבונאצ'י|fibonacci|ממוצע|average|סכום|sum|מחלק|gcd|בינארי|binary|"
    r"עצרת|factorial|המרה|convert|\d)", re.I)
_MATH_STRICT = re.compile(r"^\s*[-+*/%^().,\d\s]+$|\d+\s*(?:[+\-*/%^]|\*\*)\s*\d+", re.I)


_EXT_WORDS = (
    (("פייתון", "python", "py"), "py"),
    (("ג'אווה סקריפט", "javascript", "js"), "js"),
    (("טקסט", "text", "txt"), "txt"),
    (("json", "ג'סון"), "json"),
    (("csv", "טבלה"), "csv"),
    (("תמונ", "image", "picture", "png", "jpg"), "png"),
    (("יומן", "log", "לוג"), "log"),
    (("yaml", "yml"), "yaml"),
    (("markdown", "md", "תיעוד"), "md"),
    (("html", "דף"), "html"),
)


def _num(x: Any) -> str:
    """Render a number the way a person would: 36.0 -> 36, 12.5 -> 12.5."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return f"{round(x, 6):g}"
    return str(x)


def _search_extension(text: str) -> str:
    """Map a spoken file type ("קבצי פייתון") onto the extension fs.search wants.

    Latin keywords are matched on word boundaries, otherwise "json" contains "js"
    and a JSON request would be searched as JavaScript.
    """
    low = (text or "").lower()
    for words, ext in _EXT_WORDS:
        for w in words:
            if w.isascii():
                if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low):
                    return ext
            elif w in low:
                return ext
    return ""


_PATH_HINT = re.compile(r"(?:בקובץ|לקובץ|את הקובץ|הקובץ|בתיקי+(?:ה|ת)|מתיקי+(?:ה|ת)"
                        r"|אל התיקי+(?:ה|ת)|את|של|file|path|folder|directory)\s+([^\s,;:()]+)", re.I)
# a bare token that is unmistakably a path: it has a separator or a file extension
_PATH_TOKEN = re.compile(r"([^\s,;:()\"'“”]+/[^\s,;:()\"'“”]+|"
                         r"[^\s,;:()\"'“”]+\.[A-Za-z]{1,6})(?![\w/])")


def _extract_path(text: str) -> str:
    """Pull the file/folder a request is about, keeping extensions and separators.

    Order matters: a quoted path, then any token that is *shaped* like a path
    ("docs/PLAN.md", "output.json"), and only then a hint word — otherwise the
    generic "את" in "כתוב לקובץ output.json את הטקסט" would win over the real
    target and we would try to open a file called "הטקסט".
    """
    t = text or ""
    m = re.search(r"['\"“]([^'\"”]{1,120})['\"”]", t)
    if m:
        return m.group(1).strip().rstrip(".")
    for cand in _PATH_TOKEN.findall(t):
        cand = cand.rstrip(".")
        if "/" in cand or re.search(r"\.[A-Za-z]{1,6}$", cand):
            return cand
    m = _PATH_HINT.search(t)
    if m:
        cand = m.group(1).rstrip(".")
        if len(cand) > 1 and not cand.endswith(("?", "!")):
            return cand
    return ""


def _looks_like_math(text: str) -> bool:
    if _MATH_STRICT.search(text):
        return True
    if not _MATH_HINT.search(text):
        return False
    digits = sum(c.isdigit() for c in text)
    return digits >= 1 and len(text) < 140


class IntentRouter:
    """Deterministic first-pass understanding. No model call required."""

    def __init__(self, knowledge=None, skills=None, config=None) -> None:
        self.knowledge = knowledge
        self.skills = skills
        self.config = config
        self.custom_rules: List[Tuple[re.Pattern, Callable[[re.Match, str], Route]]] = []

    # ------------------------------------------------------------ main entry --
    def route(self, text: str) -> Route:
        t = (text or "").strip()
        low = t.lower()
        if not t:
            return Route("UNKNOWN", 0.0, "empty input")

        # 1. clock and reminders carry digits but are NOT arithmetic
        if _TIME_Q.search(t):
            return Route("TIME", 0.95, "time/date question pattern", skill="time.now")
        if _REMIND.search(t):
            return Route("SYSTEM", 0.9, "reminder request", skill="scheduler.remind",
                         args=self._parse_reminder(t), risk="SAFE")

        # 2. arithmetic — the model must never guess numbers
        if _looks_like_math(t):
            r = self._route_math(t)
            if r:
                return r

        # 3. safety / permissions
        if _SAFETY.search(t):
            return Route("SAFETY", 0.8, "security or permission question")

        # 4. explicit system intents
        if _SCREEN.search(t):
            return Route("SYSTEM", 0.95, "screenshot request", skill="screen.capture", risk="WRITE")
        if _CLIP.search(t):
            skill = "clipboard.set" if re.search(r"(העתק|copy|שמור ללוח|כתוב ללוח)", t, re.I) else "clipboard.get"
            return Route("SYSTEM", 0.9, "clipboard request", skill=skill, risk="WRITE")
        if _MEDIA.search(t):
            return Route("SYSTEM", 0.85, "media control request", skill=self._media_skill(t), risk="SAFE")
        if _SYS_Q.search(t):
            return Route("SYSTEM", 0.93, "telemetry/process question", skill="sys.telemetry")

        # 5. apps and files (WRITE / CRITICAL)
        if _APPS.search(t):
            return self._route_app(t)
        if _FILES.search(t):
            return self._route_files(t)
        # a bare path plus a file verb needs no keyword: "קרא את docs/PLAN.md"
        if _FILE_VERB.search(t) and _extract_path(t):
            return self._route_files(t)

        # 6. memory
        if _MEM_WRITE.search(t):
            return Route("MEMORY_WRITE", 0.9, "explicit remember request",
                         args=self._parse_memory_write(t))
        if _MEM_QUERY.search(t):
            return Route("MEMORY_QUERY", 0.85, "recall request")

        # 7. code
        if _CODE.search(t):
            return Route("CODE", 0.88, "programming request", agent="hephaestus")

        # 8. help / identity / greeting / thanks
        if _HELP_Q.search(t):
            return Route("HELP", 0.9, "capability question")
        if _IDENTITY.search(t):
            return Route("IDENTITY", 0.95, "identity question")
        if _GREET.match(t):
            return Route("GREETING", 0.9, "greeting")
        if _THANKS.search(t) and len(t) < 30:
            return Route("SMALLTALK", 0.8, "gratitude")

        # 9. grounded knowledge lookup
        if self.knowledge is not None:
            qa = self.knowledge.search_qa(t, k=1, min_score=0.55)
            if qa:
                hit = qa[0]
                return Route("KNOWLEDGE", min(0.99, hit["score"]),
                             f"known QA pair ({hit['fact_id']})",
                             reply_he=hit["answer"], grounded=True, value=hit)
            hits = self.knowledge.search(t, k=1, min_score=0.22)
            if hits:
                return Route("KNOWLEDGE", min(0.9, 0.4 + hits[0]["score"]),
                             f"knowledge base entry {hits[0]['id']}",
                             reply_he=hits[0]["text_he"], grounded=True, value=hits[0])

        # 10. custom rules registered by subsystems
        for pattern, fn in self.custom_rules:
            m = pattern.search(t)
            if m:
                return fn(m, t)

        return Route("UNKNOWN", 0.25, "no deterministic route — neural core will answer")

    # ------------------------------------------------------------- add rule --
    def add_rule(self, pattern: str | re.Pattern, fn: Callable[[re.Match, str], Route]) -> None:
        self.custom_rules.append((re.compile(pattern) if isinstance(pattern, str) else pattern, fn))

    # ----------------------------------------------------------------- math --
    # Keyword -> skill. Checked BEFORE raw expression extraction, because
    # "השורש הריבועי של 144" would otherwise degrade to the bare number 144.
    KEYWORD_SKILLS: Tuple[Tuple[Tuple[str, ...], str, Dict[str, Any]], ...] = (
        (("שורש", "sqrt", "shoresh"), "math.sqrt", {}),
        (("ראשוני", "prime"), "math.is_prime", {}),
        (("פיבונאצ", "fibonacci", "fib"), "math.fib", {}),
        (("בינארי", "binary", "בסיס 2"), "math.base", {"to": 2}),
        (("הקסדצימלי", "hexadecimal", "בסיס 16"), "math.base", {"to": 16}),
        (("עצרת", "factorial"), "math.factorial", {}),
        (("ממוצע", "average", "mean"), "math.stats", {"op": "mean"}),
        (("מחלק משותף", "gcd"), "math.gcd", {}),
    )

    def _route_math(self, text: str) -> Optional[Route]:
        low = text.lower()

        # a) explicit function-call syntax, e.g. gcd(48, 18) / sqrt(144)
        m = re.search(r"\b(sqrt|abs|log|log2|log10|ln|exp|sin|cos|tan|gcd|lcm|hypot|"
                      r"factorial|isqrt|mean|median|min|max|fact)\s*\(([^()]*)\)", text, re.I)
        if m:
            expr = f"{m.group(1)}({m.group(2)})"
            value = try_evaluate(expr)
            if value is not None:
                return Route("MATH", 0.99, f"explicit function {expr}", skill="math.eval",
                             args={"expr": expr}, value=value,
                             reply_he=_phrase_math(text, expr, value), grounded=True)

        # b) two-argument helpers (percent, gcd from natural language)
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|אחוזים?|אחוז|percent|per cent)\s*"
                      r"(?:מ|מתוך|מן|of)?\s*[-–—־]?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m:
            p, of = float(m.group(1)), float(m.group(2))
            return Route("MATH", 0.97, "percentage", skill="math.percent", args={"p": p, "of": of},
                         value=p * of / 100.0,
                         reply_he=f"{_num(p)} אחוז מ־{_num(of)} הם {_num(p * of / 100.0)}.",
                         grounded=True)
        if ("מחלק משותף" in text or "gcd" in low) and re.search(r"\d+\D+\d+", text):
            nums = [int(x) for x in re.findall(r"\d+", text)][:2]
            if len(nums) == 2:
                from brain.math_engine import gcd as _gcd
                return Route("MATH", 0.96, "greatest common divisor", skill="math.gcd",
                             args={"a": nums[0], "b": nums[1]}, value=_gcd(*nums),
                             reply_he=f"המחלק המשותף המקסימלי של {nums[0]} ו־{nums[1]} הוא {_gcd(*nums)}.",
                             grounded=True)

        # c) keyword + single number
        for words, skill, extra in self.KEYWORD_SKILLS:
            if any(w in low for w in words):
                nums = re.findall(r"\d+(?:\.\d+)?", text)
                if nums:
                    arg = float(nums[0]) if "." in nums[0] else int(nums[0])
                    if skill == "math.stats":
                        args = {"values": [float(x) for x in nums], **extra}
                    elif skill == "math.gcd" and len(nums) >= 2:
                        args = {"a": int(nums[0]), "b": int(nums[1])}
                    else:
                        args = {"n": arg, "x": arg, **extra}
                    from brain.math_engine import MATH_OPS
                    op = skill.split(".", 1)[1]
                    try:
                        value = MATH_OPS[op](**args)
                    except Exception:
                        value = None
                    reply = _phrase_keyword_math(skill, args, value) if value is not None else ""
                    return Route("MATH", 0.97, f"keyword '{next(w for w in words if w in low)}' -> {skill}",
                                 skill=skill, args=args, value=value,
                                 reply_he=reply, grounded=value is not None)

        # d) plain arithmetic expression
        expr = extract_expression(text)
        if expr:
            value = try_evaluate(expr)
            if value is not None:
                return Route("MATH", 0.99, f"exact evaluation of {expr!r}",
                             skill="math.eval", args={"expr": expr}, value=value,
                             reply_he=_phrase_math(text, expr, value), grounded=True)
        return None

    # ----------------------------------------------------------------- apps --
    def _route_app(self, text: str) -> Route:
        close = bool(re.search(r"(סגור|כבה|kill|close|stop)", text, re.I))
        m = re.search(r"(?:את|the)?\s*([\w\u05d0-\u05ea .-]{2,40})$", text.strip().rstrip(".!?"))
        target = (m.group(1).strip() if m else text).strip()
        target = re.sub(r"^(פתח|סגור|הפעל|open|close|launch|start|kill)\s+(את)?\s*", "", target, flags=re.I).strip()
        return Route("APPS", 0.9, "application launch/close",
                     skill="sys.close_app" if close else "sys.launch_app",
                     args={"name": target or "unknown"},
                     risk="CRITICAL" if close else "WRITE")

    # ---------------------------------------------------------------- files --
    def _route_files(self, text: str) -> Route:
        path = _extract_path(text)
        # "what is in the folder" is a listing, not a read of one file
        if re.search(r"(מה יש|מה נמצא|רשימת (ה)?קבצים|הצג (את )?הקבצים|תוכן התיקייה"
                     r"|מה בתיקייה|list (the )?(files|folder|directory)|what('s| is) in|contents of)",
                     text, re.I):
            return Route("FILES", 0.9, "directory listing", skill="fs.tree",
                         args={"path": path or ".", "depth": 2}, risk="SAFE")
        if re.search(r"(מחק|delete|remove)", text, re.I):
            if not path:
                return Route("FILES", 0.9, "delete request without a target",
                             reply_he="איזה קובץ למחוק, אדוני? תן לי נתיב מדויק — מחיקה עוברת דרך חומת ההרשאות.",
                             grounded=True, risk="CRITICAL")
            return Route("FILES", 0.88, "file delete request", skill="fs.delete",
                         args={"path": path}, risk="CRITICAL")
        if re.search(r"(כתוב|שמור|write|save)", text, re.I):
            if not path:
                return Route("FILES", 0.9, "write request without a target",
                             reply_he="לאן לכתוב, אדוני? ציין נתיב קובץ ואת התוכן ואשמור אותו.",
                             grounded=True, risk="WRITE")
            return Route("FILES", 0.88, "file write request", skill="fs.write",
                         args={"path": path}, risk="WRITE")
        if re.search(r"(חפש|find|search)", text, re.I):
            # "find python files" must search for *.py — handing the whole Hebrew
            # sentence to fs.search as a glob matches nothing at all
            ext = _search_extension(text)
            return Route("FILES", 0.9, "file search request", skill="fs.search",
                         args={"pattern": path or "*", "extension": ext, "limit": 60}, risk="SAFE")
        if not path:
            # invoking fs.read with an empty path only yields "missing required
            # arg" — asking for the path is both friendlier and more honest
            return Route("FILES", 0.9, "read request without a target",
                         reply_he="איזה קובץ לקרוא, אדוני? תן לי נתיב (או שם תיקייה ואפרט את תוכנה).",
                         grounded=True, risk="SAFE")
        return Route("FILES", 0.85, "file read request", skill="fs.read",
                     args={"path": path}, risk="SAFE")

    # --------------------------------------------------------------- media --
    @staticmethod
    def _media_skill(text: str) -> str:
        if re.search(r"(הגבר|העלה|up|louder)", text, re.I):
            return "media.volume_up"
        if re.search(r"(הנמך|הורד|down|quieter)", text, re.I):
            return "media.volume_down"
        if re.search(r"(השהה|pause|עצור)", text, re.I):
            return "media.pause"
        if re.search(r"(נגן|play|המשך)", text, re.I):
            return "media.play"
        return "media.next"

    # ------------------------------------------------------------ reminders --
    @staticmethod
    def _parse_reminder(text: str) -> Dict[str, Any]:
        m = re.search(r"(\d+)\s*(דקות|minutes|שעות|hours|שנייה|seconds)", text, re.I)
        if not m:
            return {"minutes": 10, "message": text}
        n, unit = int(m.group(1)), m.group(2).lower()
        minutes = n if "דק" in unit or "min" in unit else (n * 60 if "שע" in unit or "hour" in unit else max(1, n // 60))
        msg = re.sub(r".*(שתזכיר|להזכיר|remind me|remind me to|remind me that)", "", text, flags=re.I).strip()
        return {"minutes": minutes, "message": msg or text}

    @staticmethod
    def _parse_memory_write(text: str) -> Dict[str, Any]:
        body = re.sub(r"^(תזכור|תזכרי|זכור|remember that|remember)\s*(ש)?", "", text.strip(), flags=re.I)
        return {"text": body or text}


def _phrase_keyword_math(skill: str, args: Dict[str, Any], value: Any) -> str:
    n = args.get("n", args.get("x", ""))
    if skill == "math.sqrt":
        return f"השורש הריבועי של {n} הוא {value}."
    if skill == "math.is_prime":
        return (f"כן, {n} הוא מספר ראשוני — הוא מתחלק רק ב־1 ובעצמו." if value
                else f"לא, {n} אינו מספר ראשוני — יש לו מחלקים נוספים.")
    if skill == "math.fib":
        return f"מספר פיבונאצ'י במקום {n} הוא {value}."
    if skill == "math.factorial":
        return f"העצרת של {n} היא {value}."
    if skill == "math.base":
        base = args.get("to", 2)
        name = {2: "בינארי", 8: "אוקטלי", 16: "הקסדצימלי"}.get(base, f"בסיס {base}")
        return f"{n} בבסיס {name} הוא {value}."
    if skill == "math.gcd":
        return f"המחלק המשותף המקסימלי של {args.get('a')} ו־{args.get('b')} הוא {value}."
    if skill == "math.stats":
        return f"הממוצע הוא {value}."
    return f"התוצאה היא {value}."


def _phrase_math(original: str, expr: str, value: Any) -> str:
    """Natural Hebrew phrasing for an exact computed answer."""
    shown = f"{value:,}".replace(",", ",") if isinstance(value, int) else _num(value)
    if "שורש" in original:
        return f"השורש הריבועי הוא {shown}. חושב במנוע המתמטיקה המדויק, לא בניחוש."
    if "אחוז" in original or "percent" in original.lower():
        return f"התוצאה היא {shown} אחוז."
    if "חזק" in original:
        return f"התוצאה היא {shown}."
    return f"{expr} שווה {shown}."
