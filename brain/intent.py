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
    "MATH", "TIME", "SYSTEM", "FILES", "APPS", "CODE", "KNOWLEDGE", "RAG",
    "VISION", "RECORDS", "PLAN", "MEMORY_WRITE", "MEMORY_QUERY", "IDENTITY",
    "GREETING", "SMALLTALK",
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
_TIME_Q = re.compile(r"(מה (ה)?שעה|שעה עכשיו|what time|השעה כרגע|מה התאריך|איזה תאריך|"
                     r"התאריך (היום|של היום)|איזה יום|מה היום|what date|today's date)", re.I)
_HELP_Q = re.compile(r"(מה אתה (יודע|מסוגל|עושה|טוב)|מה (ה)?תפקיד(ך| שלך)|למה אתה מסוגל"
                     r"|עזרה|help|מה אפשר|היכולות שלך|הכישורים שלך"
                     r"|what can you do|what do you do|your (abilities|skills))", re.I)
_GREET = re.compile(r"^(שלום|היי|הי|אהלן|בוקר טוב|ערב טוב|צהריים טובים|hello|hi|hey|good (morning|evening))\b", re.I)
_THANKS = re.compile(r"(תודה|thanks|thank you|מעולה|יופי|all good)", re.I)
# Everyday Hebrew openers. Before this pattern existed the router's only
# SMALLTALK rule was gratitude, so "מה איתך" and "מה קורה" — the two most
# natural things a person says to an assistant — fell through to UNKNOWN 0.25
# and drew the canned "I have no certain answer" refusal. ReasoningEngine's
# SMALLTALK_CATEGORIES already knew how to answer these; it was simply never
# reached, because the category lookup only runs *after* the router has
# already decided SMALLTALK. Routing and answering must agree.
_SMALLTALK = re.compile(
    r"(מה איתך|מה אצלך|מה קורה|מה חדש|מה העניינים|מה המצב"
    r"|איך עבר עליך|איך היום שלך|איך אתה מסתדר|איך היה היום שלך"
    r"|מה שלומך|מה נשמע|מה מצבך|איך אתה מרגיש"
    # Lines that were measured routing to UNKNOWN 0.25 and drawing the canned
    # refusal. ReasoningEngine has a canned answer for each of these; without a
    # pattern here the category is never named and the answer never runs.
    r"|עייף|עייפה|מותש|אין לי כוח"
    r"|יש לי רעיון|רעיון חדש|חשבתי על"
    r"|מה (ה)?תוכניות|מה יש לנו היום|מה מתוכנן|סדר (לי )?את היום"
    r"|קשה לי|לא בסדר|מרגיש רע|יום גרוע"
    r"|מתכון|איך מכינים"
    r"|מה יכולות"
    r"|what'?s up|how'?s it going|how are you"
    r"|I have an idea|got an idea|what'?s the plan|plans for today"
    r"|recipe|how do I (make|cook)"
    r"|so tired|exhausted|burnt out|having a hard time|bad day)", re.I)
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
_MEM_QUERY = re.compile(r"(מה (אמרתי|סיפרתי|ביקשתי)|זוכר (מה|את)|recall|מה דיברנו|לפני (שעה|יום|שבוע)|"
                        r"מה (הסיסמה|השם|העיר|הכתובת|הפרויקט|ההעדפה|ההעדפות) (שלי|של המשתמש)|"
                        r"איפה אני גר|היכן אני גר|מה שם (הכלב|החתול|הבן|הבת|הילד) שלי|"
                        r"מה אני (אוהב|מעדיף|לומד|עובד)|על איזה פרויקט אני עובד|"
                        r"what (did i say|is my (name|password))|where do i live)", re.I)

# Subset of `_MEM_QUERY` that is unambiguously a recall request — it names the act
# of having said something, not a topic. Checked ahead of the system intents
# because those match bare keywords at priority 4, and memory only ran at 6, so
# "מה אמרתי על מוזיקה" was claimed by `_MEDIA` and answered by skipping a track.
# Same shape as the CPU/telemetry hijack: a keyword route stealing a question it
# cannot answer. Only the explicit forms are promoted; the possessive ones
# ("הסיסמה שלי") stay at priority 6, because "הקובץ שלי" must still reach FILES.
_MEM_EXPLICIT = re.compile(
    r"(מה (אמרתי|סיפרתי|ביקשתי|שאלתי)|זוכר (מה|את)|מה דיברנו|recall|"
    r"what did i (say|ask|tell)|do you remember)", re.I)

# Mode switching. Routed before the research and system intents because "מצב
# מנתח" contains the word a telemetry question would use, and "מצב מורה" the word
# a definition would — the posture change must win over the thing it resembles.
_MODE_Q = re.compile(
    r"(מצב (חוקר|מחקר|מתכנת|קוד|תכנות|אבטחה|שומר|מנהל|בית|מזכיר|יומן|מנתח|ניתוח|"
    r"מורה|לימוד|כתיבה|סופר|מנצח|משימות|צופה|תצפית|בלשן|תרגום|שיחה|חבר|לווייה)|"
    r"(עבור|תעבור|עברי|switch|change|set) (לי )?(ל)?מצב|באיזה מצב אתה|מה המצב שלך|"
    r"אילו מצבים יש|רשימת מצבים|research mode|code mode|security mode)", re.I)

# The researcher. Deliberately requires an explicit dig verb or an "everything
# about" shape, so a plain question still takes its normal specialist route and
# the researcher stays a posture you ask for, not a grab-bag that swallows chat.
_RESEARCH_Q = re.compile(
    r"(תחקור|חקור|תחפש|חפש לי|חפשו לי|מה יש לך על|מה אתה יודע על|תמצא לי כל|"
    r"כל מה שיש (לך )?על|research|find (me )?(everything|all|anything) (about|on))",
    re.I)

# Shell execution. `shell.exec` was registered as CRITICAL and described in its own
# module as "the most dangerous skill JARVIS has, so it is the most heavily guarded
# one" — but no route ever emitted it, so the blocklist, the executable allowlist,
# the confirmation prompt and the audit trail were exercised only by tests. Asked to
# run "rm -rf /", JARVIS answered UNKNOWN: the guard never even saw the command.
# `shell.preview` existing as a SAFE sibling was the tell — the intended flow was
# preview -> confirm -> execute, and it was unreachable.
#
# Deliberately requires an explicit command marker ("פקודה"/"command"/"terminal").
# Matching a bare run-verb would let ordinary phrasing fall into arbitrary command
# execution, and the cost of a false negative here is only that JARVIS asks you to
# say "פקודה". "הרץ קוד" still belongs to CODE, which is checked separately.
_SHELL = re.compile(
    r"(הרץ את הפקודה|הרץ פקודה|הפעל את הפקודה|הפעל פקודה|בצע את הפקודה|בצע פקודה|"
    r"תריץ את הפקודה|תריץ פקודה|פקודת (של|shell)|בשורת הפקודה|"
    r"run (the |this )?command|execute (the |this )?command|run shell|"
    r"run (it )?in (the )?terminal|shell command)", re.I)

# Stripped from the utterance to leave the command itself. Ordered longest-first so
# "הרץ את הפקודה" is removed whole rather than leaving a stray "את" glued to the
# front of the command, which would make the allowlist check the wrong executable.
_SHELL_LEAD = re.compile(
    r"^\s*(בבקשה\s+)?(נא\s+)?("
    r"הרץ את הפקודה|הרץ פקודה|הפעל את הפקודה|הפעל פקודה|בצע את הפקודה|בצע פקודה|"
    r"תריץ את הפקודה|תריץ פקודה|הרץ את|הרץ|הפעל|בצע|תריץ|"
    r"run (the |this )?command|execute (the |this )?command|run shell|"
    r"run (it )?in (the )?terminal|shell command|run|execute"
    r")\s*[:\-]?\s*", re.I)

# Same alternation without the `^` anchor, for the case where the marker appears
# mid-utterance ("תוכל להרץ את הפקודה git status"). Kept as a separate compiled
# pattern rather than made optional inside _SHELL_LEAD, because the anchored form
# is what makes the common leading case unambiguous.
_SHELL_LEAD_UNANCHORED = re.compile(
    r"(בבקשה\s+)?(נא\s+)?("
    r"הרץ את הפקודה|הרץ פקודה|הפעל את הפקודה|הפעל פקודה|בצע את הפקודה|בצע פקודה|"
    r"תריץ את הפקודה|תריץ פקודה|הרץ את|הרץ|הפעל|בצע|תריץ|"
    r"run (the |this )?command|execute (the |this )?command|run shell|"
    r"run (it )?in (the )?terminal|shell command|run|execute"
    r")\s*[:\-]?\s*", re.I)
_SYS_Q = re.compile(r"(מצב (ה)?מחשב|טלמטריה|cpu|ram|זיכרון פנוי|מעבד|דיסק|סוללה|temperature|"
                    r"system status|מה קורה עם המחשב|תהליכים|processes)", re.I)

# A question about what a component *is* must not be answered with what that
# component is *currently doing*. `_SYS_Q` matches the bare nouns cpu/ram/דיסק/
# מעבד, so "מה זה CPU" used to route to sys.telemetry at 0.93 and reply "מעבד
# 2.0 אחוז, זיכרון 17.5 אחוז" — technically a true statement about the machine
# and a complete non-answer to the question, while KNOWLEDGE sat at priority 9
# holding the definition. Definitional phrasing therefore vetoes the telemetry
# route and lets the question fall through to the knowledge base.
_DEFINITION = re.compile(
    r"(מה זה|מה (הוא|היא) |מה זה אומר|מה הכוונה|הסבר|תסביר|הגדר|תגדיר|"
    r"מה ההבדל|מה היתרון|למה משמש|בשביל מה|what (is|are) |define |explain )",
    re.I,
)

# Narrower sibling of `_DEFINITION`, and the distinction is deliberate.
#
# `_DEFINITION` only ever *vetoes* the telemetry route, so a false positive is
# harmless — the question simply falls through to whatever matches next. It can
# therefore afford to include bare "הסבר/תסביר/explain".
#
# `_WHAT_IS` instead *claims* a route for the knowledge base, ahead of CODE and
# SYSTEM. A false positive there steals the question from a specialist that could
# actually handle it: "תסביר את הקוד" must reach HEPHAESTUS, not the KB. So this
# pattern is restricted to explicit definition requests, and even a match only
# wins when the knowledge base actually holds an answer — otherwise routing
# continues unchanged.
_WHAT_IS = re.compile(
    r"(מה זה|מה זה אומר|מה הכוונה (ב|של)|הגדר (לי )?|תגדיר|"
    r"מה ההבדל בין|מה היתרון (של|של)|למה משמש|בשביל מה (משמש|טוב)|"
    r"what (is|are) |what does .* mean|define )",
    re.I,
)
_FILES = re.compile(r"(קובץ|קבצים|קבצי(?=[\s־\-.,!?]|$)|תיקי(?:יה|ית|ות|ה|ת)|תת־תיקייה"
                     r"|file|folder|directory|json|txt|csv|"
                     r"לקרוא את|לכתוב את|לשמור את|למחוק את|לחפש|חפש)", re.I)
# Retrieval over the user's own files. Deliberately narrow: every branch requires
# an explicit "in my files / documents / folder", so this cannot hijack the
# filesystem intents ("קרא את הקובץ" stays a file read) or the knowledge base.
_RAG_ASK = re.compile(
    r"(חפש (לי )?ב(תוך )?(ה)?(קבצים|מסמכים|תיקייה|הערות)"
    r"|מצא (לי )?ב(תוך )?(ה)?(קבצים|מסמכים|תיקייה)"
    r"|מה כתוב ב(תוך )?(ה)?(קבצים|מסמכים)"
    r"|(לפי|מתוך|על פי) (ה)?(קבצים|מסמכים|הערות) (שלי|שלך)"
    r"|(ה)?(קבצים|מסמכים) (שלי|שלך) (אומרים|מראים)"
    r"|search (my|the|in my|in the) (files|documents|docs|notes)"
    r"|grep my (files|documents|notes)"
    r"|find (it |this )?in my (files|documents|notes)"
    r"|according to my (files|documents|notes)"
    r"|what (do|does) my (files|documents|notes) say)", re.I)
# Questions about the room itself. These are unambiguous — nobody says "מי מולי"
# meaning anything else — so they sit ahead of the security/permission branch,
# which would otherwise catch "מה מותר לי" and answer it from the persona
# instead of from the camera.
_VISION_Q = re.compile(
    r"(מי (מולי|מול המצלמה|זה|אתה רואה|נמצא (פה|כאן|מולי))"
    r"|מה (אתה רואה|יש מולך|המצלמה רואה|רואה מולי)"
    r"|(תאר|תסתכל על) (את )?(החדר|המצלמה)"
    r"|מי (רשום|אתה מכיר|הבעלים)"
    r"|(מה|איזו) (מותר לי|ההרשאה( שלי)?|רמת ההרשאה)"
    r"|למה (אתה לא מרשה|אסור לי)"
    r"|who (is (this|there|in front)|do you see|am i)"
    r"|what (do you see|level am i)"
    r"|list (the )?(faces|enrolled))", re.I)
# Enrolment is an act, not a question, so it is matched ahead of _VISION_Q —
# "רשום אותי" must not be answered with "there is someone in front of the camera".
# The name is extracted separately; the brain never invents one.
_ENROLL_Q = re.compile(
    r"(רשום (אותי|את הפנים|את הפרצוף|אותו|אותה|אותנו)"
    r"|תרשום (אותי|את הפנים|את הפרצוף)"
    r"|תכיר (אותי|את הפנים|את הפרצוף)"
    r"|תזכור (אותי|את הפנים|את הפרצוף)"
    r"|תוסיף (אותי|את הפנים) (לגלריה|לרשימה|לזיהוי)"
    r"|הרשמה (לזיהוי פנים|לפנים)"
    r"|אני רוצה להירשם"
    r"|(enroll|register|remember) (me|my face|this face)"
    r"|add (me|my face) to (the )?(gallery|faces))", re.I)

# Where a name can hide in the request. Deliberately explicit — a bare "אני"
# would happily capture the next verb as somebody's name.
_NAME_PATTERNS = (
    re.compile(r"(?:בשם|בשמי|שמי|שמי הוא|השם שלי(?: הוא)?|קוראים לי|קוראים לו|קוראים לה)"
               r"[\s:]+([^\s,.;:!?]+)", re.I),
    re.compile(r"\bmy name is\s+([^\s,.;:!?]+)", re.I),
    re.compile(r"\bcall me\s+([^\s,.;:!?]+)", re.I),
    re.compile(r"\bname (?:him|her|them|it)\s+([^\s,.;:!?]+)", re.I),
)

# ediyel records — the people dossier. Matched ahead of KNOWLEDGE and RAG,
# because "what do you know about Dana" is a question about a person you told
# JARVIS about, not a question about the world or about the files on disk.
_RECORDS_ADD = re.compile(
    r"(תוסיף (לרשומות|רשומה|אותו|אותה) (לרשומות)?)"
    r"|תשמור (רשומה|אותו|אותה)"
    r"|תתעד (את )?(האיש|הבחור|האישה|אותו|אותה)"
    r"|תפתח (רשומה|תיק) (על|עבור)"
    r"|add (?:a )?(?:record|dossier)|remember this person", re.I)

# "מי רשום" alone is ambiguous between the face gallery and the dossier, and the
# gallery wins: it is the thing that actually decides who may act. The dossier
# list needs the word "רשומות" to be claimed.
_RECORDS_LIST = re.compile(
    r"(מי רשום ברשומות|רשימת (ה)?אנשים|כל הרשומות|מי יש (לך )?ברשומות)"
    r"|(list (the )?people in (the )?records|who is in the records)", re.I)

_RECORDS_WHOAMI = re.compile(
    r"(מי אני|אתה יודע מי אני|תגיד לי מי אני)"
    r"|(who am i)", re.I)

# The negative lookahead matters. "ספר לי על עצמך" is a question about JARVIS,
# not a dossier lookup, and this branch runs well ahead of the IDENTITY one — so
# without it the assistant answers "tell me about yourself" by searching its own
# people file and reporting that it has no record of you.
_RECORDS_FIND = re.compile(
    r"(?:מה אתה יודע על|תגיד לי על|ספר לי על|מי זה|מי זאת|מה יש לך על|"
    r"רשומה (?:של|על)|who is|tell me about|what do you know about)"
    r"\s+(?!(?:עצמך|עליך|אותך|עצמי|yourself|you\b|me\b))", re.I)

_RAG_INDEX = re.compile(
    r"(תאנדקס|תסרוק את התיקייה|אנדקס את|בנה אינדקס|אינדוקס מחדש|רענן את האינדקס"
    r"|index (my|the) (files|documents|folder)"
    r"|rebuild the index|rescan my files|scan (my|the) folder)", re.I)
_RAG_STATUS = re.compile(
    r"(מה מצב האינדקס|כמה קבצים (אינדקסת|יש באינדקס)|מה יש באינדקס"
    r"|index status|how many files (are|is) indexed|what is indexed)", re.I)
_APPS = re.compile(r"(פתח את|סגור את|הפעל את|open |close |launch |start |kill |הרג את|מחשבון|דפדפן)", re.I)
_FILE_VERB = re.compile(r"(קרא|הצג|שמור|כתוב|מחק|ערוך|read|show|save|write|delete|edit|list)", re.I)
_SCREEN = re.compile(r"(צילום מסך|צלם (את )?(ה)?מסך|תצלם (את )?(ה)?מסך|המסך לצלם|"
                     r"screenshot|capture the screen|screen shot)", re.I)
_CLIP = re.compile(r"(לוח|clipboard|העתק|הדבק|copy|paste)", re.I)
_MEDIA = re.compile(r"(מוזיקה|שיר|רצועה|ווליום|עוצמת קול|נגן|השהה|"
                    r"volume|music|pause|play|next|previous|track|song)", re.I)
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


_UNIT_RE = None


def _unit_pattern():
    """One alternation over every known unit name, longest first.

    Longest-first matters: "מטר" is a substring of "קילומטר", so scanning
    left-to-right with short names first would match the tail of the longer unit
    and report two units where the user named one.
    """
    global _UNIT_RE
    if _UNIT_RE is None:
        from brain.math_engine import UNIT_ALIASES
        names = sorted(UNIT_ALIASES.keys(), key=len, reverse=True)
        _UNIT_RE = re.compile("|".join(re.escape(n) for n in names), re.I)
    return _UNIT_RE


def _extract_shell_command(text: str) -> str:
    """Pull the command out of a shell request, or return '' if there isn't one.

    `_SHELL` is unanchored but the lead-stripper is anchored with `^`, and using
    the two together let "מה זה run command" through: the marker matched mid-string,
    nothing was stripped, and the whole Hebrew question became the command. The
    firewall's allowlist would have blocked it — "מה" is not an executable — but it
    would have blocked it *after* raising a CRITICAL confirmation prompt asking the
    user to approve running "מה זה run command", which is a worse failure than not
    routing at all.

    So the marker is removed wherever it occurs, and what remains has to start with
    something an executable could plausibly look like: an ASCII letter or digit, a
    path separator, or a dot. Hebrew-only residue is rejected here rather than
    being handed to the firewall.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    cmd = _SHELL_LEAD_UNANCHORED.sub(" ", raw, count=1)
    cmd = re.sub(r"^\s*(בבקשה|נא|please|kindly)\s*[:\-]?\s*", "", cmd, flags=re.I)
    cmd = cmd.strip().strip(".!?;،،").strip()
    if not cmd:
        return ""
    # Take from the first character an executable could start with, rather than
    # rejecting on a Hebrew first character. Polite wrappers survive marker
    # stripping — "תוכל להרץ את הפקודה git status" leaves "תוכל ל git status" —
    # and rejecting that would drop a perfectly clear request. Commands are ASCII
    # in practice, so cutting to the first ASCII token keeps the command and drops
    # the wrapper. Pure-Hebrew residue such as "מה זה" has no ASCII at all and is
    # rejected here, before it can raise a CRITICAL confirmation prompt.
    m = re.search(r"[A-Za-z0-9_./\\~-]", cmd)
    if not m:
        return ""
    cmd = cmd[m.start():].strip().strip(".!?;").strip()
    return cmd


def _match_conversion(text: str) -> Optional[Tuple[float, str, str]]:
    """Pull (value, source_unit, target_unit) out of a conversion request.

    Requires two *distinct* units and a number. Demanding two units is what keeps
    this from swallowing ordinary quantity questions — "מה זה 5 קילו" names one
    unit and falls through untouched, while "המר 100 צלזיוס לפרנהייט" and
    "5 קילומטר במיילים" both name a source and a target. Source is the unit that
    appears first in the sentence, which is how both Hebrew and English phrase it.
    """
    units = [(m.start(), m.group(0)) for m in _unit_pattern().finditer(text)]
    if len(units) < 2:
        return None
    src = units[0][1]
    dst = next((u for _, u in units[1:] if u.lower() != src.lower()), None)
    if not dst:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        value = float(nums[0])
    except ValueError:
        return None
    return value, src, dst


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

        # 2a-i. enrolment. Ahead of the room questions because "רשום אותי" is an
        # act, not a question — routing it to vision.who would answer "there is
        # someone in front of the camera" and do nothing. CRITICAL because the
        # first face into an empty gallery becomes the owner with every
        # permission; the firewall asks a human before it runs.
        if _ENROLL_Q.search(t):
            name = ""
            for pat in _NAME_PATTERNS:
                m = pat.search(t)
                if m:
                    name = m.group(1).strip("״׳\"'")
                    break
            return Route("VISION", 0.93, "face enrolment request",
                         skill="vision.enroll", args={"name": name}, risk="CRITICAL")

        # 2a-ii. ediyel records. Ahead of KNOWLEDGE/RAG: "what do you know about
        # Dana" is about a person you told JARVIS about, not about the world.
        # Adding is WRITE and is routed with an empty name — the skill refuses
        # rather than inventing one, for the same reason enrolment does.
        if _RECORDS_WHOAMI.search(t):
            return Route("RECORDS", 0.9, "who am I", skill="records.whoami")
        if _RECORDS_LIST.search(t):
            return Route("RECORDS", 0.9, "list dossiers", skill="records.list")
        if _RECORDS_ADD.search(t):
            return Route("RECORDS", 0.88, "add a dossier", skill="records.add",
                         args={"name": ""}, risk="WRITE")
        if _RECORDS_FIND.search(t):
            m = re.search(
                r"(?:מה אתה יודע על|תגיד לי על|ספר לי על|מי זה|מי זאת|"
                r"מה יש לך על|רשומה (?:של|על)|who is|tell me about|"
                r"what do you know about)\s+([^\s?.!,;]+)", t, re.I)
            query = m.group(1).strip("״׳\"'") if m else ""
            return Route("RECORDS", 0.88 if query else 0.7, "look up a person",
                         skill="records.find", args={"query": query})

        # 2a. questions about the room. Ahead of SAFETY, because "מה מותר לי"
        # would otherwise be answered from the persona instead of from the camera
        # — and the camera is the thing that actually determines the answer.
        if _VISION_Q.search(t):
            skill, conf, why = "vision.who", 0.9, "who is present"
            if re.search(r"(מה אתה רואה|מה יש מולך|המצלמה רואה|תאר (את )?החדר|"
                         r"what do you see|describe)", t, re.I):
                skill, conf, why = "vision.scene", 0.9, "scene description"
            elif re.search(r"(מי רשום|מי אתה מכיר|מי הבעלים|רשימת הפנים|"
                           r"list (the )?(faces|enrolled)|who is (enrolled|the owner))",
                           t, re.I):
                skill, conf, why = "vision.gallery", 0.9, "gallery listing"
            elif re.search(r"(מותר לי|ההרשאה|רמת ההרשאה|למה (אתה לא מרשה|אסור)|"
                           r"what level|why not allowed|what am i allowed)", t, re.I):
                skill, conf, why = "vision.permission", 0.9, "active permission"
            return Route("VISION", conf, why, skill=skill,
                         args={"risk": "WRITE"} if skill == "vision.permission" else {})

        # 2b. retrieval over the user's own files (RAG). Sits above SAFETY and
        # the specialist intents because every branch here requires the user to
        # have explicitly scoped the question to their files ("מה כתוב בקבצים
        # על חומת הרשאות"). Without that scoping a security word would win and
        # the question would be answered from the persona instead of the source.
        # Index/status first: they are the more specific ask.
        if _RAG_INDEX.search(t):
            return Route("RAG", 0.92, "index rebuild request", skill="rag.index",
                         args={"roots": _extract_path(t), "force": True}, risk="WRITE")
        if _RAG_STATUS.search(t):
            return Route("RAG", 0.9, "index status request", skill="rag.status")
        if _RAG_ASK.search(t):
            return Route("RAG", 0.88, "local-file retrieval request",
                         skill="rag.ask", args={"query": t}, agent="argus")

        # 3. safety / permissions
        if _SAFETY.search(t):
            return Route("SAFETY", 0.8, "security or permission question")

        # 3b. an explicit definition request consults the knowledge base BEFORE
        # the specialist intents below. Those match on bare nouns — _SYS_Q on
        # cpu/ram/דיסק/מעבד, _CODE on באג/תוכנה/אלגוריתם — so without this a
        # question about what something *is* gets answered with what it is
        # currently *doing*: "מה זה CPU" returned telemetry, "מה זה באג" was
        # handed to HEPHAESTUS as a programming request. Both were confident and
        # both were non-answers.
        #
        # This only claims the route when the KB genuinely holds an answer. A
        # definitional question the KB cannot answer falls through untouched, so
        # "מה זה דיסק" with no disk entry still reaches whatever specialist
        # matches, exactly as before.
        if self.knowledge is not None and _WHAT_IS.search(t):
            qa = self.knowledge.search_qa(t, k=1, min_score=0.55)
            if qa:
                hit = qa[0]
                return Route("KNOWLEDGE", min(0.99, hit["score"]),
                             f"definition request, known QA pair ({hit['fact_id']})",
                             reply_he=hit["answer"], grounded=True, value=hit)

        # 3c. an explicit recall request outranks the keyword-based system intents
        # below. "מה אמרתי על מוזיקה" contains "מוזיקה", and `_MEDIA` runs at
        # priority 4 while memory ran at 6, so the question was answered by trying
        # to skip a track. Promoting only the unambiguous forms keeps "הקובץ שלי"
        # on the FILES path and "נגן מוזיקה" on the media path.
        if _MEM_EXPLICIT.search(t):
            return Route("MEMORY_QUERY", 0.9, "explicit recall request")

        # 3d. posture and research. Both are explicit requests about *how* JARVIS
        # should behave rather than about the world, so they sit above the subject
        # specialists: "מצב מנתח" must change the stance, not report CPU load.
        if _MODE_Q.search(t):
            if re.search(r"(באיזה מצב אתה|מה המצב שלך)", t, re.I):
                return Route("MODE", 0.9, "mode status question", skill="mode.get")
            if re.search(r"(אילו מצבים יש|רשימת מצבים)", t, re.I):
                return Route("MODE", 0.9, "mode catalogue request", skill="mode.list")
            return Route("MODE", 0.92, "mode switch request",
                         skill="mode.set", args={"mode": t})
        if _RESEARCH_Q.search(t):
            return Route("RESEARCH", 0.88, "research request",
                         skill="research.query", args={"topic": t})

        # 4. explicit system intents
        # 4a. shell command. Routed at CRITICAL so the firewall runs check_shell
        # (destructive-pattern blocklist, then executable allowlist) and then
        # demands explicit human confirmation in the HUD before anything runs.
        # Falls through when stripping the marker leaves nothing, so "מה זה
        # run command" cannot become an execution request.
        if _SHELL.search(t):
            cmd = _extract_shell_command(t)
            if cmd:
                return Route("SYSTEM", 0.95, "shell command request",
                             skill="shell.exec", args={"command": cmd}, risk="CRITICAL")

        if _SCREEN.search(t):
            return Route("SYSTEM", 0.95, "screenshot request", skill="screen.capture", risk="WRITE")
        if _CLIP.search(t):
            skill = "clipboard.set" if re.search(r"(העתק|copy|שמור ללוח|כתוב ללוח)", t, re.I) else "clipboard.get"
            return Route("SYSTEM", 0.9, "clipboard request", skill=skill, risk="WRITE")
        if _MEDIA.search(t):
            return Route("SYSTEM", 0.85, "media control request", skill=self._media_skill(t), risk="SAFE")
        if _SYS_Q.search(t) and not _DEFINITION.search(t):
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
        # Short social openers, checked after the specific ones so that a
        # greeting or an identity question still wins when both match.
        if _SMALLTALK.search(t) and len(t) < 40:
            return Route("SMALLTALK", 0.82, "social opener")

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
        (("מחלק משותף", "מחלק המשותף", "gcd"), "math.gcd", {}),
    )

    def _route_math(self, text: str) -> Optional[Route]:
        low = text.lower()

        # a0) unit conversion. math.convert was registered and described in
        # Hebrew, but nothing ever routed to it — the keyword table below covers
        # sqrt/prime/fib/base/factorial/stats/gcd and has no conversion entry.
        # The only caller that supplied a source unit was forge_corpus.py, so the
        # training corpus taught a tool call the deterministic router could not
        # emit and the neural core does not reliably emit either. The tool was
        # dead on arrival, and worse, it failed with a message that looked like an
        # unsupported conversion rather than a missing argument.
        conv = _match_conversion(text)
        if conv is not None:
            value, src, dst = conv
            from brain.math_engine import canonical_unit, convert as _convert_units
            out = _convert_units(value, src, dst)
            if out is not None:
                cs, cd = canonical_unit(src), canonical_unit(dst)
                return Route("MATH", 0.97, f"unit conversion {src}->{dst}",
                             skill="math.convert",
                             args={"v": value, "from": src, "to": dst},
                             value=out,
                             reply_he=f"{_num(value)} {cs} הם {_num(out)} {cd}.",
                             grounded=True)

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

        # b0) Two questions the engine previously got wrong in the worst way —
        # it answered a *different* question at confidence 0.99.
        #
        # "מה השארית של 17 חלקי 5" answered 3.4. That is the quotient. The
        # question asked for the remainder, which is 2. WORD_OPS maps שארית to
        # the % operator, but no rule recognised the phrasing, so the generic
        # arithmetic path parsed "17 חלקי 5" and answered division confidently.
        m = re.search(r"ה?שארית[^\d]{0,12}(\d+(?:\.\d+)?)\s*(?:חלקי|ב|מ|mod|מודולו)?\s*(\d+(?:\.\d+)?)",
                      text, re.I)
        if m:
            a_, b_ = float(m.group(1)), float(m.group(2))
            if b_:
                v = a_ % b_
                return Route("MATH", 0.99, "remainder", skill="math.eval",
                             args={"expr": f"{_num(a_)} % {_num(b_)}"}, value=v,
                             reply_he=f"השארית של {_num(a_)} חלקי {_num(b_)} היא {_num(v)}.",
                             grounded=True)

        # "כמה אחוז זה 25 מתוך 200" — the inverse of the percent rule below. It
        # used to fall through to UNKNOWN, and the fallback then offered an
        # unrelated knowledge-base fact about cache memory as though it were an
        # answer. Wrong *and* dressed as grounded is worse than either alone.
        m = re.search(r"כמה\s*(?:אחוז|אחוזים|%)\s*(?:זה|הוא)?\s*(\d+(?:\.\d+)?)"
                      r"\s*(?:מתוך|מ|מן| out of |of)\s*(\d+(?:\.\d+)?)", text, re.I)
        if not m:
            m = re.search(r"what\s+percent(?:age)?\s+(?:is|of)\s+(\d+(?:\.\d+)?)"
                          r"\s*(?:out of|of)\s*(\d+(?:\.\d+)?)", text, re.I)
        if m:
            part, whole = float(m.group(1)), float(m.group(2))
            if whole:
                v = part / whole * 100.0
                return Route("MATH", 0.99, "percent inverse", skill="math.eval",
                             args={"expr": f"{_num(part)}/{_num(whole)}*100"}, value=v,
                             reply_he=f"{_num(part)} מתוך {_num(whole)} הם {_num(v)} אחוז.",
                             grounded=True)

        # b) two-argument helpers (percent, gcd from natural language)
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|אחוזים?|אחוז|percent|per cent)\s*"
                      r"(?:מ|מתוך|מן|of)?\s*[-–—־]?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m:
            # Guard: this rule answers *only* the percent part. If the sentence
            # carries more arithmetic — "15% מ־240 ותוסיף 30" — returning here
            # silently discards the rest and answers 36 to a question whose
            # answer is 66, at confidence 0.97. A confident wrong number is
            # worse than no number. Let the general evaluator take it instead;
            # it parses the whole expression and gets 66.
            tail = text[m.end():]
            head = text[:m.start()]
            leftover = re.sub(r"[\s.,;:!?·׃]", "", head + tail)
            leftover = re.sub(r"(?i)^(כמה|מה|תחשב|חשב|תן|calculate|compute|what|howmuch|is)", "", leftover)
            extra_ops = bool(re.search(
                r"(?:ועוד|ותוסיף|תוסיף|ומוסיף|הוסף|פחות|ותחסר|תחסר|מינוס|כפול|ותכפול|תכפול"
                r"|חלקי|ותחלק|תחלק|\+|-|\*|/)", leftover))
            if not extra_ops:
                p, of = float(m.group(1)), float(m.group(2))
                return Route("MATH", 0.97, "percentage", skill="math.percent", args={"p": p, "of": of},
                             value=p * of / 100.0,
                             reply_he=f"{_num(p)} אחוז מ־{_num(of)} הם {_num(p * of / 100.0)}.",
                             grounded=True)
        if (re.search(r"מחלק\s+(?:ה)?משותף", text) or "gcd" in low) and re.search(r"\d+\D+\d+", text):
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
        # previous-track had no branch at all, so "השיר הקודם" fell through to the
        # `return` below and skipped *forward* — the opposite of what was asked,
        # while media.prev sat registered and unreachable. Checked before play
        # because "נגן את השיר הקודם" contains both verbs.
        if re.search(r"(הקודם|הקודמת|previous|prev\b|back)", text, re.I):
            return "media.prev"
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
