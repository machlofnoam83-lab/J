"""Intent router + math engine conformance tests (Hebrew & English)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.intent import IntentRouter  # noqa: E402
from brain.knowledge import KnowledgeStore  # noqa: E402
from brain.math_engine import evaluate, is_prime, prime_factors, to_base, convert  # noqa: E402

MATH_CASES = [
    ("כמה זה 17 כפול 23", "MATH", "math.eval", 391),
    ("מה השורש הריבועי של 144", "MATH", "math.sqrt", 12),
    ("10 אחוז מ 200", "MATH", "math.percent", 20.0),
    ("האם 97 ראשוני", "MATH", "math.is_prime", True),
    ("gcd(48,18)", "MATH", "math.eval", 6),
    ("sqrt(144)", "MATH", "math.eval", 12),
    ("מספר פיבונאצ'י במקום 20", "MATH", "math.fib", 6765),  # F(0)=0 convention
    ("העצרת של 6", "MATH", "math.factorial", 720),
    ("המר 255 לבינארי", "MATH", "math.base", "11111111"),
    ("2 ** 10", "MATH", "math.eval", 1024),
    ("מה השורש של 625", "MATH", "math.sqrt", 25),
    ("100 חלקי 8", "MATH", "math.eval", 12.5),
]

ROUTE_CASES = [
    ("מה השעה", "TIME", "time.now"),
    ("מי אתה", "IDENTITY", ""),
    ("תכתוב לי פונקציה שבודקת ראשוניות", "CODE", ""),
    ("פתח את המחשבון", "APPS", "sys.launch_app"),
    ("סגור את הדפדפן", "APPS", "sys.close_app"),
    ("מחק את הקובץ tmp.txt", "FILES", "fs.delete"),
    ("קרא את הקובץ notes.txt", "FILES", "fs.read"),
    ("חפש קבצים בשם report", "FILES", "fs.search"),
    ("מה מצב המחשב", "SYSTEM", "sys.telemetry"),
    ("צלם את המסך", "SYSTEM", "screen.capture"),
    ("תזכור שאני מעדיף type hints", "MEMORY_WRITE", ""),
    ("מה דיברנו אתמול", "MEMORY_QUERY", ""),
    ("בוקר טוב", "GREETING", ""),
    ("תודה", "SMALLTALK", ""),
    ("איך אתה מדבר בלי API", "KNOWLEDGE", ""),
    ("מה זה רקורסיה", "KNOWLEDGE", ""),
    ("זה לא מסוכן לתת לך גישה?", "SAFETY", ""),
    ("תזכיר לי בעוד 20 דקות", "SYSTEM", "scheduler.remind"),
    ("הגבר עוצמת קול", "SYSTEM", "media.volume_up"),
]

ENGINE_CASES = [
    ("17 * 23", 391), ("2 ** 3 ** 2", 512), ("(2 + 3) * 4", 20),
    ("100 / 8", 12.5), ("log10(1000)", 3), ("5!", 120), ("gcd(48, 18)", 6),
    ("sqrt(2)", round(2 ** 0.5, 12)), ("10 אחוז מ 50", 5.0),
    ("שורש של 81", 9), ("3 pi", round(3 * 3.141592653589793, 12)),
]


def main() -> int:
    ok = fail = 0

    def check(label: str, cond: bool, info: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {label} {info}")
        else:
            fail += 1
            print(f"  FAIL  {label} {info}")

    print("== math engine (direct) ==")
    for expr, want in ENGINE_CASES:
        got = evaluate(expr)
        check(f"evaluate({expr!r})", got == want, f"-> {got!r} want {want!r}")

    check("is_prime(2)", is_prime(2) is True)
    check("is_prime(91)", is_prime(91) is False, "(7*13)")
    check("prime_factors(360)", prime_factors(360) == [2, 2, 2, 3, 3, 5])
    check("to_base(255,16)", to_base(255, 16) == "FF")
    check("convert(100,C,F)", convert(100, "C", "F") == 212.0)
    check("convert(5,km,m)", convert(5, "km", "m") == 5000.0)

    print("\n== intent router: math ==")
    router = IntentRouter(knowledge=KnowledgeStore())
    for text, intent, skill, want in MATH_CASES:
        r = router.route(text)
        good = r.intent == intent and r.value == want
        check(f"{text!r}", good, f"-> {r.intent}/{r.skill}={r.value!r} want {skill}={want!r}")

    print("\n== intent router: routing ==")
    for text, intent, skill in ROUTE_CASES:
        r = router.route(text)
        good = r.intent == intent and (not skill or r.skill == skill)
        check(f"{text!r}", good, f"-> {r.intent}/{r.skill} conf={r.confidence:.2f}")

    print("\n== grounding ==")
    r = router.route("איך אתה מדבר בלי API")
    check("knowledge answer is grounded", r.grounded and bool(r.reply_he))
    r = router.route("כמה זה 17 כפול 23")
    check("math answer is grounded", r.grounded and bool(r.reply_he))
    r = router.route("מה מספר הטלפון של השכן שלי")
    check("unknown question is NOT grounded", not r.grounded, f"(intent={r.intent})")

    # ---------------------------------------------------------------- honesty --
    print("\n== honesty guards (regressions that were real bugs) ==")
    from brain.knowledge import content_tokens, shares_topic  # noqa: E402
    from brain.intent import _extract_path, _search_extension  # noqa: E402
    kb = router.knowledge

    # char-n-gram cosine alone let "ספר לי על חורים שחורים" match "ספר לי בדיחה"
    # on the shared "tell me" prefix (0.575) and publish a joke as grounded truth
    r = router.route("ספר לי על חורים שחורים")
    check("an off-topic question is not answered with the joke",
          "Halloween" not in (r.reply_he or "") and not (r.grounded and r.intent == "KNOWLEDGE"),
          f"(intent={r.intent} grounded={r.grounded} reply={(r.reply_he or '')[:36]!r})")
    r = router.route("ספר לי בדיחה")
    check("an explicit joke request still gets the joke",
          r.grounded and "Halloween" in (r.reply_he or ""), f"(intent={r.intent})")

    check("shares_topic rejects a prefix-only match",
          not shares_topic("ספר לי על חורים שחורים", "ספר לי בדיחה"))
    check("shares_topic accepts a real topical match",
          shares_topic("מה זה טוקנייזר", "מה זה טוקנייזר BPE"))
    check("shares_topic sees Hebrew inflection", shares_topic("חורים שחורים", "החור השחור"))
    check("shares_topic works in English",
          shares_topic("tell me about black holes", "black hole physics"))
    check("shares_topic separates unrelated topics", not shares_topic("תודה", "מה השעה"))
    check("content_tokens strips the request frame",
          "ספר" not in content_tokens("ספר לי על חורים שחורים"))
    check("content_tokens keeps ספר when it means 'book'",
          "ספר" in content_tokens("ספר לי על הספר שקראתי"))
    check("content_tokens drops pure politeness", content_tokens("תודה") == set())

    # some KB answers are skill templates; showing one raw leaks "{time}" to the user
    qa = kb.search_qa("מה השעה?", k=3, min_score=0.1)
    check("a {placeholder} template is never handed out as an answer",
          all("{" not in h["answer"] for h in qa), f"({[h['answer'][:24] for h in qa]})")

    # percentages with a dash or a Hebrew maqaaf used to miss the math engine
    for text, want in (("כמה זה 15 אחוז מ-240", 36.0), ("15% מ-240", 36.0),
                       ("כמה זה 20 אחוזים מתוך 350", 70.0), ("10 אחוז מ 200", 20.0)):
        r = router.route(text)
        check(f"percent: {text!r}", r.intent == "MATH" and r.value == want,
              f"-> {r.intent}/{r.value} want {want}")

    # identity / capability phrasings that used to fall through to UNKNOWN
    for text, want in (("ספר לי על עצמך", "IDENTITY"), ("tell me about yourself", "IDENTITY"),
                       ("מה התפקיד שלך", "HELP"), ("מה הכישורים שלך", "HELP"),
                       ("what do you do", "HELP")):
        r = router.route(text)
        check(f"{text!r} -> {want}", r.intent == want, f"(got {r.intent} conf={r.confidence:.2f})")

    # files: listing vs reading vs asking for the missing target
    r = router.route("מה יש בתיקייה")
    check("a listing question uses fs.tree", r.intent == "FILES" and r.skill == "fs.tree",
          f"-> {r.intent}/{r.skill}")
    r = router.route("מחק קובץ")
    check("a delete with no target asks instead of invoking with an empty path",
          r.intent == "FILES" and not r.skill and "איזה קובץ" in (r.reply_he or ""),
          f"-> skill={r.skill!r} reply={(r.reply_he or '')[:30]!r}")
    r = router.route("קרא את הקובץ README.md")
    check("a read keeps the file extension",
          r.intent == "FILES" and r.skill == "fs.read" and r.args.get("path") == "README.md",
          f"-> {r.args}")
    r = router.route("קרא את docs/PLAN.md")
    check("a bare path plus a verb is a file request",
          r.intent == "FILES" and r.args.get("path") == "docs/PLAN.md", f"-> {r.intent}/{r.args}")
    r = router.route("חפש קבצי פייתון")
    check("'find python files' searches *.py, not the whole sentence",
          r.intent == "FILES" and r.skill == "fs.search"
          and r.args.get("extension") == "py" and r.args.get("pattern") == "*", f"-> {r.args}")

    for text, want in (("קרא את docs/PLAN.md", "docs/PLAN.md"),
                       ("מה יש בתיקיית docs", "docs"),
                       ('קרא את "my file.txt"', "my file.txt"),
                       ("כתוב לקובץ output.json את הטקסט", "output.json"),
                       ("פתח את notes.txt.", "notes.txt"),
                       ("read the file config.yaml please", "config.yaml"),
                       ("כמה זה 15 אחוז מ-240", ""),
                       ("חפש קבצי פייתון", ""),
                       ("מה יש בתיקייה", "")):
        got = _extract_path(text)
        check(f"_extract_path({text!r})", got == want, f"-> {got!r} want {want!r}")
    check("_search_extension maps spoken file types", _search_extension("חפש קבצי json") == "json",
          f"-> {_search_extension('חפש קבצי json')!r}")
    check("_search_extension returns '' when no type is named",
          _search_extension("חפש קבצים") == "")

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
