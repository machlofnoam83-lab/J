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

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
