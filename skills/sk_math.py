"""Math skills — thin, safe wrappers over the deterministic math engine.

Registered so the neural core can emit ``{"tool": "math.eval", ...}`` and have
it land on a real, exact calculator instead of on its own arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain import math_engine as ME  # noqa: E402
from skills.registry import REGISTRY, SkillResult  # noqa: E402


def _wrap(fn, phrase):
    def inner(**kwargs: Any) -> SkillResult:
        try:
            value = fn(**kwargs)
        except ME.MathError as exc:
            return SkillResult(ok=False, error=f"מנוע המתמטיקה סירב: {exc}")
        except Exception as exc:
            return SkillResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        return SkillResult(ok=True, value=value, data={"summary_he": phrase(value, kwargs)})
    return inner


@REGISTRY.register("math.eval", risk="SAFE", agent="jarvis",
                   description_he="מחשב ביטוי מתמטי במדויק (כולל ניסוח עברי)",
                   required=("expr",), triggers_he=("כמה זה", "חשב"))
def _eval(expr: str = "", **kw: Any) -> SkillResult:
    try:
        value = ME.evaluate(str(expr))
    except ME.MathError as exc:
        return SkillResult(ok=False, error=f"לא הצלחתי לחשב את {expr!r}: {exc}")
    return SkillResult(ok=True, value=value,
                       data={"summary_he": f"{expr} שווה {value}.", "expr": expr})


@REGISTRY.register("math.sqrt", risk="SAFE", agent="jarvis",
                   description_he="שורש ריבועי מדויק", required=("x",))
def _sqrt(x: Any = 0, **kw: Any) -> SkillResult:
    try:
        v = ME.evaluate(f"sqrt({x})")
    except ME.MathError as exc:
        return SkillResult(ok=False, error=str(exc))
    return SkillResult(ok=True, value=v, data={"summary_he": f"השורש הריבועי של {x} הוא {v}."})


@REGISTRY.register("math.is_prime", risk="SAFE", agent="jarvis",
                   description_he="בדיקת ראשוניות", required=("n",))
def _is_prime(n: int = 0, **kw: Any) -> SkillResult:
    v = ME.is_prime(int(n))
    return SkillResult(ok=True, value=v, data={
        "summary_he": (f"כן, {n} הוא מספר ראשוני." if v else f"לא, {n} אינו מספר ראשוני."),
        "factors": ME.prime_factors(int(n))[:12]})


@REGISTRY.register("math.prime_factors", risk="SAFE", agent="jarvis",
                   description_he="פירוק לגורמים ראשוניים", required=("n",))
def _factors(n: int = 0, **kw: Any) -> SkillResult:
    f = ME.prime_factors(int(n))
    return SkillResult(ok=True, value=f, data={"summary_he": f"הגורמים הראשוניים של {n} הם {', '.join(map(str, f))}."})


@REGISTRY.register("math.fib", risk="SAFE", agent="jarvis",
                   description_he="מספר פיבונאצ'י (F(0)=0)", required=("n",))
def _fib(n: int = 0, **kw: Any) -> SkillResult:
    v = ME.fib(int(n))
    return SkillResult(ok=True, value=v, data={"summary_he": f"מספר פיבונאצ'י במקום {n} הוא {v}."})


@REGISTRY.register("math.gcd", risk="SAFE", agent="jarvis",
                   description_he="מחלק משותף מקסימלי", required=("a", "b"))
def _gcd(a: int = 0, b: int = 0, **kw: Any) -> SkillResult:
    v = ME.gcd(a, b)
    return SkillResult(ok=True, value=v, data={"summary_he": f"המחלק המשותף המקסימלי של {a} ו־{b} הוא {v}."})


@REGISTRY.register("math.percent", risk="SAFE", agent="jarvis",
                   description_he="אחוזים", required=("p", "of"))
def _percent(p: float = 0, of: float = 0, **kw: Any) -> SkillResult:
    v = ME.percent(p, of)
    return SkillResult(ok=True, value=v, data={"summary_he": f"{p} אחוז מ־{of} הם {v}."})


@REGISTRY.register("math.base", risk="SAFE", agent="jarvis",
                   description_he="המרת בסיס (בינארי/אוקטלי/הקסדצימלי)", required=("n", "to"))
def _base(n: int = 0, to: int = 2, **kw: Any) -> SkillResult:
    v = ME.to_base(int(n), int(to))
    names = {2: "בינארי", 8: "אוקטלי", 16: "הקסדצימלי"}
    return SkillResult(ok=True, value=v,
                       data={"summary_he": f"{n} בבסיס {names.get(int(to), to)} הוא {v}."})


@REGISTRY.register("math.factorial", risk="SAFE", agent="jarvis",
                   description_he="עצרת", required=("n",))
def _factorial(n: int = 0, **kw: Any) -> SkillResult:
    v = ME.evaluate(f"factorial({n})")
    return SkillResult(ok=True, value=v, data={"summary_he": f"העצרת של {n} היא {v}."})


@REGISTRY.register("math.convert", risk="SAFE", agent="jarvis",
                   description_he="המרת יחידות (אורך, משקל, זמן, נפח מידע, טמפרטורה)",
                   required=("v", "to"))
def _convert(v: float = 0, to: str = "", **kw: Any) -> SkillResult:
    # `from` is a Python keyword, so it can only ever arrive through **kw — and a
    # caller that spells it `source`, `src`, `unit` or `from_unit` is not wrong,
    # just different. Previously only "from"/"src" were read; anything else left
    # the source empty, the conversion returned None, and the error printed an
    # empty slot: "אין לי המרה מ־ ל־c". That message described a missing argument
    # as though it were an unsupported one, which sends a user down the wrong path.
    src = ""
    for key in ("from", "src", "source", "from_unit", "unit", "in"):
        if kw.get(key):
            src = str(kw[key])
            break
    if not src:
        return SkillResult(ok=False,
                           error="חסרה יחידת המקור. צריך לומר למשל 'המר 100 צלזיוס לפרנהייט' "
                                 "— כלומר ערך, יחידת מקור ויחידת יעד.")
    out = ME.convert(v, src, to)
    if out is None:
        return SkillResult(ok=False,
                           error=f"אין לי המרה מ־{src} ל־{to}. אני ממיר אורך, משקל, זמן, "
                                 f"נפח מידע וטמפרטורה.")
    cs, cd = ME.canonical_unit(src), ME.canonical_unit(to)
    return SkillResult(ok=True, value=out,
                       data={"summary_he": f"{v} {cs} הם {out} {cd}.",
                             "from": cs, "to": cd})


@REGISTRY.register("math.stats", risk="SAFE", agent="jarvis",
                   description_he="סטטיסטיקה: ממוצע, חציון, סכום, מינימום, מקסימום, סטיית תקן",
                   required=("values",))
def _stats(values: Any = (), op: str = "mean", **kw: Any) -> SkillResult:
    try:
        v = ME.stats(ME.coerce_values(values), str(op))
    except ME.MathError as exc:
        return SkillResult(ok=False, error=str(exc))
    names = {"mean": "הממוצע", "median": "החציון", "sum": "הסכום",
             "min": "המינימום", "max": "המקסימום", "stdev": "סטיית התקן"}
    return SkillResult(ok=True, value=v,
                       data={"summary_he": f"{names.get(op, op)} הוא {v}."})
