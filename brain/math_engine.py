"""JARVIS deterministic math engine.

The neural model is *never* allowed to do arithmetic. Every number that leaves
JARVIS goes through this evaluator: a hand-written recursive-descent parser
with an AST, exact rational-friendly float arithmetic, guarded exponentiation,
Hebrew + English word operators, unit conversion and small statistics.

    >>> evaluate("17 * 23")
    391
    >>> evaluate("sqrt(144)")
    12.0
    >>> evaluate("שורש של 144")
    12.0
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

Number = Union[int, float]


class MathError(ValueError):
    """Raised for any expression this engine refuses to evaluate."""


# ------------------------------------------------------------------ tokens --
_TOKEN_RE = re.compile(
    r"""
      \s*(?:
        (?P<num>\d+\.\d+|\d+\.|\.\d+|\d+)
      | (?P<name>[A-Za-z_\u05d0-\u05ea][A-Za-z_\u05d0-\u05ea0-9]*)
      | (?P<op>\*\*|//|[-+*/%^(),!\[\]])
      )
    """,
    re.VERBOSE,
)

HEBREW_NUMBERS: Dict[str, int] = {
    "אפס": 0, "אחד": 1, "אחת": 1, "שתיים": 2, "שניים": 2, "שלוש": 3, "ארבע": 4,
    "חמש": 5, "שש": 6, "שבע": 7, "שמונה": 8, "תשע": 9, "עשר": 10,
    "אחת עשרה": 11, "אחד עשר": 11, "שתים עשרה": 12, "שנים עשר": 12,
    "עשרים": 20, "שלושים": 30, "ארבעים": 40, "חמישים": 50, "שישים": 60,
    "שבעים": 70, "שמונים": 80, "תשעים": 90, "מאה": 100, "מאתיים": 200,
    "אלף": 1000, "מיליון": 1000000,
}

WORD_OPS: Dict[str, str] = {
    "ועוד": "+", "פלוס": "+", "חיבור": "+",
    "פחות": "-", "מינוס": "-", "החסרה": "-",
    "כפול": "*", "פעמים": "*", "מכפלה": "*",
    "חלקי": "/", "חלוקה": "/", "מחולק": "/",
    "בחזקת": "**", "חזקת": "**",
    "שורש": "sqrt", "שורש ריבועי": "sqrt",
    "ריבוע": "**2", "שלישית": "**3",
    "עצרת": "!", "שארית": "%", "מודולו": "%",
}

FUNCTIONS: Dict[str, Callable[..., Number]] = {
    "sqrt": math.sqrt, "shoresh": math.sqrt,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "log2": math.log2, "log10": math.log10, "ln": math.log,
    "exp": math.exp, "pow": lambda a, b: _guarded_pow(a, b),
    "gcd": math.gcd, "lcm": lambda a, b: abs(a * b) // math.gcd(a, b) if a and b else 0,
    "hypot": math.hypot, "degrees": math.degrees, "radians": math.radians,
    "factorial": math.factorial, "isqrt": math.isqrt,
    "mean": lambda *xs: sum(xs) / len(xs) if xs else 0,
    "median": _median if False else None,  # filled below (ordering)
}

CONSTANTS: Dict[str, Number] = {
    "pi": math.pi, "פאי": math.pi, "e": math.e, "tau": math.tau,
    "phi": (1 + math.sqrt(5)) / 2,
    "c": 299_792_458,          # speed of light m/s
    "g": 9.80665,              # standard gravity m/s^2
    "h": 6.62607015e-34,       # Planck constant
    "inf": math.inf,
}

MAX_EXPONENT = 512        # refuse 9**9**9 style blowups
MAX_FACTORIAL = 10000


def _median(*xs: Number) -> Number:
    if not xs:
        return 0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


FUNCTIONS["median"] = _median


def _guarded_pow(a: Number, b: Number) -> Number:
    if abs(b) > MAX_EXPONENT:
        raise MathError(f"exponent too large ({b}) — refusing")
    if a == 0 and b < 0:
        raise MathError("division by zero (0 to a negative power)")
    res = a ** b
    if isinstance(res, complex):
        raise MathError("complex result — not supported by the real math engine")
    if isinstance(res, float) and (math.isnan(res) or math.isinf(res)):
        raise MathError("result overflowed")
    return res


# ------------------------------------------------------------------ lexer ---
@dataclass
class Tok:
    kind: str      # num | name | op
    value: Any
    pos: int


def preprocess(text: str) -> str:
    """Normalise Hebrew/English maths phrasing into a parseable expression."""
    t = text.strip()
    t = t.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    t = t.replace("⁻", "-").replace("²", "**2").replace("³", "**3")
    # strip ONLY thousands separators (1,000,000) — a comma between function
    # arguments such as gcd(48, 18) must survive
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    t = re.sub(r"^.*?=\s*", "", t) if t.count("=") == 1 and t.strip().endswith("?") else t
    t = t.replace("?", "").replace("כמה זה", "").replace("כמה זה?", "")
    t = t.replace("מהו", "").replace("מה זה", "").replace("חשב", "").replace("calculate", "")
    t = t.replace("של", " ").replace("equals", "=")
    # longest-first so "שורש ריבועי" wins over "שורש"
    for word in sorted(WORD_OPS, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(word)}\b", " " + WORD_OPS[word] + " ", t)
    # sqrt of X  ->  sqrt(X)
    t = re.sub(r"(sqrt|shoresh)\s+([\d.]+)", r"\1(\2)", t)
    # "10 percent of 200" / "10 אחוז מ 200"
    t = re.sub(r"([\d.]+)\s*(?:percent|אחוז|%)\s*(?:of|מ|מתוך)\s*([\d.]+)", r"(\1*\2/100)", t)
    t = re.sub(r"([\d.]+)\s*(?:percent|אחוז|%)", r"(\1/100)", t)
    # x squared / cubed
    t = re.sub(r"([\d.]+)\s*(?:squared|בריבוע)", r"(\1**2)", t)
    t = re.sub(r"([\d.]+)\s*(?:cubed|בשלישית)", r"(\1**3)", t)
    # n factorial
    t = re.sub(r"([\d.]+)\s*(?:factorial|עצרת)", r"fact(\1)", t)
    t = re.sub(r"\b([a-z_]+)\s+of\s+([\d.()]+)", r"\1(\2)", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def tokenize(text: str) -> List[Tok]:
    toks: List[Tok] = []
    i = 0
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if not m or m.end() == m.start():
            if text[i].isspace():
                i += 1
                continue
            raise MathError(f"unexpected character {text[i]!r} at {i}")
        i = m.end()
        if m.group("num") is not None:
            v = m.group("num")
            toks.append(Tok("num", float(v) if "." in v else int(v), m.start()))
        elif m.group("name") is not None:
            toks.append(Tok("name", m.group("name"), m.start()))
        else:
            toks.append(Tok("op", m.group("op"), m.start()))
    return toks


# ------------------------------------------------------------------ parser --
class _Parser:
    """Recursive descent:  expr -> term (('+'|'-') term)*
                           term -> factor (('*'|'/'|'%'|implicit) factor)*
                           factor -> unary ('**' factor)?      (right assoc)
                           unary  -> ('-'|'+') unary | primary
                           primary-> num | name | name '(' args ')' | '(' expr ')'
    """

    def __init__(self, toks: List[Tok]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> Optional[Tok]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> Tok:
        t = self.peek()
        if t is None:
            raise MathError("unexpected end of expression")
        self.i += 1
        return t

    def expect(self, kind: str, value: Any = None) -> Tok:
        t = self.next()
        if t.kind != kind or (value is not None and t.value != value):
            raise MathError(f"expected {value or kind}, got {t.value!r}")
        return t

    # ------------------------------------------------------------- rules --
    def parse(self) -> Number:
        val = self.expr()
        if self.peek() is not None:
            raise MathError(f"trailing token {self.peek().value!r}")
        return val

    def expr(self) -> Number:
        val = self.term()
        while (t := self.peek()) and t.kind == "op" and t.value in ("+", "-"):
            self.next()
            rhs = self.term()
            val = val + rhs if t.value == "+" else val - rhs
        return val

    def term(self) -> Number:
        val = self.factor()
        while True:
            t = self.peek()
            if t is None:
                break
            if t.kind == "op" and t.value in ("*", "/", "%"):
                self.next()
                rhs = self.factor()
                if t.value == "*":
                    val = val * rhs
                elif t.value == "/":
                    if rhs == 0:
                        raise MathError("division by zero")
                    val = val / rhs
                else:
                    if rhs == 0:
                        raise MathError("modulo by zero")
                    val = val % rhs
            elif t.kind in ("num", "name") or (t.kind == "op" and t.value == "("):
                # implicit multiplication: 2(3+4), 3 pi, 2x
                val = val * self.factor()
            else:
                break
        return val

    def factor(self) -> Number:
        val = self.unary()
        t = self.peek()
        if t and t.kind == "op" and t.value == "**":
            self.next()
            exp = self.factor()          # right-associative
            return _guarded_pow(val, exp)
        if t and t.kind == "op" and t.value == "^":
            self.next()
            return _guarded_pow(val, self.factor())
        return val

    def unary(self) -> Number:
        t = self.peek()
        if t and t.kind == "op" and t.value in ("-", "+"):
            self.next()
            v = self.unary()
            return -v if t.value == "-" else v
        return self.primary()

    def primary(self) -> Number:
        t = self.next()
        if t.kind == "num":
            v = t.value
        elif t.kind == "name":
            name = str(t.value).lower()
            nxt = self.peek()
            if nxt and nxt.kind == "op" and nxt.value == "(":
                self.next()
                args: List[Number] = []
                if not (self.peek() and self.peek().kind == "op" and self.peek().value == ")"):
                    args.append(self.expr())
                    while self.peek() and self.peek().kind == "op" and self.peek().value == ",":
                        self.next()
                        args.append(self.expr())
                self.expect("op", ")")
                v = self._call(name, args)
            elif name in CONSTANTS:
                v = CONSTANTS[name]
            elif name in HEBREW_NUMBERS:
                v = HEBREW_NUMBERS[name]
            elif name.startswith("fact"):
                v = self._call("factorial", [])
            else:
                raise MathError(f"unknown name {name!r}")
        elif t.kind == "op" and t.value == "(":
            v = self.expr()
            self.expect("op", ")")
        else:
            raise MathError(f"unexpected token {t.value!r}")

        nt = self.peek()
        if nt and nt.kind == "op" and nt.value == "!":
            self.next()
            v = self._call("factorial", [v])
        return v

    def _call(self, name: str, args: List[Number]) -> Number:
        if name == "fact":
            name = "factorial"
        if name == "factorial":
            if not args:
                raise MathError("factorial needs an argument")
            n = args[0]
            if n != int(n) or n < 0:
                raise MathError("factorial requires a non-negative integer")
            if n > MAX_FACTORIAL:
                raise MathError(f"factorial too large ({n})")
            return float(math.factorial(int(n)))
        fn = FUNCTIONS.get(name)
        if fn is None:
            raise MathError(f"unknown function {name!r}")
        try:
            res = fn(*args)
        except MathError:
            raise
        except ZeroDivisionError:
            raise MathError("division by zero")
        except ValueError as exc:
            raise MathError(f"{name}: {exc}")
        if isinstance(res, float) and (math.isnan(res) or math.isinf(res)):
            raise MathError(f"{name} produced a non-finite result")
        return res


# ------------------------------------------------------------------- API ----
def evaluate(expression: str, strict: bool = True) -> Number:
    """Evaluate a (possibly Hebrew-phrased) arithmetic expression exactly."""
    prepared = preprocess(expression)
    if not prepared:
        raise MathError("empty expression")
    value = _Parser(tokenize(prepared)).parse()
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    if isinstance(value, float):
        return round(value, 12)
    return value


def try_evaluate(expression: str) -> Optional[Number]:
    try:
        return evaluate(expression)
    except (MathError, RecursionError, OverflowError):
        return None


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def prime_factors(n: int) -> List[int]:
    n, out, d = abs(int(n)), [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(max(0, int(n))):
        a, b = b, a + b
    return a


def gcd(a: int, b: int) -> int:
    return math.gcd(int(a), int(b))


def percent(p: Number, of: Number) -> Number:
    return round(float(p) * float(of) / 100.0, 6)


def to_base(n: int, base: int) -> str:
    n, base = int(n), int(base)
    if base == 2:
        return bin(n)[2:]
    if base == 8:
        return oct(n)[2:]
    if base == 16:
        return hex(n)[2:].upper()
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    neg = n < 0
    n = abs(n)
    out = []
    while n:
        out.append(digits[n % base])
        n //= base
    return ("-" if neg else "") + "".join(reversed(out))


def convert(value: Number, src: str, dst: str) -> Optional[Number]:
    """Unit conversion for the units JARVIS talks about most."""
    table: Dict[str, Dict[str, float]] = {
        "length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                   "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
        "mass": {"mg": 1e-6, "g": 0.001, "kg": 1.0, "t": 1000.0, "lb": 0.45359237, "oz": 0.028349523},
        "time": {"ms": 0.001, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0},
        "data": {"b": 1.0, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
                 "kib": 1024.0, "mib": 1024.0**2, "gib": 1024.0**3},
    }
    s, d = src.lower().strip(), dst.lower().strip()
    if s in ("c", "צלזיוס", "celsius") and d in ("f", "פרנהייט", "fahrenheit"):
        return round(float(value) * 9 / 5 + 32, 6)
    if s in ("f", "פרנהייט", "fahrenheit") and d in ("c", "צלזיוס", "celsius"):
        return round((float(value) - 32) * 5 / 9, 6)
    if s in ("c", "צלזיוס", "celsius") and d in ("k", "קלווין", "kelvin"):
        return round(float(value) + 273.15, 6)
    if s in ("k", "קלווין", "kelvin") and d in ("c", "צלזיוס", "celsius"):
        return round(float(value) - 273.15, 6)
    for group in table.values():
        if s in group and d in group:
            return round(float(value) * group[s] / group[d], 9)
    return None


def stats(values: List[Number], op: str = "mean") -> Number:
    xs = [float(v) for v in values]
    if not xs:
        raise MathError("no values")
    if op == "mean":
        return round(sum(xs) / len(xs), 9)
    if op == "sum":
        return round(sum(xs), 9)
    if op == "min":
        return min(xs)
    if op == "max":
        return max(xs)
    if op == "median":
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if op == "stdev":
        m = sum(xs) / len(xs)
        return round(math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)), 9)
    raise MathError(f"unknown stats op {op!r}")


def extract_expression(text: str) -> Optional[str]:
    """Pull the maths out of a natural-language Hebrew/English request."""
    cleaned = preprocess(text)
    if re.search(r"\d", cleaned) and re.search(r"[-+*/%()\w]", cleaned):
        # trim leading/trailing noise words that survived preprocessing
        cleaned = re.sub(r"^[^\d(.-]+", "", cleaned).strip()
        cleaned = re.sub(r"[^\d).%]+$", "", cleaned).strip()
        if cleaned:
            return cleaned
    return None


# Dispatcher used by the skill registry -----------------------------------
# Every op accepts **kwargs and picks the aliases it understands, so a caller
# (intent router, tool-call JSON, skill registry) can pass `n`, `x`, `value`
# or `expr` interchangeably without a TypeError.
def _first(kw: Dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in kw and kw[n] is not None:
            return kw[n]
    raise MathError(f"missing argument, expected one of {names}")


MATH_OPS: Dict[str, Callable[..., Any]] = {
    "eval": lambda **kw: evaluate(str(_first(kw, "expr", "expression", "text", "n"))),
    "sqrt": lambda **kw: evaluate(f"sqrt({_first(kw, 'x', 'n', 'value')})"),
    "pow": lambda **kw: evaluate(f"({_first(kw, 'a', 'base')})**({_first(kw, 'b', 'exp', 'power')})"),
    "is_prime": lambda **kw: is_prime(int(_first(kw, "n", "x", "value"))),
    "prime_factors": lambda **kw: prime_factors(int(_first(kw, "n", "x", "value"))),
    "fib": lambda **kw: fib(int(_first(kw, "n", "x", "value", "index"))),
    "gcd": lambda **kw: gcd(int(_first(kw, "a", "x")), int(_first(kw, "b", "y"))),
    "percent": lambda **kw: percent(_first(kw, "p", "percent", "rate"), _first(kw, "of", "value", "total")),
    "base": lambda **kw: to_base(int(_first(kw, "n", "x", "value")), int(_first(kw, "to", "base"))),
    "convert": lambda **kw: convert(_first(kw, "v", "value", "amount"),
                                    str(kw.get("from", kw.get("src", ""))),
                                    str(kw.get("to", kw.get("dst", "")))),
    "stats": lambda **kw: stats(list(_first(kw, "values", "nums", "xs")), str(kw.get("op", "mean"))),
    "factorial": lambda **kw: evaluate(f"factorial({_first(kw, 'n', 'x', 'value')})"),
}


def call(op: str, **kwargs: Any) -> Any:
    if op not in MATH_OPS:
        raise MathError(f"unknown math op {op!r}")
    return MATH_OPS[op](**kwargs)


if __name__ == "__main__":
    cases = [
        ("17 * 23", 391), ("2 ** 10", 1024), ("sqrt(144)", 12),
        ("שורש של 144", 12), ("כמה זה 15 כפול 4", 60),
        ("10 אחוז מ 200", 20.0), ("5!", 120), ("100 / 4", 25),
        ("2 + 3 * 4", 14), ("(2 + 3) * 4", 20), ("2 ** 3 ** 2", 512),
        ("log10(1000)", 3), ("gcd(48, 18)", 6), ("3.5 + 1.5", 5),
    ]
    bad = 0
    for text, want in cases:
        got = try_evaluate(text)
        flag = "OK " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"  {flag} {text!r:24} -> {got!r} (want {want!r})")
    print(f"\nmath engine: {len(cases)-bad}/{len(cases)} passed")
    raise SystemExit(1 if bad else 0)
