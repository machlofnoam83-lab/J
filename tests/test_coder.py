"""HEPHAESTUS — the coder agent must write, RUN and VERIFY real code."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.hephaestus import HephaestusAgent, classify_error, repair, run_python  # noqa: E402

TASKS = [
    ("תכתוב פונקציה שבודקת אם מספר ראשוני", "prime"),
    ("פונקציה שמחשבת מספר פיבונאצ׳י", "fibonacci"),
    ("בדוק אם מחרוזת היא פלינדרום", "palindrome"),
    ("ספירת תדירות מילים בטקסט", "counts"),
    ("פירוק מספר לגורמים ראשוניים", "factors"),
    ("חישוב ממוצע וחציון של רשימת מספרים", "stats"),
    ("טעינת json מקובץ בצורה בטוחה", "json"),
    ("למיין רשימת מילונים לפי מפתח", "sort"),
    ("סריקת עץ הקבצים והחזרת רשימה", "scan"),
    ("הפוך את סדר המילים במשפט", "reverse"),
]

BROKEN = [
    ("import maths\n\n\ndef area(r):\n    return math.pi * r ** 2\n\n\nprint(area(3))\n",
     "ImportError", "a misspelled stdlib module (maths -> math)"),
    ("def f(x)\n    return x\n", "Syntax", "missing colon"),
    ("def g(a, b):\n    return a / b\n\n\nprint(g(1, 0))\n", "ZeroDivision", "division by zero"),
    ("d = {'a': 1}\nprint(d['b'])\n", "KeyError", "missing key"),
    ("xs = [1, 2]\nprint(xs[5])\n", "IndexError", "out of range"),
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

    print("== sandbox executes real code ==")
    r = run_python("print(6 * 7)")
    check("runs and captures stdout", r.ok and r.stdout.strip() == "42", repr(r.stdout))
    r = run_python("raise ValueError('boom')")
    check("captures stderr and classifies", not r.ok and r.error_kind == "ValueError", r.error_kind)
    r = run_python("while True: pass", timeout=2.0)
    check("enforces timeout", not r.ok and r.error_kind == "Timeout", r.error_kind)
    r = run_python("import os\nprint(os.environ.get('JARVIS_SANDBOX'))")
    check("sandbox env is isolated", r.ok and r.stdout.strip().endswith("sandbox"), r.stdout.strip()[:60])

    print("\n== writes, runs and verifies tasks ==")
    agent = HephaestusAgent()
    for task, expect_pattern in TASKS:
        res = agent.write_and_verify(task)
        good = res.ok and res.runs and any(run.ok for run in res.runs)
        check(f"{task[:36]}", good,
              f"pattern={res.pattern} repairs={res.repairs} runs={len(res.runs)}")
        if not good and res.runs:
            print(f"        stderr: {res.runs[-1].stderr[:160]}")

    print("\n== self-repair on broken code ==")
    for code, want_kind, why in BROKEN:
        first = run_python(code)
        check(f"detects {want_kind} ({why})", first.error_kind == want_kind,
              f"got {first.error_kind}")
        patched, note = repair(code, first)
        second = run_python(patched)
        check(f"repairs {want_kind}", second.ok or second.error_kind != want_kind,
              f"note={note[:60]} -> ok={second.ok} kind={second.error_kind}")

    print("\n== explain / analyze ==")
    info = agent.explain("import math\n\n\ndef area(r: float) -> float:\n"
                         "    \"\"\"Circle area.\"\"\"\n    return math.pi * r ** 2\n")
    check("finds the function", "area" in info["functions"], str(info["functions"]))
    check("finds the import", any("math" in str(i) for i in info["imports"]), str(info["imports"]))
    check("extracts the signature", info.get("signatures") and
          info["signatures"][0]["returns"] == "float", str(info.get("signatures"))[:90])
    check("parses cleanly", info["parses"] is True)

    print("\n== procedural memory integration ==")
    try:
        from brain.memory import MemoryPalace
        import tempfile
        tmpdb = Path(tempfile.mkdtemp()) / "mem.sqlite3"
        mem = MemoryPalace(tmpdb)
        agent2 = HephaestusAgent(memory=mem)
        task = "פונקציה שבודקת אם מספר הוא ראשוני"
        res = agent2.write_and_verify(task)
        check("first solve succeeded", res.ok)
        skills = mem._conn.execute("SELECT COUNT(*) c FROM skills").fetchone()["c"]
        check("skill was learned", skills >= 1, f"({skills} skills stored)")
        hit = mem.find_skill(task)
        check("skill is retrievable", hit is not None and "is_prime" in (hit or {}).get("solution", ""),
              f"score={(hit or {}).get('score')}")
    except Exception as exc:
        check("memory integration", False, f"{type(exc).__name__}: {exc}")

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
