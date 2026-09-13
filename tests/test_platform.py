#!/usr/bin/env python3
"""JARVIS platform tests — the encoding trap that broke the Windows self-test.

On a Hebrew Windows install the console codepage is cp862/cp1255, not UTF-8. Every
child process therefore wrote Hebrew in that codepage while `tests/run_all.py`
decoded its output as UTF-8, which raised UnicodeDecodeError inside
subprocess._readerthread and killed the pipe. The suite then reported
"no RESULT line (exit 1)" for *every* module — including modules that had passed.

The fix has three layers, and each is asserted here:
  1. children are forced to UTF-8 (PYTHONUTF8 / PYTHONIOENCODING in their env),
  2. the parent decodes with errors="replace", so no byte can abort a run,
  3. the .bat launchers switch the console with `chcp 65001` and export the same
     variables, for the human reading the output as much as for Python.

Run:  python tests/test_platform.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ok = 0
fail = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {label}" + (f"  {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  ✗ {label}" + (f"  {detail}" if detail else ""))


def run(script: str, env_extra: dict | None = None, **kw):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), env=env,
                          capture_output=True, timeout=120, **kw)


# Bytes a Hebrew Windows console really produces: "שלום" in cp1255, and a cp862
# box-drawing byte. Neither is valid UTF-8, which is what used to explode.
CP1255_SHALOM = b"\xf9\xec\xe5\xed"
CP862_BYTE = b"\xf9"


def test_decoder_survives():
    print("\n── a child speaking cp1255 must not break the parent ──")
    script = ("import sys;"
              "sys.stdout.buffer.write(b'RESULT: 3 passed, 0 failed\\n');"
              f"sys.stdout.buffer.write({CP1255_SHALOM!r} + b'\\n');"
              "sys.stdout.flush()")

    # the old behaviour: strict UTF-8 decoding of a non-UTF-8 pipe
    try:
        p = run(script, text=True, encoding="utf-8")
        strict_ok, err = True, ""
    except UnicodeDecodeError as exc:
        strict_ok, err = False, str(exc)[:70]
    except Exception as exc:
        strict_ok, err = False, f"{type(exc).__name__}: {exc}"[:70]
    check("strict UTF-8 decoding does raise on cp1255 bytes (the reported bug)",
          not strict_ok, err or "no error raised")

    # the fixed behaviour: same bytes, errors="replace"
    p = run(script, text=True, encoding="utf-8", errors="replace")
    check("errors='replace' captures the same output without raising", p.returncode == 0)
    # Keep the echoed text out of the detail: this module's own output is parsed by
    # tests/run_all.py for a RESULT line, and printing the fixture's one verbatim
    # used to be read as this module's summary (3 passed instead of 23).
    survived = "RESULT: 3 passed, 0 failed" in (p.stdout or "")
    check("a RESULT line written before the bad bytes still arrives intact",
          survived, f"first line intact={survived}, bytes after it={len(p.stdout or '')}")
    check("the undecodable bytes became replacement characters, not an exception",
          "\ufffd" in (p.stdout or "") or len(p.stdout or "") > 30)

    # and with the child forced to UTF-8, the Hebrew arrives readable
    p = run("import sys; sys.stdout.write('שלום אדוני\\n'); sys.stdout.flush()",
            env_extra={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            text=True, encoding="utf-8", errors="replace")
    check("a child forced to UTF-8 delivers Hebrew undamaged",
          "שלום אדוני" in (p.stdout or ""), repr((p.stdout or "").strip()[:40]))

    # stderr must be survivable too: that is where tracebacks land
    p = run("import sys; sys.stderr.buffer.write(b'boom \\xf9\\n'); sys.exit(1)",
            text=True, encoding="utf-8", errors="replace")
    check("stderr with a stray cp862 byte is captured, not fatal",
          p.returncode == 1 and "boom" in (p.stderr or ""))


def test_runner_is_hardened():
    print("\n── tests/run_all.py is hardened ──")
    src = (ROOT / "tests" / "run_all.py").read_text(encoding="utf-8")
    check("children are told to speak UTF-8", 'PYTHONIOENCODING": "utf-8"' in src
          and 'PYTHONUTF8": "1"' in src)
    check("the child environment is actually passed to subprocess.run",
          "env=CHILD_ENV" in src)
    check("decoding can never abort a run", 'errors="replace"' in src)
    check("the runner reconfigures its own stdout so Hebrew prints on cp862",
          "reconfigure(encoding=\"utf-8\"" in src)
    # importing it must not execute the suite
    check("CHILD_ENV exists and forces UTF-8 at import time",
          "CHILD_ENV = {" in src)


def test_launchers_are_hardened():
    print("\n── Windows launchers switch the console to UTF-8 ──")
    for name in ("tools/setup_windows.bat", "JARVIS.bat"):
        p = ROOT / name
        check(f"{name} exists", p.exists())
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        check(f"{name} selects codepage 65001", "chcp 65001" in src)
        check(f"{name} exports PYTHONUTF8", "set PYTHONUTF8=1" in src)
        check(f"{name} exports PYTHONIOENCODING", "set PYTHONIOENCODING=utf-8" in src)
        head = src[:400]
        check(f"{name} sets the codepage before anything else runs",
              "chcp 65001" in head, "must precede the Python calls")


def test_skills_are_hardened():
    print("\n── every text capture in the skills tolerates bad bytes ──")
    import re
    offenders = []
    for path in list((ROOT / "skills").rglob("*.py")) + list((ROOT / "agents").rglob("*.py")) \
            + [ROOT / "tests" / "run_all.py"]:
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"(?:subprocess\.(?:run|check_output|Popen)|\.communicate)\(", text):
            tail = text[m.end():m.end() + 400]
            if "text=True" in tail.split(")")[0] and "errors=" not in tail.split(")")[0]:
                offenders.append(f"{path.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}")
    check("no text=True capture is left without errors='replace'",
          not offenders, ", ".join(offenders) or "all covered")

    # and the shell skill really does return text when the child misbehaves
    try:
        from skills.sk_shell import shell_exec as run_command
        res = run_command(f'{sys.executable} -c "import sys;'
                          'sys.stdout.buffer.write(b\'ok \\xf9\');'
                          'sys.stdout.flush()"')
        out = (getattr(res, "data", None) or {}).get("stdout", "")
        check("sk_shell survives a child emitting cp862",
              getattr(res, "ok", False) is True and "ok" in out and "\ufffd" in out,
              f"stdout={out!r}"[:70])
    except Exception as exc:
        check("sk_shell survives a child emitting cp862", False, f"{type(exc).__name__}: {exc}"[:80])


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — platform / encoding")
    print("═" * 68)
    test_decoder_survives()
    test_runner_is_hardened()
    test_launchers_are_hardened()
    test_skills_are_hardened()
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
