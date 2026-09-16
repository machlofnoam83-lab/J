#!/usr/bin/env python3
"""Run every JARVIS test module and print one honest summary.

Each test file is a standalone script that prints `RESULT: n passed, m failed`.
Run:  python tests/run_all.py [--quick]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = re.compile(r"RESULT:\s*(\d+)\s*passed,\s*(\d+)\s*failed")

# order matters only for readability: brain → cognition → action → integration
ORDER = [
    "test_brain_smoke.py",
    "test_intent_math.py",
    "test_skills_security.py",
    "test_coder.py",
    "test_orchestrator.py",
    "test_rag.py",
    "test_scene.py",
    "test_presence.py",
    "test_enroll.py",
    "test_access.py",
    "test_records.py",
    "test_deliberate.py",
    "test_voice.py",
    "test_server.py",
    "test_voice_session.py",
]


def discover() -> list[Path]:
    found = sorted(p.name for p in HERE.glob("test_*.py"))
    ordered = [n for n in ORDER if n in found]
    return [HERE / n for n in ordered + [f for f in found if f not in ordered]]


def main() -> int:
    # Windows: the console codepage (cp862/cp1255 on a Hebrew system) is not
    # UTF-8, so a child printing Hebrew wrote bytes this parent could not decode —
    # UnicodeDecodeError killed the reader thread and every module reported
    # "no RESULT line" while actually passing. Force both ends to UTF-8 and never
    # let an undecodable byte abort a test run.
    CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    # The access gate refuses to act with no face in frame, which is exactly
    # right in production and exactly wrong for a test that is checking RAG or
    # the voice pipeline and has no camera attached. So headless modules run
    # with the gate off, and the one module that tests the gate turns it back
    # on itself. Disabling it here rather than in each test keeps the intent in
    # one place: "these suites are not about access control."
    GATE_OFF = {**CHILD_ENV, "JARVIS_ACCESS_GATE": "0"}
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the slow integration suites")
    ap.add_argument("--only", default="", help="comma separated file names")
    args = ap.parse_args()

    skip = ({"test_orchestrator.py", "test_server.py", "test_voice_session.py"}
            if args.quick else set())
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    files = [f for f in discover() if f.name not in skip and (not only or f.name in only)]

    print("═" * 68)
    print(" J.A.R.V.I.S. test suite")
    print("═" * 68)

    total_ok = total_fail = 0
    broken: list[str] = []
    t_all = time.perf_counter()

    for f in files:
        print(f"\n── {f.name} " + "─" * max(0, 60 - len(f.name)))
        t0 = time.perf_counter()
        try:
            env = CHILD_ENV if f.name == "test_access.py" else GATE_OFF
            proc = subprocess.run([sys.executable, str(f)], cwd=str(HERE.parent),
                                  capture_output=True, text=True, timeout=1800,
                                  encoding="utf-8", errors="replace", env=env)
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 1800s")
            broken.append(f.name)
            continue
        ms = (time.perf_counter() - t0) * 1000
        tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
        # The LAST match, not the first: a module may legitimately print
        # RESULT-shaped text while testing something else (test_platform pipes a
        # fake result line through a decoder to prove the pipe survives), and a
        # real summary is always the last thing a module writes.
        found = RESULT.findall(out)
        m = found[-1] if found else None
        if m:
            p, fl = int(m[0]), int(m[1])
            total_ok += p
            total_fail += fl
            mark = "✓" if fl == 0 else "✗"
            print(f"  {mark} {p} passed, {fl} failed  ({ms/1000:.1f}s)")
            if fl:
                broken.append(f.name)
                for ln in out.splitlines():
                    if ln.strip().startswith("FAIL"):
                        print("      " + ln.strip()[:110])
        else:
            total_fail += 1
            broken.append(f.name)
            print(f"  ✗ no RESULT line (exit {proc.returncode})  ({ms/1000:.1f}s)")
            for ln in tail:
                print("      " + ln[:110])

    print("\n" + "═" * 68)
    print(f" TOTAL: {total_ok} passed, {total_fail} failed   ({(time.perf_counter()-t_all):.1f}s)")
    if broken:
        print(" failing modules: " + ", ".join(broken))
    print("═" * 68)
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
