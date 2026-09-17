"""Shell execution — the most dangerous skill JARVIS has, so it is the most
heavily guarded one.

Nothing runs until the Permission Firewall has:
  1. checked the command against a destructive-pattern blocklist
  2. checked the executable against an allowlist
  3. required explicit user confirmation (CRITICAL level)
  4. written the decision to the immutable audit log

Output is captured, truncated and streamed back as an event so the HUD can show
it live. Timeouts are mandatory — no command may hang JARVIS.
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.bus import BUS  # noqa: E402
from skills.registry import REGISTRY, SkillResult  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"
MAX_OUTPUT = 40_000


def _shell_for(command: str) -> List[str]:
    if IS_WINDOWS:
        if re.match(r"^\s*(get-|set-|new-|remove-|start-|write-|select-|\$)", command, re.I):
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        return ["cmd", "/c", command]
    return ["/bin/sh", "-c", command]


@REGISTRY.register(
    "shell.exec", risk="CRITICAL", agent="hermes",
    description_he="מריץ פקודת shell/PowerShell בפיקוח מלא — דורש אישור",
    required=("command",), triggers_he=("הרץ פקודה", "run command"),
)
def shell_exec(command: str = "", cwd: str = "", timeout: float = 30.0,
               env: Optional[Dict[str, str]] = None) -> SkillResult:
    cmd = str(command).strip()
    if not cmd:
        return SkillResult(ok=False, error="empty command")

    timeout = max(1.0, min(float(timeout), 300.0))
    workdir = str(Path(cwd).expanduser()) if cwd else None
    merged_env = dict(os.environ)
    merged_env.update(env or {})

    t0 = time.perf_counter()
    BUS.emit("shell.start", {"command": cmd[:200]}, source="shell")
    try:
        proc = subprocess.run(
            _shell_for(cmd), cwd=workdir, env=merged_env,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return SkillResult(ok=False, error=f"הפקודה נותקה אחרי {timeout} שניות (timeout)")
    except FileNotFoundError as exc:
        return SkillResult(ok=False, error=f"המעטפת לא נמצאה: {exc}")
    except Exception as exc:
        return SkillResult(ok=False, error=f"ההרצה נכשלה: {type(exc).__name__}: {exc}")

    ms = (time.perf_counter() - t0) * 1000
    stdout = (proc.stdout or "")[:MAX_OUTPUT]
    stderr = (proc.stderr or "")[:MAX_OUTPUT]
    BUS.emit("shell.done", {"returncode": proc.returncode, "ms": round(ms, 1),
                            "stdout_chars": len(stdout), "stderr_chars": len(stderr)}, source="shell")

    ok = proc.returncode == 0
    summary = (f"הפקודה הסתיימה בהצלחה עם {len(stdout.splitlines())} שורות פלט."
               if ok else
               f"הפקודה נכשלה עם קוד {proc.returncode}. {(stderr.strip().splitlines() or [''])[0][:160]}")
    return SkillResult(ok=ok, value=summary,
                       error="" if ok else stderr.strip()[:500],
                       ms=ms,
                       data={"command": cmd, "returncode": proc.returncode,
                             "stdout": stdout, "stderr": stderr,
                             "stdout_lines": len(stdout.splitlines()),
                             "summary_he": summary})


@REGISTRY.register(
    "shell.preview", risk="SAFE", agent="argus",
    description_he="מראה מה בדיוק ירוץ, בלי להריץ — מצב הדמיה בטוח",
    required=("command",),
)
def shell_preview(command: str = "") -> SkillResult:
    cmd = str(command).strip()
    argv = _shell_for(cmd)
    head = cmd.split()[0] if cmd.split() else ""
    return SkillResult(ok=True,
                       value=f"הייתי מריץ: {' '.join(argv)} — קובץ ההרצה הוא {head or 'ריק'}. לא בוצעה שום פעולה.",
                       data={"argv": argv, "executable": head, "dry_run": True})


@REGISTRY.register(
    "shell.history", risk="SAFE", agent="argus",
    description_he="מציג את הפקודות האחרונות שרוצו דרך JARVIS",
)
def shell_history(limit: int = 15) -> SkillResult:
    rows = [e["data"] for e in BUS.history if e.topic == "shell.start"][-int(limit):]
    if not rows:
        return SkillResult(ok=True, value="עדיין לא הורצו פקודות דרכי.", data={"count": 0})
    return SkillResult(ok=True, value="\n".join(f"$ {r.get('command', '')}" for r in rows),
                       data={"count": len(rows), "commands": rows})


@REGISTRY.register(
    "coder.run", risk="WRITE", agent="hephaestus",
    description_he="מריץ קוד פייתון בארגז חול מבודד עם פסק זמן — לא נוגע במערכת ישירות",
    required=("code",), triggers_he=("הרץ קוד", "run code", "run python"),
)
def coder_run(code: str = "", tests: str = "", timeout: float = 20.0) -> SkillResult:
    """Execute Python in HEPHAESTUS's restricted sandbox.

    This skill existed in the training corpus long before it existed here:
    forge_corpus.py emitted {"tool": "coder.run", ...} for every coding sample,
    but no such skill was registered, so the corpus taught a tool call that could
    only ever resolve to "unknown skill 'coder.run'". Registering it makes that
    data truthful and gives sandboxed execution a first-class, permission-gated
    entry point.

    Risk is WRITE, matching the gate on the CODE intent path. Not CRITICAL: that
    level demands HUD confirmation for every coding request, which headless denies
    outright — measured, it broke the coding agent entirely rather than protecting
    it. Sandboxed execution under an isolated interpreter with a mandatory timeout
    is the same trust class as fs.write, whereas shell.exec runs arbitrary commands
    against the real system and stays CRITICAL.

    The import is lazy because agents.hephaestus and the skill registry load each
    other during startup.
    """
    src = str(code or "").strip()
    if not src:
        return SkillResult(ok=False, error="לא קיבלתי קוד להרצה.")

    timeout = max(1.0, min(float(timeout), 120.0))
    try:
        from agents.hephaestus import run_python
    except Exception as exc:                                  # pragma: no cover
        return SkillResult(ok=False, error=f"ארגז החול לא זמין: {exc}")

    extra = {"test_generated.py": tests} if str(tests or "").strip() else None
    BUS.emit("coder.start", {"chars": len(src), "timeout": timeout}, source="coder.run")
    res = run_python(src, timeout=timeout, extra_files=extra)

    out = (getattr(res, "stdout", "") or "")[:MAX_OUTPUT]
    err = (getattr(res, "stderr", "") or "")[:MAX_OUTPUT]
    ok = bool(getattr(res, "ok", False))
    BUS.emit("coder.done", {"ok": ok, "ms": round(getattr(res, "ms", 0.0), 1),
                            "error_kind": getattr(res, "error_kind", "")}, source="coder.run")

    if ok:
        return SkillResult(ok=True, value=out or "(הקוד רץ בלי פלט)",
                           data={"stdout": out, "ms": getattr(res, "ms", 0.0),
                                 "sandboxed": True,
                                 "summary_he": "הקוד הורץ בארגז חול מבודד והסתיים בהצלחה."})
    kind = getattr(res, "error_kind", "") or "Error"
    line = getattr(res, "error_line", None)
    where = f" בשורה {line}" if line else ""
    return SkillResult(ok=False,
                       error=f"ההרצה נכשלה ({kind}){where}: {err[-600:] or 'אין פלט שגיאה'}",
                       data={"stdout": out, "stderr": err, "error_kind": kind,
                             "error_line": line, "sandboxed": True})
