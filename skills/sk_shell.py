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
