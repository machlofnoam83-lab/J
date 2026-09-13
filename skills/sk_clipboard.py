"""Clipboard skills with a local history ring. No cloud, no sync, no leaks."""

from __future__ import annotations

import platform
import subprocess
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skills.registry import REGISTRY, SkillResult  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"
HISTORY: Deque[Dict[str, Any]] = deque(maxlen=64)
_LOCK = threading.Lock()

try:
    import pyperclip  # type: ignore
except Exception:  # pragma: no cover
    pyperclip = None  # type: ignore


def _push(text: str, direction: str) -> None:
    with _LOCK:
        HISTORY.appendleft({"text": text[:2000], "direction": direction, "ts": time.time()})


def _native_get() -> Optional[str]:
    try:
        if pyperclip is not None:
            return pyperclip.paste()
    except Exception:
        pass
    try:
        if IS_WINDOWS:
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                                  capture_output=True, text=True, errors="replace", timeout=8)
            return proc.stdout if proc.returncode == 0 else None
        if shutil.which("xclip"):
            proc = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                                  capture_output=True, text=True, errors="replace", timeout=8)
            return proc.stdout if proc.returncode == 0 else None
        if shutil.which("wl-paste"):
            proc = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True, errors="replace", timeout=8)
            return proc.stdout if proc.returncode == 0 else None
        if shutil.which("pbpaste"):
            proc = subprocess.run(["pbpaste"], capture_output=True, text=True, errors="replace", timeout=8)
            return proc.stdout if proc.returncode == 0 else None
    except Exception:
        return None
    return None


def _native_set(text: str) -> bool:
    try:
        if pyperclip is not None:
            pyperclip.copy(text)
            return True
    except Exception:
        pass
    try:
        if IS_WINDOWS:
            escaped = text.replace("'", "''")
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", f"Set-Clipboard -Value '{escaped}'"],
                                  capture_output=True, text=True, errors="replace", timeout=8)
            return proc.returncode == 0
        if shutil.which("xclip"):
            proc = subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, errors="replace", timeout=8)
            return proc.returncode == 0
        if shutil.which("wl-copy"):
            proc = subprocess.run(["wl-copy", text], capture_output=True, timeout=8)
            return proc.returncode == 0
        if shutil.which("pbcopy"):
            proc = subprocess.run(["pbcopy"], input=text, text=True, errors="replace", timeout=8)
            return proc.returncode == 0
    except Exception:
        return False
    return False


@REGISTRY.register(
    "clipboard.get", risk="SAFE", agent="argus",
    description_he="קורא את תוכן לוח ההעתקה", triggers_he=("מה יש בלוח", "הדבק"),
)
def clipboard_get() -> SkillResult:
    text = _native_get()
    if text is None:
        return SkillResult(ok=False,
                           error="אין גישה ללוח ההעתקה בסביבה הזו. ב־Windows יש להתקין pyperclip "
                                 "(tools/setup_windows.bat) או להריץ עם הרשאות שולחן עבודה.")
    _push(text, "read")
    preview = text.strip()
    short = preview[:200] + ("…" if len(preview) > 200 else "")
    return SkillResult(ok=True, value=f"בלוח ההעתקה יש {len(preview)} תווים: {short}",
                       data={"chars": len(preview), "text": text[:20000]})


@REGISTRY.register(
    "clipboard.set", risk="WRITE", agent="hermes",
    description_he="כותב טקסט ללוח ההעתקה", required=("text",), triggers_he=("העתק ללוח", "שמור ללוח"),
)
def clipboard_set(text: str = "") -> SkillResult:
    if not _native_set(str(text)):
        return SkillResult(ok=False, error="הכתיבה ללוח נכשלה — אין מנגנון לוח זמין בסביבה הזו")
    _push(str(text), "write")
    return SkillResult(ok=True, value=f"העתקתי {len(str(text))} תווים ללוח.",
                       data={"chars": len(str(text))})


@REGISTRY.register(
    "clipboard.history", risk="SAFE", agent="argus",
    description_he="מציג את היסטוריית הלוח שנשמרה מקומית בלבד",
)
def clipboard_history(limit: int = 10) -> SkillResult:
    with _LOCK:
        items = list(HISTORY)[: int(limit)]
    if not items:
        return SkillResult(ok=True, value="היסטוריית הלוח ריקה עדיין.", data={"count": 0})
    lines = [f"[{time.strftime('%H:%M:%S', time.localtime(i['ts']))}] {i['direction']}: {i['text'][:60]}"
             for i in items]
    return SkillResult(ok=True, value="\n".join(lines), data={"count": len(items), "items": items})
