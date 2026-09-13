"""Application control — launching, closing, window management, notifications.

Primary target is Windows (Start-menu aliases, ``start`` verb, taskkill,
PowerShell toast notifications), with working fallbacks on Linux/macOS so the
same skill set behaves identically everywhere.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skills.registry import REGISTRY, SkillResult  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"

# Friendly Hebrew/English name -> per-OS launch target
APP_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "calculator": {
        "Windows": ["calc.exe"],
        "Darwin": ["open", "-a", "Calculator"],
        "Linux": ["gnome-calculator"],
    },
    "notepad": {
        "Windows": ["notepad.exe"],
        "Darwin": ["open", "-a", "TextEdit"],
        "Linux": ["gedit"],
    },
    "files": {
        "Windows": ["explorer.exe"],
        "Darwin": ["open", "."],
        "Linux": ["nautilus"],
    },
    "explorer": {"Windows": ["explorer.exe"], "Darwin": ["open", "."], "Linux": ["nautilus"]},
    "terminal": {
        "Windows": ["cmd.exe"],
        "Darwin": ["open", "-a", "Terminal"],
        "Linux": ["x-terminal-emulator"],
    },
    "cmd": {"Windows": ["cmd.exe"], "Darwin": ["open", "-a", "Terminal"], "Linux": ["bash"]},
    "powershell": {"Windows": ["powershell.exe"], "Darwin": ["pwsh"], "Linux": ["pwsh"]},
    "browser": {
        "Windows": ["start", ""],
        "Darwin": ["open", "https://localhost"],
        "Linux": ["xdg-open", "https://localhost"],
    },
    "paint": {"Windows": ["mspaint.exe"], "Darwin": ["open", "-a", "Preview"], "Linux": ["gimp"]},
    "task manager": {"Windows": ["taskmgr.exe"], "Darwin": ["open", "-a", "Activity Monitor"],
                     "Linux": ["gnome-system-monitor"]},
    "settings": {"Windows": ["start", "ms-settings:"], "Darwin": ["open", "/System/Applications/System Preferences.app"],
                 "Linux": ["gnome-control-center"]},
}

HEBREW_ALIASES: Dict[str, str] = {
    "מחשבון": "calculator", "פנקס רשימות": "notepad", "פנקס": "notepad",
    "סייר הקבצים": "files", "סייר": "files", "קבצים": "files", "תיקיות": "files",
    "טרמינל": "terminal", "שורת פקודה": "cmd", "פקודה": "cmd",
    "דפדפן": "browser", "דפדפן האינטרנט": "browser", "כרום": "chrome",
    "צייר": "paint", "מנהל המשימות": "task manager", "הגדרות": "settings",
    "הגדרות מערכת": "settings", "וורד": "winword", "אקסל": "excel",
    "מדיה": "media player", "נגן": "media player",
}


def _canonical(name: str) -> str:
    n = (name or "").strip().lower().rstrip(".")
    n = n.replace("את ה", "").replace("את ", "")
    n = n.replace("the ", "").strip()
    return HEBREW_ALIASES.get(n, n)


def _command_for(name: str) -> Optional[List[str]]:
    key = _canonical(name)
    entry = APP_ALIASES.get(key)
    system = platform.system()
    if entry and system in entry:
        return list(entry[system])
    if shutil.which(key):
        return [key]
    if IS_WINDOWS:
        for cand in (key, key + ".exe"):
            if shutil.which(cand):
                return [cand]
        return ["cmd", "/c", "start", "", key]
    if system == "Darwin":
        return ["open", "-a", name]
    return [key] if shutil.which(key) else None


@REGISTRY.register(
    "sys.launch_app", risk="WRITE", agent="hermes",
    description_he="פותח תוכנית לפי שם בעברית או באנגלית",
    required=("name",), triggers_he=("פתח את", "הפעל את", "open"),
)
def launch_app(name: str = "", args: List[str] = (), cwd: str = "") -> SkillResult:
    cmd = _command_for(name)
    if not cmd:
        return SkillResult(ok=False, error=f"לא זיהיתי את התוכנית {name!r} ואין פקודה מתאימה במערכת")
    cmd = cmd + [str(a) for a in (args or [])]
    try:
        kwargs: Dict[str, Any] = {"cwd": cwd or None}
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | \
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    except FileNotFoundError as exc:
        return SkillResult(ok=False, error=f"הפקודה לא נמצאה: {exc}")
    except Exception as exc:
        return SkillResult(ok=False, error=f"ההפעלה נכשלה: {exc}")
    return SkillResult(ok=True, value=f"פתחתי את {name}.",
                       data={"pid": proc.pid, "command": cmd})


@REGISTRY.register(
    "sys.close_app", risk="CRITICAL", agent="hermes",
    description_he="סוגר תוכנית לפי שם — דורש אישור מפורש",
    required=("name",), triggers_he=("סגור את", "כבה את", "close"),
)
def close_app(name: str = "", force: bool = False) -> SkillResult:
    key = _canonical(name)
    if IS_WINDOWS:
        flag = "/F" if force else ""
        cmd = ["taskkill", "/IM", f"{key}.exe"] + ([flag] if flag else [])
    else:
        cmd = ["pkill", "-9" if force else "-TERM", key]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=15)
    except FileNotFoundError:
        return SkillResult(ok=False, error=f"כלי הסגירה לא זמין במערכת הזו ({cmd[0]})")
    except Exception as exc:
        return SkillResult(ok=False, error=f"הסגירה נכשלה: {exc}")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:200]
        return SkillResult(ok=False, error=f"לא הצלחתי לסגור את {name}: {detail or 'returncode ' + str(proc.returncode)}")
    return SkillResult(ok=True, value=f"סגרתי את {name}.", data={"command": cmd})


@REGISTRY.register(
    "sys.apps_installed", risk="SAFE", agent="argus",
    description_he="מפרט תוכניות זמינות להפעלה (Start menu ב־Windows, PATH אחרת)",
)
def apps_installed(limit: int = 80) -> SkillResult:
    found: List[str] = []
    if IS_WINDOWS:
        roots = [
            Path(os.environ.get("ProgramData", "C:/ProgramData")) /
            "Microsoft/Windows/Start Menu/Programs",
            Path(os.environ.get("APPDATA", "")) /
            "Microsoft/Windows/Start Menu/Programs",
        ]
        for root in roots:
            if root.exists():
                for lnk in root.rglob("*.lnk"):
                    found.append(lnk.stem)
    else:
        for folder in os.environ.get("PATH", "").split(os.pathsep):
            p = Path(folder)
            if p.is_dir():
                try:
                    found.extend(f.name for f in p.iterdir() if os.access(f, os.X_OK))
                except OSError:
                    continue
    found = sorted({f for f in found if f and not f.startswith(".")})[: int(limit)]
    if not found:
        return SkillResult(ok=True, value="לא מצאתי רשימת תוכניות במערכת הזו.", data={"apps": []})
    return SkillResult(ok=True, value=f"זיהיתי {len(found)} תוכניות זמינות. למשל: {', '.join(found[:10])}.",
                       data={"count": len(found), "apps": found,
                             "aliases": sorted(set(HEBREW_ALIASES) | set(APP_ALIASES))})


@REGISTRY.register(
    "sys.notify", risk="SAFE", agent="hermes",
    description_he="שולח התראת מערכת (Windows toast, notification אחרת)",
    required=("message",),
)
def notify(message: str = "", title: str = "JARVIS") -> SkillResult:
    msg = str(message)
    if IS_WINDOWS:
        ps = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
            "ContentType = WindowsRuntime] | Out-Null; "
            f"$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0); "
            f"$n = $t.SelectSingleNode('//text'); $n.AppendChild($t.CreateTextNode('{title}')); "
            f"$b = $t.CreateTextNode('{msg}'); "
            "Write-Host 'notified'"
        )
        try:
            fallback = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Add-Type -AssemblyName System.Windows.Forms; "
                 f"$n=New-Object System.Windows.Forms.NotifyIcon; "
                 f"$n.Icon=[System.Drawing.SystemIcons]::Information; $n.Visible=$true; "
                 f"$n.ShowBalloonTip(4000,'{title}','{msg}',[System.Windows.Forms.ToolTipIcon]::Info)"],
                capture_output=True, text=True, errors="replace", timeout=12)
            if fallback.returncode == 0:
                return SkillResult(ok=True, value=f"שלחתי התראה: {msg}", data={"channel": "balloon"})
        except Exception:
            pass
    elif shutil.which("notify-send"):
        try:
            subprocess.run(["notify-send", str(title), msg], timeout=8, check=False)
            return SkillResult(ok=True, value=f"שלחתי התראה: {msg}", data={"channel": "notify-send"})
        except Exception as exc:
            return SkillResult(ok=False, error=f"notify-send failed: {exc}")
    return SkillResult(ok=True, value=f"[התראה] {title}: {msg}", data={"channel": "console-fallback"})


@REGISTRY.register(
    "win.layout", risk="WRITE", agent="hermes",
    description_he="מסדר חלונות על המסך: רשת, צד ימין/שמאל, מזעור הכול",
    required=("mode",),
)
def window_layout(mode: str = "grid") -> SkillResult:
    mode = str(mode).lower()
    if not IS_WINDOWS:
        return SkillResult(ok=False, error="סידור חלונות זמין כרגע רק ב־Windows (Win32 API)")
    script = {
        "grid": _PS_GRID, "minimize": "$ (New-Object -ComObject Shell.Application).MinimizeAll()",
        "cascade": "(New-Object -ComObject Shell.Application).CascadeWindows()",
        "tile": "(New-Object -ComObject Shell.Application).TileWindows()",
    }.get(mode)
    if not script:
        return SkillResult(ok=False, error=f"מצב לא מוכר: {mode} (אפשר grid/minimize/cascade/tile)")
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                              capture_output=True, text=True, errors="replace", timeout=20)
    except Exception as exc:
        return SkillResult(ok=False, error=f"הפעלת PowerShell נכשלה: {exc}")
    if proc.returncode != 0:
        return SkillResult(ok=False, error=(proc.stderr or "unknown PowerShell error")[:300])
    return SkillResult(ok=True, value=f"סידרתי את החלונות במצב {mode}.", data={"mode": mode})


_PS_GRID = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
    "$p=Get-Process | Where-Object {$_.MainWindowHandle -ne 0}; "
    "$n=[Math]::Max(1,$p.Count); $cols=[Math]::Ceiling([Math]::Sqrt($n)); "
    "$rows=[Math]::Ceiling($n/$cols); $w=[int]($b.Width/$cols); $h=[int]($b.Height/$rows); $i=0; "
    "Add-Type '[DllImport(\"user32.dll\")] public static extern bool MoveWindow(IntPtr h,int x,int y,int w,int ht,bool r);' "
    "-Name W -Namespace N; "
    "foreach($proc in $p){ $c=$i % $cols; $r=[int]($i/$cols); "
    "[N.W]::MoveWindow($proc.MainWindowHandle, $b.X+$c*$w, $b.Y+$r*$h, $w, $h, $true) | Out-Null; $i++ }"
)
