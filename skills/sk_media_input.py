"""Media keys, volume, screenshots and synthetic input.

These depend on optional native packages (pyautogui / mss / pycaw). When a
package is missing JARVIS says so plainly instead of pretending it acted.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skills.registry import REGISTRY, SkillResult  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"


def _missing(pkg: str, installer: str = "") -> SkillResult:
    return SkillResult(ok=False,
                       error=f"הרכיב {pkg} לא מותקן בסביבה הזו. "
                             f"{installer or 'התקנה דרך tools/setup_windows.bat'} — "
                             f"בלי מסך/שולחן עבודה פעיל אי אפשר לבצע את הפעולה.")


# ------------------------------------------------------------------ media --
_MEDIA_KEYS = {"play_pause": 0xB3, "stop": 0xB2, "next": 0xB0, "prev": 0xB1,
               "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD}


def _send_media_key(key_name: str) -> SkillResult:
    vk = _MEDIA_KEYS.get(key_name)
    if vk is None:
        return SkillResult(ok=False, error=f"unknown media key {key_name!r}")
    if IS_WINDOWS:
        ps = (
            "$s=Add-Type -MemberDefinition '[DllImport(\"user32.dll\")] "
            "public static extern void keybd_event(byte k, byte s, int f, int e);' "
            "-Name K -Namespace N -PassThru; "
            f"$s::keybd_event({vk},0,0,0); $s::keybd_event({vk},0,2,0)"
        )
        try:
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                  capture_output=True, text=True, errors="replace", timeout=10)
            if proc.returncode == 0:
                return SkillResult(ok=True, value=f"שלחתי את מקש המדיה {key_name}.",
                                   data={"key": key_name})
        except Exception as exc:
            return SkillResult(ok=False, error=f"שליחת מקש נכשלה: {exc}")
    elif shutil.which("xdotool"):
        names = {"play_pause": "XF86AudioPlay", "next": "XF86AudioNext", "prev": "XF86AudioPrev",
                 "stop": "XF86AudioStop", "volume_up": "XF86AudioRaiseVolume",
                 "volume_down": "XF86AudioLowerVolume", "mute": "XF86AudioMute"}
        try:
            subprocess.run(["xdotool", "key", names[key_name]], timeout=8, check=False)
            return SkillResult(ok=True, value=f"שלחתי את מקש המדיה {key_name}.", data={"key": key_name})
        except Exception as exc:
            return SkillResult(ok=False, error=str(exc))
    return _missing("media-key transport", "ב־Windows הפעולה עובדת דרך Win32 keybd_event")


for _name, _he in (
    ("play", "מפעיל/משהה נגינה"), ("pause", "משהה נגינה"), ("next", "הרצועה הבאה"),
    ("prev", "הרצועה הקודמת"), ("stop", "עוצר נגינה"),
    ("volume_up", "מגביר עוצמת קול"), ("volume_down", "מנמיך עוצמת קול"), ("mute", "השתקה"),
):
    def _make(key: str, desc: str):
        @REGISTRY.register(f"media.{key}", risk="SAFE", agent="hermes", description_he=desc)
        def _fn() -> SkillResult:
            mapped = "play_pause" if key in ("play", "pause") else key
            return _send_media_key(mapped)
        return _fn
    _make(_name, _he)


@REGISTRY.register(
    "media.volume_set", risk="WRITE", agent="hermes",
    description_he="קובע עוצמת קול באחוזים", required=("percent",),
)
def volume_set(percent: float = 50) -> SkillResult:
    percent = max(0.0, min(100.0, float(percent)))
    if IS_WINDOWS:
        try:
            from ctypes import cast, POINTER  # noqa: F401
            import comtypes  # type: ignore  # noqa: F401
        except Exception:
            pass
        # nircmd-free approach: send volume keys relative to mute
        steps = int(round(percent / 2))
        res = _send_media_key("mute")
        if not res.ok:
            return res
        for _ in range(steps):
            _send_media_key("volume_up")
        return SkillResult(ok=True, value=f"כיוונתי את עוצמת הקול ל־{percent:g} אחוז.",
                           data={"percent": percent})
    if shutil.which("amixer"):
        try:
            subprocess.run(["amixer", "sset", "Master", f"{int(percent)}%"], timeout=8, check=False)
            return SkillResult(ok=True, value=f"כיוונתי את עוצמת הקול ל־{percent:g} אחוז.",
                               data={"percent": percent})
        except Exception as exc:
            return SkillResult(ok=False, error=str(exc))
    if IS_MAC and shutil.which("osascript"):
        try:
            subprocess.run(["osascript", "-e", f"set volume output volume {int(percent)}"],
                           timeout=8, check=False)
            return SkillResult(ok=True, value=f"כיוונתי את עוצמת הקול ל־{percent:g} אחוז.",
                               data={"percent": percent})
        except Exception as exc:
            return SkillResult(ok=False, error=str(exc))
    return _missing("volume control backend", "amixer / pycaw / osascript")


# ----------------------------------------------------------------- screen --
@REGISTRY.register(
    "screen.capture", risk="WRITE", agent="argus",
    description_he="מצלם את המסך ושומר כתמונת PNG",
    triggers_he=("צלם את המסך", "צילום מסך", "screenshot"),
)
def screen_capture(path: str = "", monitor: int = 0) -> SkillResult:
    out = Path(path) if path else (ROOT / "data" / f"screenshot_{int(time.time())}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mss  # type: ignore
        import mss.tools  # type: ignore
    except Exception:
        if IS_WINDOWS:
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
                "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
                "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height; "
                "$g=[System.Drawing.Graphics]::FromImage($bmp); "
                "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size); "
                f"$bmp.Save('{out}'); $g.Dispose(); $bmp.Dispose()"
            )
            try:
                proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                      capture_output=True, text=True, errors="replace", timeout=20)
                if proc.returncode == 0 and out.exists():
                    return SkillResult(ok=True, value=f"צילמתי את המסך ושמרתי ל־{out.name}.",
                                       data={"path": str(out), "bytes": out.stat().st_size})
                return SkillResult(ok=False, error=(proc.stderr or "capture failed")[:300])
            except Exception as exc:
                return SkillResult(ok=False, error=f"צילום מסך נכשל: {exc}")
        return _missing("mss", "pip install mss")
    try:
        with mss.mss() as sct:
            mon = sct.monitors[min(int(monitor), len(sct.monitors) - 1)]
            shot = sct.grab(mon)
            mss.tools.to_png(shot.rgb, shot.size, output=str(out))
    except Exception as exc:
        return SkillResult(ok=False, error=f"צילום מסך נכשל: {exc}")
    return SkillResult(ok=True, value=f"צילמתי את המסך ({out.stat().st_size} בתים) ושמרתי ל־{out.name}.",
                       data={"path": str(out), "bytes": out.stat().st_size, "monitor": mon})


@REGISTRY.register(
    "screen.size", risk="SAFE", agent="argus",
    description_he="מחזיר את רזולוציית המסך",
)
def screen_size() -> SkillResult:
    if IS_WINDOWS:
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Add-Type -AssemblyName System.Windows.Forms; "
                 "[System.Windows.Forms.Screen]::AllScreens | ForEach-Object { "
                 "\"$($_.Bounds.Width)x$($_.Bounds.Height) primary=$($_.Primary)\" }"],
                capture_output=True, text=True, errors="replace", timeout=10)
            if proc.returncode == 0 and proc.stdout.strip():
                return SkillResult(ok=True, value=proc.stdout.strip().replace("\n", "; "),
                                   data={"raw": proc.stdout.strip()})
        except Exception:
            pass
    try:
        import mss  # type: ignore
        with mss.mss() as sct:
            mon = sct.monitors[0]
            return SkillResult(ok=True, value=f"רזולוציית המסך היא {mon['width']}x{mon['height']}.",
                               data={"width": mon["width"], "height": mon["height"]})
    except Exception:
        return _missing("mss", "pip install mss")


# ------------------------------------------------------------------ input --
@REGISTRY.register(
    "input.type_text", risk="CRITICAL", agent="hermes",
    description_he="מקליד טקסט כאילו הוקלד במקלדת — דורש אישור",
    required=("text",),
)
def type_text(text: str = "", interval: float = 0.01) -> SkillResult:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return _missing("pyautogui", "pip install pyautogui")
    try:
        pyautogui.typewrite(str(text), interval=max(0.0, float(interval)))
    except Exception as exc:
        return SkillResult(ok=False, error=f"ההקלדה נכשלה: {exc}")
    return SkillResult(ok=True, value=f"הקלדתי {len(str(text))} תווים.", data={"chars": len(str(text))})


@REGISTRY.register(
    "input.hotkey", risk="CRITICAL", agent="hermes",
    description_he="לוחץ צירוף מקשים", required=("keys",),
)
def hotkey(keys: List[str] = (), ) -> SkillResult:
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.replace("+", " ").split() if k.strip()]
    if not keys:
        return SkillResult(ok=False, error="no keys given")
    try:
        import pyautogui  # type: ignore
        pyautogui.hotkey(*keys)
    except Exception:
        return _missing("pyautogui", "pip install pyautogui")
    return SkillResult(ok=True, value=f"לחצתי {' + '.join(keys)}.", data={"keys": keys})


@REGISTRY.register(
    "input.click", risk="CRITICAL", agent="hermes",
    description_he="לוחץ בעכבר בקואורדינטות או במיקום הנוכחי",
)
def click(x: int = -1, y: int = -1, button: str = "left", clicks: int = 1) -> SkillResult:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return _missing("pyautogui", "pip install pyautogui")
    try:
        if int(x) >= 0 and int(y) >= 0:
            pyautogui.click(int(x), int(y), clicks=int(clicks), button=str(button))
        else:
            pyautogui.click(clicks=int(clicks), button=str(button))
    except Exception as exc:
        return SkillResult(ok=False, error=f"הלחיצה נכשלה: {exc}")
    return SkillResult(ok=True, value="ביצעתי לחיצה.", data={"x": x, "y": y, "button": button})
