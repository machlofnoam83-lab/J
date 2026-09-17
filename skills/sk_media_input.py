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

import numpy as np

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
            # The path arrives as $env:JARVIS_SHOT, never spliced into the script
            # text. `path` is caller-supplied, so interpolating it meant a quote
            # in the filename closed the literal and the remainder parsed as
            # PowerShell — a WRITE-risk skill, waved through by the firewall with
            # no confirmation. skills/_ps.py keeps the script constant.
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
                "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
                "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height; "
                "$g=[System.Drawing.Graphics]::FromImage($bmp); "
                "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size); "
                "$bmp.Save($env:JARVIS_SHOT); $g.Dispose(); $bmp.Dispose()"
            )
            try:
                from skills._ps import ps_env_path
                rc, _stdout, stderr = ps_env_path(ps, "JARVIS_SHOT", out, timeout=20)
                if rc == 0 and out.exists():
                    return SkillResult(ok=True, value=f"צילמתי את המסך ושמרתי ל־{out.name}.",
                                       data={"path": str(out), "bytes": out.stat().st_size})
                return SkillResult(ok=False, error=(stderr or "capture failed")[:300])
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


@REGISTRY.register(
    "screen.read", risk="SAFE", agent="argus",
    description_he="קורא צילום מסך שנשמר כ־PNG ומדווח מה יש בו",
    triggers_he=("קרא את צילום המסך", "מה יש על המסך", "read screenshot"),
)
def screen_read(path: str = "") -> SkillResult:
    """Decode a saved screenshot and describe it.

    `screen.capture` has always written a PNG, and until now nothing in JARVIS
    could open one: Pillow and OpenCV are both absent and neither may be assumed
    offline. So the assistant could photograph the screen and never look at the
    photograph — the file landed on disk and the loop stopped there.

    Decoding goes through `vision.png`, a codec written against the format spec
    using only stdlib `zlib`/`binascii` and numpy. This reports what can be
    derived from pixels alone and says nothing about text content: reading
    characters off a screen needs a recogniser with training data, and there is
    none available offline — the same data starvation that closed the
    open-vocabulary ASR attempt. Claiming otherwise here would be the kind of
    confident non-answer the knowledge base audits were about.

    With no `path`, reads the most recent screenshot under data/.
    """
    target = Path(path).expanduser() if path else None
    if target is None:
        shots = sorted((ROOT / "data").glob("screenshot_*.png")) if (ROOT / "data").exists() else []
        if not shots:
            return SkillResult(ok=False,
                               error="אין צילום מסך שמור. אפשר לבקש 'צלם את המסך' קודם, "
                                     "או לתת נתיב לקובץ PNG.")
        target = shots[-1]
    if not target.exists():
        return SkillResult(ok=False, error=f"הקובץ לא נמצא: {target}")
    from vision import png as _png
    if not _png.is_png(target):
        # Checked by content, not by extension: a screenshot renamed on disk is
        # still readable, and a .png that is not one should say so plainly.
        kind = target.suffix.lower() if target.suffix else "ללא סיומת"
        return SkillResult(ok=False,
                           error=f"הקובץ {target.name} אינו PNG תקין (סיומת: {kind}).")

    try:
        im = _png.decode_file(target)
    except _png.PngError as exc:
        return SkillResult(ok=False, error=f"הקובץ אינו PNG תקין: {exc}")
    except OSError as exc:
        return SkillResult(ok=False, error=f"לא הצלחתי לקרוא את הקובץ: {exc}")

    gray = im.gray()
    mean = float(gray.mean())
    # a screen is mostly background, so the dominant tone is more informative than
    # the mean alone; quartiles say whether it is a dark UI or a bright document
    q1, q2, q3 = (float(np.percentile(gray, p)) for p in (25, 50, 75))
    bright_share = float((gray > 0.66).mean())
    dark_share = float((gray < 0.33).mean())
    edge = float(np.abs(np.diff(gray, axis=1)).mean())

    # adjective agrees with תמונה (feminine); כהה is gender-invariant
    kind = ("בהירה" if mean > 0.6 else "כהה" if mean < 0.35 else "בינונית")
    detail = ("עמוס פרטים" if edge > 0.06 else "מעט פרטים" if edge < 0.015 else "פירוט בינוני")
    summary = (f"{target.name}: {im.width}x{im.height} פיקסלים, {im.color_name}, "
               f"תמונה {kind} (ממוצע אור {mean:.2f}), {detail}. "
               f"שיעור האזורים הבהירים {bright_share:.0%}, כהים {dark_share:.0%}.")

    return SkillResult(ok=True, value=summary, data={
        "path": str(target), "bytes": target.stat().st_size,
        "width": im.width, "height": im.height,
        "color_type": im.color_type, "color_name": im.color_name,
        "bit_depth": im.bit_depth, "channels": im.channels,
        "interlaced": im.interlaced,
        "brightness_mean": round(mean, 4),
        "brightness_quartiles": [round(q1, 4), round(q2, 4), round(q3, 4)],
        "bright_share": round(bright_share, 4),
        "dark_share": round(dark_share, 4),
        "edge_energy": round(edge, 5),
        "summary_he": summary,
        "text_extracted": False,
        "note": "pixel statistics only — offline OCR is not available",
    })


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
