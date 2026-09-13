"""System observation skills — the ARGUS agent's eyes.

Cross-platform by design: works on Windows (the deployment target) and on
Linux/macOS. Optional dependencies degrade gracefully instead of crashing, so
JARVIS always answers with something true rather than something broken.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skills.registry import REGISTRY, SkillResult  # noqa: E402

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def _human(num: float, unit: str = "B") -> str:
    for step in ("", "K", "M", "G", "T", "P"):
        if abs(num) < 1024:
            return f"{num:.1f}{step}{unit}" if step else f"{int(num)}{unit}"
        num /= 1024
    return f"{num:.1f}E{unit}"


@REGISTRY.register(
    "sys.info", risk="SAFE", agent="argus",
    description_he="מידע כללי על המחשב: מערכת הפעלה, מעבד, זיכרון, שם מכונה",
    triggers_he=("מידע על המחשב", "system info"),
)
def sys_info() -> SkillResult:
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()[:40]})",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        "cpus": os.cpu_count(),
        "uptime_s": _uptime(),
    }
    if psutil:
        vm = psutil.virtual_memory()
        info["ram_total"] = _human(vm.total)
        info["ram_used_percent"] = vm.percent
        try:
            info["cpu_freq_mhz"] = round(psutil.cpu_freq().current) if psutil.cpu_freq() else None
        except Exception:
            info["cpu_freq_mhz"] = None
    he = (f"מערכת ההפעלה היא {info['os']}, שם המכונה {info['hostname']}, "
          f"{info['cpus']} ליבות מעבד, זיכרון כולל {info.get('ram_total', 'לא ידוע')} "
          f"בניצול {info.get('ram_used_percent', '?')} אחוז. זמן פעילות {_fmt_uptime(info['uptime_s'])}.")
    return SkillResult(ok=True, value=he, data=info)


def _uptime() -> Optional[float]:
    if psutil:
        try:
            return time.time() - psutil.boot_time()
        except Exception:
            pass
    try:
        with open("/proc/uptime", encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except Exception:
        return None


def _fmt_uptime(seconds: Optional[float]) -> str:
    if not seconds:
        return "לא ידוע"
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    parts = []
    if d:
        parts.append(f"{d} ימים")
    if h:
        parts.append(f"{h} שעות")
    parts.append(f"{m} דקות")
    return " ".join(parts)


@REGISTRY.register(
    "sys.telemetry", risk="SAFE", agent="argus",
    description_he="טלמטריה חיה: מעבד, זיכרון, דיסק, רשת, תהליכים מובילים",
    triggers_he=("מצב המחשב", "טלמטריה", "cpu", "זיכרון"),
)
def sys_telemetry(top: int = 5) -> SkillResult:
    data: Dict[str, Any] = {"ts": time.time()}
    if psutil is None:
        data["warning"] = "psutil is not installed — run tools/setup_windows.bat for full telemetry"
        try:
            total, used, free = shutil.disk_usage(str(Path.home()))
            data["disk"] = {"total": _human(total), "used": _human(used), "free": _human(free)}
        except Exception:
            pass
        return SkillResult(ok=True, value="טלמטריה חלקית: psutil לא מותקן, אבל אני עדיין כאן.", data=data)

    data["cpu_percent"] = psutil.cpu_percent(interval=0.25)
    vm = psutil.virtual_memory()
    data["ram"] = {"percent": vm.percent, "used": _human(vm.used), "total": _human(vm.total),
                   "available": _human(vm.available)}
    try:
        du = shutil.disk_usage(psutil.disk_partitions()[0].mountpoint)
        data["disk"] = {"percent": round(du.used / du.total * 100, 1), "free": _human(du.free),
                        "total": _human(du.total)}
    except Exception:
        pass
    try:
        nio = psutil.net_io_counters()
        data["net"] = {"sent": _human(nio.bytes_sent), "recv": _human(nio.bytes_recv)}
    except Exception:
        pass
    try:
        data["battery"] = (lambda b: None if not b else {"percent": b.percent, "plugged": b.power_plugged})(
            psutil.sensors_battery())
    except Exception:
        data["battery"] = None
    try:
        procs = sorted(psutil.process_iter(["name", "cpu_percent", "memory_percent"]),
                       key=lambda p: (p.info.get("cpu_percent") or 0), reverse=True)[:top]
        data["top_processes"] = [{"name": p.info["name"],
                                  "cpu": p.info.get("cpu_percent"),
                                  "mem": round(p.info.get("memory_percent") or 0, 1)} for p in procs]
    except Exception:
        data["top_processes"] = []

    parts = [f"מעבד {data['cpu_percent']} אחוז",
             f"זיכרון {data['ram']['percent']} אחוז מתוך {data['ram']['total']}"]
    if "disk" in data:
        parts.append(f"דיסק {data['disk']['percent']} אחוז תפוס")
    if data.get("top_processes"):
        lead = data["top_processes"][0]
        parts.append(f"התהליך המוביל הוא {lead['name']}")
    he = ". ".join(parts) + "."
    data["summary_he"] = he
    return SkillResult(ok=True, value=he, data=data)


@REGISTRY.register(
    "proc.list", risk="SAFE", agent="argus",
    description_he="מפרט תהליכים רצים לפי צריכת מעבד",
    triggers_he=("תהליכים", "processes"),
)
def proc_list(limit: int = 12) -> SkillResult:
    if psutil is None:
        return SkillResult(ok=False, error="psutil is not installed")
    rows = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            rows.append({"pid": p.info["pid"], "name": p.info["name"],
                         "cpu": p.info.get("cpu_percent") or 0.0,
                         "mem": round(p.info.get("memory_percent") or 0.0, 2)})
        except Exception:
            continue
    rows.sort(key=lambda r: r["cpu"], reverse=True)
    rows = rows[: int(limit)]
    he = "; ".join(f"{r['name']} (pid {r['pid']}, {r['cpu']}% CPU)" for r in rows) or "לא נמצאו תהליכים."
    return SkillResult(ok=True, value=he, data={"count": len(rows), "processes": rows})


@REGISTRY.register(
    "proc.kill", risk="CRITICAL", agent="hermes",
    description_he="עוצר תהליך לפי PID או שם — דורש אישור מפורש",
    required=("target",),
    triggers_he=("הרוג תהליך", "סגור תהליך", "kill"),
)
def proc_kill(target: str = "", force: bool = False) -> SkillResult:
    if psutil is None:
        return SkillResult(ok=False, error="psutil is not installed")
    matches: List[Any] = []
    if str(target).isdigit():
        try:
            matches = [psutil.Process(int(target))]
        except Exception as exc:
            return SkillResult(ok=False, error=f"no process with pid {target}: {exc}")
    else:
        needle = str(target).lower()
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if needle in (p.info["name"] or "").lower():
                    matches.append(p)
            except Exception:
                continue
    if not matches:
        return SkillResult(ok=False, error=f"no process matching {target!r}")
    if len(matches) > 1 and not force:
        names = ", ".join(f"{p.info['name']}({p.info['pid']})" for p in matches[:6])
        return SkillResult(ok=False, error=f"{len(matches)} processes match {target!r}: {names}. "
                                           f"Pass force=true or a pid.")
    killed = []
    for p in matches[:20]:
        try:
            p.terminate()
            killed.append(p.info["pid"])
        except Exception as exc:
            return SkillResult(ok=False, error=f"could not terminate {p.info['pid']}: {exc}")
    return SkillResult(ok=True, value=f"עצרתי {len(killed)} תהליכים: {killed}.",
                       data={"killed": killed})


@REGISTRY.register(
    "net.info", risk="SAFE", agent="argus",
    description_he="כתובות רשת מקומיות ומצב חיבור",
)
def net_info() -> SkillResult:
    data: Dict[str, Any] = {"hostname": socket.gethostname()}
    try:
        data["local_ip"] = socket.gethostbyname(socket.gethostname())
    except Exception:
        data["local_ip"] = None
    if psutil:
        try:
            addrs = {nic: [a.address for a in v if a.family == socket.AF_INET]
                     for nic, v in psutil.net_if_addrs().items()}
            data["interfaces"] = {k: v for k, v in addrs.items() if v}
        except Exception:
            pass
        try:
            conns = psutil.net_connections(kind="inet")
            data["connections"] = len(conns)
            data["listening"] = len([c for c in conns if c.status == "LISTEN"])
        except Exception:
            pass
    he = f"שם המכונה {data['hostname']}, כתובת מקומית {data.get('local_ip') or 'לא זוהתה'}"
    if "listening" in data:
        he += f", {data['listening']} פורטים בהאזנה ו־{data['connections']} חיבורים פעילים"
    return SkillResult(ok=True, value=he + ".", data=data)
