"""Time, date and scheduling skills. Pure standard library — always available."""

from __future__ import annotations

import datetime as dt
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.bus import BUS  # noqa: E402
from skills.registry import REGISTRY, SkillResult  # noqa: E402

HE_DAYS = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
HE_MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
             "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]

REMINDERS: List[Dict[str, Any]] = []


def _part_of_day(now: dt.datetime) -> str:
    h = now.hour
    if 5 <= h < 11:
        return "בוקר טוב"
    if 11 <= h < 14:
        return "צהריים טובים"
    if 14 <= h < 18:
        return "אחר הצהריים טובים"
    if 18 <= h < 22:
        return "ערב טוב"
    return "לילה טוב"


@REGISTRY.register(
    "time.now", risk="SAFE", agent="jarvis",
    description_he="מחזיר את השעה המדויקת לפי שעון המחשב, מנוסחת בעברית",
    description_en="Returns the current local time phrased in Hebrew",
    triggers_he=("מה השעה", "שעה עכשיו"),
)
def time_now() -> SkillResult:
    now = dt.datetime.now()
    he = f"{_part_of_day(now)}, אדוני. השעה היא {now.hour:02d}:{now.minute:02d}."
    return SkillResult(ok=True, value=he, data={"iso": now.isoformat(timespec="seconds"),
                                                "hour": now.hour, "minute": now.minute})


@REGISTRY.register(
    "time.today", risk="SAFE", agent="jarvis",
    description_he="מחזיר את התאריך והיום בשבוע בעברית",
    triggers_he=("איזה יום היום", "מה התאריך"),
)
def time_today() -> SkillResult:
    now = dt.datetime.now()
    he = (f"היום הוא יום {HE_DAYS[now.weekday()]}, "
          f"{now.day} ב{HE_MONTHS[now.month - 1]} {now.year}.")
    return SkillResult(ok=True, value=he, data={"date": now.strftime("%Y-%m-%d"),
                                                "weekday": HE_DAYS[now.weekday()]})


@REGISTRY.register(
    "time.countdown", risk="SAFE", agent="jarvis",
    description_he="מחשב כמה זמן נשאר עד תאריך יעד",
    required=("target",),
)
def time_countdown(target: str = "") -> SkillResult:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M"):
        try:
            when = dt.datetime.strptime(target, fmt)
            break
        except ValueError:
            continue
    else:
        return SkillResult(ok=False, error=f"could not parse date {target!r}")
    delta = when - dt.datetime.now()
    total = int(delta.total_seconds())
    if total < 0:
        return SkillResult(ok=True, value=f"התאריך {target} כבר עבר לפני {abs(total) // 86400} ימים.")
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days} ימים")
    if hours:
        parts.append(f"{hours} שעות")
    parts.append(f"{minutes} דקות")
    return SkillResult(ok=True, value=f"נותרו {' ו'.join(parts)} עד {target}.",
                       data={"seconds": total})


# ------------------------------------------------------------- scheduler ----
class _Scheduler(threading.Thread):
    """A tiny dependency-free scheduler for reminders and delayed tasks."""

    daemon = True

    def __init__(self) -> None:
        super().__init__(name="jarvis-scheduler")
        self._stop = threading.Event()
        self.jobs: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, when_ts: float, message: str, callback: Optional[Callable[[str], None]] = None) -> int:
        with self._lock:
            job_id = len(self.jobs) + 1
            self.jobs.append({"id": job_id, "ts": when_ts, "message": message,
                              "callback": callback, "done": False})
        return job_id

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            due = []
            with self._lock:
                for job in self.jobs:
                    if not job["done"] and job["ts"] <= now:
                        job["done"] = True
                        due.append(job)
            for job in due:
                BUS.emit("scheduler.fire", {"id": job["id"], "message": job["message"]}, source="scheduler")
                cb = job.get("callback")
                if cb:
                    try:
                        cb(job["message"])
                    except Exception as exc:
                        BUS.emit("scheduler.error", {"error": str(exc)}, source="scheduler")
            self._stop.wait(0.5)


SCHEDULER = _Scheduler()
SCHEDULER.start()


@REGISTRY.register(
    "scheduler.remind", risk="SAFE", agent="jarvis",
    description_he="קובע תזכורת שתיורה בעוד מספר דקות",
    required=("minutes",),
    triggers_he=("תזכיר לי", "בעוד דקות"),
)
def scheduler_remind(minutes: float = 10, message: str = "תזכורת", **kwargs: Any) -> SkillResult:
    minutes = max(0.05, float(minutes))
    text = message or "תזכורת"
    job_id = SCHEDULER.add(time.time() + minutes * 60, text)
    REMINDERS.append({"id": job_id, "in_minutes": minutes, "message": text})
    he = f"קבעתי תזכורת בעוד {minutes:g} דקות: {text}."
    return SkillResult(ok=True, value=he, data={"job_id": job_id, "minutes": minutes})


@REGISTRY.register(
    "scheduler.list", risk="SAFE", agent="jarvis",
    description_he="מציג תזכורות ממתינות",
)
def scheduler_list() -> SkillResult:
    pending = [j for j in SCHEDULER.jobs if not j["done"]]
    if not pending:
        return SkillResult(ok=True, value="אין תזכורות ממתינות, אדוני.", data={"count": 0})
    lines = [f"{j['message']} (עוד {max(0, int(j['ts'] - time.time()) // 60)} דקות)" for j in pending]
    return SkillResult(ok=True, value=f"{len(pending)} תזכורות ממתינות: " + "; ".join(lines),
                       data={"count": len(pending), "jobs": pending})
