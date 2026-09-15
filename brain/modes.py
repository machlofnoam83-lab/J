"""Operational modes — twelve postures JARVIS can hold, chosen by voice or HUD.

A mode is not a skin. It changes what JARVIS reaches for first: which skills a
vague request is allowed to resolve to, which opener frames the reply, and which
sources the researcher digs through. The user asked for "a researcher that finds
me everything I need" and eleven more like it, so each entry here is a real
behavioural stance backed by tools that already exist offline, not a label.

Modes are stored in data/mode.json so the posture survives a restart — waking up
in a different stance than the one you were left in would be a small lie about
continuity.

Nothing here talks to the network. The researcher searches local sources only
(knowledge base, memory palace, transcript, file tree, skill catalogue, corpus);
an offline assistant that "researches" by phoning home is not offline.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "mode.json"


@dataclass(frozen=True)
class Mode:
    id: str
    he: str                 # display name
    en: str                 # short tag
    desc_he: str            # one line, shown in the HUD
    persona_he: str         # opener used when the mode takes an action
    focus: tuple            # skill-name prefixes this mode prefers
    triggers: tuple         # phrases that switch into it
    research: bool = False  # digs through every local source by default


MODES: Dict[str, Mode] = {m.id: m for m in (
    Mode("researcher", "חוקר", "RESEARCH",
         "חופר בכל המקורות המקומיים — ידע, זיכרון, קבצים, כלים — ומחזיר תמצית עם מקורות",
         "בדקתי בכל המקורות שיש לי כאן, אדוני.",
         ("research.", "knowledge", "memory", "fs."),
         ("מצב חוקר", "תעבור למצב חוקר", "research mode", "מצב מחקר"),
         research=True),
    Mode("coder", "מתכנת", "CODE",
         "כותב, מריץ ומאמת קוד דרך HEPHAESTUS בארגז חול",
         "HEPHAESTUS מוכן, אדוני.",
         ("coder.", "fs.read", "fs.write"),
         ("מצב מתכנת", "מצב קוד", "code mode", "מצב תכנות")),
    Mode("sentinel", "שומר סף", "SECURITY",
         "עירני להרשאות, חומת אש ויון ביקורת — כל פעולה נבחנת פעמיים",
         "עומד על המשמר, אדוני.",
         ("security.", "sys."),
         ("מצב אבטחה", "שומר סף", "security mode")),
    Mode("steward", "מנהל בית", "STEWARD",
         "מסדר קבצים, תוכניות ומשימות תחזוקה במחשב",
         "הבית בידיים טובות, אדוני.",
         ("fs.", "app.", "win."),
         ("מצב מנהל", "מצב בית", "steward mode")),
    Mode("secretary", "מזכיר", "SECRETARY",
         "זמן, תזכורות וסדר יום — לא נותן לדברים ליפול בין הכיסאות",
         "היומן פתוח, אדוני.",
         ("time.", "scheduler."),
         ("מצב מזכיר", "מצב יומן", "secretary mode")),
    Mode("analyst", "מנתח", "ANALYST",
         "מספרים, מדידות וחישובים — אף ספרה לא מנוחשת",
         "המספרים לפניך, אדוני.",
         ("math.", "sys.telemetry", "screen.read"),
         ("מצב מנתח", "מצב ניתוח", "analyst mode")),
    Mode("tutor", "מורה", "TUTOR",
         "מסביר שלב אחרי שלב מתוך בסיס הידע, בלי לקפוץ למסקנה",
         "נתחיל מהיסוד, אדוני.",
         ("knowledge", "math."),
         ("מצב מורה", "מצב לימוד", "tutor mode")),
    Mode("scribe", "סופר", "SCRIBE",
         "טיוטות, ניסוחים וסיכומים — מילים הן הכלי כאן",
         "הקולמוס מושחז, אדוני.",
         ("fs.write", "clipboard."),
         ("מצב כתיבה", "מצב סופר", "scribe mode")),
    Mode("conductor", "מנצח", "CONDUCTOR",
         "מפרק בקשה גדולה לצעדים ומנגן אותם לפי הסדר",
         "התזמורת מוכנה, אדוני.",
         ("scheduler.", "fs.", "coder."),
         ("מצב מנצח", "מצב משימות", "conductor mode")),
    Mode("watcher", "צופה", "WATCHER",
         "מודע למסך ולמצלמה — מה נראה, מי נמצא, מה השתנה",
         "עיניי פקוחות, אדוני.",
         ("screen.", "vision.", "face."),
         ("מצב צופה", "מצב תצפית", "watcher mode")),
    Mode("linguist", "בלשן", "LINGUIST",
         "שפה, תרגום ודקדוק — עברית, אנגלית ומה שביניהן",
         "המילים בידיים טובות, אדוני.",
         ("knowledge", "clipboard."),
         ("מצב בלשן", "מצב תרגום", "linguist mode")),
    Mode("companion", "בן לווייה", "COMPANION",
         "שיחה רגילה, בלי משימה — לפעמים זה כל מה שנדרש",
         "אני כאן, אדוני.",
         ("knowledge", "memory"),
         ("מצב שיחה", "מצב חבר", "companion mode")),
)}

DEFAULT_MODE = "companion"


class ModeState:
    """The posture JARVIS is currently holding, persisted across restarts."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else STATE_PATH
        self._lock = threading.Lock()
        self.current = DEFAULT_MODE
        self.since = time.time()
        self.history: List[Dict[str, Any]] = []
        self._load()

    # ---------------------------------------------------------------- load --
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("current") in MODES:
                self.current = raw["current"]
                self.since = float(raw.get("since", time.time()))
                self.history = list(raw.get("history", []))[-40:]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a corrupt state file must not stop the assistant starting

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"current": self.current, "since": self.since,
                 "history": self.history[-40:]}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        except OSError:
            pass

    # ----------------------------------------------------------------- api --
    def get(self) -> Mode:
        with self._lock:
            return MODES[self.current]

    def set(self, mode_id: str) -> Optional[Mode]:
        """Switch posture. Unknown ids return None rather than raising: a
        mis-heard voice command should be reported, not crash the loop."""
        mid = str(mode_id or "").strip().lower()
        if mid not in MODES:
            return None
        with self._lock:
            if mid == self.current:
                return MODES[mid]
            prev = self.current
            self.current = mid
            self.since = time.time()
            self.history.append({"from": prev, "to": mid, "ts": self.since})
            self.history = self.history[-40:]
            self._save()
        return MODES[mid]

    def by_trigger(self, text: str) -> Optional[Mode]:
        """The mode a spoken phrase asks for, if any."""
        t = (text or "").strip().lower()
        if not t:
            return None
        for m in MODES.values():
            for trig in m.triggers:
                if trig.lower() in t:
                    return m
        return None

    def schema(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.current
        return [{"id": m.id, "he": m.he, "en": m.en, "desc": m.desc_he,
                 "persona": m.persona_he, "research": m.research,
                 "active": m.id == cur} for m in MODES.values()]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"current": self.current, "he": MODES[self.current].he,
                    "since": self.since, "held_seconds": round(time.time() - self.since, 1),
                    "switches": len(self.history), "count": len(MODES)}


STATE = ModeState()
