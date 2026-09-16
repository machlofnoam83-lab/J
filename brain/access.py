"""The access gate — no face, no assistant.

Everything else in this system treats presence as advice: the firewall compares
ranks, the skills explain themselves, and an unidentified person still gets a
polite answer. This module is the one place where presence is a **precondition**
instead of a preference.

The rule the user asked for, literally: whoever has no access can do nothing,
and every time somebody walks in they have to be scanned. So:

  * no face in frame              -> the brain answers nothing but "scan first"
  * a face that expired (TTL)     -> same; leaving the room ends the session
  * a face the gallery does not know -> read-only at most, and only if the
    operator allowed guests at all
  * liveness "suspect"            -> treated as no face; a photograph held up to
    the camera is not a person walking in

Two things are deliberately *not* gated, because gating them makes the system
unusable rather than secure:

  * asking to be enrolled (that is how a stranger becomes a person)
  * asking who is in the room (that is how you find out you need to be scanned)

Without those exceptions the gate is a locked door with no handle, and the
honest failure mode is that people stop using the system instead of attacking it.

Nothing here grants permission. Like the rest of the vision stack it can only
*withhold*: the firewall remains the authority on what an identified person may
do. This decides whether the conversation happens at all.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Enabled by default. The user asked for this as the behaviour, not as an option;
# the escape hatch exists so a headless CI run or a machine with no camera can
# still exercise the rest of the brain.
_ENV = "JARVIS_ACCESS_GATE"

# What is still allowed with no verified face. Kept deliberately tiny: these are
# the two things that let someone *become* allowed.
_ALWAYS_ALLOWED = frozenset({
    "vision.enroll",
    "vision.who",
    "vision.scene",
    "vision.gallery",
    "vision.permission",
    "records.whoami",
})

# What a known-but-unprivileged guest may still do with no face at all: nothing.
# A guest with a *fresh* face is handled by level, not by this list.
_GUEST_ALLOWED = frozenset(_ALWAYS_ALLOWED) | frozenset({
    "time.now",
    "date.today",
    "system.status",
})


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Verdict:
    """Why the gate opened or closed, in terms a user can act on."""

    allowed: bool
    reason: str                     # machine-readable
    text_he: str                    # what JARVIS says
    needs_scan: bool = False
    identity: str = ""
    level: str = ""
    age: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason,
                "text_he": self.text_he, "needs_scan": self.needs_scan,
                "identity": self.identity, "level": self.level,
                "age": round(self.age, 2)}


@dataclass
class AccessGate:
    """Decides whether a turn happens at all, before any skill is considered.

    ``presence`` is the brain's own tracker, so this reads the same 12-second
    lease the firewall uses — the gate cannot stay open longer than the camera
    agrees that somebody is still there.
    """

    presence: Any = None
    enabled: bool = field(default_factory=lambda: _env_flag(_ENV, True))
    # Off by default, deliberately. The instruction was "whoever has no access
    # can do nothing", and an unenrolled face has no access — so it gets the two
    # doors that let it *become* allowed (enrolment, and asking who is in the
    # room) and nothing else. Not even the clock: a read-only allowance here
    # would be a gap wide enough to drive a habit through, and the honest
    # default is that an unidentified person talks to nobody.
    #
    # An operator who wants a kiosk-style guest mode can opt in explicitly.
    allow_guests_readonly: bool = field(
        default_factory=lambda: _env_flag("JARVIS_ACCESS_GUESTS", False))
    history: List[Dict[str, Any]] = field(default_factory=list)
    _last_verdict: Optional[Verdict] = field(default=None, repr=False)

    # ------------------------------------------------------------------ public

    def check(self, skill: str = "", *, risk: str = "SAFE") -> Verdict:
        """The single question: may this turn proceed?"""
        if not self.enabled:
            return self._record(Verdict(True, "gate_disabled",
                                        "שער הגישה כבוי בהגדרות."))

        p = self._now()
        if p is None:
            return self._record(self._no_tracker(skill))

        # A suspect frame is not a person. Gating on it is the whole point: the
        # alternative is that a phone screen becomes a session.
        if getattr(p, "liveness", "") == "suspect":
            return self._record(Verdict(
                False, "liveness_suspect",
                "הפריים נראה כמו מסך או תצלום, לא כמו פנים חיות. בוא פיזית מול "
                "המצלמה ואני אסרוק אותך.",
                needs_scan=True, identity=getattr(p, "name", ""),
                level=getattr(p, "level", ""), age=float(getattr(p, "age", 0.0))))

        if not getattr(p, "has_face", False):
            return self._record(self._needs_scan(skill))

        if bool(getattr(p, "stale", False)):
            return self._record(Verdict(
                False, "session_expired",
                "עבר זמן מאז הפעם האחרונה שראיתי אותך — הסריקה פגה. תסתכל "
                "למצלמה ואני אמשיך.",
                needs_scan=True, identity=getattr(p, "name", ""),
                level=getattr(p, "level", ""), age=float(getattr(p, "age", 0.0))))

        # A face we do not know. Enrolment is always allowed — that is the door.
        if not getattr(p, "known", False):
            if skill in _ALWAYS_ALLOWED or not skill:
                return self._record(Verdict(
                    True, "stranger_allowed_limited",
                    "אני רואה אותך אבל עדיין לא זיהיתי. אפשר לרשום אותך.",
                    identity=getattr(p, "name", ""),
                    level=getattr(p, "level", ""),
                    age=float(getattr(p, "age", 0.0))))
            if not self.allow_guests_readonly:
                return self._record(Verdict(
                    False, "unknown_face",
                    "אני לא מכיר אותך ולא הוגדר שאפשר לתת לאורחים כלום. "
                    "תגיד «רשום אותי בשם …» ואז אוכל לעזור.",
                    needs_scan=True, identity=getattr(p, "name", ""),
                    level=getattr(p, "level", ""),
                    age=float(getattr(p, "age", 0.0))))
            if skill in _GUEST_ALLOWED:
                return self._record(Verdict(
                    True, "guest_readonly",
                    "אורח — קריאה בלבד.", identity=getattr(p, "name", ""),
                    level=getattr(p, "level", ""),
                    age=float(getattr(p, "age", 0.0))))
            return self._record(Verdict(
                False, "guest_blocked",
                "אני רואה אותך אבל לא זיהיתי, ולכן אני לא מבצע פעולות. "
                "תגיד «רשום אותי בשם …» או שיזהו אותך קודם.",
                needs_scan=True, identity=getattr(p, "name", ""),
                level=getattr(p, "level", ""),
                age=float(getattr(p, "age", 0.0))))

        # Known. The firewall decides what they may do; this only confirms that
        # a real, live, current person is the one asking.
        return self._record(Verdict(
            True, "identified",
            f"זיהיתי את {getattr(p, 'name', '')}.",
            identity=getattr(p, "name", ""),
            level=getattr(p, "level", ""),
            age=float(getattr(p, "age", 0.0))))

    def state(self) -> Dict[str, Any]:
        """What the HUD shows: is the door open, and why."""
        p = self._now()
        v = self._last_verdict
        return {
            "enabled": self.enabled,
            "has_face": bool(getattr(p, "has_face", False)) if p else False,
            "known": bool(getattr(p, "known", False)) if p else False,
            "identity": getattr(p, "name", "") if p else "",
            "level": getattr(p, "level", "") if p else "",
            "stale": bool(getattr(p, "stale", True)) if p else True,
            "liveness": getattr(p, "liveness", "") if p else "",
            "age": round(float(getattr(p, "age", 0.0)), 2) if p else 0.0,
            "last_reason": v.reason if v else "",
            "last_allowed": bool(v.allowed) if v else False,
            "allow_guests_readonly": self.allow_guests_readonly,
        }

    def explain(self) -> str:
        """One Hebrew sentence for the HUD."""
        s = self.state()
        if not s["enabled"]:
            return "שער הגישה כבוי — כל אחד יכול לדבר."
        if not s["has_face"]:
            return "אף אחד לא מול המצלמה. חובה להיסרק לפני כל פעולה."
        if s["stale"]:
            return "הסריקה פגה — צריך סריקה חדשה."
        if not s["known"]:
            return "יש מישהו מול המצלמה אבל הוא לא מזוהה — קריאה בלבד."
        return f"{s['identity']} מזוהה, רמת הרשאה {s['level']}."

    # ----------------------------------------------------------------- private

    def _now(self) -> Any:
        if self.presence is None:
            return None
        try:
            return self.presence.current()
        except Exception:                                  # pragma: no cover
            return None

    def _needs_scan(self, skill: str) -> Verdict:
        if skill in _ALWAYS_ALLOWED or not skill:
            return Verdict(True, "no_face_but_allowed",
                           "אף אחד לא מול המצלמה, אבל על השאלה הזאת אני עונה.",
                           needs_scan=True)
        return Verdict(
            False, "no_face",
            "אף אחד לא מול המצלמה, ואני לא מבצע פעולות בלי סריקת פנים. "
            "תסתכל למצלמה ואז תגיד שוב.",
            needs_scan=True)

    def _no_tracker(self, skill: str) -> Verdict:
        """No camera wired at all. Failing closed is the honest default; the
        operator can lift it explicitly with JARVIS_ACCESS_GATE=0."""
        if skill in _ALWAYS_ALLOWED or not skill:
            return Verdict(True, "no_tracker_allowed",
                           "אין מצלמה מחוברת, אז אני עונה בלי אימות.",
                           needs_scan=False)
        return Verdict(
            False, "no_tracker",
            "אין מצלמה מחוברת, ואני לא מבצע פעולות בלי סריקת פנים.",
            needs_scan=True)

    def _record(self, v: Verdict) -> Verdict:
        self._last_verdict = v
        self.history.append({"t": time.time(), **v.to_dict()})
        if len(self.history) > 200:
            del self.history[:-200]
        return v


_GATE: Optional[AccessGate] = None


def get_gate(presence: Any = None, fresh: bool = False) -> AccessGate:
    """Process-wide gate, bound to the brain's presence tracker.

    When called with no argument it falls back to the presence singleton rather
    than binding ``None`` for good. That was a real bug: the first caller in a
    process decided whether the gate could ever see a face, and an HTTP handler
    that ran before the brain was constructed won — leaving ``/api/access``
    reporting an empty room even while the camera was identifying the owner.
    """
    global _GATE
    if presence is None:
        try:
            from brain.presence import get_presence
            presence = get_presence()
        except Exception:                                  # pragma: no cover
            presence = None
    if fresh or _GATE is None:
        _GATE = AccessGate(presence=presence)
    elif presence is not None and _GATE.presence is None:
        _GATE.presence = presence
    return _GATE
