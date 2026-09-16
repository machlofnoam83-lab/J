"""Presence: the bridge between what the camera sees and what the brain knows.

Until this module existed, ``vision/`` and ``brain/`` did not touch. ``FaceGate``
pushed a permission level into the firewall and that was the whole connection —
the reasoning engine had no idea anybody was there. It could not greet you by
name, could not tell you who it thought was in the room, and could not refuse a
stranger for any reason other than a firewall rule it did not understand.

This module is deliberately thin. It does not run detection; ``vision/`` does
that and calls :meth:`PresenceTracker.observe` with the result. What it adds is
the part the brain actually needs:

  * a single current answer to "who is in front of me right now", with an age
  * the same lease semantics the gate uses, so the brain and the firewall cannot
    disagree about whether you are still in the room
  * a risk check phrased in the brain's terms, so a refusal can be *explained*
    rather than surfacing as an opaque firewall block
  * a Hebrew phrase for the situation, because the brain speaks to a person

Nothing here grants anything. It reports, and it can only ever recommend
*less* than the firewall already allows.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEVELS = ("SAFE", "WRITE", "CRITICAL")
_RANK = {name: i for i, name in enumerate(LEVELS)}

# How long an observation stays believable without a fresh frame. Matches the
# default gate lease so the brain and the firewall expire on the same clock —
# two different numbers here would mean the brain greeting someone the firewall
# had already forgotten.
DEFAULT_TTL = 12.0


@dataclass
class Presence:
    """Who the camera believes is in front of the machine, right now."""

    identity: Optional[str] = None
    name: str = "—"
    level: str = "SAFE"
    role: str = "guest"
    confidence: float = 0.0
    known: bool = False
    people: int = 0
    quality: float = 0.0
    liveness: str = ""
    summary_he: str = ""
    age: float = 0.0
    withheld: str = ""
    # The encoded face from the frame that produced this answer. This is what
    # makes enrolment possible from the brain at all: before it, recognize()
    # computed a vector and discarded it, so there was nothing left to enrol
    # from and "register me" could only ever be a refusal.
    # Never serialised — 4096 floats must not ride out to the HUD every frame.
    vector: Optional[Any] = None

    @property
    def has_face(self) -> bool:
        """True when there is an encoded face available to enrol."""
        return self.vector is not None and not self.stale

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    @property
    def stale(self) -> bool:
        return self.age > DEFAULT_TTL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity, "name": self.name, "level": self.level,
            "role": self.role, "is_owner": self.is_owner, "known": self.known,
            "confidence": round(float(self.confidence), 4), "people": self.people,
            "quality": round(float(self.quality), 3), "liveness": self.liveness,
            "summary_he": self.summary_he, "age": round(float(self.age), 2),
            "stale": self.stale, "withheld": self.withheld,
        }


def _empty() -> Presence:
    return Presence()


class PresenceTracker:
    """Holds the current presence and answers the brain's questions about it."""

    def __init__(self, ttl: float = DEFAULT_TTL) -> None:
        self.ttl = float(ttl)
        self._lock = threading.RLock()
        self._p = Presence()
        self._seen: float = 0.0
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- observe --
    def observe(self, match: Any = None, scene: Any = None,
                gate: Any = None) -> Presence:
        """Feed one camera frame's result.

        ``match`` is a ``vision.faces.Match``, ``scene`` a ``vision.scene.Scene``;
        both optional so a caller with only one of them still updates coherently.
        ``gate`` lets the firewall's own view win when it disagrees, because the
        firewall is what actually enforces anything.
        """
        with self._lock:
            now = time.perf_counter()
            p = self._p

            if match is not None:
                known = bool(getattr(match, "known", False))
                p.known = known
                p.confidence = float(getattr(match, "confidence", 0.0) or 0.0)
                p.people = int(getattr(match, "faces", 0) or 0)
                # Keep the encoded face, whatever the identification verdict was.
                # An unknown stranger is exactly the case enrolment exists for, so
                # dropping the vector when known is False would defeat the purpose.
                v = getattr(match, "vector", None)
                p.vector = v if v is not None else None
                if known:
                    p.identity = getattr(match, "identity", None)
                    p.name = getattr(match, "name", "") or "—"
                    p.level = str(getattr(match, "level", "SAFE") or "SAFE").upper()
                    p.role = str(getattr(match, "role", "guest") or "guest").lower()
                else:
                    p.identity = None
                    p.name = getattr(match, "name", "") or "לא מזוהה"
                    p.level = "SAFE"
                    p.role = "guest"

            if scene is not None:
                p.people = int(getattr(scene, "people", p.people) or 0)
                p.quality = float(getattr(scene, "quality", 0.0) or 0.0)
                p.summary_he = str(getattr(scene, "summary_he", "") or "")
                lv = getattr(scene, "liveness", None)
                p.liveness = str(getattr(lv, "verdict", "") or "") if lv else ""

            # Liveness can only subtract. See vision/scene.py for why this is
            # one-directional: a false "suspect" costs a re-enrolment, a false
            # "live" hands CRITICAL to a photograph.
            if p.liveness == "suspect" and p.level != "SAFE":
                p.withheld = "liveness"
                p.level = "SAFE"

            # The firewall is the authority. If it has already fallen back, the
            # brain must not keep addressing someone by name.
            if gate is not None:
                try:
                    gl = str(getattr(gate, "level", "") or "").upper()
                    if gl in _RANK and _RANK[gl] < _RANK.get(p.level, 0):
                        p.level = gl
                except Exception:                                  # pragma: no cover
                    pass

            if p.people or p.known:
                self._seen = now
            p.age = (now - self._seen) if self._seen else float(self.ttl + 1)

            if p.age > self.ttl:
                # Expired means expired. Keeping the head count or the scene text
                # would let the brain assert "someone is here" on evidence that
                # has already gone stale — the honest answer is that the room is
                # unknown again.
                self._p = Presence(age=p.age)
                p = self._p

            self._history.append({"ts": time.time(), **p.to_dict()})
            if len(self._history) > 120:
                self._history = self._history[-120:]
            return p

    # -------------------------------------------------------------- queries --
    def current(self) -> Presence:
        """The present answer, aged. Never raises, never returns None."""
        with self._lock:
            p = self._p
            p.age = (time.perf_counter() - self._seen) if self._seen else float(self.ttl + 1)
            if p.age > self.ttl and (p.known or p.identity or p.people):
                self._p = Presence(age=p.age)
                p = self._p
            return p

    def history(self, k: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history[-k:])

    def clear(self) -> None:
        with self._lock:
            self._p = Presence()
            self._seen = 0.0

    # ------------------------------------------------------------ reasoning --
    def allows(self, risk: str) -> bool:
        """Would the person in front of the camera be allowed ``risk``?"""
        risk = str(risk or "SAFE").upper()
        if risk not in _RANK:
            return True
        return _RANK[risk] <= _RANK.get(self.current().level, 0)

    def explain(self, risk: str) -> str:
        """A Hebrew reason the brain can actually say out loud.

        The firewall already produces a reason, but it is phrased for a log
        ("blocked: action is CRITICAL but the firewall is set to SAFE"). This is
        phrased for the person who has to act on it.
        """
        p = self.current()
        risk = str(risk or "SAFE").upper()
        if self.allows(risk):
            return ""
        if p.people == 0:
            return "אף אחד לא מול המצלמה, אז אני נשאר ברמת SAFE."
        if not p.known:
            return (f"יש מישהו מול המצלמה אבל לא זיהיתי אותו, ולכן ההרשאה היא SAFE "
                    f"ו‑{risk} דורש יותר מזה.")
        if p.withheld == "liveness":
            return ("הפריים נראה כמו מסך או תמונה ולא כמו פנים חיות, "
                    "אז לא העליתי הרשאה.")
        return (f"{p.name} מורשה עד {p.level}, ו‑{risk} מעל זה.")

    def address(self) -> str:
        """How to refer to the person in the room, in Hebrew."""
        p = self.current()
        if p.known and p.name and p.name != "—":
            return p.name
        if p.people:
            return ""
        return ""

    def greeting_hint(self) -> str:
        """A short Hebrew clause the brain may fold into a greeting.

        Empty when there is nothing worth saying — the brain must not invent a
        person who is not there.
        """
        p = self.current()
        if p.known and p.name and p.name != "—":
            if p.is_owner:
                return f"שלום {p.name}."
            return f"שלום {p.name} — ההרשאה שלך כרגע {p.level}."
        if p.people and not p.known:
            return "אני רואה מישהו מול המצלמה אבל לא מזהה אותו."
        return ""


# ─────────────────────────────────────────────────────────── process-wide ──
# One tracker per process, shared by the server (which feeds it every camera
# frame) and the brain (which reads it every turn). Two instances would let the
# brain greet someone the server had already stopped seeing.
_TRACKER: Optional[PresenceTracker] = None
_LOCK = threading.RLock()


def get_presence(fresh: bool = False) -> PresenceTracker:
    """The process-wide presence tracker."""
    global _TRACKER
    with _LOCK:
        if _TRACKER is None or fresh:
            _TRACKER = PresenceTracker()
        return _TRACKER
