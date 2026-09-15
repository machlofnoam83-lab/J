"""The sentinel — anomaly detection over JARVIS's own behaviour.

The firewall judges each action alone: is this skill, at this level, allowed now?
That misses the shape of a sequence. Six WRITE actions in a minute is a different
story from six in an hour; four denials in two minutes means something is probing
for the edge of the policy; the same CRITICAL skill three times in a minute means
a loop or a stuck retry, not a user. None of those are visible to a per-call check.

So the sentinel watches the event bus instead of the call: tool invocations,
denials and audit entries arrive here as they happen, and three rules score the
recent window. An alert is written to data/sentinel.jsonl, emitted on the bus for
the HUD, and counted in stats().

Actuation is deliberately separate from detection and off by default. Tripping a
rule can engage the firewall's dry-run for a cooldown — a real, reversible
protective move — but automatically changing the security posture of a running
assistant is the kind of surprise that turns a safety feature into an outage, so
it happens only when explicitly armed (JARVIS_SENTINEL_ACT=1 or sentinel.arm()).
Detection never depends on it.

Everything is in-process and offline. The sentinel has no network, no model and
no weights: three counters with windows, which is exactly what the threat here
is — a runaway local loop, not a remote adversary.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "sentinel.jsonl"

# Windows and thresholds, in seconds and counts. Measured against the test
# suites first: a legitimate session never issues six non-SAFE actions inside a
# minute, and the suites' denial bursts stay under four in two minutes, so these
# catch abnormal shapes without firing on honest use.
BURST_WINDOW = 60.0
BURST_COUNT = 6
DENY_WINDOW = 120.0
DENY_COUNT = 4
REPEAT_WINDOW = 60.0
REPEAT_COUNT = 3
COOLDOWN = 120.0


class Sentinel:
    """Watches invocation shapes and raises alerts on abnormal ones."""

    def __init__(self, log_path: Optional[Path] = None, acting: Optional[bool] = None,
                 firewall: Any = None) -> None:
        self.log_path = Path(log_path) if log_path else LOG_PATH
        self._lock = threading.Lock()
        self._actions: Deque[float] = deque()          # timestamps of non-SAFE calls
        self._names: Deque[tuple] = deque()            # (ts, skill) for repeat rule
        self._denials: Deque[float] = deque()
        self.alerts: List[Dict[str, Any]] = []
        self.firewall = firewall
        self._acting = bool(acting if acting is not None
                            else os.environ.get("JARVIS_SENTINEL_ACT", "") in ("1", "true", "yes"))
        self._cooldown_until = 0.0
        self._timer: Optional[threading.Timer] = None
        self._attached = False

    # ------------------------------------------------------------ wiring --
    def attach(self, bus: Any) -> None:
        """Subscribe to the live event bus. Idempotent."""
        if self._attached or bus is None:
            return
        bus.on("brain.tool.call", lambda ev: self.observe_call(
            str((ev.data or {}).get("skill", "")), str((ev.data or {}).get("risk", ""))))
        bus.on("security.permission.deny", lambda ev: self.observe_denial())
        self._attached = True

    def arm(self, acting: bool = True) -> None:
        with self._lock:
            self._acting = bool(acting)

    @property
    def acting(self) -> bool:
        with self._lock:
            return self._acting

    # --------------------------------------------------------- observation --
    def observe_call(self, skill: str, risk: str = "") -> Optional[Dict[str, Any]]:
        """One tool invocation. Only non-SAFE shapes feed the rules.

        SAFE calls are ignored entirely, repeat rule included: twenty time.now
        calls in a minute is a chatty session, not a threat, and a watchdog that
        cries over reading the clock trains the user to ignore it.
        """
        if risk == "SAFE":
            return None
        now = time.time()
        alert = None
        with self._lock:
            self._actions.append(now)
            self._names.append((now, skill))
            self._purge(now)
            recent_actions = len(self._actions)
            repeats = sum(1 for ts, nm in self._names if nm == skill)
            if recent_actions >= BURST_COUNT:
                alert = self._alert("burst", now, {
                    "count": recent_actions, "window": BURST_WINDOW,
                    "skills": [nm for _, nm in list(self._names)[-BURST_COUNT:]]})
            elif repeats >= REPEAT_COUNT and skill:
                alert = self._alert("repeat", now, {"skill": skill, "count": repeats,
                                                    "window": REPEAT_WINDOW})
        return alert

    def observe_denial(self) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            self._denials.append(now)
            self._purge(now)
            if len(self._denials) >= DENY_COUNT:
                return self._alert("denial_streak", now,
                                   {"count": len(self._denials), "window": DENY_WINDOW})
        return None

    def _purge(self, now: float) -> None:
        while self._actions and now - self._actions[0] > BURST_WINDOW:
            self._actions.popleft()
        while self._denials and now - self._denials[0] > DENY_WINDOW:
            self._denials.popleft()
        while self._names and now - self._names[0][0] > BURST_WINDOW:
            self._names.popleft()

    # -------------------------------------------------------------- alerts --
    def _alert(self, kind: str, now: float, detail: Dict[str, Any]) -> Dict[str, Any]:
        # one alert per kind per window, or a stuck loop would log forever
        for a in reversed(self.alerts):
            if a["kind"] == kind and now - a["ts"] < BURST_WINDOW:
                return a
        rec = {"kind": kind, "ts": now, "detail": detail,
               "acted": False, "action": ""}
        if self._acting and self.firewall is not None and not self.firewall.dry_run:
            self._engage_cooldown(rec)
        self.alerts.append(rec)
        if len(self.alerts) > 200:
            self.alerts = self.alerts[-200:]
        self._write(rec)
        try:
            from core.bus import BUS
            BUS.emit("sentinel.alert", rec, source="sentinel")
        except Exception:
            pass
        return rec

    def _engage_cooldown(self, rec: Dict[str, Any]) -> None:
        """Flip the firewall into dry-run for COOLDOWN seconds, then restore."""
        fw = self.firewall
        fw.dry_run = True
        rec["acted"] = True
        rec["action"] = f"dry-run engaged for {COOLDOWN:.0f}s"
        self._cooldown_until = time.time() + COOLDOWN
        if self._timer is not None:
            self._timer.cancel()

        def _restore() -> None:
            with self._lock:
                if time.time() >= self._cooldown_until - 0.5:
                    fw.dry_run = False

        self._timer = threading.Timer(COOLDOWN, _restore)
        self._timer.daemon = True
        self._timer.start()

    def _write(self, rec: Dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # --------------------------------------------------------------- stats --
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            self._purge(now)
            return {"acting": self._acting,
                    "alerts": len(self.alerts),
                    "recent_non_safe": len(self._actions),
                    "recent_denials": len(self._denials),
                    "cooldown_active": now < self._cooldown_until,
                    "kinds": sorted({a["kind"] for a in self.alerts}),
                    "log": str(self.log_path)}

    def tail(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.alerts)[-n:]


SENTINEL = Sentinel()
