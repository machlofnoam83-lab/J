"""JARVIS event bus.

Every subsystem — brain, voice, agents, skills, UI — talks through this bus.
Nothing calls anything else directly. That is what makes the HUD able to
visualise the mind in real time: the *bus is the observability layer*.

Supports both sync callbacks and asyncio coroutines, wildcard subscriptions
(``brain.*``), ring-buffered history (for the UI timeline) and typed events.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

_SEQ = itertools.count(1)


@dataclass
class Event:
    topic: str
    data: Any = None
    source: str = "system"
    ts: float = field(default_factory=time.time)
    id: int = field(default_factory=lambda: next(_SEQ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "source": self.source,
            "ts": self.ts,
            "data": _jsonable(self.data),
        }


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return repr(obj)


class EventBus:
    """Thread-safe pub/sub with wildcard topics and history."""

    def __init__(self, history_size: int = 2000) -> None:
        self._subs: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self.history: Deque[Event] = deque(maxlen=history_size)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.stats: Dict[str, int] = {}

    # ---------------------------------------------------------- subscribe --
    def on(self, topic: str, handler: Callable) -> Callable:
        """Register ``handler`` for ``topic`` (``*`` wildcards supported)."""
        with self._lock:
            self._subs.setdefault(topic, []).append(handler)
        return handler

    def off(self, topic: str, handler: Callable) -> None:
        with self._lock:
            if topic in self._subs and handler in self._subs[topic]:
                self._subs[topic].remove(handler)

    def once(self, topic: str) -> Any:
        """Block until ``topic`` fires; returns the event. (diagnostics)"""
        box: List[Event] = []
        done = threading.Event()

        def _h(ev: Event) -> None:
            box.append(ev)
            done.set()

        self.on(topic, _h)
        try:
            done.wait(timeout=30)
        finally:
            self.off(topic, _h)
        return box[0] if box else None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------- publish --
    def emit(self, topic: str, data: Any = None, source: str = "system") -> Event:
        ev = Event(topic=topic, data=data, source=source)
        self.history.append(ev)
        self.stats[topic] = self.stats.get(topic, 0) + 1

        for pattern, handlers in list(self._subs.items()):
            if not _matches(pattern, topic):
                continue
            for h in list(handlers):
                try:
                    if inspect.iscoroutinefunction(h):
                        self._schedule_async(h, ev)
                    else:
                        h(ev)
                except Exception as exc:  # a broken subscriber must not kill the bus
                    self.emit("bus.error", {"topic": topic, "error": str(exc)}, source="bus")
        return ev

    def _schedule_async(self, coro_fn: Callable, ev: Event) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro_fn(ev), self._loop)
        else:
            try:
                asyncio.run(coro_fn(ev))
            except RuntimeError:
                pass

    # ------------------------------------------------------------- helpers --
    def recent(self, n: int = 50, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        items: Iterable[Event] = self.history
        if topic:
            items = [e for e in items if _matches(topic, e.topic)]
        return [e.to_dict() for e in list(items)[-n:]]

    def clear(self) -> None:
        self.history.clear()
        self.stats.clear()


def _matches(pattern: str, topic: str) -> bool:
    if pattern == topic or pattern == "*":
        return True
    if pattern.endswith(".*"):
        return topic.startswith(pattern[:-1])
    if pattern.endswith("*"):
        return topic.startswith(pattern[:-1])
    return False


# Global singleton used across the whole application.
BUS = EventBus()


# ---------------------------------------------------------------- topics ----
class T:
    """Canonical topic names. Import this instead of typing strings."""

    BOOT = "system.boot"
    READY = "system.ready"
    SHUTDOWN = "system.shutdown"

    USER_TEXT = "input.user.text"
    USER_VOICE = "input.user.voice"
    WAKE = "input.wake"

    BRAIN_THINK_START = "brain.think.start"
    BRAIN_STEP = "brain.think.step"
    BRAIN_TOKEN = "brain.token"
    BRAIN_PLAN = "brain.plan"
    BRAIN_TOOL_CALL = "brain.tool.call"
    BRAIN_TOOL_RESULT = "brain.tool.result"
    BRAIN_VERIFY = "brain.verify"
    BRAIN_ANSWER = "brain.answer"

    SPEAK_START = "voice.speak.start"
    SPEAK_CHUNK = "voice.speak.chunk"
    SPEAK_END = "voice.speak.end"
    LISTEN_START = "voice.listen.start"
    LISTEN_END = "voice.listen.end"

    MEMORY_WRITE = "memory.write"
    MEMORY_RECALL = "memory.recall"

    AGENT_DISPATCH = "agent.dispatch"
    AGENT_RESULT = "agent.result"

    SKILL_CALL = "skill.call"
    SKILL_RESULT = "skill.result"
    PERMISSION_ASK = "security.permission.ask"
    PERMISSION_DENY = "security.permission.deny"
    AUDIT = "security.audit"

    TELEMETRY = "sys.telemetry"
    UI_STATE = "ui.state"
    ERROR = "error"
