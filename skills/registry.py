"""JARVIS skill registry.

A *skill* is a callable the brain can invoke through a JSON tool-call. Every
skill declares:
  * a machine name (``fs.read``),
  * a JSON argument schema (validated by our own validator — no jsonschema dep),
  * a **risk level** consumed by the Permission Firewall,
  * Hebrew + English trigger phrases used by the Intent Router,
  * whether it is read-only (SAFE), mutating (WRITE) or dangerous (CRITICAL).

Adding a capability to JARVIS = adding one decorated function. Nothing else.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

SAFE, WRITE, CRITICAL = "SAFE", "WRITE", "CRITICAL"


@dataclass
class SkillResult:
    ok: bool
    value: Any = None
    error: str = ""
    ms: float = 0.0
    skill: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    risk: str = SAFE

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "value": self.value, "error": self.error,
                "ms": round(self.ms, 2), "skill": self.skill, "risk": self.risk, **self.data}


@dataclass
class Skill:
    name: str
    fn: Callable[..., Any]
    description_he: str = ""
    description_en: str = ""
    args: Dict[str, str] = field(default_factory=dict)     # name -> type hint
    required: Sequence[str] = ()
    risk: str = SAFE
    triggers_he: Sequence[str] = ()
    triggers_en: Sequence[str] = ()
    agent: str = "hermes"
    enabled: bool = True

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description_he or self.description_en,
            "args": self.args,
            "required": list(self.required),
            "risk": self.risk,
            "agent": self.agent,
        }


class SkillRegistry:
    """Central catalogue of everything JARVIS is able to do."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    # ---------------------------------------------------------- registration --
    def register(
        self,
        name: str,
        *,
        risk: str = SAFE,
        description_he: str = "",
        description_en: str = "",
        required: Sequence[str] = (),
        triggers_he: Sequence[str] = (),
        triggers_en: Sequence[str] = (),
        agent: str = "hermes",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            hints = {
                k: _hint_name(v) for k, v in inspect.signature(fn).parameters.items()
                if k not in ("self", "kwargs")
            }
            self._skills[name] = Skill(
                name=name, fn=fn, description_he=description_he, description_en=description_en,
                args=hints, required=tuple(required), risk=risk,
                triggers_he=triggers_he, triggers_en=triggers_en, agent=agent,
            )
            return fn
        return deco

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    # ------------------------------------------------------------- lookup ----
    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def names(self) -> List[str]:
        return sorted(self._skills)

    def all(self) -> List[Skill]:
        return [self._skills[n] for n in self.names()]

    def by_agent(self, agent: str) -> List[Skill]:
        return [s for s in self.all() if s.agent == agent]

    def by_risk(self, risk: str) -> List[Skill]:
        return [s for s in self.all() if s.risk == risk]

    def schema(self) -> List[Dict[str, Any]]:
        """Machine-readable catalogue fed to the model as the tool contract."""
        return [s.describe() for s in self.all() if s.enabled]

    # -------------------------------------------------------------- invoke ---
    def invoke(self, name: str, args: Optional[Dict[str, Any]] = None,
               permission_granted: bool = False) -> SkillResult:
        skill = self._skills.get(name)
        args = dict(args or {})
        t0 = time.perf_counter()
        if skill is None:
            return SkillResult(ok=False, error=f"unknown skill '{name}'", skill=name,
                               ms=(time.perf_counter() - t0) * 1000)
        if not skill.enabled:
            return SkillResult(ok=False, error=f"skill '{name}' is disabled", skill=name)

        problem = validate_args(skill, args)
        if problem:
            return SkillResult(ok=False, error=problem, skill=name,
                               ms=(time.perf_counter() - t0) * 1000)

        if skill.risk != SAFE and not permission_granted:
            return SkillResult(
                ok=False,
                error=f"permission required: '{name}' is a {skill.risk} action",
                skill=name, data={"risk": skill.risk, "needs_permission": True},
                ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            value = skill.fn(**args)
            if isinstance(value, SkillResult):
                value.skill = name
                value.ms = (time.perf_counter() - t0) * 1000
                return value
            return SkillResult(ok=True, value=value, skill=name, risk=skill.risk,
                               ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:  # a failing skill must never crash the brain
            return SkillResult(ok=False, error=f"{type(exc).__name__}: {exc}", skill=name,
                               data={"risk": skill.risk},
                               ms=(time.perf_counter() - t0) * 1000)


def _hint_name(hint: Any) -> str:
    return getattr(hint, "__name__", str(hint)).replace("typing.", "")


# ------------------------------------------------------------- validation ---
_COERCE = {"int": int, "float": float, "str": str, "bool": bool,
           "list": list, "dict": dict, "Any": lambda v: v}


def validate_args(skill: Skill, args: Dict[str, Any]) -> str:
    """Our own tiny schema validator. Returns '' when valid, else a message."""
    for req in skill.required:
        if req not in args or args[req] in (None, ""):
            return f"missing required argument '{req}' for {skill.name}"
    unknown = set(args) - set(skill.args)
    if unknown and "kwargs" not in skill.args:
        return f"unexpected arguments {sorted(unknown)} for {skill.name}"
    for key, value in list(args.items()):
        hint = skill.args.get(key)
        if hint in (None, "Any", "object"):
            continue
        fn = _COERCE.get(hint)
        if fn is None:
            continue
        try:
            if hint == "bool" and isinstance(value, str):
                args[key] = value.lower() in ("1", "true", "yes", "כן")
            elif hint == "list" and isinstance(value, str):
                args[key] = [v.strip() for v in value.split(",") if v.strip()]
            else:
                args[key] = fn(value)
        except (TypeError, ValueError):
            return f"argument '{key}' of {skill.name} must be {hint}, got {value!r}"
    return ""


# The single global registry used by every subsystem.
REGISTRY = SkillRegistry()
