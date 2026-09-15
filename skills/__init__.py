"""Skill loader — importing this package registers every capability JARVIS has.

``load_all()`` is idempotent and returns a report so the HUD/boot sequence can
show exactly which hands are attached and which are unavailable on this machine.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.registry import REGISTRY, Skill, SkillResult  # noqa: E402,F401

MODULES = (
    "skills.sk_math",
    "skills.sk_time",
    "skills.sk_system",
    "skills.sk_fs",
    "skills.sk_apps",
    "skills.sk_clipboard",
    "skills.sk_shell",
    "skills.sk_media_input",
    "skills.sk_research",
    "skills.sk_rag",
)

_loaded = False


def load_all(verbose: bool = False) -> Dict[str, Any]:
    global _loaded
    report: Dict[str, Any] = {"modules": {}, "skills": [], "errors": []}
    for name in MODULES:
        try:
            importlib.import_module(name)
            report["modules"][name] = "ok"
        except Exception as exc:
            report["modules"][name] = f"failed: {exc}"
            report["errors"].append({"module": name, "error": str(exc)})
        if verbose:
            print(f"  [skills] {name}: {report['modules'][name]}")
    report["skills"] = REGISTRY.schema()
    report["count"] = len(report["skills"])
    _loaded = True
    return report


def schema() -> List[Dict[str, Any]]:
    if not _loaded:
        load_all()
    return REGISTRY.schema()


def summary() -> str:
    if not _loaded:
        load_all()
    by_risk: Dict[str, int] = {}
    by_agent: Dict[str, int] = {}
    for s in REGISTRY.all():
        by_risk[s.risk] = by_risk.get(s.risk, 0) + 1
        by_agent[s.agent] = by_agent.get(s.agent, 0) + 1
    risk = ", ".join(f"{k}={v}" for k, v in sorted(by_risk.items()))
    agent = ", ".join(f"{k}={v}" for k, v in sorted(by_agent.items()))
    return f"{len(REGISTRY.names())} skills ({risk}) across agents: {agent}"
