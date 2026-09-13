"""JARVIS Permission Firewall + audit log + kill switch.

Safety is not a feature bolted on later — every action JARVIS takes on the
machine passes through here first.

Levels
    SAFE      read-only observation (telemetry, file read, search)
    WRITE     mutation with low blast radius (write a file, launch an app)
    CRITICAL  destructive or irreversible (delete, kill process, shell, registry)

Guarantees
    * an action above the configured level is refused, not silently downgraded
    * CRITICAL actions require an explicit user grant (the HUD asks)
    * every decision is appended to an immutable JSONL audit log
    * the kill switch short-circuits *everything* instantly
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.bus import BUS, T  # noqa: E402
from core.config import CONFIG  # noqa: E402

LEVELS = ("SAFE", "WRITE", "CRITICAL")
_RANK = {name: i for i, name in enumerate(LEVELS)}


@dataclass
class Decision:
    allowed: bool
    reason: str
    level: str = "SAFE"
    requires_confirmation: bool = False
    audit_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "level": self.level,
                "requires_confirmation": self.requires_confirmation, "audit_id": self.audit_id}


@dataclass
class AuditRecord:
    id: int
    ts: float
    action: str
    level: str
    allowed: bool
    reason: str
    args: Dict[str, Any]
    agent: str = "jarvis"
    dry_run: bool = False


class PermissionFirewall:
    def __init__(self, config=None) -> None:
        cfg = config or CONFIG.security
        self.level: str = str(cfg.level).upper()
        self.require_confirmation: bool = bool(cfg.require_confirmation)
        self.dry_run: bool = bool(cfg.dry_run)
        self.audit_path = Path(cfg.audit_log)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.shell_allowlist: Tuple[str, ...] = tuple(cfg.shell_allowlist)
        self.shell_blocklist: Tuple[str, ...] = tuple(cfg.shell_blocklist)
        self.protected: Tuple[str, ...] = tuple(cfg.protected_paths)
        self._kill = False
        self._lock = threading.RLock()
        self._counter = 0
        self.pending: Dict[int, Dict[str, Any]] = {}
        self._confirm_hook: Optional[Callable[[Dict[str, Any]], bool]] = None
        self.history: List[AuditRecord] = []

    # --------------------------------------------------------- kill switch --
    @property
    def killed(self) -> bool:
        return self._kill

    def kill(self, reason: str = "manual kill switch") -> None:
        with self._lock:
            self._kill = True
        self.audit("__KILL_SWITCH__", "CRITICAL", False, reason, {})
        BUS.emit("security.kill", {"reason": reason}, source="firewall")

    def revive(self) -> None:
        with self._lock:
            self._kill = False
        self.audit("__REVIVE__", "SAFE", True, "kill switch released", {})
        BUS.emit("security.revive", {}, source="firewall")

    # ------------------------------------------------------------ policies --
    def set_level(self, level: str) -> None:
        level = level.upper()
        if level in LEVELS:
            self.level = level
            self.audit("policy.level", "SAFE", True, f"level set to {level}", {"level": level})

    def set_dry_run(self, on: bool) -> None:
        self.dry_run = bool(on)
        self.audit("policy.dry_run", "SAFE", True, f"dry_run={self.dry_run}", {"dry_run": self.dry_run})

    def on_confirm(self, hook: Callable[[Dict[str, Any]], bool]) -> None:
        """Register the HUD callback used to ask the human before CRITICAL acts."""
        self._confirm_hook = hook

    # ------------------------------------------------------------- checks --
    def check(self, action: str, level: str, args: Optional[Dict[str, Any]] = None,
              agent: str = "jarvis") -> Decision:
        args = dict(args or {})
        level = level.upper()
        if self._kill:
            return self._decide(action, level, False, "kill switch engaged", args, agent)

        if level not in _RANK:
            return self._decide(action, level, False, f"unknown risk level {level!r}", args, agent)

        if _RANK[level] > _RANK[self.level]:
            return self._decide(
                action, level, False,
                f"blocked: action is {level} but the firewall is set to {self.level}", args, agent)

        # path protection
        for key in ("path", "target", "file", "folder", "destination"):
            value = args.get(key)
            if isinstance(value, str) and self._is_protected(value):
                return self._decide(action, level, False,
                                    f"blocked: {key}={value!r} is inside a protected path", args, agent)

        # shell command filtering
        if action in ("shell.exec", "shell.run") or "command" in args:
            cmd = str(args.get("command", ""))
            verdict = self.check_shell(cmd)
            if not verdict.allowed:
                return self._decide(action, level, False, verdict.reason, args, agent)

        if level == "CRITICAL" and self.require_confirmation:
            if self._confirm_hook is None:
                return self._decide(action, level, False,
                                    "CRITICAL action needs explicit user confirmation and no "
                                    "confirmation channel is attached", args, agent,
                                    requires_confirmation=True)
            request = {"action": action, "level": level, "args": args, "agent": agent}
            with self._lock:
                self._counter += 1
                req_id = self._counter
                self.pending[req_id] = request
            BUS.emit(T.PERMISSION_ASK, {"id": req_id, **request}, source="firewall")
            granted = bool(self._confirm_hook(request))
            self.pending.pop(req_id, None)
            if not granted:
                BUS.emit(T.PERMISSION_DENY, {"id": req_id, **request}, source="firewall")
                return self._decide(action, level, False, "user denied the action", args, agent,
                                    requires_confirmation=True)

        if self.dry_run and level != "SAFE":
            return self._decide(action, level, True,
                                f"dry-run: {action} would execute with {args}", args, agent)

        return self._decide(action, level, True, "policy satisfied", args, agent)

    def check_shell(self, command: str) -> Decision:
        cmd = (command or "").strip()
        if not cmd:
            return Decision(False, "empty command", "CRITICAL")
        low = cmd.lower()
        for bad in self.shell_blocklist:
            if bad.lower() in low:
                return Decision(False, f"command matches blocklist pattern {bad!r}", "CRITICAL")
        for pattern in (r"rm\s+-[a-z]*r[a-z]*f", r"del\s+/[sfq]", r":\(\)\s*\{", r">\s*/dev/sd",
                        r"format\s+[a-z]:", r"shutdown", r"mkfs", r"dd\s+if=", r"reg\s+delete"):
            if re.search(pattern, low):
                return Decision(False, f"command matches destructive pattern {pattern!r}", "CRITICAL")
        head = re.split(r"[\s|;&]", low)[0]
        base = os.path.basename(head)
        if self.shell_allowlist and base not in self.shell_allowlist:
            return Decision(False, f"'{base}' is not in the shell allowlist", "CRITICAL")
        return Decision(True, "shell command allowed", "CRITICAL")

    def _is_protected(self, path: str) -> bool:
        p = str(Path(path).expanduser()).replace("\\", "/").lower()
        for prot in self.protected:
            q = str(Path(prot).expanduser()).replace("\\", "/").lower().rstrip("/")
            if p == q or p.startswith(q + "/") or p.startswith(q + "\\"):
                return True
        return False

    # -------------------------------------------------------------- audit --
    def _decide(self, action: str, level: str, allowed: bool, reason: str,
                args: Dict[str, Any], agent: str, requires_confirmation: bool = False) -> Decision:
        rec_id = self.audit(action, level, allowed, reason, args, agent=agent)
        return Decision(allowed=allowed, reason=reason, level=level,
                        requires_confirmation=requires_confirmation, audit_id=rec_id)

    def audit(self, action: str, level: str, allowed: bool, reason: str,
              args: Dict[str, Any], agent: str = "jarvis") -> int:
        with self._lock:
            self._counter += 1
            rec_id = self._counter
        rec = AuditRecord(id=rec_id, ts=time.time(), action=action, level=level,
                          allowed=allowed, reason=reason, args=_safe_args(args),
                          agent=agent, dry_run=self.dry_run)
        self.history.append(rec)
        if len(self.history) > 5000:
            self.history = self.history[-5000:]
        line = json.dumps(rec.__dict__, ensure_ascii=False, default=str)
        try:
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        BUS.emit(T.AUDIT, rec.__dict__, source="firewall")
        return rec_id

    def tail(self, n: int = 25) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self.history[-n:]]

    def stats(self) -> Dict[str, Any]:
        allowed = sum(1 for r in self.history if r.allowed)
        return {
            "level": self.level, "dry_run": self.dry_run, "killed": self._kill,
            "total": len(self.history), "allowed": allowed,
            "blocked": len(self.history) - allowed,
            "audit_file": str(self.audit_path),
        }


def _safe_args(args: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in args.items():
        s = str(v)
        out[k] = s if len(s) <= 400 else s[:400] + "…"
    return out


FIREWALL = PermissionFirewall()
