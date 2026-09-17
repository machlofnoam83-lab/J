#!/usr/bin/env python3
"""Sentinel + memory consolidation tests.

Two additions that only mean something if their failure modes are pinned down:

The sentinel watches sequence shapes the per-call firewall cannot see. The traps
are a watchdog that cries over harmless chatter (twenty SAFE calls is a chatty
session, not a threat) and one that silently misses the shapes it exists for, so
both directions are tested: bursts, repeats and denial streaks alert; SAFE
traffic never does; a stuck loop logs once per window rather than flooding; and
actuation — flipping the firewall into dry-run — stays off unless explicitly
armed, because a safety feature that changes the security posture unasked is an
outage wearing a helmet.

Consolidation promotes recurring conversational themes into semantic facts. Its
trap is promoting sentence fragments and forms of address: the first cut grouped
by crude stem and crowned "איכ" and "דיברנו" as topics about the user's world.
These tests therefore check what gets promoted, not merely that something does.

Run:  python tests/test_sentinel_memory.py
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ok = 0
fail = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {label}" + (f"  {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  ✗ {label}" + (f"  {detail}" if detail else ""))


def _fw(tmp: Path):
    from security.permissions import PermissionFirewall
    cfg = types.SimpleNamespace(level="WRITE", require_confirmation=False,
                                dry_run=False, audit_log=str(tmp / "audit.jsonl"),
                                shell_allowlist=(), shell_blocklist=(),
                                protected_paths=())
    return PermissionFirewall(cfg)


def test_sentinel(tmp: Path) -> None:
    print("\n[1] sentinel: sequence shapes")
    from security.sentinel import Sentinel

    s = Sentinel(log_path=tmp / "s1.jsonl", firewall=_fw(tmp))
    a = None
    for i in range(6):
        a = s.observe_call(f"fs.write{i}", "WRITE") or a
    check("six non-SAFE actions in a minute raise a burst alert",
          a is not None and a["kind"] == "burst", str(a and a["detail"].get("count")))

    s2 = Sentinel(log_path=tmp / "s2.jsonl", firewall=_fw(tmp))
    r = None
    for _ in range(3):
        r = s2.observe_call("shell.exec", "CRITICAL") or r
    check("the same CRITICAL skill three times raises a repeat alert",
          r is not None and r["kind"] == "repeat" and r["detail"]["skill"] == "shell.exec")

    s3 = Sentinel(log_path=tmp / "s3.jsonl", firewall=_fw(tmp))
    d = None
    for _ in range(4):
        d = s3.observe_denial() or d
    check("four denials in two minutes raise a streak alert",
          d is not None and d["kind"] == "denial_streak")

    s4 = Sentinel(log_path=tmp / "s4.jsonl", firewall=_fw(tmp))
    for _ in range(20):
        s4.observe_call("time.now", "SAFE")
    check("twenty SAFE calls raise nothing — chatter is not a threat",
          len(s4.alerts) == 0, f"alerts={len(s4.alerts)}")

    s5 = Sentinel(log_path=tmp / "s5.jsonl", firewall=_fw(tmp))
    for _ in range(12):
        s5.observe_call("fs.write", "WRITE")
    kinds = sorted({x["kind"] for x in s5.alerts})
    check("a stuck loop alerts once per kind per window, not twelve times",
          len(s5.alerts) == len(kinds) and len(s5.alerts) <= 2,
          f"{len(s5.alerts)} alerts {kinds}")

    check("alerts are written to the sentinel log",
          (tmp / "s5.jsonl").exists() and (tmp / "s5.jsonl").stat().st_size > 0)

    st = s5.stats()
    check("stats expose the window counts and alert total",
          st["recent_non_safe"] > 0 and st["alerts"] == len(s5.alerts), str(st)[:70])


def test_sentinel_actuation(tmp: Path) -> None:
    print("\n[2] sentinel actuation is opt-in")
    from security.sentinel import Sentinel
    fw = _fw(tmp)

    s = Sentinel(log_path=tmp / "a1.jsonl", firewall=fw)
    for i in range(6):
        s.observe_call(f"sk{i}", "CRITICAL")
    check("unarmed: an alert does not touch the firewall",
          fw.dry_run is False and s.alerts[-1]["acted"] is False)
    check("unarmed is the default", s.acting is False)

    fw2 = _fw(tmp)
    s2 = Sentinel(log_path=tmp / "a2.jsonl", firewall=fw2, acting=True)
    for i in range(6):
        s2.observe_call(f"sk{i}", "CRITICAL")
    check("armed: tripping engages dry-run as a reversible cooldown",
          fw2.dry_run is True and s2.alerts[-1]["acted"] is True,
          s2.alerts[-1]["action"])
    s2._timer.cancel()
    fw2.dry_run = False

    s3 = Sentinel(log_path=tmp / "a3.jsonl", firewall=_fw(tmp))
    s3.arm(True)
    check("arm() flips actuation at runtime", s3.acting is True)


def test_consolidation(tmp: Path) -> None:
    print("\n[3] memory consolidation promotes topics, not fragments")
    from brain.memory import MemoryPalace
    mp = MemoryPalace(tmp / "m.db")

    for _ in range(4):
        mp.remember_episode("דיברנו על המוזיקה שלי ואיך המוזיקה מסודרת בתיקיות",
                            kind="dialogue")
    mp.remember_episode("אדוני, שלום — הבקשה בוצעה בבקשה תודה", kind="dialogue")
    mp.remember_episode("הקובץ של הדוחות נמצא בתיקיה", kind="dialogue")

    rep = mp.consolidate()
    joined = " ".join(rep["promoted"])
    check("a recurring topic is promoted to a semantic fact",
          "מוזיקה" in joined, joined[:80])
    check("prefix variants count as one theme (המוזיקה/מוזיקה)",
          joined.count("מוזיק") == 1, joined[:80])
    check("forms of address never consolidate",
          "אדוני" not in joined and "שלום" not in joined)
    check("sentence fragments never consolidate",
          not any(x in joined for x in ("איכ", "דיברנו", "שלי", "ואיכ")))
    check("a single-mention theme stays episodic",
          "דוח" not in joined and "דוחות" not in joined)
    check("the promoted fact is readable Hebrew with its support count",
          "עלה ב־4" in (mp.recall_fact("consolidated.מוזיקה") or ""),
          (mp.recall_fact("consolidated.מוזיקה") or "")[:70])
    check("the fact records where it came from",
          any(f.get("source") == "consolidation"
              for f in mp.all_facts() if f.get("key") == "consolidated.מוזיקה"))

    rep2 = mp.consolidate()
    check("a second pass refreshes instead of duplicating",
          len(rep2["promoted"]) == 0 and len(rep2["refreshed"]) == len(rep["promoted"]),
          f"promoted={len(rep2['promoted'])} refreshed={len(rep2['refreshed'])}")
    check("confidence grows with support but stays under 1",
          all(0 < f.get("confidence", 0) < 1 for f in mp.all_facts()
              if str(f.get("key", "")).startswith("consolidated.")))

    empty = MemoryPalace(tmp / "empty.db")
    rep3 = empty.consolidate()
    check("an empty palace consolidates to nothing, cleanly",
          rep3["promoted"] == [] and rep3["episodes_scanned"] == 0)
    check("prune defaults to dry-run", rep["pruned"]["dry_run"] is True)


def test_consolidate_skill(tmp: Path) -> None:
    print("\n[4] the consolidation skill")
    import skills
    skills.load_all()
    from skills.registry import REGISTRY
    from core.config import CONFIG

    sk = REGISTRY.get("memory.consolidate")
    check("memory.consolidate is registered SAFE", sk is not None and sk.risk == "SAFE")

    # The skill builds its palace from CONFIG's db path. Pointing it at the real
    # one here would consolidate the sandbox's own chatter into the user's
    # semantic memory as a side effect of running a test, so the path is
    # redirected for the duration and restored after.
    original = CONFIG.memory.db_path
    seeded = MemoryPalaceSeeded(tmp / "skill.db")
    CONFIG.memory.db_path = str(tmp / "skill.db")
    try:
        res = REGISTRY.invoke("memory.consolidate", {}, permission_granted=True)
        check("the skill runs and answers in Hebrew", res.ok and bool(res.value),
              res.value[:60])
        check("the skill reports its scan in data",
              "episodes_scanned" in (res.data or {}))
        check("the skill read the redirected palace, not the real one",
              res.data.get("episodes_scanned") == seeded.count,
              f"scanned={res.data.get('episodes_scanned')} seeded={seeded.count}")
    finally:
        CONFIG.memory.db_path = original


class MemoryPalaceSeeded:
    """A temp palace with a recurring theme, so the skill has something to find."""

    def __init__(self, path: Path) -> None:
        from brain.memory import MemoryPalace
        mp = MemoryPalace(path)
        for _ in range(3):
            mp.remember_episode("הדפסתי את הדוח והדוח יצא יפה", kind="dialogue")
        self.count = 3


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — sentinel & memory consolidation")
    print("═" * 68)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_sentinel(tmp)
        test_sentinel_actuation(tmp)
        test_consolidation(tmp)
        test_consolidate_skill(tmp)
    print(f"\n completed in {time.perf_counter() - t0:.1f}s")
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
