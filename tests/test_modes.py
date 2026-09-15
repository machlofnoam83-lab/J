#!/usr/bin/env python3
"""JARVIS operational modes + the researcher — tests.

Twelve postures were requested ("a researcher that finds me everything I need,
and eleven more like it"). A posture that only changes a label would be theatre,
so these tests hold the two claims that make it real:

  · switching a mode is observable and durable — it survives a restart, an
    unknown id is refused rather than half-applied, and a spoken phrase reaches
    the same switch as the HUD button;
  · the researcher actually searches, and knows when it found nothing. Recall is
    checked against topics the local sources genuinely cover; precision against
    strings that appear nowhere, because a search that always "finds" something
    is a confidence machine, not a researcher.

The boot-timeout tests cover the fix for a reported symptom: on a slow machine
the HUD sat on a black overlay for minutes because the boot POST never streamed.
Each check now runs under a hard timeout on a daemon thread and results stream
through boot_progress(), so the interface can fill while measurement continues.

Run:  python tests/test_modes.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import skills  # noqa: E402
from skills.registry import REGISTRY  # noqa: E402

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


def test_modes_registry() -> None:
    print("\n[1] the twelve postures")
    from brain import modes as M
    check("twelve modes are defined", len(M.MODES) == 12, str(len(M.MODES)))
    check("the researcher is one of them and flagged as such",
          "researcher" in M.MODES and M.MODES["researcher"].research is True)
    check("every mode carries Hebrew display, description and persona",
          all(m.he and m.desc_he and m.persona_he and m.en for m in M.MODES.values()))
    check("every mode has at least one spoken trigger",
          all(m.triggers for m in M.MODES.values()))
    ids = [m.id for m in M.MODES.values()]
    check("mode ids are unique", len(set(ids)) == len(ids))
    check("the default posture needs no privileges", M.DEFAULT_MODE in M.MODES)


def test_mode_state(tmp: Path) -> None:
    print("\n[2] posture state, switching, persistence")
    from brain.modes import MODES, ModeState
    st = ModeState(path=tmp / "mode.json")
    check("starts at the default", st.get().id == "companion", st.get().id)

    m = st.set("researcher")
    check("switching to researcher works", m is not None and st.get().id == "researcher")
    check("the switch is recorded in history",
          st.history and st.history[-1]["from"] == "companion"
          and st.history[-1]["to"] == "researcher")

    check("an unknown id is refused, not half-applied",
          st.set("no_such_posture") is None and st.get().id == "researcher")

    st2 = ModeState(path=tmp / "mode.json")
    check("the posture survives a restart", st2.get().id == "researcher", st2.get().id)

    st2.set("companion")
    broken = tmp / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    st3 = ModeState(path=broken)
    check("a corrupt state file degrades to the default, not a crash",
          st3.get().id == "companion")

    trig = st2.by_trigger("תעבור למצב חוקר בבקשה")
    check("a spoken phrase resolves to the researcher", trig is not None and trig.id == "researcher")
    check("unrelated speech resolves to nothing", st2.by_trigger("כמה זה שתיים ועוד שתיים") is None)

    sch = st2.schema()
    check("the schema marks exactly one mode active",
          sum(1 for r in sch if r["active"]) == 1 and len(sch) == 12)
    # st2 is the instance that ended on companion; `st` still holds its own
    # in-memory researcher, which is correct isolation, not a bug.
    stats = st2.stats()
    check("stats report the held posture and switch count",
          stats["current"] == "companion" and stats["switches"] >= 2, str(stats)[:70])
    del MODES  # keep linters quiet about the import being used


def test_routing() -> None:
    print("\n[3] intent routing for posture and research")
    from brain.intent import IntentRouter
    r = IntentRouter(knowledge=None, skills=REGISTRY, config=None)

    for txt, want, skill in [
        ("תחקור לי כל מה שיש לך על זיכרון", "RESEARCH", "research.query"),
        ("חפש לי כל מה שקשור לצילום מסך", "RESEARCH", "research.query"),
        ("מצב חוקר", "MODE", "mode.set"),
        ("עבור למצב מנתח", "MODE", "mode.set"),
        ("באיזה מצב אתה", "MODE", "mode.get"),
        ("אילו מצבים יש לך", "MODE", "mode.list"),
    ]:
        rt = r.route(txt)
        check(f"{txt!r} routes to {want}", rt.intent == want and rt.skill == skill,
              f"{rt.intent}/{rt.skill}")

    # the posture change must beat the subject it resembles
    rt = r.route("מצב מנתח")
    check("'מצב מנתח' changes posture rather than reporting telemetry",
          rt.intent == "MODE", f"{rt.intent}/{rt.skill}")

    for txt in ["כמה זה 12 כפול 12", "מה השעה", "נגן מוזיקה", "מה זה באג"]:
        rt = r.route(txt)
        check(f"{txt!r} is NOT swallowed by the researcher", rt.intent != "RESEARCH",
              rt.intent)


def test_research(tmp: Path) -> None:
    print("\n[4] the researcher searches, and admits emptiness")
    res = REGISTRY.invoke("research.query", {"topic": "זיכרון ושיכחה"}, permission_granted=True)
    check("a covered topic yields hits", res.ok and res.data["hits"] > 0,
          f"hits={res.data.get('hits')}")
    groups = res.data.get("groups", {})
    check("hits span more than one source", len(groups) >= 2, ", ".join(groups))
    check("every hit carries provenance",
          all(("id" in h or "path" in h or "skill" in h or "key" in h or "question" in h
               or "user" in h or "text" in h)
              for rows in groups.values() for h in rows))
    check("the digest names the sources it used",
          all(g in res.value for g in list(groups)[:2]))
    check("the result states it is offline", "offline" in res.data.get("note", ""))

    kb = groups.get("knowledge", [])
    check("knowledge hits carry the KB record id", bool(kb) and bool(kb[0].get("id")),
          str(kb[0].get("id")) if kb else "no kb hits")

    # A literal nonsense string in this file would be found by the file sweep —
    # in this very test — which is correct provenance, not a hallucination. So
    # the strict zero-hits case uses a topic generated at runtime, guaranteed to
    # appear in no source, while the literal case asserts only that the *stored
    # knowledge* sources did not invent anything.
    import uuid
    # letters only: the tokenizer reads words, so a nonce full of digits would be
    # refused as "no words to search" rather than searched and missed.
    uniq = "zz" + "".join(chr(97 + int(c, 16)) for c in uuid.uuid4().hex)
    empty = REGISTRY.invoke("research.query", {"topic": uniq}, permission_granted=True)
    check("a topic found nowhere reports zero hits", empty.ok and empty.data["hits"] == 0,
          f"hits={empty.data.get('hits')} groups={list(empty.data.get('groups', {}))}")
    check("the empty answer says so plainly instead of inventing",
          "לא מצאתי" in empty.value, empty.value[:60])

    literal = REGISTRY.invoke("research.query",
                              {"topic": "זברה כחולה מקפצת 9137"}, permission_granted=True)
    lg = literal.data.get("groups", {})
    check("nonsense never hallucinates from stored knowledge or skills",
          not any(g in lg for g in ("knowledge", "knowledge_qa", "memory",
                                    "memory_facts", "skills")),
          ", ".join(lg) or "no groups")

    blank = REGISTRY.invoke("research.query", {"topic": "   "}, permission_granted=True)
    check("an empty topic is refused with a question, not a crash",
          not blank.ok and "נושא" in (blank.error or ""))

    t0 = time.perf_counter()
    REGISTRY.invoke("research.query", {"topic": "טוקנייזר"}, permission_granted=True)
    ms = (time.perf_counter() - t0) * 1000
    check("a full five-source sweep stays under two seconds", ms < 2000, f"{ms:.0f}ms")


def test_mode_skills() -> None:
    print("\n[5] the posture skills behind voice and HUD")
    from brain.modes import STATE
    r = REGISTRY.invoke("mode.set", {"mode": "מצב חוקר"}, permission_granted=True)
    check("a spoken phrase switches the posture", r.ok and r.data["mode"] == "researcher",
          r.data.get("mode"))
    check("the reply states the new posture in Hebrew", "חוקר" in r.value)

    bad = REGISTRY.invoke("mode.set", {"mode": "מצב שאין כזה"}, permission_granted=True)
    check("an unknown posture is reported with the list of real ones",
          not bad.ok and "חוקר" in (bad.error or ""))

    g = REGISTRY.invoke("mode.get", {}, permission_granted=True)
    check("mode.get reports the held posture", g.ok and "חוקר" in g.value)

    lst = REGISTRY.invoke("mode.list", {}, permission_granted=True)
    check("mode.list shows all twelve with the active one marked",
          lst.ok and lst.data["count"] == 12 and "פעיל" in lst.value)

    STATE.set("companion")
    for name in ("research.query", "mode.set", "mode.get", "mode.list"):
        sk = REGISTRY.get(name)
        check(f"{name} is registered SAFE", sk is not None and sk.risk == "SAFE",
              str(sk.risk) if sk else "missing")


def test_boot_timeout() -> None:
    print("\n[6] a hung boot check cannot hold the sequence")
    from agents.jarvis import JarvisAgent

    class Stub(JarvisAgent):
        def __init__(self) -> None:  # no heavy construction for a timing test
            import threading
            self._boot_lock = threading.Lock()
            self._boot_progress = []
            self._boot_done = False
            self._boot_started = 0.0

        def _check(self, key):
            if key == "slow":
                time.sleep(5)
                return True, "should never be reported"
            return True, f"{key} ok"

    stub = Stub()
    t0 = time.perf_counter()
    okflag, detail = stub._run_check_timed("slow", timeout=0.4)
    ms = (time.perf_counter() - t0) * 1000
    check("an overrunning check is cut off near its timeout", ms < 1500, f"{ms:.0f}ms")
    check("the overrun is reported as a failure with a reason",
          okflag is False and "לא הסתיימה" in detail, detail[:50])

    okflag2, detail2 = stub._run_check_timed("fast", timeout=5)
    check("a fast check still reports its real result", okflag2 is True and "fast ok" in detail2)

    class Boom(JarvisAgent):
        def __init__(self) -> None:
            import threading
            self._boot_lock = threading.Lock()
            self._boot_progress = []
            self._boot_done = False
            self._boot_started = 0.0

        def _check(self, key):
            raise RuntimeError("exploded mid-check")

    okflag3, detail3 = Boom()._run_check_timed("x", timeout=5)
    check("an exception inside a check becomes a reported failure",
          okflag3 is False and "RuntimeError" in detail3, detail3[:50])


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — operational modes & the researcher")
    print("═" * 68)
    skills.load_all()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_modes_registry()
        test_mode_state(tmp)
        test_routing()
        test_research(tmp)
        test_mode_skills()
        test_boot_timeout()
    print(f"\n completed in {time.perf_counter() - t0:.1f}s")
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
