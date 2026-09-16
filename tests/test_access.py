"""The access gate — no face, no assistant.

This is the rule the user asked for in their words: whoever has no access can do
nothing, and every time somebody walks in they have to be scanned. Everything
else in the system treats presence as *advice* — the firewall compares ranks,
the skills explain themselves, and an unidentified person still gets a polite
answer. This module is the one place where presence is a precondition.

The properties worth pinning are the negative ones, because a gate that opens by
accident is worse than no gate:

* no face in frame            -> nothing runs
* a lease that expired        -> the session is over, not merely stale
* liveness "suspect"          -> a photograph is not a person walking in
* an unenrolled face          -> gets nothing, by default not even the clock
* enrolment and "who is here" -> always allowed, or the door has no handle

That last one is the design tension. A gate with no exception is unusable, and
an unusable gate gets turned off; the two exemptions are the difference between
a lock and a wall.
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_scene import identity, render, screen_like  # noqa: E402

import core.server as srv  # noqa: E402
from agents.jarvis import JarvisAgent  # noqa: E402
from brain.access import AccessGate, get_gate  # noqa: E402
from brain.presence import PresenceTracker, get_presence  # noqa: E402
from brain.reasoning import ReasoningEngine  # noqa: E402
from brain.intent import IntentRouter  # noqa: E402
from security.permissions import PermissionFirewall  # noqa: E402
from vision.faces import FaceGate, FaceStore, encode_frame  # noqa: E402
from vision.scene import describe  # noqa: E402


class _Room:
    """A throwaway gallery plus the tracker and gate that watch it."""

    def __enter__(self):
        self._saved = (srv.FACE_STATE.get("store"), srv.FACE_STATE.get("gate"))
        self.store = FaceStore(path=Path(tempfile.mkdtemp(prefix="acc_")) / "g.json")
        self.gate = FaceGate(self.store, firewall=None, debounce=1)
        srv.FACE_STATE["store"], srv.FACE_STATE["gate"] = self.store, self.gate
        # The singleton, not a private tracker. ``_who_is_here()`` and the
        # skills both read ``get_presence()``, so a harness that observed into
        # its own tracker would leave the singleton empty and every identity
        # check would compare '' to '' — passing for the wrong reason.
        self.presence = get_presence(fresh=True)
        # Every verdict is audited to disk now, so the harness must point the
        # log somewhere disposable — otherwise a test run quietly appends to the
        # real logs/access.jsonl.
        self.audit = Path(tempfile.mkdtemp(prefix="accaudit_")) / "access.jsonl"
        self.g = AccessGate(presence=self.presence, audit_path=self.audit)
        return self

    def audit_lines(self):
        import json as _json
        if not self.audit.exists():
            return []
        return [_json.loads(x) for x in
                self.audit.read_text(encoding="utf-8").strip().split("\n") if x]

    def __exit__(self, *exc):
        srv.FACE_STATE["store"], srv.FACE_STATE["gate"] = self._saved
        return False

    def see(self, img, with_scene=False):
        m = self.store.recognize(img)
        self.presence.observe(m, describe(img, identity=m) if with_scene else None,
                              self.gate)
        return m

    def enrol_owner(self, img, name="OSCAR"):
        self.store.enroll(name, encode_frame(img).vector)


class GateUnitTest(unittest.TestCase):
    """The gate on its own, before the brain is involved."""

    def test_no_face_blocks_action(self):
        with _Room() as r:
            v = r.g.check("files.read")
            self.assertFalse(v.allowed)
            self.assertEqual(v.reason, "no_face")
            self.assertTrue(v.needs_scan)

    def test_enrolment_is_allowed_with_no_face(self):
        """Without this the gate is a locked door with no handle."""
        with _Room() as r:
            self.assertTrue(r.g.check("vision.enroll").allowed)
            self.assertTrue(r.g.check("vision.who").allowed)
            self.assertTrue(r.g.check().allowed)   # the general question

    def test_an_unknown_face_gets_nothing_by_default(self):
        """The instruction was 'can do nothing' — not even the clock."""
        with _Room() as r:
            self.assertFalse(r.g.allow_guests_readonly)
            r.see(render(identity(11), seed=5))
            v = r.g.check("time.now")
            self.assertFalse(v.allowed)
            self.assertEqual(v.reason, "unknown_face")

    def test_an_unknown_face_may_still_ask_to_be_enrolled(self):
        with _Room() as r:
            r.see(render(identity(11), seed=5))
            self.assertTrue(r.g.check("vision.enroll").allowed)

    def test_an_identified_face_may_act(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            r.see(img)
            v = r.g.check("files.read")
            self.assertTrue(v.allowed)
            self.assertEqual(v.reason, "identified")
            self.assertEqual(v.identity, "OSCAR")

    def test_a_photograph_is_not_a_person(self):
        with _Room() as r:
            img = screen_like(render(identity(31), seed=7), period=4)
            r.see(img, with_scene=True)
            v = r.g.check("files.read")
            self.assertFalse(v.allowed)
            self.assertEqual(v.reason, "liveness_suspect")

    def test_leaving_the_room_ends_the_session(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            short = PresenceTracker(ttl=0.05)
            short.observe(r.store.recognize(img), None, r.gate)
            g = AccessGate(presence=short)
            time.sleep(0.08)
            v = g.check("files.read")
            self.assertFalse(v.allowed)
            self.assertIn(v.reason, ("no_face", "session_expired"))

    def test_no_tracker_fails_closed(self):
        """No camera wired: actions refuse, questions still answer."""
        g = AccessGate(presence=None)
        self.assertFalse(g.check("files.read").allowed)
        self.assertEqual(g.check("files.read").reason, "no_tracker")
        self.assertTrue(g.check("vision.who").allowed)

    def test_guest_mode_is_opt_in_not_default(self):
        with _Room() as r:
            r.see(render(identity(11), seed=5))
            r.g.allow_guests_readonly = True
            self.assertTrue(r.g.check("time.now").allowed)
            self.assertFalse(r.g.check("files.write").allowed)

    def test_state_and_explain_are_readable(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            r.see(img)
            r.g.check("time.now")
            s = r.g.state()
            self.assertTrue(s["has_face"])
            self.assertTrue(s["known"])
            self.assertEqual(s["identity"], "OSCAR")
            self.assertIsInstance(r.g.explain(), str)
            self.assertIn("OSCAR", r.g.explain())

    def test_history_is_bounded(self):
        with _Room() as r:
            for _ in range(250):
                r.g.check("files.read")
            self.assertLessEqual(len(r.g.history), 200)


class BrainAccessTest(unittest.TestCase):
    """The gate wired into think(), which is where it has to actually bite."""

    @classmethod
    def setUpClass(cls):
        from skills import REGISTRY, load_all
        load_all()
        cls.reg = REGISTRY

    def _engine(self, presence):
        eng = ReasoningEngine(skills=self.reg, router=IntentRouter(),
                              presence=presence)
        eng.access = AccessGate(presence=presence)
        return eng

    def test_no_scan_means_no_action(self):
        with _Room() as r:
            eng = self._engine(r.presence)
            a = eng.think("הפעל את המחשבון")
            self.assertFalse(a.grounded)
            self.assertIn("סריקת פנים", a.text)

    def test_the_trace_shows_the_gate(self):
        with _Room() as r:
            eng = self._engine(r.presence)
            a = eng.think("מה השעה")
            self.assertIn("access", [s["kind"] for s in a.trace["steps"]])

    def test_enrolment_still_reaches_the_skill(self):
        """The exemption must survive the trip through the brain."""
        with _Room() as r:
            eng = self._engine(r.presence)
            a = eng.think("רשום אותי בשם דנה")
            self.assertEqual(a.skill, "vision.enroll")
            self.assertNotIn("סריקת פנים", a.text)

    def test_a_stranger_cannot_ask_the_time(self):
        with _Room() as r:
            r.see(render(identity(77), seed=3))
            eng = self._engine(r.presence)
            a = eng.think("מה השעה")
            self.assertFalse(a.grounded)
            self.assertNotIn("השעה היא", a.text)

    def test_an_identified_owner_can_ask_the_time(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            r.see(img)
            eng = self._engine(r.presence)
            a = eng.think("מה השעה")
            self.assertTrue(a.grounded)
            self.assertIn("השעה", a.text)

    def test_disabling_the_gate_reopens_everything(self):
        """The escape hatch has to work, or a headless run cannot use the brain."""
        with _Room() as r:
            eng = self._engine(r.presence)
            eng.access.enabled = False
            a = eng.think("מה השעה")
            self.assertTrue(a.grounded)


class _BridgeStub:
    """The minimum shape _confirm_bridge touches, without booting the agent.

    Constructing a real JarvisAgent builds a neural core, a knowledge store and
    a coder — minutes of work to test one method. The stub carries exactly the
    attributes the method reads, and the real ``_confirm_bridge`` and
    ``_who_is_here`` are bound onto it unbound, so the code under test is the
    shipping code and not a reimplementation of it.
    """

    def __init__(self, firewall, timeout=2.0):
        import itertools
        import threading as _t
        self.firewall = firewall
        self.confirm_timeout = timeout
        self.name = "jarvis"
        self._confirm_events = {}
        self._confirm_verdicts = {}
        self._confirm_requests = {}
        self._confirm_seq = itertools.count(1)

    _confirm_bridge = JarvisAgent._confirm_bridge
    _who_is_here = staticmethod(JarvisAgent._who_is_here)


class ConfirmBindingTest(unittest.TestCase):
    """A CRITICAL approval belongs to the person who was in frame when it was asked.

    Without this the prompt is a token that outlives the person: the owner asks
    for something CRITICAL, walks away, and whoever sits down next can approve a
    request they never made.
    """

    def _make(self, r, change, verdict=True):
        """Wire a real firewall to a stub agent whose confirm bridge is the
        shipping ``JarvisAgent._confirm_bridge``.

        ``change`` runs while the prompt is open, which is the whole scenario
        under test: the person in front of the camera is not necessarily the
        person who was there when the action was asked for.
        """
        fw = PermissionFirewall()
        fw.set_level("WRITE")
        stub = _BridgeStub(fw)
        box = {"stub": stub, "change": change}
        box["bridge"] = lambda req: JarvisAgent._confirm_bridge(stub, req)

        def hook(request):
            def answer():
                time.sleep(0.05)
                box["change"]()
                rid = next(iter(stub._confirm_events))
                stub._confirm_verdicts[rid] = verdict
                stub._confirm_events[rid].set()
            import threading as _t
            _t.Thread(target=answer, daemon=True).start()
            return box["bridge"](request)

        fw.on_confirm(hook)
        return fw

    def test_approval_is_honoured_when_the_same_person_is_still_there(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            r.see(img)
            fw = self._make(r, lambda: None)          # nobody moves
            d = fw.check("system.shutdown", "CRITICAL", {})
            self.assertTrue(d.allowed, d.reason)

    def test_approval_is_discarded_when_the_face_changed(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            r.see(img)
            other = render(identity(31), seed=9)
            fw = self._make(r, lambda: r.see(other))   # someone else sits down
            d = fw.check("system.shutdown", "CRITICAL", {})
            self.assertFalse(d.allowed)

    def test_approval_is_discarded_when_the_room_empties(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            short = PresenceTracker(ttl=0.05)
            short.observe(r.store.recognize(img), None, r.gate)
            import brain.presence as bp
            saved = bp._TRACKER
            bp._TRACKER = short
            try:
                fw = self._make(r, lambda: time.sleep(0.08))   # lease expires
                d = fw.check("system.shutdown", "CRITICAL", {})
                self.assertFalse(d.allowed)
            finally:
                bp._TRACKER = saved

    def test_a_headless_run_is_unaffected(self):
        """No presence at all compares '' to '' and behaves as before."""
        with _Room() as r:
            import brain.presence as bp
            saved = bp._TRACKER
            bp._TRACKER = None
            try:
                fw = self._make(r, lambda: None)
                d = fw.check("system.shutdown", "CRITICAL", {})
                self.assertTrue(d.allowed, d.reason)
            finally:
                bp._TRACKER = saved


class AuditTrailTest(unittest.TestCase):
    """Every decision leaves a trace — including the ones that said yes.

    An audit trail that records only refusals cannot answer the question an
    access control is actually asked later, which is "who got in".
    """

    def test_refusals_and_grants_both_reach_disk(self):
        with _Room() as r:
            r.g.check("files.read")            # no face -> refused
            r.g.check("vision.enroll")         # no face -> allowed (the door)
            lines = r.audit_lines()
            self.assertEqual(len(lines), 2)
            self.assertFalse(lines[0]["allowed"])
            self.assertTrue(lines[1]["allowed"])

    def test_the_record_names_what_was_attempted(self):
        """A verdict without the skill is half a log line."""
        with _Room() as r:
            r.g.check("files.write")
            rec = r.audit_lines()[0]
            self.assertEqual(rec["skill"], "files.write")
            self.assertEqual(rec["reason"], "no_face")
            self.assertTrue(rec["needs_scan"])

    def test_an_identified_person_is_recorded_by_name(self):
        with _Room() as r:
            img = render(identity(11), seed=5)
            r.enrol_owner(img)
            r.see(img)
            r.g.check("time.now")
            rec = r.audit_lines()[-1]
            self.assertTrue(rec["allowed"])
            self.assertEqual(rec["reason"], "identified")
            self.assertEqual(rec["identity"], "OSCAR")

    def test_a_suspect_frame_is_audited_as_such(self):
        with _Room() as r:
            r.see(screen_like(render(identity(31), seed=7), period=4),
                  with_scene=True)
            r.g.check("files.write")
            self.assertEqual(r.audit_lines()[-1]["reason"], "liveness_suspect")

    def test_an_unwritable_log_never_stops_the_gate(self):
        """A full disk must not become a denial-of-service on the front door."""
        with _Room() as r:
            g = AccessGate(presence=r.presence,
                           audit_path=Path("/no/such/dir/x.jsonl"))
            v = g.check("time.now")
            self.assertIsNotNone(v.reason)

    def test_history_is_bounded_in_memory(self):
        with _Room() as r:
            for _ in range(250):
                r.g.check("files.read")
            self.assertLessEqual(len(r.g.history), 200)
            self.assertEqual(len(r.g.tail(5)), 5)


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — the access gate")
    print("═" * 68)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"\nRESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
