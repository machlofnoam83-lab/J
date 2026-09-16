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
from brain.access import AccessGate, get_gate  # noqa: E402
from brain.presence import PresenceTracker, get_presence  # noqa: E402
from brain.reasoning import ReasoningEngine  # noqa: E402
from brain.intent import IntentRouter  # noqa: E402
from vision.faces import FaceGate, FaceStore, encode_frame  # noqa: E402
from vision.scene import describe  # noqa: E402


class _Room:
    """A throwaway gallery plus the tracker and gate that watch it."""

    def __enter__(self):
        self._saved = (srv.FACE_STATE.get("store"), srv.FACE_STATE.get("gate"))
        self.store = FaceStore(path=Path(tempfile.mkdtemp(prefix="acc_")) / "g.json")
        self.gate = FaceGate(self.store, firewall=None, debounce=1)
        srv.FACE_STATE["store"], srv.FACE_STATE["gate"] = self.store, self.gate
        self.presence = PresenceTracker(ttl=12.0)
        self.g = AccessGate(presence=self.presence)
        return self

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
