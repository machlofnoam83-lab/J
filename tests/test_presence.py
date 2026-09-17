"""Face recognition reaching the brain, and the brain acting on what it checks.

Two regressions are pinned here, both of them real defects that shipped:

1. ``brain/`` never imported from ``vision/``. ``FaceGate`` pushed a level into
   the firewall behind the reasoning engine's back, so the brain could not say
   who was in the room, could not greet by name, and could not explain a refusal
   in terms the user could act on. ``brain/presence.py`` is the wire; these tests
   fail without it.

2. ``_verify_and_pack`` computed ``(ok, issues, cleaned)`` from the ``Verifier``
   and then returned the answer at full confidence regardless. Having a verifier
   and ignoring its verdict is not the same as thinking — the answer now changes
   when the check fails.

Synthetic faces only. As in the other vision tests, nothing here claims a real
camera was used.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_vision import identity, render  # noqa: E402

from brain.presence import Presence, PresenceTracker, get_presence  # noqa: E402
from brain.reasoning import ReasoningEngine  # noqa: E402
from vision.faces import FaceStore, FaceGate, encode_frame  # noqa: E402
from vision.scene import describe  # noqa: E402


def _tmp_store() -> FaceStore:
    return FaceStore(path=Path(tempfile.mkdtemp(prefix="pres_")) / "gallery.json")


class _Gate:
    """Stand-in for FaceGate when only its level matters."""

    def __init__(self, level: str, armed: bool = True) -> None:
        self.level = level
        self.armed = armed


class _Liveness:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict


class _Scene:
    def __init__(self, verdict: str = "live", summary: str = "", quality: float = 0.9) -> None:
        self.liveness = _Liveness(verdict) if verdict else None
        self.summary_he = summary
        self.quality = quality

    def to_dict(self):
        return {"summary_he": self.summary_he, "quality": self.quality}


def _match(name, identity, level="WRITE", role="guest", conf=0.92, people=1):
    from vision.faces import Match
    return Match(identity=f"{name}-id", name=name, level=level, confidence=conf,
                 residual=0.15, faces=people, known=True, role=role)


class PresenceLayerTest(unittest.TestCase):
    """The bridge itself: what the brain believes about the room."""

    def test_empty_room_invents_nobody(self):
        t = PresenceTracker()
        p = t.current()
        self.assertEqual(p.name, "—")
        self.assertEqual(p.people, 0)
        self.assertFalse(p.known)
        # the critical one: an empty room must not produce a greeting
        self.assertEqual(t.greeting_hint(), "")

    def test_owner_frame_is_reported_as_owner(self):
        t = PresenceTracker()
        t.observe(_match("OSCAR", identity(11), level="CRITICAL", role="owner"),
                  _Scene(summary="OSCAR מול המצלמה"), _Gate("CRITICAL"))
        p = t.current()
        self.assertEqual(p.name, "OSCAR")
        self.assertTrue(p.is_owner)
        self.assertEqual(p.level, "CRITICAL")
        self.assertIn("OSCAR", t.greeting_hint())

    def test_still_allows_the_owners_risk(self):
        t = PresenceTracker()
        t.observe(_match("OSCAR", identity(11), level="CRITICAL", role="owner"),
                  _Scene(), _Gate("CRITICAL"))
        self.assertTrue(t.allows("CRITICAL"))
        self.assertTrue(t.allows("WRITE"))
        self.assertEqual(t.explain("CRITICAL"), "")

    def test_stranger_is_denied_critical_with_a_reason(self):
        t = PresenceTracker()
        t.observe(_match("GUEST", identity(31), level="WRITE", role="guest"),
                  _Scene(), _Gate("WRITE"))
        self.assertFalse(t.allows("CRITICAL"))
        self.assertTrue(t.allows("SAFE"))
        # the explanation names the level the guest actually has, not a generic one
        why = t.explain("CRITICAL")
        self.assertIn("WRITE", why or "")
        self.assertNotEqual(why, "")

    def test_liveness_can_only_subtract(self):
        """An owner in front of a screen is not an owner in front of the machine."""
        t = PresenceTracker()
        t.observe(_match("OSCAR", identity(11), level="CRITICAL", role="owner"),
                  _Scene(verdict="suspect", summary="נראה כמו מסך"), _Gate("CRITICAL"))
        p = t.current()
        self.assertEqual(p.level, "SAFE")
        self.assertEqual(p.withheld, "liveness")
        self.assertFalse(t.allows("WRITE"))
        self.assertNotEqual(t.explain("WRITE"), "")

    def test_firewall_level_caps_the_match(self):
        """A CRITICAL identity seen while the firewall is SAFE stays SAFE."""
        t = PresenceTracker()
        t.observe(_match("OSCAR", identity(11), level="CRITICAL", role="owner"),
                  _Scene(), _Gate("SAFE"))
        self.assertEqual(t.current().level, "SAFE")

    def test_ttl_expires_the_room(self):
        t = PresenceTracker(ttl=0.0)
        t.observe(_match("OSCAR", identity(11), level="CRITICAL", role="owner"),
                  _Scene(), _Gate("CRITICAL"))
        p = t.current()
        self.assertEqual(p.people, 0)
        self.assertFalse(p.known)
        self.assertEqual(p.name, "—")

    def test_singleton_is_one_per_process(self):
        first = get_presence()
        self.assertIs(get_presence(), first)
        fresh = get_presence(fresh=True)
        self.assertIsNot(fresh, first)
        self.assertIs(get_presence(), fresh)

    def test_presence_serialises_for_the_hud(self):
        t = PresenceTracker()
        t.observe(_match("OSCAR", identity(11), level="CRITICAL", role="owner"),
                  _Scene(summary="סצנה"), _Gate("CRITICAL"))
        d = t.current().to_dict()
        for key in ("name", "level", "role", "is_owner", "known", "people",
                    "quality", "withheld", "age", "summary_he"):
            self.assertIn(key, d)


class VisionSkillsTest(unittest.TestCase):
    """The brain's way in: the skills must exist and answer in Hebrew."""

    @classmethod
    def setUpClass(cls):
        from skills import REGISTRY, load_all
        load_all()
        cls.reg = REGISTRY

    def test_the_vision_skills_are_registered(self):
        names = {s.name for s in self.reg.all() if s.name.startswith("vision.")}
        self.assertEqual(names, {"vision.who", "vision.scene", "vision.gallery",
                                 "vision.permission", "vision.enroll"})

    def test_asking_about_the_room_is_never_privileged(self):
        """Reading the camera must not need a permission; changing it must."""
        for name in ("vision.who", "vision.scene", "vision.gallery", "vision.permission"):
            self.assertEqual(self.reg.get(name).risk, "SAFE", name)

    def test_enrolment_is_the_one_privileged_vision_act(self):
        self.assertEqual(self.reg.get("vision.enroll").risk, "CRITICAL")

    def test_who_reports_the_owner(self):
        get_presence(fresh=True).observe(
            _match("OSCAR", identity(11), level="CRITICAL", role="owner"),
            _Scene(summary="OSCAR מול המצלמה"), _Gate("CRITICAL"))
        r = self.reg.invoke("vision.who")
        self.assertTrue(r.ok)
        self.assertIn("OSCAR", r.value)
        self.assertTrue(r.data["presence"]["is_owner"])

    def test_who_in_an_empty_room_says_so(self):
        get_presence(fresh=True)
        r = self.reg.invoke("vision.who")
        self.assertTrue(r.ok)
        self.assertIn("אף אחד", r.value)

    def test_permission_explains_a_denial(self):
        get_presence(fresh=True).observe(
            _match("GUEST", identity(31), level="WRITE", role="guest"),
            _Scene(), _Gate("SAFE"))
        r = self.reg.invoke("vision.permission", {"risk": "CRITICAL"})
        self.assertTrue(r.ok)
        self.assertFalse(r.data["allowed"])
        self.assertNotEqual(r.data["reason_he"], "")


class VisionIntentTest(unittest.TestCase):
    """The router must send room questions to the camera, not to the persona."""

    @classmethod
    def setUpClass(cls):
        from brain.intent import IntentRouter
        cls.r = IntentRouter()

    def test_room_questions_route_to_vision(self):
        for q in ("מי מולי", "מה אתה רואה", "מי רשום לזיהוי פנים", "מה מותר לי עכשיו"):
            self.assertEqual(self.r.route(q).intent, "VISION", q)

    def test_each_room_question_picks_its_own_skill(self):
        self.assertEqual(self.r.route("מי מולי").skill, "vision.who")
        self.assertEqual(self.r.route("מה אתה רואה").skill, "vision.scene")
        self.assertEqual(self.r.route("מי הבעלים").skill, "vision.gallery")
        self.assertEqual(self.r.route("מה מותר לי").skill, "vision.permission")

    def test_permission_question_beats_the_safety_branch(self):
        """"מה מותר לי" contains a security word; the camera answers it better."""
        self.assertEqual(self.r.route("מה מותר לי לעשות").intent, "VISION")

    def test_other_intents_are_unaffected(self):
        self.assertEqual(self.r.route("מה השעה").intent, "TIME")
        self.assertEqual(self.r.route("כמה זה 2 ועוד 2").intent, "MATH")
        self.assertEqual(self.r.route("מה כתוב בקבצים על חומת הרשאות").intent, "RAG")


class BrainSeesTheRoomTest(unittest.TestCase):
    """End to end: a frame must change what the brain says."""

    @classmethod
    def setUpClass(cls):
        from skills import REGISTRY, load_all
        from brain.intent import IntentRouter
        load_all()
        cls.eng = ReasoningEngine(skills=REGISTRY, router=IntentRouter())

    def _feed(self, match, scene, gate):
        get_presence(fresh=True).observe(match, scene, gate)

    def test_brain_names_the_owner(self):
        self._feed(_match("OSCAR", identity(11), level="CRITICAL", role="owner"),
                   _Scene(summary="OSCAR מול המצלמה"), _Gate("CRITICAL"))
        a = self.eng.think("מי מולי")
        self.assertEqual(a.intent, "VISION")
        self.assertIn("OSCAR", a.text)

    def test_presence_is_in_the_trace(self):
        """If presence is absent from the trace the wire is not connected."""
        a = self.eng.think("מה השעה")
        kinds = [s["kind"] for s in (a.trace.get("steps") or [])]
        self.assertIn("presence", kinds)

    def test_empty_room_is_reported_as_empty(self):
        get_presence(fresh=True)
        a = self.eng.think("מי מולי")
        self.assertIn("אף אחד", a.text)

    def test_real_store_round_trip(self):
        """Through the actual FaceStore and FaceGate, not stand-ins."""
        store = _tmp_store()
        store.enroll("OSCAR", encode_frame(render(identity(11), seed=5)).vector,
                     level="SAFE")
        gate = FaceGate(store, firewall=None, debounce=1)
        img = render(identity(11), seed=5)
        match = store.recognize(img)
        self._feed(match, describe(img, identity=match), gate)
        a = self.eng.think("מי מולי")
        self.assertIn("OSCAR", a.text)
        # the owner tier, which the store owns, survives the round trip
        self.assertEqual(store.owner().role, "owner")
        self.assertEqual(store.owner().level, "CRITICAL")


class VerifierActsTest(unittest.TestCase):
    """The verdict must change the answer, not just be logged."""

    @classmethod
    def setUpClass(cls):
        from skills import REGISTRY, load_all
        from brain.intent import IntentRouter
        from brain.reasoning import Verifier
        load_all()
        cls.eng = ReasoningEngine(skills=REGISTRY, router=IntentRouter())
        cls.verifier = Verifier()

    def test_clean_answer_is_marked_verified(self):
        a = self.eng.think("כמה זה 7 כפול 8")
        self.assertTrue(a.data.get("verified"))
        self.assertTrue(a.grounded)
        self.assertGreater(a.confidence, 0.9)

    def test_unverified_number_strips_grounding(self):
        """The defect: a number no tool produced must not ride out as grounded."""
        from brain.intent import Route
        from brain.reasoning import Trace
        route = Route("KNOWLEDGE", 0.8, "test", grounded=True)
        trace = Trace()
        ans = self.eng._verify_and_pack("המהירות היא 4217 מטרים לשנייה.",
                                        route, trace, 0.0, numbers=[1, 2, 3])
        self.assertFalse(ans.data.get("verified"))
        self.assertFalse(ans.grounded)
        self.assertLessEqual(ans.confidence, 0.35)

    def test_verified_answer_keeps_grounding(self):
        from brain.intent import Route
        from brain.reasoning import Trace
        route = Route("MATH", 0.99, "test", grounded=True)
        ans = self.eng._verify_and_pack("התוצאה היא 56.", route, Trace(), 0.0,
                                        numbers=[56])
        self.assertTrue(ans.data.get("verified"))
        self.assertTrue(ans.grounded)

    def test_issues_are_surfaced_not_hidden(self):
        from brain.intent import Route
        from brain.reasoning import Trace
        route = Route("KNOWLEDGE", 0.8, "test")
        ans = self.eng._verify_and_pack("ערך לא בדוק 9182 יחידות.", route, Trace(),
                                        0.0, numbers=[1, 2, 3])
        self.assertTrue(ans.data.get("verify_issues"))

    def test_verifier_still_flags_untraced_figures(self):
        ok, issues, _ = self.verifier.check("הערך הוא 8842.", numbers=[1])
        self.assertFalse(ok)
        self.assertTrue(any("unverified number" in i for i in issues))


def main() -> int:
    """Emit the RESULT line tests/run_all.py parses.

    The rest of the suite is script-style; this file is unittest-style, but the
    runner only understands one contract, so honour it rather than being counted
    as a module that produced no result.
    """
    print("═" * 68)
    print(" J.A.R.V.I.S. — presence: face recognition reaching the brain")
    print("═" * 68)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"\nRESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
