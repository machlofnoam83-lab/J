"""Face enrolment reaching the brain — and the guards that keep it safe.

The gap this pins: ``vision/`` could identify a face and the brain could ask who
was in the room, but nothing let the brain *register* anyone. ``Match`` had no
``vector`` field, so ``recognize()`` computed an encoded face and threw it away —
by the time the reasoning engine looked, there was nothing left to enrol from.
"Register me" was not a feature that was broken, it was a feature that did not
exist.

Three properties are worth more than the happy path:

* **A name is a fact, not a guess.** No usable name in the request means a
  refusal, never an invented identity.
* **A photograph is never enrolled.** ``liveness == suspect`` blocks enrolment
  outright, because planting a picture's identity in the gallery is permanent in
  a way a single failed unlock is not.
* **Enrolment is CRITICAL.** The first face into an empty gallery becomes the
  owner with every permission in the system, so the firewall must ask a human
  before it happens.

Tests inject through ``core.server.FACE_STATE`` — the real injection point — and
restore it afterwards, because an earlier version of this file monkeypatched an
attribute that ``get_faces()`` never reads and quietly enrolled test faces into
the production gallery.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_scene import identity, render, screen_like  # noqa: E402

import core.server as srv  # noqa: E402
from brain.intent import IntentRouter  # noqa: E402
from brain.presence import get_presence  # noqa: E402
from brain.reasoning import ReasoningEngine  # noqa: E402
from security.permissions import PermissionFirewall  # noqa: E402
from vision.faces import FaceGate, FaceStore  # noqa: E402
from vision.scene import describe  # noqa: E402


class _Harness:
    """Swap in a throwaway gallery, and always put the real one back."""

    def __enter__(self):
        self._saved = (srv.FACE_STATE.get("store"), srv.FACE_STATE.get("gate"))
        self.store = FaceStore(path=Path(tempfile.mkdtemp(prefix="enr_")) / "g.json")
        self.fw = PermissionFirewall()
        self.fw.set_level("WRITE")
        self.gate = FaceGate(self.store, firewall=self.fw, debounce=1)
        srv.FACE_STATE["store"], srv.FACE_STATE["gate"] = self.store, self.gate
        self.presence = get_presence(fresh=True)
        return self

    def __exit__(self, *exc):
        srv.FACE_STATE["store"], srv.FACE_STATE["gate"] = self._saved
        get_presence(fresh=True)
        return False

    def see(self, img, with_scene=False):
        m = self.store.recognize(img)
        self.presence.observe(m, describe(img, identity=m) if with_scene else None,
                              self.gate)
        return m


def _enroll(args):
    from skills import REGISTRY
    return REGISTRY.invoke("vision.enroll", args, permission_granted=True)


class EnrolSkillTest(unittest.TestCase):
    """The skill itself, with the firewall already satisfied."""

    @classmethod
    def setUpClass(cls):
        from skills import REGISTRY, load_all
        load_all()
        cls.reg = REGISTRY

    def test_skill_exists_and_is_critical(self):
        """Enrolment grants permissions, so it must not be a SAFE action."""
        s = self.reg.get("vision.enroll")
        self.assertIsNotNone(s)
        self.assertEqual(s.risk, "CRITICAL")
        self.assertIn("name", s.required)

    def test_refuses_when_nobody_is_there(self):
        with _Harness():
            r = _enroll({"name": "Dana"})
            self.assertFalse(r.ok)
            self.assertEqual(r.error, "no face available to enrol")
            self.assertIn("אין לי פנים", r.data["text_he"])

    def test_refuses_without_a_usable_name(self):
        with _Harness() as h:
            h.see(render(identity(11), seed=5))
            r = _enroll({"name": "אותי בבקשה"})
            self.assertFalse(r.ok)
            self.assertEqual(r.error, "no usable name given")
            self.assertEqual(h.store.list(), [])

    def test_first_enrolment_takes_the_owner_slot(self):
        with _Harness() as h:
            h.see(render(identity(11), seed=5))
            r = _enroll({"name": "Dana"})
            self.assertTrue(r.ok, r.error)
            self.assertTrue(r.data["is_owner"])
            self.assertEqual(r.data["level"], "CRITICAL")
            self.assertEqual(len(h.store.list()), 1)
            self.assertEqual(h.store.owner().name, "Dana")

    def test_second_enrolment_is_a_guest_not_an_owner(self):
        with _Harness() as h:
            h.see(render(identity(11), seed=5))
            _enroll({"name": "Dana"})
            h.see(render(identity(31), seed=9))
            r = _enroll({"name": "Guest"})
            self.assertTrue(r.ok, r.error)
            self.assertFalse(r.data["is_owner"])
            self.assertEqual(h.store.owner().name, "Dana")
            self.assertEqual(len(h.store.list()), 2)

    def test_a_display_is_never_enrolled(self):
        """The permanent-damage guard: a photo's identity must not enter the gallery."""
        with _Harness() as h:
            img = screen_like(render(identity(31), seed=7), period=4)
            h.see(img, with_scene=True)
            self.assertEqual(h.presence.current().liveness, "suspect")
            r = _enroll({"name": "Photo"})
            self.assertFalse(r.ok)
            self.assertEqual(r.error, "liveness suspect — refusing to enrol")
            self.assertEqual(h.store.list(), [])

    def test_enrolled_person_is_immediately_recognisable(self):
        """The point of the whole path: after enrolling, the camera knows them."""
        with _Harness() as h:
            img = render(identity(11), seed=5)
            h.see(img)
            _enroll({"name": "Dana"})
            m = h.store.recognize(img)
            self.assertTrue(m.known)
            self.assertEqual(m.name, "Dana")
            self.assertEqual(m.role, "owner")

    def test_vector_survives_recognize(self):
        """The root cause: without this field there was nothing to enrol from."""
        with _Harness() as h:
            m = h.store.recognize(render(identity(11), seed=5))
            self.assertIsNotNone(m.vector)
            self.assertEqual(m.vector.size, 4096)

    def test_vector_never_leaks_to_the_hud(self):
        with _Harness() as h:
            m = h.store.recognize(render(identity(11), seed=5))
            self.assertNotIn("vector", m.to_dict())
            h.presence.observe(m, None, h.gate)
            self.assertNotIn("vector", h.presence.current().to_dict())


class EnrolRoutingTest(unittest.TestCase):
    """The brain must hear "register me" as an act, not answer it as a question."""

    @classmethod
    def setUpClass(cls):
        cls.r = IntentRouter()

    def test_enrolment_requests_route_to_the_enrol_skill(self):
        for q in ("רשום אותי בשם דנה", "תכיר אותי, קוראים לי OSCAR",
                  "my name is Dana, enroll me", "register my face"):
            rt = self.r.route(q)
            self.assertEqual(rt.skill, "vision.enroll", q)
            self.assertEqual(rt.risk, "CRITICAL", q)

    def test_the_name_is_extracted(self):
        self.assertEqual(self.r.route("רשום אותי בשם דנה").args["name"], "דנה")
        self.assertEqual(self.r.route("תכיר אותי, קוראים לי OSCAR").args["name"], "OSCAR")
        self.assertEqual(self.r.route("call me Noam and enroll me").args["name"], "Noam")

    def test_a_missing_name_is_passed_empty_not_invented(self):
        """The router must not guess an identity to fill the gap."""
        self.assertEqual(self.r.route("רשום אותי").args["name"], "")

    def test_room_questions_are_still_questions(self):
        self.assertEqual(self.r.route("מי מולי").skill, "vision.who")
        self.assertEqual(self.r.route("מה אתה רואה").skill, "vision.scene")

    def test_other_intents_are_unaffected(self):
        self.assertEqual(self.r.route("מה השעה").intent, "TIME")
        self.assertEqual(self.r.route("כמה זה 2 ועוד 2").intent, "MATH")


class BrainEnrolTest(unittest.TestCase):
    """End to end through think(), with the firewall actually attached."""

    @classmethod
    def setUpClass(cls):
        from skills import REGISTRY, load_all
        load_all()
        cls.reg = REGISTRY

    def _engine(self, fw):
        return ReasoningEngine(skills=self.reg, router=IntentRouter(), firewall=fw)

    def test_critical_enrolment_is_blocked_without_a_confirm_channel(self):
        """The security property: nobody becomes owner without a human saying yes."""
        with _Harness() as h:
            h.see(render(identity(11), seed=5))
            eng = self._engine(h.fw)
            a = eng.think("רשום אותי בשם דנה")
            self.assertEqual(h.store.list(), [])
            self.assertIn("אישור", a.text)

    def test_enrolment_runs_when_the_human_confirms(self):
        with _Harness() as h:
            h.see(render(identity(11), seed=5))
            h.fw.on_confirm(lambda req: True)
            eng = self._engine(h.fw)
            a = eng.think("רשום אותי בשם דנה")
            self.assertEqual(len(h.store.list()), 1)
            self.assertEqual(h.store.owner().name, "דנה")
            self.assertIn("דנה", a.text)

    def test_the_brain_then_knows_who_it_is(self):
        """Closing the loop: enrol through the brain, be recognised by the brain."""
        with _Harness() as h:
            img = render(identity(11), seed=5)
            h.see(img)
            h.fw.on_confirm(lambda req: True)
            eng = self._engine(h.fw)
            eng.think("רשום אותי בשם דנה")
            h.see(img)
            a = eng.think("מי מולי")
            self.assertIn("דנה", a.text)

    def test_an_unidentified_stranger_at_safe_cannot_self_enrol(self):
        """SAFE is a hard ceiling — the state a stranger produces must not allow it."""
        with _Harness() as h:
            h.fw.set_level("SAFE")
            h.fw.on_confirm(lambda req: True)
            h.see(render(identity(11), seed=5))
            eng = self._engine(h.fw)
            eng.think("רשום אותי בשם דנה")
            self.assertEqual(h.store.list(), [])


class BlockPhrasingTest(unittest.TestCase):
    """Regression: ``_phrase_block`` shipped broken for a whole commit.

    It was converted from a ``@staticmethod`` to an instance method so it could
    consult presence, but the decorator was left behind — so ``self`` consumed
    the ``route`` argument and every firewall block raised ``TypeError`` instead
    of explaining itself. 1091 tests passed with that in the tree, because not
    one of them drove a block through the engine. This is the test that should
    have existed.
    """

    @classmethod
    def setUpClass(cls):
        from skills import REGISTRY, load_all
        load_all()
        cls.reg = REGISTRY

    def test_a_firewall_block_produces_an_explanation_not_a_crash(self):
        with _Harness() as h:
            h.fw.set_level("SAFE")
            h.fw.on_confirm(lambda req: True)
            h.see(render(identity(11), seed=5))
            eng = ReasoningEngine(skills=self.reg, router=IntentRouter(),
                                  firewall=h.fw)
            a = eng.think("רשום אותי בשם דנה")   # CRITICAL vs a SAFE firewall
            self.assertIsInstance(a.text, str)
            self.assertTrue(a.text.strip())
            self.assertEqual(h.store.list(), [])

    def test_the_block_is_explained_in_terms_of_who_is_there(self):
        """The whole point of making it an instance method."""
        with _Harness() as h:
            h.fw.set_level("SAFE")
            h.fw.on_confirm(lambda req: True)
            h.see(render(identity(11), seed=5))
            eng = ReasoningEngine(skills=self.reg, router=IntentRouter(),
                                  firewall=h.fw)
            a = eng.think("רשום אותי בשם דנה")
            self.assertTrue(any("מצלמה" in s.get("detail", "") or "presence"
                                == s.get("kind") for s in a.trace["steps"]))
            self.assertNotIn("Traceback", a.text)

    def test_phrase_block_is_callable_as_an_instance_method(self):
        """Direct pin on the exact defect: the decorator must be gone."""
        from brain.intent import Route
        eng = ReasoningEngine(skills=self.reg, router=IntentRouter())
        self.assertNotIsInstance(
            ReasoningEngine.__dict__["_phrase_block"], staticmethod,
            "_phrase_block is a staticmethod again but takes self")
        out = eng._phrase_block(Route("VISION", 0.9, "t", skill="vision.enroll"),
                                "blocked: action is CRITICAL but the firewall is "
                                "set to SAFE", "CRITICAL")
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())


class ProductionGalleryUntouchedTest(unittest.TestCase):
    """These tests must not write to the real gallery the way an earlier run did."""

    def test_real_gallery_is_still_empty(self):
        from vision.faces import FaceStore
        real = FaceStore()
        self.assertEqual(real.list(), [],
                         "a test leaked into data/faces/gallery.json")
        self.assertIsNone(real.owner())


def main() -> int:
    """Emit the RESULT line tests/run_all.py parses."""
    print("═" * 68)
    print(" J.A.R.V.I.S. — face enrolment reaching the brain")
    print("═" * 68)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"\nRESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
