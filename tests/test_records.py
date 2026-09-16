"""ediyel records — the people dossier.

The face gallery answers "have I seen this face before?", which is a vector
comparison and nothing else. This holds the rest: the story, the age, the
birthday, the note you actually want when somebody walks in.

Two things are worth testing beyond the happy path:

* **A stored age is a lie the day after you write it.** When a birthday is
  known it wins, because that one never needs updating.
* **A photo path that does not exist is refused, not stored.** A dossier that
  claims a picture it cannot show is worse than one that admits it has none.

Everything here is a local JSON file. The tests use a temp directory so they
cannot touch the real store, and one test asserts the real store was untouched.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from agents.records import (Person, RecordsAgent, RecordsStore,  # noqa: E402
                            _years_since)


class _Store(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="rec_"))
        self.st = RecordsStore(path=self.dir / "people.json")


class DossierTest(_Store):
    def test_add_needs_a_name(self):
        self.assertIsNone(self.st.add(name=""))
        self.assertIsNone(self.st.add(name="   "))
        self.assertEqual(self.st.list(), [])

    def test_add_and_get(self):
        p = self.st.add(name="דנה", age=34, relationship="אחות",
                        story="אוהבת לרוץ בבוקר.")
        self.assertIsNotNone(p)
        self.assertEqual(self.st.get(p.id).name, "דנה")
        self.assertEqual(len(self.st.list()), 1)

    def test_a_dossier_without_a_face_is_legal(self):
        """You can keep a record of your mother without pointing a camera at her."""
        p = self.st.add(name="אמא", story="")
        self.assertEqual(p.face_id, "")
        self.assertFalse(p.has_photo)

    def test_birthday_beats_a_stale_stored_age(self):
        p = self.st.add(name="ילד", age=5, birthday="2020-01-01")
        self.assertNotEqual(p.age_now, 5)
        self.assertEqual(p.age_now, _years_since("2020-01-01"))

    def test_year_parsing_accepts_two_honest_shapes_and_rejects_rubbish(self):
        self.assertEqual(_years_since("1991-03-14"), _years_since("14/03/1991"))
        self.assertIsNone(_years_since("garbage"))
        self.assertIsNone(_years_since(""))
        self.assertIsNone(_years_since("1850-01-01"))   # out of range

    def test_impossible_dates_are_rejected_not_clamped(self):
        self.assertIsNone(_years_since("1991-13-45"))

    def test_story_is_truncated_not_rejected(self):
        p = self.st.add(name="ארוך", story="א" * 9000)
        self.assertLessEqual(len(p.story), 4000)

    def test_summary_is_one_hebrew_sentence(self):
        p = self.st.add(name="דנה", age=34, relationship="אחות",
                        story="אוהבת לרוץ. יש לה שני חתולים.")
        s = p.summary_he
        self.assertIn("דנה", s)
        self.assertIn("אחות", s)
        self.assertIn("אוהבת לרוץ", s)
        self.assertNotIn("חתולים", s)          # only the first sentence

    def test_update_ignores_unknown_fields(self):
        p = self.st.add(name="דנה")
        self.st.update(p.id, note="x", not_a_field="y")
        self.assertEqual(self.st.get(p.id).note, "x")
        self.assertNotIn("not_a_field", self.st.get(p.id).to_dict())

    def test_update_cannot_blank_a_name(self):
        p = self.st.add(name="דנה")
        self.st.update(p.id, name="")
        self.assertEqual(self.st.get(p.id).name, "דנה")

    def test_remove(self):
        p = self.st.add(name="דנה")
        self.assertTrue(self.st.remove(p.id))
        self.assertFalse(self.st.remove(p.id))
        self.assertIsNone(self.st.get(p.id))

    def test_survives_reload(self):
        p = self.st.add(name="דנה", story="x")
        again = RecordsStore(path=self.dir / "people.json")
        self.assertEqual(len(again.list()), 1)
        self.assertEqual(again.get(p.id).story, "x")

    def test_a_corrupt_file_does_not_take_the_assistant_down(self):
        bad = self.dir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        st = RecordsStore(path=bad)
        self.assertEqual(st.list(), [])
        self.assertTrue(bad.with_suffix(".corrupt.json").is_file())

    def test_stats(self):
        self.st.add(name="א", story="x")
        self.st.add(name="ב")
        s = self.st.stats()
        self.assertEqual(s["people"], 2)
        self.assertEqual(s["with_story"], 1)
        self.assertEqual(s["with_face"], 0)


class SearchTest(_Store):
    def setUp(self):
        super().setUp()
        self.st.add(name="דנה", relationship="אחות", tags=["משפחה", "רצה"],
                    story="אוהבת לרוץ בבוקר.")
        self.st.add(name="נועם", relationship="חבר",
                    story="הכרתי אותו דרך דנה בעבודה.")

    def test_exact_name_beats_a_story_mention(self):
        hits = self.st.find("דנה")
        self.assertEqual(hits[0].name, "דנה")
        self.assertEqual(len(hits), 2)          # נועם mentions her

    def test_finds_by_relationship_and_tag(self):
        self.assertEqual([p.name for p in self.st.find("אחות")], ["דנה"])
        self.assertEqual([p.name for p in self.st.find("רצה")], ["דנה"])

    def test_no_match_is_empty_not_an_error(self):
        self.assertEqual(self.st.find("מישהו שלא קיים"), [])
        self.assertEqual(self.st.find(""), [])

    def test_face_join(self):
        p = self.st.find("דנה")[0]
        self.st.link_face(p.id, "face-xyz")
        self.assertEqual(self.st.by_face("face-xyz").name, "דנה")
        self.assertIsNone(self.st.by_face("nope"))
        self.assertIsNone(self.st.by_face(""))


class SkillTest(unittest.TestCase):
    """Through the registry, which is how the brain actually calls them."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ["JARVIS_RECORDS_DIR"] = tempfile.mkdtemp(prefix="recsk_")
        from skills import REGISTRY, load_all
        load_all()
        cls.reg = REGISTRY
        from agents.records import get_store
        get_store(fresh=True)

    def invoke(self, skill, **kw):
        # The parameter cannot be called ``name``: that collides with the
        # ``name=`` argument every records skill takes.
        return self.reg.invoke(skill, kw, permission_granted=True)

    def test_the_dossier_skills_are_registered(self):
        names = {s.name for s in self.reg.all() if s.name.startswith("records.")}
        self.assertEqual(names, {"records.add", "records.find", "records.list",
                                 "records.whoami", "records.update",
                                 "records.remove", "records.link",
                                 "records.upcoming"})

    def test_reading_is_safe_and_writing_is_not(self):
        for name in ("records.find", "records.list", "records.whoami",
                     "records.upcoming"):
            self.assertEqual(self.reg.get(name).risk, "SAFE", name)
        for name in ("records.add", "records.update", "records.remove",
                     "records.link"):
            self.assertEqual(self.reg.get(name).risk, "WRITE", name)

    def test_add_find_remove_round_trip(self):
        r = self.invoke("records.add", name="טלי", story="מנגנת בצ'לו.")
        self.assertTrue(r.ok, r.error)
        pid = r.value["id"]
        r = self.invoke("records.find", query="טלי")
        self.assertTrue(r.ok)
        self.assertEqual(r.value["count"], 1)
        self.assertIn("צ'לו", r.data["text_he"])
        r = self.invoke("records.remove", person_id=pid)
        self.assertTrue(r.ok)

    def test_a_missing_photo_is_refused_not_stored(self):
        r = self.invoke("records.add", name="נועם", photo="/no/such/file.jpg")
        self.assertFalse(r.ok)
        self.assertIn("photo not found", r.error)
        self.assertEqual(self.invoke("records.find", query="נועם").value["count"], 0)

    def test_whoami_answers_even_with_no_camera(self):
        r = self.invoke("records.whoami")
        self.assertTrue(r.ok)
        self.assertIn("עדיין לא יודע", r.data["text_he"])

    def test_every_skill_refuses_empty_arguments(self):
        for name in ("records.add", "records.update", "records.remove",
                     "records.find", "records.link"):
            r = self.invoke(name)
            self.assertFalse(r.ok, name)
            self.assertTrue(r.error, name)

    def test_update_reaches_the_store(self):
        """Regression: **kwargs was rejected by the registry validator, so every
        real update failed with 'unexpected arguments'."""
        r = self.invoke("records.add", name="מיקי")
        pid = r.value["id"]
        r = self.invoke("records.update", person_id=pid, note="הערה חדשה")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.value["note"], "הערה חדשה")

    def test_age_is_coerced_and_range_checked(self):
        r = self.invoke("records.add", name="גיל")
        pid = r.value["id"]
        self.assertEqual(self.invoke("records.update", person_id=pid,
                                     age="40").value["age"], 40)
        self.assertIsNone(self.invoke("records.update", person_id=pid,
                                      age="999").value["age"])


class AgentContractTest(unittest.TestCase):
    def test_the_agent_matches_the_shape_the_orchestrator_expects(self):
        st = RecordsStore(path=Path(tempfile.mkdtemp()) / "p.json")
        st.add(name="דנה", story="אוהבת לרוץ.")
        ag = RecordsAgent(store=st)
        self.assertEqual(ag.name, "ediyel_records")
        self.assertTrue(ag.role)
        out = ag.handle("דנה")
        self.assertTrue(out["ok"])
        self.assertIn("דנה", out["text_he"])
        d = ag.to_dict()
        self.assertEqual(d["name"], "ediyel_records")
        self.assertEqual(d["people"], 1)

    def test_an_unknown_person_is_not_an_error(self):
        ag = RecordsAgent(store=RecordsStore(path=Path(tempfile.mkdtemp()) / "p.json"))
        out = ag.handle("מישהו")
        self.assertFalse(out["ok"])
        self.assertIn("אין לי רשומה", out["text_he"])


class RoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from brain.intent import IntentRouter
        cls.r = IntentRouter()

    def test_records_requests_route_to_records(self):
        cases = {"מי אני": "records.whoami",
                 "מי רשום ברשומות": "records.list",
                 "תוסיף לרשומות את דנה": "records.add",
                 "מה אתה יודע על דנה": "records.find"}
        for q, skill in cases.items():
            rt = self.r.route(q)
            self.assertEqual(rt.skill, skill, q)
            self.assertEqual(rt.intent, "RECORDS", q)

    def test_the_name_is_extracted_for_a_lookup(self):
        self.assertEqual(self.r.route("מה אתה יודע על דנה").args["query"], "דנה")
        self.assertEqual(self.r.route("ספר לי על נועם").args["query"], "נועם")

    def test_adding_never_invents_a_name(self):
        self.assertEqual(self.r.route("תוסיף לרשומות את דנה").args["name"], "")

    def test_enrolment_is_not_confused_with_a_dossier(self):
        self.assertEqual(self.r.route("רשום אותי בשם דנה").skill, "vision.enroll")

    def test_other_intents_are_unaffected(self):
        self.assertEqual(self.r.route("מה השעה").intent, "TIME")
        self.assertEqual(self.r.route("כמה זה 2 ועוד 2").intent, "MATH")
        self.assertEqual(self.r.route("מי מולי").skill, "vision.who")


class BirthdayTest(unittest.TestCase):
    """Upcoming birthdays — the payoff for having stored a date at all.

    Reading is SAFE and needs no face. Scheduling is a side effect, so it stays
    behind the same permission as any other dossier write. A date that will not
    parse is skipped rather than guessed at: a reminder on the wrong day teaches
    the user to ignore all the others.
    """

    @classmethod
    def setUpClass(cls):
        import os
        from skills import load_all
        cls._saved_env = os.environ.get("JARVIS_RECORDS_DIR")
        load_all()

    @classmethod
    def tearDownClass(cls):
        import os
        if cls._saved_env is None:
            os.environ.pop("JARVIS_RECORDS_DIR", None)
        else:
            os.environ["JARVIS_RECORDS_DIR"] = cls._saved_env

    def setUp(self):
        import datetime as dt
        import os
        os.environ["JARVIS_RECORDS_DIR"] = tempfile.mkdtemp(prefix="bday_")
        from agents.records import get_store
        self.st = get_store(fresh=True)
        self.today = dt.date.today()

    def _rel(self, n):
        import datetime as dt
        d = self.today + dt.timedelta(days=n)
        return f"{d.year}-{d.month:02d}-{d.day:02d}"

    def _inv(self, **kw):
        from skills import REGISTRY
        return REGISTRY.invoke("records.upcoming", kw, permission_granted=True)

    def test_days_until_parses_both_shapes_and_rejects_rubbish(self):
        import datetime as dt
        from agents.records import days_until_birthday as d
        t = dt.date(2026, 9, 16)
        self.assertEqual(d("1991-03-14", t), d("14/03/1991", t))
        self.assertEqual(d("2020-09-16", t), 0)
        self.assertEqual(d("2020-09-20", t), 4)
        self.assertIsNone(d("", t))
        self.assertIsNone(d("garbage", t))
        self.assertIsNone(d("2020-13-45", t))

    def test_feb_29_rolls_to_feb_28_in_a_common_year(self):
        """Someone born on the 29th still has a birthday."""
        import datetime as dt
        from agents.records import days_until_birthday as d
        # 2027 is a common year
        self.assertEqual(d("1988-02-29", dt.date(2027, 2, 27)), 1)

    def test_upcoming_is_sorted_nearest_first(self):
        self.st.add(name="רחוק", birthday=self._rel(20))
        self.st.add(name="דנה", birthday=self._rel(4), relationship="אחות")
        self.st.add(name="נועם", birthday=self._rel(0), relationship="חבר")
        r = self._inv(days=30)
        self.assertTrue(r.ok, r.error)
        self.assertEqual([p["name"] for p in r.value["people"]],
                         ["נועם", "דנה", "רחוק"])
        self.assertIn("היום", r.data["text_he"])
        self.assertIn("דנה", r.data["text_he"])

    def test_unparseable_dates_are_skipped_not_guessed(self):
        self.st.add(name="זבל", birthday="garbage")
        self.st.add(name="ריק")
        r = self._inv(days=365)
        self.assertEqual(r.value["count"], 0)
        self.assertIn("אין ימי הולדת", r.data["text_he"])

    def test_the_window_is_clamped_not_crashed(self):
        self.assertEqual(self._inv(days="abc").value["window_days"], 30)
        self.assertEqual(self._inv(days=99999).value["window_days"], 365)
        self.assertEqual(self._inv(days=-5).value["window_days"], 0)

    def test_remind_schedules_only_future_birthdays(self):
        """Today's birthday is already here; a reminder for now is just noise."""
        self.st.add(name="נועם", birthday=self._rel(0))
        self.st.add(name="דנה", birthday=self._rel(4))
        r = self._inv(days=30, remind=True)
        self.assertEqual([s["name"] for s in r.value["scheduled"]], ["דנה"])
        self.assertTrue(r.value["scheduled"][0]["ok"])

    def test_reading_needs_no_permission_and_no_face(self):
        from skills import REGISTRY
        self.assertEqual(REGISTRY.get("records.upcoming").risk, "SAFE")
        r = REGISTRY.invoke("records.upcoming", {})
        self.assertTrue(r.ok, r.error)


class RealStoreUntouchedTest(unittest.TestCase):
    def test_the_production_dossier_was_not_written_to(self):
        from agents.records import DEFAULT_FILE
        if DEFAULT_FILE.is_file():
            import json
            data = json.loads(DEFAULT_FILE.read_text(encoding="utf-8"))
            self.assertEqual(data.get("people", []), [],
                             "a test leaked into data/records/people.json")


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — ediyel records")
    print("═" * 68)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"\nRESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
