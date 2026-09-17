"""Multi-step planning with verification — the loop that makes it think.

``think()`` was a single pass: route, act, answer. Ask it for two things and the
first pattern that matched won; the rest of the sentence was dropped without a
trace. And when a step came back ungrounded there was no second attempt, only
whatever the fallback had.

These tests pin the three things that make ``brain/deliberate.py`` deliberation
rather than a string split:

* it engages **only** when a request genuinely decomposes — "כמה זה 2 ועוד 2"
  contains a conjunction and must stay one arithmetic question;
* every subgoal goes through the ordinary pipeline, so the firewall still stands
  in front of each one;
* an unresolved step is reported as unresolved. The combined answer says how many
  of the steps it actually finished instead of smoothing over the gap.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.deliberate import Deliberator, split_steps  # noqa: E402
from brain.intent import INTENTS, IntentRouter  # noqa: E402
from brain.knowledge import KnowledgeStore  # noqa: E402
from brain.reasoning import ReasoningEngine  # noqa: E402


def _engine() -> ReasoningEngine:
    from skills import REGISTRY, load_all
    load_all()
    kb = KnowledgeStore()
    return ReasoningEngine(skills=REGISTRY, router=IntentRouter(knowledge=kb),
                           knowledge=kb)


class SplitTest(unittest.TestCase):
    """Decomposition must be conservative or it manufactures failed steps."""

    def test_arithmetic_with_a_conjunction_stays_one_question(self):
        """The case a naive split gets wrong: "2 ועוד 2" is not two tasks."""
        self.assertEqual(len(split_steps("כמה זה 2 ועוד 2")), 1)

    def test_ordinary_questions_are_never_split(self):
        for q in ("מה השעה", "ספר לי בדיחה", "מה מצב המערכת", "כמה זה 7 כפול 8"):
            self.assertEqual(len(split_steps(q)), 1, q)

    def test_a_sequenced_request_splits_in_order(self):
        steps = split_steps("חשב 17 כפול 23 ואז מה השעה")
        self.assertEqual(len(steps), 2)
        self.assertIn("17", steps[0])
        self.assertIn("שעה", steps[1])

    def test_the_marker_itself_is_not_a_step(self):
        """No empty or dangling fragment may survive as a step of its own."""
        for s in split_steps("פתח את המחשבון ואחר כך תגיד לי מה השעה"):
            self.assertGreaterEqual(len(s.split()), 2, s)

    def test_step_count_is_capped(self):
        q = " ".join(["חשב 2 ועוד 2 ואז"] * 12)
        self.assertLessEqual(len(split_steps(q)), 4)

    def test_empty_input_is_safe(self):
        self.assertEqual(split_steps(""), [""])


class GateTest(unittest.TestCase):
    """The planner must not engage for a request that is not really compound."""

    @classmethod
    def setUpClass(cls):
        cls.eng = _engine()

    def test_single_questions_do_not_engage(self):
        for q in ("מה השעה", "כמה זה 2 ועוד 2", "ספר לי בדיחה"):
            self.assertFalse(self.eng.deliberator.wants(q), q)

    def test_a_real_compound_request_engages(self):
        self.assertTrue(self.eng.deliberator.wants("חשב 17 כפול 23 ואז מה השעה"))

    def test_a_compound_with_an_unanswerable_piece_does_not(self):
        """One solvable step plus noise is not a plan; splitting would only add
        a manufactured failure to the trace."""
        self.assertFalse(self.eng.deliberator.wants("חשב 2 ועוד 2 ואז מה מספר הטלפון של הרופא שלי"))


class PlanExecutionTest(unittest.TestCase):
    """End to end: both steps get done, and the answer says so."""

    @classmethod
    def setUpClass(cls):
        cls.eng = _engine()

    def test_both_steps_are_executed(self):
        a = self.eng.think("חשב 17 כפול 23 ואז מה השעה")
        self.assertEqual(a.intent, "PLAN")
        self.assertEqual(a.data.get("total"), 2)
        self.assertEqual(a.data.get("resolved"), 2)
        # each subgoal's own answer must appear in the combined reply
        self.assertIn("391", a.text)
        self.assertIn("שלב", a.text)

    def test_the_plan_is_visible_in_the_trace(self):
        a = self.eng.think("חשב 17 כפול 23 ואז מה השעה")
        kinds = [s["kind"] for s in (a.trace.get("steps") or [])]
        self.assertIn("plan", kinds)
        plan_step = next(s for s in a.trace["steps"] if s["kind"] == "plan")
        self.assertEqual(len(plan_step["data"]["steps"]), 2)

    def test_each_step_is_recorded_individually(self):
        a = self.eng.think("חשב 17 כפול 23 ואז מה השעה")
        steps = [s for s in a.trace["steps"] if str(s.get("detail", "")).startswith("step ")]
        self.assertEqual(len(steps), 2)
        for s in steps:
            self.assertIn("resolved", s["data"])
            self.assertIn("attempts", s["data"])

    def test_speech_is_prose_not_a_numbered_list(self):
        """The numbered layout is for the screen; TTS must not read "1." aloud."""
        a = self.eng.think("חשב 17 כפול 23 ואז מה השעה")
        self.assertNotIn("\n", a.speak)
        self.assertFalse(a.speak.strip().startswith("1."))
        self.assertIn("391", a.speak)

    def test_an_ordinary_question_is_untouched(self):
        """The gate is the safety property: no planning overhead, same answer."""
        a = self.eng.think("כמה זה 2 ועוד 2")
        self.assertEqual(a.intent, "MATH")
        self.assertTrue(a.grounded)
        kinds = [s["kind"] for s in (a.trace.get("steps") or [])]
        self.assertNotIn("plan", kinds)

    def test_plan_intent_is_declared(self):
        self.assertIn("PLAN", INTENTS)

    def test_a_subgoal_is_never_split_again(self):
        """Re-entrancy: without the guard a plan could recurse into itself."""
        a = self.eng.think("חשב 17 כפול 23 ואז מה השעה")
        self.assertFalse(getattr(self.eng, "_deliberating", False))
        self.assertEqual(a.data.get("total"), 2)

    def test_planning_costs_more_than_one_step_but_stays_fast(self):
        t0 = time.perf_counter()
        a = self.eng.think("חשב 17 כפול 23 ואז מה השעה")
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 8000.0, f"planning took {elapsed:.0f}ms")
        self.assertEqual(a.data.get("resolved"), 2)


class HonestReportingTest(unittest.TestCase):
    """An unfinished step must be reported, not hidden."""

    @classmethod
    def setUpClass(cls):
        cls.eng = _engine()

    def test_an_unresolvable_step_is_said_to_be_unresolved(self):
        eng = self.eng
        # Drive the combine step directly with a result the pipeline could not
        # resolve, which is the state worth pinning: the answer must name the gap.
        from brain.reasoning import Trace
        d = Deliberator(eng)
        results = [
            {"step": "חשב 2 ועוד 2", "resolved": True, "attempts": 1,
             "repaired": False, "text": "2 + 2 שווה 4.", "grounded": True,
             "confidence": 0.99, "intent": "MATH"},
            {"step": "מה מספר הטלפון של הרופא שלי", "resolved": False,
             "attempts": 2, "repaired": False, "text": "", "grounded": False,
             "confidence": 0.25, "intent": "UNKNOWN"},
        ]
        ans = d._combine("x", results, Trace(), time.perf_counter())
        self.assertEqual(ans.data["resolved"], 1)
        self.assertEqual(ans.data["total"], 2)
        self.assertIn("לא הצלחתי", ans.text)
        self.assertIn("1 מתוך 2", ans.text)
        # a partly finished plan must not claim full grounding
        self.assertLessEqual(ans.confidence, 0.55)

    def test_a_fully_resolved_plan_claims_grounding(self):
        from brain.reasoning import Trace
        d = Deliberator(self.eng)
        results = [
            {"step": "a", "resolved": True, "attempts": 1, "repaired": False,
             "text": "תשובה אחת.", "grounded": True, "confidence": 0.99,
             "intent": "MATH"},
            {"step": "b", "resolved": True, "attempts": 1, "repaired": False,
             "text": "תשובה שנייה.", "grounded": True, "confidence": 0.95,
             "intent": "TIME"},
        ]
        ans = d._combine("x", results, Trace(), time.perf_counter())
        self.assertTrue(ans.grounded)
        self.assertEqual(ans.data["resolved"], 2)
        self.assertNotIn("לא הצלחתי", ans.text)

    def test_is_resolved_rejects_the_honest_dunno(self):
        d = Deliberator(self.eng)
        self.assertFalse(d._is_resolved(None))
        a = self.eng.think("מה מספר הטלפון של הרופא שלי")
        self.assertEqual(a.intent, "UNKNOWN")
        self.assertFalse(d._is_resolved(a))

    def test_repair_never_fabricates_when_there_is_no_index(self):
        """With no RAG index there is nothing to repair from, and the planner must
        return None rather than invent a second answer."""
        d = Deliberator(self.eng)
        out = d._repair("מה כתוב בקבצים על נושא שלא קיים בשום מקום")
        self.assertIsNone(out)


def main() -> int:
    """Emit the RESULT line tests/run_all.py parses."""
    print("═" * 68)
    print(" J.A.R.V.I.S. — deliberation: plan, act, verify, repair")
    print("═" * 68)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"\nRESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
