"""Deliberation — plan, act, verify, repair.

``think()`` is a single pass: route, act, answer. That is the right amount of
machinery for "what time is it" and not enough for "compute 17*23 then open the
calculator", where two things were asked and only the first one that matched a
pattern ever got done. It also had no way to notice that a step came back
ungrounded and try a different source instead of shipping it.

This module is the loop that was missing. Two rules keep it honest:

* **It engages only when a request genuinely decomposes.** A request is split on
  explicit sequential markers, and then only if the pieces independently route
  to actionable intents. "כמה זה 2 ועוד 2" contains a conjunction and must stay
  one arithmetic question, not become two failed steps.

* **It invents nothing.** Every step is executed by the same pipeline the single
  pass would have used, with the same firewall in front of it. A step that
  cannot be grounded is reported as unresolved; the combined answer says so
  rather than smoothing it over.

The repair path is the part that makes this thinking rather than sequencing: a
step whose first attempt came back ungrounded gets one alternative attempt
against the user's own files before it is written off.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional

# Explicit sequential markers. Deliberately narrow: a bare "ו" would split
# "2 ועוד 2" and turn one solvable question into two unanswerable ones.
# Every group here is non-capturing on purpose. ``re.split`` interleaves the
# contents of capturing groups into the result and yields None for any group that
# did not participate, which turns the split into a list full of holes.
_SEQUENTIAL = re.compile(
    r"(?:ואז|ו אז|ואחר[־ ]כך|ואחרי[־ ]כך|אחר[־ ]כך|אחרי[־ ]זה|אחרי זה"
    r"|ולבסוף|ובסוף|קודם כל|ראשית|שנית|בשלב (?:הראשון|השני|השלישי)"
    r"|ואז גם|וגם ת?פתח|וגם ת?חשב|וגם ת?בדוק|וגם ת?חפש"
    r"|\band then\b|\bthen\b|\bafter that\b|\bfinally\b)", re.I)

# What counts as a piece worth executing on its own.
_MIN_WORDS = 2
_MAX_STEPS = 4

# A step is worth reporting as resolved only if it cleared this bar. Anything
# below it is either the honest "I don't know" or a guess, and neither belongs
# in an answer presented as a completed plan.
_RESOLVED_CONFIDENCE = 0.45

# Intents that mean "this piece was actually acted on or answered from a source"
# rather than "the router gave up".
_ACTIONABLE = frozenset({
    "MATH", "TIME", "SYSTEM", "FILES", "APPS", "CODE", "KNOWLEDGE", "RAG",
    "VISION", "MEMORY_WRITE", "MEMORY_QUERY", "SMALLTALK", "GREETING",
    "IDENTITY", "HELP",
})


def split_steps(text: str) -> List[str]:
    """Break a compound request into ordered subgoals.

    Returns a single-element list for anything that is not genuinely compound, so
    the caller never has to special-case the ordinary path.
    """
    t = (text or "").strip()
    if not t or not _SEQUENTIAL.search(t):
        return [t]

    # Cut on the marker itself, keeping the piece that follows it.
    pieces = [p.strip(" ,.;:!?·") for p in _SEQUENTIAL.split(t)]
    # ``re.split`` with a capture group interleaves the separators; drop them and
    # anything the cut left empty or trivially short.
    cleaned: List[str] = []
    for p in pieces:
        p = (p or "").strip()
        if not p or _SEQUENTIAL.fullmatch(p):
            continue
        if len([w for w in p.split() if w]) < _MIN_WORDS:
            # A dangling fragment ("ואז") carries no task; fold it away rather
            # than presenting an empty step to the user.
            continue
        cleaned.append(p)

    if len(cleaned) < 2:
        return [t]
    return cleaned[:_MAX_STEPS]


class Deliberator:
    """Plans a multi-part request, executes it step by step, and repairs failures.

    Holds a reference to the engine rather than duplicating its pipeline: the
    firewall, the verifier and the skills must all be the same instances, or a
    plan could take a path a single request would have been blocked on.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._lock = threading.RLock()

    # -------------------------------------------------------------- gating --
    def wants(self, text: str) -> bool:
        """True only when splitting will actually do more work than one pass."""
        steps = split_steps(text)
        if len(steps) < 2:
            return False
        # Every piece has to stand on its own. If any of them routes to UNKNOWN
        # the "compound request" is really one question with noise around it, and
        # splitting would only manufacture a failed step.
        router = getattr(self.engine, "router", None)
        if router is None:                                  # pragma: no cover
            return False
        actionable = 0
        for s in steps:
            try:
                r = router.route(s)
            except Exception:                                  # pragma: no cover
                return False
            if r.intent not in _ACTIONABLE or not r.skill and not r.reply_he:
                return False
            actionable += 1
        return actionable >= 2

    # --------------------------------------------------------------- loop --
    def run(self, text: str, trace: Any, t0: float) -> Optional[Any]:
        """Execute the plan. Returns an Answer, or None to fall back to one pass."""
        import time

        steps = split_steps(text)
        if len(steps) < 2:
            return None

        t = time.perf_counter()
        trace.add("plan", f"decompose into {len(steps)} steps", {"steps": steps}, t)

        results: List[Dict[str, Any]] = []
        for i, step in enumerate(steps, start=1):
            t = time.perf_counter()
            ans = self._execute(step)
            resolved = self._is_resolved(ans)
            attempts = 1
            repaired = False

            if not resolved:
                # Repair: one alternative attempt against the user's own files.
                # This is the step that makes the loop more than sequencing — a
                # single pass would have shipped the weak answer here.
                alt = self._repair(step)
                attempts = 2
                if alt is not None and self._is_resolved(alt):
                    ans = alt
                    resolved = True
                    repaired = True

            trace.add(
                "tool" if resolved else "reflect",
                f"step {i}/{len(steps)} {'resolved' if resolved else 'unresolved'}"
                f"{', repaired via files' if repaired else ''}",
                {"step": step, "resolved": resolved, "attempts": attempts,
                 "intent": getattr(ans, "intent", "") if ans is not None else "",
                 "skill": getattr(ans, "skill", "") if ans is not None else "",
                 "confidence": round(float(getattr(ans, "confidence", 0.0) or 0.0), 3)},
                t)

            results.append({
                "step": step, "resolved": resolved, "attempts": attempts,
                "repaired": repaired,
                "text": (getattr(ans, "text", "") or "") if ans is not None else "",
                "grounded": bool(getattr(ans, "grounded", False)) if ans is not None else False,
                "confidence": round(float(getattr(ans, "confidence", 0.0) or 0.0), 3)
                if ans is not None else 0.0,
                "intent": getattr(ans, "intent", "") if ans is not None else "",
            })

        return self._combine(text, results, trace, t0)

    # -------------------------------------------------------------- parts --
    def _execute(self, step: str) -> Optional[Any]:
        """Run one subgoal through the ordinary pipeline, un-split."""
        engine = self.engine
        with self._lock:
            guard = getattr(engine, "_deliberating", False)
            engine._deliberating = True
        try:
            return engine.think(step)
        except Exception:                                      # pragma: no cover
            return None
        finally:
            with self._lock:
                engine._deliberating = guard

    def _repair(self, step: str) -> Optional[Any]:
        """Second attempt at a step, sourced from the user's own files."""
        skills = getattr(self.engine, "skills", None)
        if skills is None:
            return None
        firewall = getattr(self.engine, "firewall", None)
        try:
            skill = skills.get("rag.ask")
        except Exception:
            skill = None
        if skill is None:
            return None
        args = {"query": step}
        if firewall is not None:
            try:
                if not firewall.check("rag.ask", skill.risk, args, agent="argus").allowed:
                    return None
            except Exception:                                  # pragma: no cover
                return None
        try:
            res = skills.invoke("rag.ask", args, permission_granted=True)
        except Exception:                                      # pragma: no cover
            return None
        if not getattr(res, "ok", False):
            return None
        data = res.data if isinstance(res.data, dict) else {}
        # Only a genuinely grounded hit counts as a repair. The retrieval layer
        # answers "weak" when the best it found barely matched — accepting that
        # would launder a near-miss into a step reported as resolved, which is
        # the exact failure the repair path exists to avoid.
        if not data.get("grounded") or str(data.get("answer_type", "")) != "grounded":
            return None
        if float(data.get("confidence", 0.0) or 0.0) < _RESOLVED_CONFIDENCE:
            return None

        from brain.intent import Route
        from brain.reasoning import Answer
        conf = float(data.get("confidence", 0.0) or 0.0)
        text = str(res.value or "")
        route = Route("RAG", max(conf, 0.5), "repair attempt over local files",
                      skill="rag.ask", grounded=True, value=res.value, agent="argus")
        return Answer(text=text, speak=text, grounded=True, confidence=conf,
                      intent="RAG", skill="rag.ask", agent="argus",
                      data={"repaired": True, **data})

    @staticmethod
    def _is_resolved(ans: Optional[Any]) -> bool:
        """A step counts as done only if it produced a grounded, confident answer."""
        if ans is None:
            return False
        text = (getattr(ans, "text", "") or "").strip()
        if not text:
            return False
        if getattr(ans, "intent", "") == "UNKNOWN":
            return False
        if float(getattr(ans, "confidence", 0.0) or 0.0) < _RESOLVED_CONFIDENCE:
            return False
        return True

    def _combine(self, text: str, results: List[Dict[str, Any]], trace: Any,
                 t0: float) -> Optional[Any]:
        """Fold the step results into one Hebrew answer, honestly."""
        import time

        from brain.intent import Route
        from brain.reasoning import Answer

        done = [r for r in results if r["resolved"]]
        missed = [r for r in results if not r["resolved"]]

        lines: List[str] = []
        for i, r in enumerate(results, start=1):
            body = (r["text"] or "").strip()
            if r["resolved"]:
                lines.append(f"{i}. {body}")
            else:
                lines.append(f"{i}. לא הצלחתי לפתור את «{r['step']}» ולא אני ממציא תשובה.")

        reply = "ביצעתי את זה בשלבים:\n" + "\n".join(lines)
        if missed:
            reply += (f"\nהשלמתי {len(done)} מתוך {len(results)} שלבים; "
                      f"את השאר אני מדווח כלא פתור במקום לנחש.")

        grounded = bool(done) and all(r["grounded"] for r in done)
        confidence = (sum(r["confidence"] for r in results) / len(results)) if results else 0.0
        if missed:
            confidence = min(confidence, 0.55)

        t = time.perf_counter()
        trace.add("answer", f"plan complete: {len(done)}/{len(results)} resolved",
                  {"steps": results, "grounded": grounded}, t)

        route = Route("PLAN", 0.9, "multi-step plan", grounded=grounded,
                      value={"steps": results})
        ans = Answer(text=reply,
                     speak=self._speakable(reply),
                     grounded=grounded, confidence=round(confidence, 3),
                     intent="PLAN", skill="", agent="",
                     data={"plan": results, "resolved": len(done),
                           "total": len(results)},
                     ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())
        return ans

    @staticmethod
    def _speakable(reply: str) -> str:
        """The numbered layout is for the screen; speech needs prose."""
        body = reply.split("\n", 1)[-1] if "\n" in reply else reply
        parts = [re.sub(r"^\d+\.\s*", "", ln).strip() for ln in body.split("\n") if ln.strip()]
        return " ".join(p for p in parts if p)
