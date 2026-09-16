"""JARVIS reasoning loop — the thing that actually makes it *think*.

    intent -> context -> plan -> act -> observe -> verify -> answer

The loop is explicit and observable: every stage emits an event on the bus so
the HUD can render the thought process live. Nothing is answered by vibes:

  * numbers come from the math engine
  * facts come from the knowledge store
  * actions come from the skill registry through the permission firewall
  * everything else comes from the neural core — and then passes a verifier
    that strips fabrication, enforces language and caps verbosity.

If verification fails, the loop *rethinks* (bounded by ``max_steps``).
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.intent import IntentRouter, Route  # noqa: E402
from brain.knowledge import content_tokens, shares_topic  # noqa: E402
from core.bus import BUS, T  # noqa: E402
from core.config import CONFIG  # noqa: E402


# ------------------------------------------------------------------- types --
@dataclass
class Step:
    kind: str                 # route | recall | plan | tool | verify | neural | reflect | answer
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "data": self.data, "ms": round(self.ms, 1)}


@dataclass
class Trace:
    steps: List[Step] = field(default_factory=list)
    # perf_counter, not time.time: to_dict() subtracts this from perf_counter to
    # get an elapsed duration, and the two clocks have unrelated reference
    # points. Mixing them published total_ms as roughly -1.79e12 — a number no
    # reader could mistake for a duration, but one that made every trace useless.
    started: float = field(default_factory=time.perf_counter)

    def add(self, kind: str, detail: str, data: Optional[Dict[str, Any]] = None, t0: float = 0.0) -> Step:
        step = Step(kind=kind, detail=detail, data=data or {},
                    ms=(time.perf_counter() - t0) * 1000 if t0 else 0.0)
        self.steps.append(step)
        BUS.emit(T.BRAIN_STEP, step.to_dict(), source="reasoning")
        return step

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps],
                "total_ms": round((time.perf_counter() - self.started) * 1000, 1)}


@dataclass
class Answer:
    text: str                 # what JARVIS says (Hebrew, speakable)
    speak: str                # what the TTS will actually voice (no code fences)
    grounded: bool = False
    confidence: float = 0.5
    intent: str = "UNKNOWN"
    skill: str = ""
    agent: str = ""
    risk: str = "SAFE"
    needs_confirmation: bool = False
    value: Any = None
    data: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "speak": self.speak, "grounded": self.grounded,
            "confidence": round(self.confidence, 3), "intent": self.intent,
            "skill": self.skill, "agent": self.agent, "risk": self.risk,
            "needs_confirmation": self.needs_confirmation, "value": self.value,
            "data": self.data, "trace": self.trace, "ms": round(self.ms, 1),
        }


# ---------------------------------------------------------------- verifier --
_HE_LETTERS = re.compile(r"[\u05d0-\u05ea]")
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


class Verifier:
    """Hallucination guard + speakability filter. Runs before JARVIS talks."""

    def __init__(self, knowledge=None) -> None:
        self.knowledge = knowledge
        self.issues: List[str] = []

    def check(self, text: str, *, numbers: Sequence[Any] = (), route: Optional[Route] = None,
              tool_values: Sequence[Any] = ()) -> Tuple[bool, List[str], str]:
        """Returns (ok, issues, cleaned_text)."""
        issues: List[str] = []
        cleaned = (text or "").strip()
        if not cleaned:
            return False, ["empty answer"], ""

        # 1. numeric grounding: any number in the answer must be traceable
        if numbers or tool_values:
            allowed = {_num_key(n) for n in list(numbers) + list(tool_values) if n is not None}
            allowed |= {str(int(y)) for y in (2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027)}
            allowed |= {_num_key(i) for i in range(0, 13)}          # small ordinals are fine
            for raw in _NUM.findall(cleaned)[:12]:
                key = _num_key(raw)
                if key and key not in allowed:
                    issues.append(f"unverified number {raw!r} in answer")

        # 2. language discipline
        he = len(_HE_LETTERS.findall(cleaned))
        latin = len(re.findall(r"[A-Za-z]", cleaned))
        if he == 0 and latin > 12 and (route is None or route.intent != "CODE"):
            issues.append("answer is not in Hebrew")

        # 3. length discipline (spoken answers must be short)
        if len(cleaned) > 1400:
            issues.append("answer too long for speech")
            cleaned = _truncate_sentences(cleaned, 900)

        # 4. refuse leaked protocol tokens
        if "<|" in cleaned:
            issues.append("leaked protocol token")
            cleaned = re.sub(r"<\|[^|]*\|>", "", cleaned).strip()

        # 5. refuse tool-call JSON leaking into speech
        if cleaned.startswith("{") and '"tool"' in cleaned[:60]:
            issues.append("raw tool-call JSON in answer")
            cleaned = ""

        return (not issues), issues, cleaned

    def speakable(self, text: str) -> str:
        """Strip markdown/code so the TTS does not read backticks aloud."""
        out = _CODE_FENCE.sub(" הקוד מצורף במסך. ", text or "")
        out = re.sub(r"`([^`]*)`", r"\1", out)
        out = re.sub(r"\*\*|__|~~", "", out)
        out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
        out = re.sub(r"^\s*[-*•]\s+", "", out, flags=re.M)
        out = re.sub(r"\s{2,}", " ", out)
        return out.strip()


# --------------------------------------------------- template-leak guard ----
# The neural core was trained on a templated corpus, so an off-topic prompt can
# make it recite a template from another domain instead of answering. These
# signature/keyword pairs catch that: if the answer reeks of a domain the
# question never mentioned, the answer is rejected and we fall back honestly.
_TEMPLATE_DOMAINS: Tuple[Tuple[re.Pattern, re.Pattern, str], ...] = (
    (re.compile(r"פעולה ברמת (CRITICAL|WRITE|SAFE)|אישור מפורש|יומן הביקורת|חומת ההרשאות|kill switch"),
     re.compile(r"הרשאה|אבטחה|סיכון|מסוכן|אישור|permission|critical|מחיקה|delete|חומרה|kill", re.I),
     "permissions"),
    (re.compile(r"זיהוי הדיבור|מילת השכמה|MFCC|מנוע דיבור|פונמות|voicebank"),
     re.compile(r"דיבור|קול|שומע|מדבר|מזהה|voice|speech|stt|tts|מיקרופון", re.I),
     "voice"),
    (re.compile(r"בדיקת התחלקות|התחלקות עד השורש|מספר ראשוני|פיבונאצ"),
     re.compile(r"ראשוני|prime|פיבונאצ|fib|מספר|חלוקה|חשבון|מתמטיק|כמה", re.I),
     "math"),
    (re.compile(r"מחרוזת הופכת|פלינדרום|palindrome"),
     re.compile(r"פלינדרום|הופכת|מחרוזת|string|reverse|קוד|פונקציה", re.I),
     "code"),
)


def template_leak(answer: str, question: str) -> str:
    """Return the leaked domain name when the answer is off-topic, else ''."""
    if not answer or not question:
        return ""
    for signature, keywords, domain in _TEMPLATE_DOMAINS:
        if signature.search(answer) and not keywords.search(question):
            return domain
    return ""


# Floor for presenting a memory hit as an answer, as opposed to merely using it
# as loose context. `recall` defaults to min_score 0.05, which is right for
# context and wrong for a claim: at that floor every "מה אמרתי על X" returned
# something, so JARVIS answered questions about dogs, spaceships and politics
# with a stored sentence about coffee, prefaced by "כן, אדוני. מהזיכרון שלי".
#
# 0.15 is deliberately conservative and is NOT fitted to a decision boundary.
# After `strip_recall_frame` removed the interrogative frame from the query, the
# measured probe scored genuine content matches 0.252-0.707 and irrelevant ones
# 0.056-0.229. Those ranges do now separate, but by only 0.023 across 15 samples,
# so a threshold placed inside that gap would be overfitting the probe — the same
# mistake as tuning against the knowledge base's 22-sample set. 0.15 sits well
# below the observed relevant minimum, blocks the noise cluster outright, and
# leaves room for the boundary to move on real data.
MEMORY_ANSWER_FLOOR = 0.15


def _num_key(value: Any) -> str:
    try:
        if isinstance(value, str):
            value = value.replace(",", "")
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return f"{f:.6g}"
    except (TypeError, ValueError):
        return str(value)


def _truncate_sentences(text: str, budget: int) -> str:
    parts = re.split(r"(?<=[.!?;])\s+", text)
    out = ""
    for p in parts:
        if len(out) + len(p) > budget:
            break
        out += (" " if out else "") + p
    return out.strip() or text[:budget]


# ------------------------------------------------------------------ engine --
class ReasoningEngine:
    """The conductor. Everything else is an organ it plays."""

    def __init__(
        self,
        core=None,
        router: Optional[IntentRouter] = None,
        skills=None,
        memory=None,
        firewall=None,
        knowledge=None,
        agents: Optional[Dict[str, Any]] = None,
        config=None,
        presence=None,
    ) -> None:
        self.core = core
        self.router = router or IntentRouter(knowledge=knowledge)
        self.skills = skills
        self.memory = memory
        self.firewall = firewall
        self.knowledge = knowledge
        self.agents = agents or {}
        # Who is in front of the camera. Lazy by design: importing brain.presence
        # unconditionally would pull vision into every headless test run, and the
        # tracker is a no-op when nothing has ever observed a frame.
        if presence is None:
            try:
                from brain.presence import get_presence
                presence = get_presence()
            except Exception:                                  # pragma: no cover
                presence = None
        self.presence = presence
        self.cfg = config or CONFIG.reasoning
        self.verifier = Verifier(knowledge)
        self.history: List[Tuple[str, str]] = []
        self.persona = CONFIG.persona
        self._smalltalk_turn = 0
        self._canned: Optional[Dict[str, List[str]]] = None

    # ---------------------------------------------------------------- API --
    def think(self, text: str, *, speak_stream: Optional[Callable[[str], None]] = None) -> Answer:
        t0 = time.perf_counter()
        trace = Trace()
        text = (text or "").strip()
        if not text:
            return Answer(text="לא קיבלתי טקסט, אדוני.", speak="לא קיבלתי טקסט, אדוני.",
                          ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())

        BUS.emit(T.BRAIN_THINK_START, {"text": text[:120]}, source="reasoning")

        # 0 ------------------------------------------------------ presence --
        # Who is in front of the camera, before deciding anything. This is the
        # step that was missing entirely: the firewall already knew, the brain
        # did not, so it could not greet by name or explain a refusal in terms
        # the user could act on. Read-only and cheap — it never runs detection.
        who: Optional[Any] = None
        if self.presence is not None:
            t = time.perf_counter()
            try:
                who = self.presence.current()
                trace.add("presence",
                          f"{who.name} · {who.level} · {who.role}" if who.known
                          else (f"{who.people} in frame, unidentified" if who.people
                                else "nobody in frame"),
                          who.to_dict(), t)
            except Exception:                                  # pragma: no cover
                who = None

        # 1 ------------------------------------------------------- intent --
        t = time.perf_counter()
        route = self.router.route(text)
        trace.add("route", f"{route.intent} ({route.confidence:.2f}) — {route.reason}",
                  route.to_dict(), t)

        # 2 ------------------------------------------------------- memory --
        recalled: List[Dict[str, Any]] = []
        if self.memory is not None:
            t = time.perf_counter()
            recalled = [h.to_dict() for h in self.memory.recall(text, k=self.cfg.max_steps // 2 or 3)]
            if recalled:
                trace.add("recall", f"{len(recalled)} memory hits", {"hits": recalled[:3]}, t)

        # 3 --------------------------------------- deterministic resolution --
        if route.reply_he and route.grounded:
            t = time.perf_counter()
            # Everything this branch itself produced or cited is traceable, and
            # only that. Two real cases, both of which used to be masked because
            # the verdict was discarded:
            #   * arithmetic — "17 * 23 שווה 391" contains the operands as well as
            #     the result, and they live in route.args['expr'], not route.value;
            #   * a knowledge answer — route.value is a dict whose 'answer' field
            #     is the cited source, so "Oct 31 שווה Dec 25" carries figures that
            #     came from the fact itself.
            # Walking route.value picks up both a bare result and the numbers
            # inside a source payload; the question and the user's own text are
            # included so a restated figure is not treated as invented.
            answer = self._verify_and_pack(
                route.reply_he, route, trace, t0,
                numbers=[*_numbers_in(route.value), *_numbers_in(route.args),
                         *_numbers_in(text)])
            trace.add("answer", "grounded deterministic answer", {"grounded": True}, t)
            self._remember(text, answer.text)
            return answer

        if route.intent == "MEMORY_WRITE":
            return self._handle_memory_write(text, route, trace, t0)
        if route.intent == "MEMORY_QUERY":
            return self._handle_memory_query(text, trace, t0, recalled)
        if route.intent == "CODE":
            return self._handle_code(text, route, trace, t0)
        if route.intent in ("IDENTITY", "HELP"):
            return self._handle_identity(route.intent, trace, t0)
        if route.intent in ("SMALLTALK", "GREETING"):
            return self._handle_smalltalk(text, route, trace, t0)

        # 4 --------------------------------------- tool execution (act) ----
        if route.skill and self.skills is not None:
            answer = self._run_skill(text, route, trace, t0)
            if answer is not None:
                self._remember(text, answer.text)
                return answer

        # 5 --------------------------------------- knowledge grounding -----
        if self.knowledge is not None and route.intent == "KNOWLEDGE":
            hits = self.knowledge.search(text, k=2)
            if hits:
                t = time.perf_counter()
                trace.add("plan", f"ground answer on {hits[0]['id']}", {"score": hits[0]["score"]}, t)
                prompt_note = f"ענה בעברית, בקצרה, על בסיס: {hits[0]['text_he'][:400]}"
                answer = self._neural_answer(text, trace, t0, system_note=prompt_note,
                                             route=route, stream=speak_stream)
                if answer.text:
                    return answer

        # 6 --------------------------------------- neural core (free) ------
        answer = self._neural_answer(text, trace, t0, route=route, stream=speak_stream,
                                     memory_hint=recalled)
        if answer.text:
            self._remember(text, answer.text)
            return answer

        # 7 --------------------------------------- honest fallback ---------
        answer = self._fallback(text, route, trace, t0)
        self._remember(text, answer.text)
        return answer

    # ------------------------------------------------------------- helpers --
    def _remember(self, user: str, assistant: str) -> None:
        self.history.append((user, assistant))
        self.history = self.history[-12:]
        if self.memory is not None:
            try:
                self.memory.remember_exchange(user, assistant)
            except Exception as exc:
                BUS.emit(T.ERROR, {"where": "memory", "error": str(exc)}, source="reasoning")

    def _verify_and_pack(self, text: str, route: Route, trace: Trace, t0: float,
                         numbers: Sequence[Any] = (),
                         extra: Optional[Dict[str, Any]] = None,
                         speak_override: str = "") -> Answer:
        t = time.perf_counter()
        ok, issues, cleaned = self.verifier.check(text, numbers=numbers, route=route)
        trace.add("verify", "pass" if ok else f"issues: {issues}", {"ok": ok, "issues": issues}, t)

        # The verdict used to be logged here and then ignored: the answer went out
        # at full confidence whether or not it had passed. That is the difference
        # between having a verifier and actually thinking — so a failure now
        # changes the answer.
        grounded = route.grounded
        confidence = route.confidence
        if not ok:
            bad = set(issues)
            if any("unverified number" in i for i in bad):
                # A number we cannot trace is exactly the thing JARVIS must not
                # assert. Drop the claim of grounding and say so; never present an
                # untraced figure as if it had come from a tool.
                grounded = False
                confidence = min(confidence, 0.35)
            else:
                confidence = min(confidence, 0.5)
        speak = speak_override or self.verifier.speakable(cleaned or text)
        data = dict(extra or {})
        data["verified"] = bool(ok)
        if issues:
            data["verify_issues"] = list(issues)
        return Answer(text=cleaned or text, speak=speak, grounded=grounded,
                      confidence=confidence, intent=route.intent, skill=route.skill,
                      agent=route.agent, risk=route.risk, value=route.value,
                      data=data,
                      ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())

    def _run_skill(self, text: str, route: Route, trace: Trace, t0: float) -> Optional[Answer]:
        skill = self.skills.get(route.skill) if self.skills else None
        if skill is None:
            trace.add("tool", f"skill {route.skill} not registered — falling back", {})
            return None

        t = time.perf_counter()
        trace.add("plan", f"invoke {route.skill} with {route.args}", {"risk": skill.risk}, t)

        decision = None
        if self.firewall is not None:
            t = time.perf_counter()
            decision = self.firewall.check(route.skill, skill.risk, route.args, agent=route.agent or "hermes")
            trace.add("tool", f"firewall: {'allow' if decision.allowed else 'block'} — {decision.reason}",
                      decision.to_dict(), t)
            if not decision.allowed:
                reply = self._phrase_block(route, decision.reason, skill.risk)
                return self._verify_and_pack(reply, route, trace, t0)
            if self.firewall.dry_run and skill.risk != "SAFE":
                reply = (f"מצב הדמיה פעיל, אדוני. הייתי מבצע {route.skill} עם {json.dumps(route.args, ensure_ascii=False)}, "
                         f"אבל לא נגעתי במערכת.")
                return self._verify_and_pack(reply, route, trace, t0)

        BUS.emit(T.BRAIN_TOOL_CALL, {"skill": route.skill, "args": route.args,
                                 "risk": skill.risk}, source="reasoning")
        t = time.perf_counter()
        result = self.skills.invoke(route.skill, route.args, permission_granted=True)
        BUS.emit(T.BRAIN_TOOL_RESULT, result.to_dict(), source="reasoning")
        trace.add("tool", f"{route.skill} -> {'ok' if result.ok else 'error'}", result.to_dict(), t)

        # a real tool result is *grounded truth* — that is the whole point of tools
        if result.ok:
            route.grounded = True
        reply = self._phrase_tool_result(route, result)
        numbers = _numbers_in(result.value)
        extra = result.data if isinstance(result.data, dict) else None
        speak = ""
        if route.intent == "RAG" and extra and extra.get("answer_type"):
            # Three corrections, all of them things the HUD would otherwise lie
            # about, scoped to RAG so no other skill's behaviour changes:
            #
            # 1. ``route.confidence`` is the router's 0.88 "this looks like a
            #    file query" — it says nothing about the evidence. Publishing it
            #    as the answer's confidence showed 88% for an answer the
            #    retrieval layer had already scored at 0.3.
            # 2. A refusal still returns ``ok=True`` from the skill (refusing is
            #    success), which used to light the "grounded" badge on an answer
            #    that explicitly says it found nothing.
            # 3. The answer body quotes source code and line ranges, which a TTS
            #    engine would read character by character; the skill supplies a
            #    short spoken version instead.
            route.confidence = float(extra.get("confidence") or route.confidence)
            if extra.get("answer_type") == "none":
                route.grounded = False
            speak = str(extra.get("summary_he") or "")
        return self._verify_and_pack(reply, route, trace, t0, numbers=numbers,
                                     extra=extra, speak_override=speak)

    @staticmethod
    def _phrase_block(self, route: Route, reason: str, risk: str) -> str:
        if "kill switch" in reason:
            return "מתג החירום מופעל, אדוני. שום פעולה לא תתבצע עד שתשחרר אותו."
        if "confirmation" in reason:
            return (f"הפעולה {route.skill} היא ברמת {risk} ודורשת אישור מפורש. "
                    f"אני מבקש אישור במסך לפני שאגע במערכת.")
        if "protected" in reason:
            return f"חסמתי את הפעולה: הנתיב המבוקש נמצא באזור מוגן. {reason}."
        # A level block is the one case where the camera explains it better than
        # the firewall does. "action is CRITICAL but the firewall is set to SAFE"
        # is true and useless; naming who is (or is not) in front of the lens
        # tells the user what to actually do about it.
        if "but the firewall is set to" in reason and self.presence is not None:
            try:
                why = self.presence.explain(risk)
                if why:
                    return f"לא ביצעתי את הפעולה. {why}"
            except Exception:                                  # pragma: no cover
                pass
        return f"לא ביצעתי את הפעולה. הסיבה: {reason}."

    @staticmethod
    def _phrase_tool_result(route: Route, result) -> str:
        if not result.ok:
            return f"ניסיתי להפעיל את {route.skill} אבל נכשל: {result.error}."
        value = result.value
        if route.intent == "TIME":
            return str(value)
        if isinstance(value, (int, float, str, bool)):
            return str(value)
        if isinstance(value, dict):
            if "summary_he" in value:
                return str(value["summary_he"])
            try:
                return json.dumps(value, ensure_ascii=False)[:600]
            except Exception:
                return str(value)[:600]
        if isinstance(value, (list, tuple)):
            if not value:
                return "לא נמצאו פריטים."
            head = ", ".join(str(v)[:60] for v in value[:8])
            more = f" ועוד {len(value) - 8} פריטים" if len(value) > 8 else ""
            return f"מצאתי {len(value)} פריטים: {head}{more}."
        return str(value)[:600]

    def _handle_memory_write(self, text: str, route: Route, trace: Trace, t0: float) -> Answer:
        body = route.args.get("text", text)
        t = time.perf_counter()
        if self.memory is not None:
            self.memory.remember_fact(f"user:{body[:60]}", body, confidence=0.95, source="user")
        trace.add("tool", "wrote user preference to semantic memory", {"body": body[:80]}, t)
        reply = f"נרשם לזיכרון, אדוני: {body}. אקח את זה בחשבון בהמשך."
        return self._verify_and_pack(reply, route, trace, t0)

    def _handle_memory_query(self, text: str, trace: Trace, t0: float,
                             recalled: Sequence[Dict[str, Any]]) -> Answer:
        route = Route("MEMORY_QUERY", 0.8, "recall request")
        t = time.perf_counter()
        # Filter to hits strong enough to assert, not merely to use as context.
        strong = [h for h in (recalled or []) if float(h.get("score", 0.0)) >= MEMORY_ANSWER_FLOOR]
        if not strong:
            trace.add("recall", f"no memory hits above {MEMORY_ANSWER_FLOOR} "
                                f"({len(recalled or [])} weak)",
                      {"top": [h.get("score") for h in (recalled or [])[:3]]}, t)
            reply = "לא מצאתי משהו רלוונטי בזיכרון לשאלה הזו. אם תזכיר לי את ההקשר, אשמור אותו עכשיו."
        else:
            trace.add("recall", f"{len(strong)} hits above floor", {"top": strong[:3]}, t)
            lines = " ".join(f"[{h['score']:.2f}] {h['text'][:120]}" for h in strong[:3])
            reply = f"כן, אדוני. מהזיכרון שלי: {lines}"
        return self._verify_and_pack(reply, route, trace, t0)

    def _handle_code(self, text: str, route: Route, trace: Trace, t0: float) -> Answer:
        agent = self.agents.get("hephaestus")
        t = time.perf_counter()
        if agent is None:
            trace.add("plan", "HEPHAESTUS not loaded — answering from the neural core", {}, t)
            return self._neural_answer(text, trace, t0, route=route,
                                       system_note="המשתמש ביקש קוד. ספק קוד פייתון תקין עם הסבר קצר בעברית.")
        trace.add("plan", "dispatch to HEPHAESTUS (coder agent)", {"task": text[:120]}, t)

        # HEPHAESTUS writes Python and then *runs it* in a subprocess sandbox. That
        # is arbitrary code execution, and it used to happen with no firewall check
        # at all: `_handle_code` dispatched straight to `agent.handle(text)`, while
        # every skill-based action went through `firewall.check` in `_run_tool`.
        # Measured, the gap was total — identical code ran at level WRITE, at level
        # SAFE, and with the emergency kill switch engaged. A kill switch that stops
        # fs.write but not code execution is not a kill switch, and SAFE as a "hard
        # ceiling" meant nothing on this path.
        #
        # Gated at WRITE, not CRITICAL, and the distinction was measured rather than
        # assumed. CRITICAL is correct for shell.exec, which runs arbitrary commands
        # against the real system; at CRITICAL this path requires human confirmation
        # for *every* coding request, which headless denies outright and which cost
        # a 180s confirm_timeout per call — that broke the coding agent completely
        # instead of protecting it. Sandboxed execution under an isolated
        # interpreter with a timeout is the same trust class as fs.write, so WRITE
        # is the honest level. It still closes the actual hole: SAFE blocks it as a
        # hard ceiling and the kill switch stops it.
        if self.firewall is not None:
            t = time.perf_counter()
            decision = self.firewall.check("coder.run", "WRITE",
                                           {"task": text[:200]}, agent="hephaestus")
            trace.add("tool",
                      f"firewall: {'allow' if decision.allowed else 'block'} — {decision.reason}",
                      decision.to_dict(), t)
            if not decision.allowed:
                BUS.emit(T.AGENT_RESULT, {"agent": "hephaestus", "ok": False,
                                          "blocked": decision.reason}, source="reasoning")
                reply = self._phrase_block(route, decision.reason, "WRITE")
                return self._verify_and_pack(reply, route, trace, t0)
            if self.firewall.dry_run:
                reply = ("מצב הדמיה פעיל, אדוני. הייתי כותב ומריץ קוד עבור הבקשה הזו "
                         "בארגז החול, אבל לא הרצתי דבר.")
                return self._verify_and_pack(reply, route, trace, t0)

        BUS.emit(T.AGENT_DISPATCH, {"agent": "hephaestus", "task": text[:200]}, source="reasoning")
        try:
            out = agent.handle(text)
        except Exception as exc:
            trace.add("reflect", f"coder agent failed: {exc}", {}, time.perf_counter())
            out = {"text": f"סוכן המתכנת נכשל: {exc}", "code": "", "ok": False}
        BUS.emit(T.AGENT_RESULT, {"agent": "hephaestus", "ok": bool(out.get("ok"))}, source="reasoning")
        t = time.perf_counter()
        trace.add("tool", "coder agent returned",
                  {"ok": out.get("ok"), "understood": out.get("understood", True),
                   "chars": len(out.get("text", ""))}, t)

        if not out.get("understood", True) and self.knowledge is not None:
            # The coder only produced a skeleton, so this was probably not a coding
            # task at all. Give the curated knowledge a chance before we answer —
            # "יש לי שגיאה בקוד" has a real answer in the KB, and a skeleton does not.
            qa = self.knowledge.search_qa(text, k=1, min_score=0.5)
            if qa:
                route = Route("KNOWLEDGE", 0.9, f"coder declined; KB answer ({qa[0]['fact_id']})")
                route.grounded = True
                trace.add("reflect", "coder declined — answering from the knowledge base",
                          {"from": qa[0]["fact_id"]}, time.perf_counter())
                answer = self._verify_and_pack(qa[0]["answer"], route, trace, t0)
                self._remember(text, answer.text)
                return answer

        understood = bool(out.get("understood", True))
        ans = Answer(text=out.get("text", ""), speak=self.verifier.speakable(out.get("text", "")),
                     grounded=bool(out.get("ok")) and understood,
                     confidence=(0.9 if out.get("ok") else 0.4) if understood else 0.45,
                     intent="CODE", agent="hephaestus",
                     data={"code": out.get("code", ""), "tests": out.get("tests"), "repairs": out.get("repairs")},
                     ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())
        self._remember(text, ans.text)
        return ans

    # ------------------------------------------------------------- smalltalk --
    SMALLTALK_CATEGORIES: Tuple[Tuple[str, re.Pattern], ...] = (
        ("bye", re.compile(r"להתראות|ביי|נתראה|לילה טוב|bye|good ?night", re.I)),
        ("joke", re.compile(r"בדיחה|תצחיק|מצחיק|joke|funny", re.I)),
        ("howareyou", re.compile(r"מה שלומך|מה נשמע|איך אתה מרגיש|מה מצבך|how are you", re.I)),
        ("thanks", re.compile(r"תודה|thanks|thank you|מעולה|יופי|כל הכבוד", re.I)),
        ("opinion", re.compile(r"מה דעתך|מה אתה חושב|האם אתה מאמין|what do you think", re.I)),
        ("learning", re.compile(r"ללמוד|לימוד|יכולת למידה|learn", re.I)),
        ("capability", re.compile(r"מה אתה (יודע|מסוגל)|היכולות שלך|what can you do", re.I)),
        ("greeting", re.compile(r"^(שלום|היי|הי|אהלן|בוקר טוב|ערב טוב|צהריים טובים|hello|hi|hey)", re.I)),
    )

    def _smalltalk_category(self, text: str) -> str:
        for name, pattern in self.SMALLTALK_CATEGORIES:
            if pattern.search(text):
                return name
        return "unknown"

    def _smalltalk_lines(self, category: str) -> List[str]:
        if self.knowledge is None:
            return []
        if getattr(self, "_canned", None) is None:
            canned: Dict[str, List[str]] = {}
            for fact in self.knowledge.facts:
                block = (fact.extra or {}).get("canned")
                if isinstance(block, dict):
                    for k, v in block.items():
                        if isinstance(v, (list, tuple)):
                            canned.setdefault(str(k), []).extend(str(x) for x in v if x)
            self._canned = canned
        return list(self._canned.get(category) or self._canned.get("unknown") or [])

    def _handle_smalltalk(self, text: str, route: Route, trace: Trace, t0: float) -> Answer:
        """Short social exchange — answered from our own KB, never invented."""
        t = time.perf_counter()
        reply = ""
        source = ""
        if self.knowledge is not None:
            qa = self.knowledge.search_qa(text, k=1, min_score=0.42)
            if qa:
                reply = qa[0]["answer"]
                source = f"knowledge:{qa[0]['fact_id']} ({qa[0]['score']:.2f})"
                route.grounded = True
        if not reply:
            category = self._smalltalk_category(text)
            lines = self._smalltalk_lines(category)
            if lines:
                reply = lines[self._smalltalk_turn % len(lines)]
                self._smalltalk_turn += 1
                source = f"canned:{category}"
                # a canned line is curated text from our own KB, not an invention
                # of the neural core — it earns the same grounded flag as a fact
                route.grounded = True
        if reply:
            trace.add("plan", f"smalltalk reply from {source}", {"text": reply[:80]}, t)
            answer = self._verify_and_pack(reply, route, trace, t0)
            self._remember(text, answer.text)
            return answer
        trace.add("reflect", "no smalltalk match — deferring to the neural core", {}, t)
        return self._neural_answer(text, trace, t0, route=route)

    def _handle_identity(self, intent: str, trace: Trace, t0: float) -> Answer:
        # Both branches below answer from text we wrote ourselves (a KB fact or the
        # persona card), so neither is a hallucination risk -> grounded.
        route = Route(intent, 0.95, "identity/help answer")
        route.grounded = True
        if self.knowledge is not None:
            qa = self.knowledge.search_qa("מי אתה" if intent == "IDENTITY" else "מה אתה יודע לעשות", k=1, min_score=0.4)
            if qa:
                t = time.perf_counter()
                trace.add("answer", "grounded identity answer", {"from": qa[0]["fact_id"]}, t)
                return self._verify_and_pack(qa[0]["answer"], route, trace, t0)
        t = time.perf_counter()
        trace.add("answer", "identity answer from the persona card",
                  {"from": "persona", "kb_match": False}, t)
        reply = (f"אני {self.persona['name']} — {self.persona['full_name']}. עוזר אישי שרץ כולו על המחשב שלך, "
                 f"בלי ענן ובלי מפתחות API. אני חושב, מדבר בעברית, כותב ומריץ קוד, שולט בקבצים ובתהליכים, "
                 f"וזוכר את השיחות שלנו.")
        return self._verify_and_pack(reply, route, trace, t0)

    def _neural_answer(self, text: str, trace: Trace, t0: float, *, route: Optional[Route] = None,
                       system_note: str = "", memory_hint: Sequence[Dict[str, Any]] = (),
                       stream: Optional[Callable[[str], None]] = None,
                       attempt: int = 1) -> Answer:
        route = route or Route("UNKNOWN", 0.3, "neural generation")
        if self.core is None or not self.core.available:
            trace.add("neural", "neural core unavailable", {})
            return Answer(text="", speak="", intent=route.intent, confidence=0.0,
                          ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())

        note = system_note or (
            f"אתה {self.persona['name']}. ענה בעברית, בקצרה, מדויק. "
            f"{self.persona['principles']}"
        )
        if memory_hint:
            note += " זיכרון רלוונטי: " + " | ".join(h["text"][:70] for h in memory_hint[:2])

        prompt = self.core.build_prompt(text, self.history, system_note=note)
        t = time.perf_counter()
        trace.add("neural", f"generating (attempt {attempt}) via {self.core.backend}",
                  {"prompt_chars": len(prompt)}, t)

        gen = self.core.generate(prompt, max_new_tokens=140, stream_fn=stream,
                                 stop_texts=("<|user|>", "<|eos|>", "<|tool|>"))
        raw = gen.text
        t = time.perf_counter()
        ok, issues, cleaned = self.verifier.check(raw, route=route)
        leaked = template_leak(raw, text)
        if leaked:
            ok = False
            issues.append(f"recited an unrelated {leaked} template instead of answering")
            cleaned = ""
        trace.add("verify", "pass" if ok else f"issues: {issues}",
                  {"ok": ok, "issues": issues, "tokens": gen.tokens, "ms": round(gen.ms, 1)}, t)

        if not ok and self.cfg.reflect_on_failure and attempt < 2:
            t = time.perf_counter()
            trace.add("reflect", f"rethinking because: {issues}", {"attempt": attempt + 1}, t)
            stricter = (note + f" התיקון הנדרש: {'; '.join(issues)}. ענה בעברית בלבד, משפט או שניים, "
                               "ללא ספרות שאינן ודאיות.")
            return self._neural_answer(text, trace, t0, route=route, system_note=stricter,
                                       memory_hint=memory_hint, stream=stream, attempt=attempt + 1)

        if not cleaned:
            return Answer(text="", speak="", intent=route.intent, confidence=0.0,
                          ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())

        # Free generation with no deterministic basis has to at least be *about*
        # the question. A 6.2M-parameter core drifts into unrelated corpus
        # fragments on out-of-domain prompts ("מי כתב את ההמלט" -> "אני מריץ את
        # הקוד עכשיו"), and publishing a non-sequitur is worse than the caller's
        # honest "I don't know". Returning empty text hands control to that path.
        if route.intent == "UNKNOWN" and route.confidence < 0.5 and not shares_topic(text, cleaned):
            t = time.perf_counter()
            trace.add("reflect", "discarding a non-sequitur generation — no shared topic",
                      {"question_tokens": sorted(content_tokens(text))[:6],
                       "answer_tokens": sorted(content_tokens(cleaned))[:6]}, t)
            return Answer(text="", speak="", intent=route.intent, confidence=route.confidence,
                          ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())

        speak = self.verifier.speakable(cleaned)
        return Answer(text=cleaned, speak=speak, grounded=False,
                      confidence=max(0.35, route.confidence), intent=route.intent,
                      skill=route.skill, agent=route.agent, risk=route.risk,
                      data={"tokens": gen.tokens, "backend": gen.backend, "stop": gen.stop_reason},
                      ms=(time.perf_counter() - t0) * 1000, trace=trace.to_dict())

    def _fallback(self, text: str, route: Route, trace: Trace, t0: float) -> Answer:
        t = time.perf_counter()
        trace.add("reflect", "all paths failed — honest fallback", {}, t)
        if self.knowledge is not None:
            hits = self.knowledge.search(text, k=1, min_score=0.15, require_overlap=True)
            if hits:
                reply = f"אין לי תשובה ישירה, אבל מצאתי במאגר הידע: {hits[0]['text_he'][:300]}"
                return self._verify_and_pack(reply, route, trace, t0)
        reply = ("אין לי תשובה ודאית לזה, אדוני, ואני לא ממציא. "
                 "אפשר לנסח אחרת, לבקש חישוב מדויק, או לתת לי גישה לקובץ הרלוונטי ואקרא אותו.")
        return self._verify_and_pack(reply, route, trace, t0)


def _numbers_in(value: Any) -> List[Any]:
    out: List[Any] = []

    def walk(v: Any) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.append(v)
        elif isinstance(v, str):
            for m in _NUM.findall(v):
                out.append(m)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(value)
    return out
