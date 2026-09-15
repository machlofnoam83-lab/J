"""JARVIS — the orchestrator agent.

Owns the whole pipeline:

    input (text or recognised speech)
      -> reasoning loop (intent / memory / plan / tools / verify)
      -> answer
      -> speech synthesis
      -> event stream for the HUD

It also runs the honest boot self-check (nothing in the boot sequence is fake —
each line is a real measurement), exposes the kill switch, and routes permission
prompts to the UI.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import skills as skills_pkg  # noqa: E402
from agents.hephaestus import HephaestusAgent  # noqa: E402
from brain.inference import NeuralCore  # noqa: E402
from brain.intent import IntentRouter  # noqa: E402
from brain.knowledge import KnowledgeStore  # noqa: E402
from brain.memory import MemoryPalace  # noqa: E402
from brain.reasoning import Answer, ReasoningEngine  # noqa: E402
from core.bus import BUS, T  # noqa: E402
from core.config import CONFIG, DATA  # noqa: E402
from security.permissions import FIREWALL, PermissionFirewall  # noqa: E402
from skills.registry import REGISTRY  # noqa: E402
from voice.tts import VoiceEngine, get_voice  # noqa: E402

BOOT_LINES = [
    ("core.bus", "אפיק האירועים"),
    ("core.config", "תצורת מערכת"),
    ("brain.tokenizer", "טוקנייזר BPE"),
    ("brain.neural", "ליבה עצבית"),
    ("brain.knowledge", "מאגר ידע"),
    ("brain.memory", "ארמון הזיכרון"),
    ("brain.intent", "מנתב כוונות"),
    ("brain.math", "מנוע מתמטיקה"),
    ("skills.registry", "מרשם כלים"),
    ("security.firewall", "חומת הרשאות"),
    ("voice.g2p", "ממיר כתב לצלילים"),
    ("voice.tts", "מנוע דיבור"),
    ("voice.stt", "זיהוי דיבור"),
    ("agent.hephaestus", "סוכן מתכנת"),
    ("sys.telemetry", "טלמטריה"),
]


@dataclass
class Turn:
    user: str
    answer: Answer
    spoke: bool = False
    ms: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"user": self.user, "answer": self.answer.to_dict(), "spoke": self.spoke,
                "ms": round(self.ms, 1), "ts": self.ts}


class JarvisAgent:
    name = "jarvis"
    role = "המנצח"

    def __init__(self, speak_out: bool = True, audio_sink: Optional[Callable[[bytes, int], bool]] = None,
                 confirm_hook: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
        self.t0 = time.perf_counter()
        self.speak_out = speak_out
        self.turns: List[Turn] = []
        # Chat history is a record of a conversation, not a cache: it is written to
        # data/transcript.jsonl as it happens and read back on start-up, so the HUD
        # shows yesterday's talk instead of an empty room.
        self.transcript_path = Path(getattr(CONFIG.memory, "transcript_path",
                                            str(DATA / "transcript.jsonl")))
        self._log: List[Dict[str, Any]] = []
        self._load_transcript()

        skills_pkg.load_all()
        self.firewall: PermissionFirewall = FIREWALL
        # The sentinel watches the shapes the per-call firewall cannot see:
        # bursts, repeats and denial streaks across the session. Detection is
        # always on; actuation stays opt-in.
        from security.sentinel import SENTINEL
        self.sentinel = SENTINEL
        SENTINEL.firewall = self.firewall
        SENTINEL.attach(BUS)
        self.voice: VoiceEngine = get_voice(audio_sink=audio_sink) if audio_sink else get_voice()
        if audio_sink is not None:
            self.voice.audio_sink = audio_sink
        self.core = NeuralCore()
        self.knowledge = KnowledgeStore()
        self.memory = MemoryPalace(CONFIG.memory.db_path,
                                   decay_half_life_days=CONFIG.memory.decay_half_life_days)
        self.router = IntentRouter(knowledge=self.knowledge, skills=REGISTRY, config=CONFIG)
        self.coder = HephaestusAgent(core=self.core, memory=self.memory)
        self.agents: Dict[str, Any] = {"hephaestus": self.coder, "mnemosyne": self.memory,
                                       "argus": self, "hermes": self}
        self.engine = ReasoningEngine(core=self.core, router=self.router, skills=REGISTRY,
                                      memory=self.memory, firewall=self.firewall,
                                      knowledge=self.knowledge, agents=self.agents)
        self.bus_history: List[Dict[str, Any]] = []
        # Boot progress is shared between the worker thread running the POST and
        # the HTTP thread serving /api/boot/progress, so it needs its own lock.
        self._boot_lock = threading.Lock()
        self._boot_progress: List[Dict[str, Any]] = []
        self._boot_done = False
        self._boot_started = 0.0

        # ── human-in-the-loop confirmation bridge (HUD ⇄ firewall) ──
        self._confirm_events: Dict[int, threading.Event] = {}
        self._confirm_verdicts: Dict[int, bool] = {}
        self._confirm_requests: Dict[int, Dict[str, Any]] = {}
        self._confirm_seq = itertools.count(1)
        self.confirm_timeout = float(getattr(CONFIG.security, "confirm_timeout", 180.0))
        self.firewall.on_confirm(confirm_hook or self._confirm_bridge)

        BUS.on("*", self._on_event)
        BUS.emit(T.READY, self.status(), source=self.name)

    # ---------------------------------------------------------------- events --
    def _on_event(self, ev) -> None:
        self.bus_history.append(ev.to_dict())
        if len(self.bus_history) > 600:
            self.bus_history = self.bus_history[-600:]

    # ------------------------------------------------------------------ boot --
    # Two of these checks are expensive by design: voice.stt builds the whole
    # template bank (~3 minutes in the test suite) and agent.hephaestus writes,
    # runs and verifies real code. On a slow Windows machine that meant the boot
    # POST ground for minutes while the HUD sat on a black overlay at a 0% bar —
    # the interface was not broken, it was faithfully waiting on a response that
    # never streamed. So each check now runs on its own daemon thread with a
    # hard timeout, results land in a progress buffer as they complete so the HUD
    # can fill live, and a check that overruns is reported as overrun and skipped
    # rather than allowed to hold the whole sequence hostage.
    CHECK_TIMEOUT = float(os.environ.get("JARVIS_BOOT_CHECK_TIMEOUT", "30"))
    BOOT_BUDGET = float(os.environ.get("JARVIS_BOOT_BUDGET", "150"))

    def boot(self) -> List[Dict[str, Any]]:
        """Honest POST: every line is a real measurement, not theatre."""
        BUS.emit(T.BOOT, {"items": len(BOOT_LINES)}, source=self.name)
        with self._boot_lock:
            self._boot_progress = []
            self._boot_done = False
            self._boot_started = time.perf_counter()
        report: List[Dict[str, Any]] = []
        for key, label in BOOT_LINES:
            elapsed = time.perf_counter() - self._boot_started
            if elapsed > self.BOOT_BUDGET:
                item = {"key": key, "label": label, "ok": False,
                        "detail": f"לא נבדק — תקציב האתחול ({self.BOOT_BUDGET:g}s) נגמר",
                        "ms": 0.0}
            else:
                t = time.perf_counter()
                ok, detail = self._run_check_timed(key, self.CHECK_TIMEOUT)
                ms = (time.perf_counter() - t) * 1000
                item = {"key": key, "label": label, "ok": ok, "detail": detail,
                        "ms": round(ms, 1)}
            report.append(item)
            with self._boot_lock:
                self._boot_progress.append(item)
            BUS.emit("boot.item", item, source=self.name)
        with self._boot_lock:
            self._boot_done = True
        BUS.emit("system.ready", {"items": len(report),
                                  "failed": sum(1 for r in report if not r["ok"])}, source=self.name)
        return report

    def boot_progress(self) -> Dict[str, Any]:
        """Items completed so far, so the HUD can fill while the POST runs."""
        with self._boot_lock:
            return {"items": list(self._boot_progress), "done": self._boot_done,
                    "total": len(BOOT_LINES),
                    "elapsed": round(time.perf_counter() - self._boot_started, 1)
                    if self._boot_started else 0.0}

    def _run_check_timed(self, key: str, timeout: float) -> Tuple[bool, str]:
        """Run one check on a daemon thread; never let it hold the sequence."""
        box: Dict[str, Tuple[bool, str]] = {}

        def work() -> None:
            try:
                box["r"] = self._check(key)
            except BaseException as exc:  # a hung check must not become a crash
                box["r"] = (False, f"{type(exc).__name__}: {exc}")

        th = threading.Thread(target=work, daemon=True, name=f"boot-{key}")
        th.start()
        th.join(timeout)
        if th.is_alive():
            # The thread is abandoned on purpose (daemon): whatever it was
            # waiting on — a subprocess, a device, a lock — keeps going in the
            # background and simply never reports. Saying so beats hiding it.
            return False, (f"הבדיקה לא הסתיימה תוך {timeout:g}s "
                           f"— ממשיכים בלעדיה")
        return box.get("r", (False, "הבדיקה לא החזירה תוצאה"))

    def _check(self, key: str) -> Tuple[bool, str]:
        try:
            if key == "core.bus":
                return True, f"{len(BUS.stats)} נושאים פעילים"
            if key == "core.config":
                return True, f"רמת הרשאות {self.firewall.level}, ערכת נושא {CONFIG.ui.theme}"
            if key == "brain.tokenizer":
                tok = self.core.tokenizer
                return (tok is not None and len(tok) > 266), f"{len(tok):,} טוקנים, {len(tok.merges):,} מיזוגים"
            if key == "brain.neural":
                d = self.core.describe()
                if not d["available"]:
                    return False, d.get("warning", "אין משקולות מאומנות")
                tr = d.get("train") or {}
                ppl = tr.get("ppl")
                return True, (f"{d['params']/1e6:.1f}M פרמטרים, {d['layers']} שכבות, "
                              f"backend={d['backend']}" + (f", perplexity={ppl}" if ppl else ""))
            if key == "brain.knowledge":
                s = self.knowledge.stats()
                return s["facts"] > 0, f"{s['facts']} רשומות, {s['qa_pairs']} זוגות שאלה־תשובה"
            if key == "brain.memory":
                s = self.memory.stats()
                return True, (f"{s['episodes']} זיכרונות, {s['facts']} עובדות, "
                              f"{s['skills']} מיומנויות")
            if key == "brain.intent":
                r = self.router.route("כמה זה 12 כפול 12")
                return r.intent == "MATH" and r.value == 144, f"כוונה {r.intent}, ערך {r.value}"
            if key == "brain.math":
                from brain.math_engine import evaluate
                checks = [evaluate("17*23") == 391, evaluate("sqrt(144)") == 12,
                          evaluate("שורש של 81") == 9, evaluate("10 אחוז מ 200") == 20.0]
                return all(checks), f"{sum(checks)}/{len(checks)} בדיקות מדויקות"
            if key == "skills.registry":
                n = len(REGISTRY.names())
                return n > 20, f"{n} כלים רשומים ({skills_pkg.summary().split('(', 1)[-1].rstrip(')')})"
            if key == "security.firewall":
                d = self.firewall.check("__selftest__", "SAFE", {})
                return d.allowed, f"רמה {self.firewall.level}, יומן {Path(self.firewall.audit_path).name}"
            if key == "voice.g2p":
                from voice.tts.g2p import convert
                ph = convert("שלום אדוני")
                return len(ph) > 4, f"{len(ph)} פונמות לדוגמה"
            if key == "voice.tts":
                st = self.voice.status()
                cov = st.get("coverage", {})
                return True, (f"מנוע {st['engine']}, {st['bank_words']} מילים מוקלטות, "
                              f"{cov.get('phones_covered', 0)}/{cov.get('phones_needed', 29)} פונמות")
            if key == "voice.stt":
                from voice.stt import get_engine
                eng = get_engine()
                st = eng.stats()
                if not st["available"]:
                    return False, "בנק התבניות לא נבנה"
                cal = st.get("calibration", {})
                return st["commands"] >= 20 and cal.get("self_ref", 9) < cal.get("cross_ref", 0), (
                    f"{st['commands']} פקודות, {st['templates']} תבניות, "
                    f"הבחנה {cal.get('self_ref', 0):.2f}/{cal.get('cross_ref', 0):.2f}"
                    + (f", {st['enrolled']} הקלטות משתמש" if st.get("enrolled") else ""))
            if key == "agent.hephaestus":
                res = self.coder.write_and_verify("פונקציה שבודקת אם מספר ראשוני")
                return res.ok, f"קוד נכתב, הורץ ואומת ({res.pattern}, {res.repairs} תיקונים)"
            if key == "sys.telemetry":
                r = REGISTRY.invoke("sys.telemetry")
                return r.ok, str(r.value)[:90]
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return False, "לא מומש"

    # ----------------------------------------------------------------- turns --
    def handle_text(self, text: str, speak: Optional[bool] = None) -> Dict[str, Any]:
        speak = self.speak_out if speak is None else speak
        t0 = time.perf_counter()
        BUS.emit(T.USER_TEXT, {"text": text[:400]}, source=self.name)
        answer: Answer = self.engine.think(text)
        spoke = False
        if speak and answer.speak:
            self._speak(answer)
            spoke = True
        turn = Turn(user=text, answer=answer, spoke=spoke, ms=(time.perf_counter() - t0) * 1000)
        self.turns.append(turn)
        if len(self.turns) > 200:
            self.turns = self.turns[-200:]
        self._persist_turn(turn.to_dict())
        BUS.emit(T.BRAIN_ANSWER, {"text": answer.text[:400], "grounded": answer.grounded,
                                  "intent": answer.intent, "ms": round(turn.ms, 1)}, source=self.name)
        return turn.to_dict()

    def handle_voice_transcript(self, text: str, confidence: float = 1.0,
                                speak: Optional[bool] = None) -> Dict[str, Any]:
        BUS.emit(T.USER_VOICE, {"text": text[:400], "confidence": confidence}, source=self.name)
        return self.handle_text(text, speak=speak)

    def listen(self, wav_bytes: bytes, speak: Optional[bool] = None,
               min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """Hear a WAV captured by the HUD microphone and act on it.

        Recognition is local template matching (see voice/stt). When the verdict
        is not confident we say so instead of guessing a command.
        """
        from voice.stt import recognize_bytes
        t0 = time.perf_counter()
        verdict = recognize_bytes(wav_bytes, min_confidence=min_confidence)
        BUS.emit("stt.verdict", {k: verdict.get(k) for k in ("ok", "label", "text", "confidence")},
                 source=self.name)
        if not verdict.get("ok") or not str(verdict.get("text", "")).strip():
            reply = ("שמעתי משהו אבל לא זיהיתי פקודה בוודאות. "
                     "הפקודות הזמינות: " + ", ".join(self.voice_commands()[:8]) + ".")
            return {"ok": False, "error": verdict.get("error", "no confident match"),
                    "confidence": verdict.get("confidence", 0.0), "label": verdict.get("label", ""),
                    "text": "", "reply": reply, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        turn = self.handle_voice_transcript(verdict["text"],
                                            float(verdict.get("confidence") or 1.0), speak=speak)
        return {"ok": True, "label": verdict.get("label", ""), "text": verdict["text"],
                "confidence": verdict.get("confidence", 0.0), "reply": turn["answer"]["text"],
                "ms": round((time.perf_counter() - t0) * 1000, 1), "turn": turn}

    def voice_commands(self) -> List[str]:
        from voice.stt import COMMANDS
        return [spec["text"] for spec in COMMANDS.values()]

    def _speak(self, answer: Answer) -> None:
        mood = "urgent" if answer.risk == "CRITICAL" else ("thinking" if not answer.grounded else "calm")
        self.voice.mood(mood)
        self.voice.speak(answer.speak)

    def speak_text(self, text: str) -> Dict[str, Any]:
        res = self.voice.speak(text)
        return res.to_dict()

    # -------------------------------------------------------------- controls --
    def kill(self, reason: str = "user") -> Dict[str, Any]:
        self.firewall.kill(reason)
        return {"killed": True, "reason": reason}

    def revive(self) -> Dict[str, Any]:
        self.firewall.revive()
        return {"killed": False}

    # ------------------------------------------------- confirmation bridge --
    def _confirm_bridge(self, request: Dict[str, Any]) -> bool:
        """Called on a worker thread by the firewall. Blocks until the human answers.

        Fail-closed: no answer inside ``confirm_timeout`` ⇒ denied.
        """
        req_id = next(self._confirm_seq)
        payload = dict(request)
        payload["id"] = req_id
        payload["timeout_s"] = self.confirm_timeout
        payload["asked_at"] = time.time()
        event = threading.Event()
        self._confirm_events[req_id] = event
        self._confirm_verdicts[req_id] = False
        self._confirm_requests[req_id] = payload
        self.firewall.pending[req_id] = payload
        BUS.emit("security.permission.request", payload, source=self.name)
        BUS.emit(T.PERMISSION_ASK, payload, source=self.name)

        answered = event.wait(self.confirm_timeout)
        verdict = bool(self._confirm_verdicts.pop(req_id, False))
        self._confirm_events.pop(req_id, None)
        self._confirm_requests.pop(req_id, None)
        self.firewall.pending.pop(req_id, None)
        if not answered:
            BUS.emit("security.permission.timeout", payload, source=self.name)
            return False
        BUS.emit("security.permission.answer" if verdict else T.PERMISSION_DENY,
                 {"id": req_id, "allow": verdict, **payload}, source=self.name)
        return verdict

    def pending_permissions(self) -> List[Dict[str, Any]]:
        return list(self._confirm_requests.values())

    def grant(self, request_id: int, allow: bool) -> Dict[str, Any]:
        """The human answered a CRITICAL prompt in the HUD."""
        allow = bool(allow)
        self._confirm_verdicts[request_id] = allow
        event = self._confirm_events.get(request_id)
        known = event is not None or request_id in self.firewall.pending
        if event is not None:
            event.set()
        self.firewall.pending.pop(request_id, None)
        return {"id": request_id, "allow": allow, "known": known,
                "pending": list(self._confirm_requests.keys())}

    def set_level(self, level: str) -> Dict[str, Any]:
        self.firewall.set_level(level)
        return {"level": self.firewall.level}

    def set_dry_run(self, on: bool) -> Dict[str, Any]:
        self.firewall.set_dry_run(on)
        return {"dry_run": self.firewall.dry_run}

    def set_theme(self, theme: str) -> Dict[str, Any]:
        CONFIG.ui.theme = theme
        BUS.emit(T.UI_STATE, {"theme": theme}, source=self.name)
        return {"theme": theme}

    # ---------------------------------------------------------------- status --
    def _stt_stats(self) -> Dict[str, Any]:
        try:
            from voice.stt import SttEngine
            if getattr(self, "_stt", None) is None:
                self._stt = SttEngine()
                self._stt.load()          # never auto-builds inside status()
            st = self._stt.stats()
            st["commands_text"] = self.voice_commands() if st.get("available") else []
            return st
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def status(self) -> Dict[str, Any]:
        tel = REGISTRY.invoke("sys.telemetry") if REGISTRY.get("sys.telemetry") else None
        return {
            "name": CONFIG.persona["name"],
            "full_name": CONFIG.persona["full_name"],
            "uptime_s": round(time.perf_counter() - self.t0, 1),
            "brain": self.core.describe() if self.core else {"available": False},
            "knowledge": self.knowledge.stats() if self.knowledge else {},
            "memory": self.memory.stats() if self.memory else {},
            "voice": self.voice.status() if self.voice else {},
            "stt": self._stt_stats(),
            "skills": len(REGISTRY.names()),
            "security": self.firewall.stats(),
            "pending_permissions": self.pending_permissions(),
            "confirm_timeout": self.confirm_timeout,
            "turns": len(self.turns),
            "events": len(self.bus_history),
            "telemetry": (tel.data if tel and tel.ok else {}),
            "theme": CONFIG.ui.theme,
        }

    def history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Persisted turns from earlier runs, then this run's live turns."""
        merged = self._log + [t.to_dict() for t in self.turns]
        return merged[-n:]

    def _load_transcript(self, limit: int = 200) -> None:
        try:
            if not self.transcript_path.exists():
                return
            rows = [json.loads(line) for line in
                    self.transcript_path.read_text(encoding="utf-8").splitlines()[-limit:]
                    if line.strip()]
            self._log = [r for r in rows if isinstance(r, dict) and r.get("user") is not None]
        except Exception:
            self._log = []

    def _persist_turn(self, turn: Dict[str, Any]) -> None:
        try:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with self.transcript_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(turn, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass                      # a full disk must never break a conversation

    def clear_transcript(self) -> Dict[str, Any]:
        """Forget the visible conversation. The memory palace is left alone —
        this clears the chat log, not what JARVIS learned about you."""
        n = len(self._log) + len(self.turns)
        self._log = []
        self.turns = []
        try:
            self.transcript_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            return {"ok": False, "cleared": n, "error": str(exc)}
        BUS.emit("chat.cleared", {"turns": n}, source=self.name)
        return {"ok": True, "cleared": n}

    def events(self, n: int = 120) -> List[Dict[str, Any]]:
        return self.bus_history[-n:]
