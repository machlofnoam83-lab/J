"""Hands-free conversation: streaming microphone audio -> turns -> spoken answers.

The HUD streams raw PCM here and this module does the turn-taking a butler would:
wait for the wake word, listen until the sentence actually ends, answer out loud,
then keep listening — no button held, no click per sentence.

Everything is local, like the rest of JARVIS:
    endpointing   voice.dsp.VAD        (energy + spectral flux + ZCR, adaptive floor)
    wake word     voice.stt.WakeListener (MFCC/DTW against our own templates)
    recognition   voice.stt.SttEngine
    the answer    brain.reasoning via agents.jarvis
    the voice     voice.tts concat engine

Two things make this behave in a real room rather than in a demo:

* **We do not listen to ourselves.** While JARVIS speaks, the microphone picks up
  the loudspeaker; without a guard the system recognises its own answer as the
  next command and talks to itself forever. Every spoken reply therefore opens a
  mute window exactly as long as the audio we just produced (plus a tail), and
  audio arriving inside that window is metered but never buffered.

* **Silence has a meaning.** After ``idle_timeout`` seconds with nothing said the
  session disarms itself, so an open microphone does not stay open forever.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.bus import BUS
from voice.dsp import VAD, resample, to_mono
from voice.stt import WakeListener, get_engine

SR = 16000
HOP_MS = 10.0                       # the VAD's frame hop — masks are 10ms per cell

IDLE, LISTENING, THINKING, SPEAKING = "idle", "listening", "thinking", "speaking"

# labels that end a conversation rather than continue it
_END_LABELS = frozenset({"stop", "silence"})


def pcm_to_float(data: Any, sr: int = SR) -> np.ndarray:
    """Accept the shapes a browser can send: Int16 LE bytes, or a float array."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(bytes(data), dtype="<i2")
        return (raw.astype(np.float32) / 32768.0)
    x = np.asarray(data, dtype=np.float32)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    # anything already in Int16 range is scaled down; floats are left alone
    if x.size and float(np.abs(x).max()) > 2.0:
        x = x / 32768.0
    return to_mono(x)


class VoiceSession:
    """One listener for one connected HUD."""

    def __init__(self, agent: Any, emit: Callable[[Dict[str, Any]], Any], sr: int = SR,
                 ack: str = "כן, אדוני?", idle_timeout: float = 25.0,
                 min_speech_ms: float = 220.0, end_silence_ms: float = 560.0,
                 max_utterance_s: float = 8.0, unheard_cues: int = 2,
                 wake_threshold: float = 0.55, level_hz: float = 8.0) -> None:
        self.agent = agent
        self.emit = emit
        self.sr = int(sr)
        self.ack = ack
        self.idle_timeout = float(idle_timeout)
        self.max_buf = int((max_utterance_s + 1.0) * self.sr)
        self.min_frames = max(1, int(min_speech_ms / HOP_MS))
        self.end_frames = max(1, int(end_silence_ms / HOP_MS))
        self.unheard_cues = int(unheard_cues)
        self.level_every = 1.0 / max(1.0, level_hz)

        self.engine = get_engine()
        self.wake = WakeListener(engine=self.engine, threshold=wake_threshold)
        self.vad = VAD(self.sr)

        self._lock = threading.RLock()
        self._buf = np.zeros(0, dtype=np.float32)
        self._state = IDLE
        self._speech_on = False
        self._start_frame = 0
        self._armed_at = 0.0
        self._last_voice = time.time()
        self._mute_until = 0.0
        self._last_level = 0.0
        self._cues_spent = 0
        self._turns = 0
        self._stopped = False

    # ------------------------------------------------------------- public ---
    @property
    def state(self) -> str:
        return self._state

    def feed(self, chunk: Any, sr: Optional[int] = None) -> None:
        """Absorb one microphone chunk. Blocking: the server runs it in a worker.

        Emits HUD events (level, state, heard, answer) through ``self.emit``.
        """
        if self._stopped:
            return
        x = pcm_to_float(chunk, self.sr)
        if x.size == 0:
            return
        if sr and int(sr) != self.sr:
            x = resample(x, int(sr), self.sr)

        with self._lock:
            self._meter(x)
            now = time.time()

            # our own voice coming back through the microphone is metered but
            # never buffered, recognised or allowed to re-trigger the wake word
            if now < self._mute_until:
                return

            if self._state == IDLE:
                self._watch_for_wake(x)
                return

            if self._state in (THINKING, SPEAKING):
                return

            self._accumulate(x)
            self._maybe_timeout(now)

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            was = self._state
            self._state = IDLE
            self._buf = np.zeros(0, dtype=np.float32)
        if was != IDLE:
            self._emit_state(IDLE, reason="session stopped")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"state": self._state, "armed": self._state != IDLE,
                    "turns": self._turns, "buffer_s": round(len(self._buf) / self.sr, 2),
                    "idle_timeout": self.idle_timeout, "muted": time.time() < self._mute_until}

    # -------------------------------------------------------------- internals
    def _emit(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("ts", time.time())
        try:
            self.emit(payload)
        except Exception:
            pass

    def _emit_state(self, state: str, **extra: Any) -> None:
        self._state = state
        self._emit({"type": "voice.state", "state": state, "armed": state != IDLE,
                    "turns": self._turns, **extra})
        BUS.emit("voice.session.state", {"state": state, **extra}, source="voice_session")

    def _meter(self, x: np.ndarray) -> None:
        now = time.time()
        if now - self._last_level < self.level_every:
            return
        self._last_level = now
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
        db = 20.0 * np.log10(max(rms, 1e-7))
        self._emit({"type": "voice.level", "rms": round(rms, 5), "db": round(float(db), 1),
                    "state": self._state})

    def _watch_for_wake(self, x: np.ndarray) -> None:
        hit = self.wake.push(x, self.sr)
        if not hit:
            return
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            self._speech_on = False
            self._armed_at = time.time()
            self._last_voice = self._armed_at
            self._cues_spent = 0
        self._emit_state(LISTENING, reason="wake word",
                         confidence=round(float(hit.get("confidence", 0.0)), 3))
        if self.ack:
            self._say(self.ack)

    def _accumulate(self, x: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, x])
        if len(self._buf) > self.max_buf:
            self._buf = self._buf[-self.max_buf:]
        self._endpoint()

    def _endpoint(self) -> None:
        """Cut the utterance when the speaker has clearly stopped talking."""
        if len(self._buf) < int(0.25 * self.sr):
            return          # not enough audio to judge speech yet
        mask = self.vad.mask(self._buf)
        if mask.size == 0:
            return
        hop = int(self.sr * HOP_MS / 1000.0)

        # "last voice" means the last moment the VAD actually heard someone talk,
        # not the last chunk that arrived — a live microphone never stops sending,
        # so keying the idle timeout on arrivals would make it unreachable
        if mask[-min(len(mask), 30):].any() or self._speech_on:
            self._last_voice = time.time()

        if not self._speech_on:
            start, run = -1, 0
            for i, active in enumerate(mask):
                if active:
                    if run == 0:
                        start = i
                    run += 1
                    if run >= self.min_frames:
                        self._speech_on = True
                        self._start_frame = start
                        break
                else:
                    run = 0
            return

        # how much silence trails the end of the buffer?
        tail = 0
        for active in mask[::-1]:
            if active:
                break
            tail += 1
        overrun = len(self._buf) >= self.max_buf
        if tail < self.end_frames and not overrun:
            return

        end_frame = max(self._start_frame + self.min_frames, len(mask) - tail)
        a = self._start_frame * hop
        b = min(len(self._buf), end_frame * hop)
        utterance = self._buf[a:b]
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech_on = False
        if len(utterance) >= int(0.2 * self.sr):
            self._handle_utterance(utterance)

    def _maybe_timeout(self, now: float) -> None:
        if self._state == LISTENING and now - self._last_voice > self.idle_timeout:
            self._emit_state(IDLE, reason="idle timeout",
                             seconds=round(now - self._armed_at, 1), turns=self._turns)

    def _handle_utterance(self, utt: np.ndarray) -> None:
        self._emit_state(THINKING, reason="endpoint reached")
        t0 = time.perf_counter()
        try:
            res = self.engine.recognize(utt, self.sr)
        except Exception as exc:
            res = None
            self._emit({"type": "voice.unheard", "error": f"{type(exc).__name__}: {exc}"})
        ms = round((time.perf_counter() - t0) * 1000, 1)

        ok = bool(res is not None and getattr(res, "ok", False))
        text = str(getattr(res, "text", "") or "").strip() if res is not None else ""
        label = str(getattr(res, "label", "") or "") if res is not None else ""
        conf = float(getattr(res, "confidence", 0.0) or 0.0) if res is not None else 0.0

        if not ok or not text:
            self._emit({"type": "voice.unheard", "confidence": round(conf, 3),
                        "label": label, "ms": ms, "seconds": round(len(utt) / self.sr, 2)})
            self._emit_state(LISTENING, reason="nothing recognised")
            if self._cues_spent < self.unheard_cues:
                self._cues_spent += 1
                self._say("לא זיהיתי פקודה, אדוני. נסה שוב.")
            return

        self._emit({"type": "voice.heard", "text": text, "label": label,
                    "confidence": round(conf, 3), "ms": ms,
                    "seconds": round(len(utt) / self.sr, 2)})

        if label in _END_LABELS:
            self._say("כמובן, אדוני." if label == "stop" else "שקט מוחלט.")
            self._emit_state(IDLE, reason=f"voice command '{label}'", turns=self._turns)
            return

        self._emit_state(SPEAKING, reason="answering")
        try:
            turn = self.agent.handle_voice_transcript(text, conf, speak=False)
        except Exception as exc:
            self._emit({"type": "error", "message": f"voice turn failed: {exc}"})
            self._emit_state(LISTENING, reason="turn failed")
            return
        self._turns += 1
        self._emit({"type": "answer", "voice": True, **turn})
        answer = (turn or {}).get("answer") or {}
        self._say(str(answer.get("speak") or answer.get("text") or ""))
        self._emit_state(LISTENING, reason="answer delivered", turns=self._turns)

    def _say(self, text: str) -> None:
        """Speak through the agent so the HUD receives the audio, and mute the
        microphone for exactly as long as it takes to say it."""
        text = (text or "").strip()
        if not text:
            self._mute_until = max(self._mute_until, time.time() + 0.3)
            return
        seconds = 0.0
        try:
            res = self.agent.speak_text(text) or {}
            seconds = float(res.get("seconds") or 0.0)
        except Exception as exc:
            self._emit({"type": "voice.unheard", "error": f"tts failed: {exc}"})
        if seconds <= 0.0:
            seconds = 0.35 + 0.055 * len(text)          # conservative estimate
        self._mute_until = time.time() + seconds + 0.35
        self._last_voice = time.time()


# ---------------------------------------------------------------- registry ---
_SESSIONS: Dict[str, VoiceSession] = {}
_SESSIONS_LOCK = threading.Lock()


def start_session(sid: str, agent: Any, emit: Callable[[Dict[str, Any]], Any],
                  **kwargs: Any) -> VoiceSession:
    """Create (or replace) the hands-free session for one client id."""
    with _SESSIONS_LOCK:
        old = _SESSIONS.get(sid)
        if old is not None:
            old.stop()
        sess = VoiceSession(agent, emit, **kwargs)
        _SESSIONS[sid] = sess
    sess._emit_state(IDLE, reason="microphone open — say the wake word")
    return sess


def get_session(sid: str) -> Optional[VoiceSession]:
    with _SESSIONS_LOCK:
        return _SESSIONS.get(sid)


def stop_session(sid: str) -> bool:
    with _SESSIONS_LOCK:
        sess = _SESSIONS.pop(sid, None)
    if sess is None:
        return False
    sess.stop()
    return True


def active_sessions() -> List[str]:
    with _SESSIONS_LOCK:
        return list(_SESSIONS)
