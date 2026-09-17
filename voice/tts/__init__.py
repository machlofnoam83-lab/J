"""JARVIS text-to-speech — our own engine, no API, no cloud, no downloads.

Two synthesis layers with an automatic cascade:

  * ``concat``   — real recorded Hebrew units from ``brain/voicebank`` joined
                   with our own prosody model (the natural voice)
  * ``formant``  — source-filter synthesis from first principles (always works,
                   even with an empty voicebank; the "machine" voice)

The public entry point is ``VoiceEngine.speak()`` / ``VoiceEngine.synthesize()``.
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.bus import BUS, T  # noqa: E402
from core.config import CONFIG, VOICEBANK  # noqa: E402
from voice.dsp import normalize, silence  # noqa: E402
from voice.tts import formant  # noqa: E402
from voice.tts.concat import ConcatEngine  # noqa: E402
from voice.tts.formant import SR_DEFAULT, units_from_g2p  # noqa: E402
from voice.tts.g2p import convert  # noqa: E402


@dataclass
class SpeechResult:
    samples: np.ndarray
    sample_rate: int
    text: str
    engine: str
    seconds: float
    ms: float
    coverage: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)
    wav_b64: str = ""

    def to_wav_bytes(self) -> bytes:
        data = np.clip(np.asarray(self.samples, dtype=np.float32), -1.0, 1.0)
        pcm = (data * 32767).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self.sample_rate)
            fh.writeframes(pcm)
        return buf.getvalue()

    def to_dict(self) -> Dict[str, Any]:
        return {"engine": self.engine, "seconds": round(self.seconds, 3),
                "ms": round(self.ms, 1), "coverage": self.coverage,
                "sample_rate": self.sample_rate, "chars": len(self.text),
                "stats": self.stats}


class VoiceEngine:
    """The mouth of JARVIS."""

    def __init__(
        self,
        bank_dir: Path | str = VOICEBANK,
        sr: int = SR_DEFAULT,
        engine: str = "auto",
        rate: float = 1.0,
        pitch: float = 1.0,
        audio_sink: Optional[Callable[[bytes, int], bool]] = None,
    ) -> None:
        self.sr = int(sr)
        self.bank_dir = Path(bank_dir)
        self.engine_pref = engine
        self.rate = float(rate)
        self.pitch = float(pitch)
        self.concat = ConcatEngine(self.bank_dir, sr=self.sr, pitch=pitch, rate=rate)
        self.audio_sink = audio_sink          # usually the UI channel
        self.engine = self._choose_engine()
        self.last_stats: Dict[str, Any] = {}

    def _choose_engine(self) -> str:
        if self.engine_pref in ("concat", "auto") and self.concat.available:
            return "concat"
        return "formant"

    def status(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "preferred": self.engine_pref,
            "concat_available": self.concat.available,
            "bank_dir": str(self.bank_dir),
            "bank_words": len(self.concat.words),
            "bank_phone_units": len(self.concat.phones),
            "coverage": self.concat.coverage,
            "sample_rate": self.sr,
            "rate": self.rate,
            "pitch": self.pitch,
        }

    # ------------------------------------------------------------ synthesis --
    def synthesize(self, text: str, rate: Optional[float] = None,
                   pitch: Optional[float] = None, engine: Optional[str] = None) -> SpeechResult:
        text = (text or "").strip()
        t0 = time.perf_counter()
        if not text:
            return SpeechResult(samples=silence(0.05, self.sr), sample_rate=self.sr, text="",
                                engine="none", seconds=0.05, ms=0.0)

        which = engine or self.engine
        coverage = 0.0
        stats: Dict[str, Any] = {}

        if which == "concat" and self.concat.available:
            samples, cstats = self.concat.synthesize(text, rate=rate, pitch=pitch)
            coverage = cstats.coverage
            stats = {"units": cstats.units, "word_units": cstats.from_word,
                     "phone_ctx": cstats.from_phone_ctx, "phone_any": cstats.from_phone_any,
                     "formant_fallback": cstats.from_formant, "coverage": coverage}
            # if the bank covered almost nothing, prefer the full formant render
            if coverage < 0.25:
                samples = formant.synthesize_units(units_from_g2p(convert(text)),
                                                   sr=self.sr, f0=formant.BASE_F0 * (pitch or self.pitch),
                                                   rate=rate or self.rate)
                which = "formant"
                stats["downgraded"] = "low concat coverage"
        else:
            samples = formant.synthesize_units(units_from_g2p(convert(text)),
                                               sr=self.sr, f0=formant.BASE_F0 * (pitch or self.pitch),
                                               rate=rate or self.rate)
            which = "formant"

        samples = normalize(np.asarray(samples, dtype=np.float32), 0.94)
        ms = (time.perf_counter() - t0) * 1000
        result = SpeechResult(samples=samples, sample_rate=self.sr, text=text, engine=which,
                              seconds=len(samples) / self.sr, ms=ms, coverage=coverage, stats=stats)
        self.last_stats = result.to_dict()
        return result

    def to_b64_wav(self, result: SpeechResult) -> str:
        return base64.b64encode(result.to_wav_bytes()).decode("ascii")

    def save(self, result: SpeechResult, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(result.to_wav_bytes())
        return p

    # -------------------------------------------------------------- output --
    def speak(self, text: str, *, rate: Optional[float] = None, pitch: Optional[float] = None,
              engine: Optional[str] = None, blocking: bool = False,
              stream_to_ui: bool = True) -> SpeechResult:
        """Synthesise and route the audio to every available output."""
        BUS.emit(T.SPEAK_START, {"text": text[:200], "engine": engine or self.engine}, source="voice")
        result = self.synthesize(text, rate=rate, pitch=pitch, engine=engine)

        delivered: List[str] = []
        if stream_to_ui and self.audio_sink is not None:
            try:
                if self.audio_sink(result.to_wav_bytes(), result.sample_rate):
                    delivered.append("ui")
            except Exception as exc:
                BUS.emit(T.ERROR, {"where": "audio_sink", "error": str(exc)}, source="voice")
        if not delivered:
            if self._play_local(result, blocking=blocking):
                delivered.append("local")
        if not delivered:
            path = ROOT / "data" / f"speech_{int(time.time() * 1000)}.wav"
            self.save(result, path)
            delivered.append(f"file:{path}")

        BUS.emit(T.SPEAK_END, {"seconds": round(result.seconds, 2), "ms": round(result.ms, 1),
                               "engine": result.engine, "delivered": delivered,
                               "coverage": result.coverage}, source="voice")
        result.stats["delivered"] = delivered
        return result

    def _play_local(self, result: SpeechResult, blocking: bool = False) -> bool:
        """Try the platform's native players. Returns True if something played."""
        wav_bytes = result.to_wav_bytes()
        # 1) python audio device
        try:
            import sounddevice as sd  # type: ignore
            sd.play(result.samples, result.sample_rate, blocking=blocking)
            return True
        except Exception:
            pass
        # 2) an external player that can read a WAV from stdin
        for cmd in (["aplay", "-q", "-"], ["paplay", "-"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
                    ["afplay", "/dev/stdin"], ["powershell", "-NoProfile", "-Command",
                                               "(New-Object Media.SoundPlayer [Console]::In.ReadToEnd()).PlaySync()"]):
            if not shutil.which(cmd[0]):
                continue
            try:
                kwargs: Dict[str, Any] = {"input": wav_bytes, "timeout": 60}
                if blocking:
                    kwargs["stdout"] = subprocess.DEVNULL
                    kwargs["stderr"] = subprocess.DEVNULL
                    subprocess.run(cmd, **kwargs)
                else:
                    kwargs["stdout"] = subprocess.DEVNULL
                    kwargs["stderr"] = subprocess.DEVNULL
                    kwargs["stdin"] = subprocess.PIPE
                    subprocess.Popen(cmd, **kwargs)
                return True
            except Exception:
                continue
        return False

    # --------------------------------------------------------------- voices --
    def set_prosody(self, rate: Optional[float] = None, pitch: Optional[float] = None) -> None:
        if rate is not None:
            self.rate = float(rate)
            self.concat.rate = self.rate
        if pitch is not None:
            self.pitch = float(pitch)
            self.concat.pitch = self.pitch
        BUS.emit("voice.prosody", {"rate": self.rate, "pitch": self.pitch}, source="voice")

    def mood(self, name: str) -> None:
        """Canned prosody presets — this is the voice's personality layer."""
        presets = {
            "calm":      dict(rate=0.96, pitch=0.98),
            "urgent":    dict(rate=1.18, pitch=1.07),
            "warm":      dict(rate=0.92, pitch=1.02),
            "alert":     dict(rate=1.10, pitch=1.12),
            "thinking":  dict(rate=0.88, pitch=0.96),
            "boot":      dict(rate=0.90, pitch=1.00),
        }
        p = presets.get(name.lower())
        if p:
            self.set_prosody(**p)


VOICE: Optional[VoiceEngine] = None


def get_voice(**kwargs: Any) -> VoiceEngine:
    global VOICE
    if VOICE is None:
        cfg = CONFIG.voice
        VOICE = VoiceEngine(
            bank_dir=kwargs.pop("bank_dir", cfg.voicebank_dir),
            sr=kwargs.pop("sr", cfg.sample_rate),
            engine=kwargs.pop("engine", cfg.engine),
            rate=kwargs.pop("rate", cfg.rate),
            pitch=kwargs.pop("pitch", cfg.pitch),
            **kwargs,
        )
    return VOICE


def speak(text: str, **kwargs: Any) -> SpeechResult:
    return get_voice().speak(text, **kwargs)


__all__ = ["VoiceEngine", "SpeechResult", "get_voice", "speak", "VOICE"]
