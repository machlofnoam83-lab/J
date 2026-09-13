"""JARVIS concatenative voice engine — real recorded units, our own prosody.

Selection cascade (best sounding first):
    1. whole recorded word            (if the word exists in the bank)
    2. recorded phone with matching left/right context
    3. any recorded phone with that symbol
    4. our formant synthesiser fills the gap

Units are joined with equal-power crossfades, pitch-shifted along a declination
contour, and time-stretched for stress — so the result has real human timbre
with controlled rhythm.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voice.dsp import (VAD, crossfade, fade_in_out, normalize, pitch_shift,  # noqa: E402
                       resample, silence, time_stretch, to_mono)
from voice.tts import formant  # noqa: E402
from voice.tts.formant import Unit, units_from_g2p  # noqa: E402
from voice.tts.g2p import Phoneme, convert  # noqa: E402


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    import wave
    with wave.open(str(path), "rb") as fh:
        sr = fh.getframerate()
        raw = fh.readframes(fh.getnframes())
        width = fh.getsampwidth()
    data = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if width == 2 else np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0)
    return data, sr


@dataclass
class ConcatStats:
    units: int = 0
    from_word: int = 0
    from_phone_ctx: int = 0
    from_phone_any: int = 0
    from_formant: int = 0
    coverage: float = 0.0


class ConcatEngine:
    def __init__(self, bank_dir: Path | str, sr: int = 24000, pitch: float = 1.0,
                 rate: float = 1.0) -> None:
        self.dir = Path(bank_dir)
        self.sr = sr
        self.pitch = pitch
        self.rate = rate
        self.index_path = self.dir / "index.json"
        self.words: Dict[str, Dict[str, Any]] = {}
        self.phones: Dict[str, Dict[str, Any]] = {}
        self.by_sym: Dict[str, List[str]] = {}
        self._cache: Dict[str, np.ndarray] = {}
        self.available = False
        self.coverage: Dict[str, Any] = {}
        self.load()

    # ---------------------------------------------------------------- load --
    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            idx = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        self.words = idx.get("words", {})
        self.phones = idx.get("phones", {})
        self.coverage = idx.get("coverage", {})
        self.by_sym = {}
        for tag, meta in self.phones.items():
            self.by_sym.setdefault(meta["sym"], []).append(tag)
        self.available = bool(self.words or self.phones)
        return self.available

    def _unit(self, filename: str) -> Optional[np.ndarray]:
        if filename in self._cache:
            return self._cache[filename]
        path = self.dir / filename
        if not path.exists():
            return None
        try:
            x, sr = _read_wav(path)
        except Exception:
            return None
        if sr != self.sr:
            x = resample(x, sr, self.sr)
        x = normalize(to_mono(x), 0.92)
        # second-line trim: never let a unit carry dead air into the sentence
        segs = VAD(self.sr, hangover=4, threshold_scale=2.6).segments(x, min_ms=40.0, pad_ms=10.0)
        if segs:
            x = x[segs[0][0]: segs[-1][1]]
        self._cache[filename] = x
        return x

    # ------------------------------------------------------------ synth ----
    def synthesize(self, text: str, rate: Optional[float] = None,
                   pitch: Optional[float] = None) -> Tuple[np.ndarray, ConcatStats]:
        rate = self.rate if rate is None else rate
        pitch = self.pitch if pitch is None else pitch
        stats = ConcatStats()
        out = np.zeros(0, dtype=np.float32)

        for word in re.findall(r"[^\s]+", text):
            core = word.strip(".,!?;:\"'()[]{}—–-־")
            tail = word[len(core) + (len(word) - len(word.lstrip(".,!?;:\"'()[]{}—–-־"))):]
            if core and core in self.words:
                meta = self.words[core]
                chunk = self._unit(meta["file"])
                if chunk is not None and len(chunk) > 8:
                    chunk = self._cap_duration(chunk, max_s=0.78, rate=rate)
                    chunk = self._prosody(chunk, 0.0, rate, pitch)
                    out = self._join(out, chunk)
                    stats.from_word += 1
                    stats.units += 1
                    out = self._join(out, silence(self._pause_for(tail), self.sr))
                    continue
            # fall back to phone-by-phone
            phones = [p for p in convert(core) if p.sym and p.sym != "_"]
            if not phones:
                continue
            chunk, s = self._render_phones(phones, rate, pitch)
            out = self._join(out, chunk)
            stats.units += s.units
            stats.from_phone_ctx += s.from_phone_ctx
            stats.from_phone_any += s.from_phone_any
            stats.from_formant += s.from_formant
            out = self._join(out, silence(self._pause_for(tail), self.sr))

        if not out.size:
            out = formant.synthesize_units(units_from_g2p(convert(text), rate=rate), sr=self.sr)
            stats.from_formant += 1
        total = max(1, stats.units)
        stats.coverage = round((stats.from_word + stats.from_phone_ctx + stats.from_phone_any) / total, 3)
        return fade_in_out(normalize(out, 0.94), self.sr, ms=6), stats

    def _render_phones(self, phones: Sequence[Phoneme], rate: float, pitch: float
                       ) -> Tuple[np.ndarray, ConcatStats]:
        stats = ConcatStats()
        out = np.zeros(0, dtype=np.float32)
        used = 0
        for i, ph in enumerate(phones):
            left = phones[i - 1].sym if i > 0 else "#"
            right = phones[i + 1].sym if i + 1 < len(phones) else "#"
            tag = f"{ph.sym}_{left}_{right}"
            chunk: Optional[np.ndarray] = None
            if tag in self.phones:
                chunk = self._unit(self.phones[tag]["file"])
                if chunk is not None:
                    stats.from_phone_ctx += 1
                    used += 1
            if chunk is None:
                for alt in self.by_sym.get(ph.sym, []):
                    chunk = self._unit(self.phones[alt]["file"])
                    if chunk is not None and len(chunk) > 8:
                        stats.from_phone_any += 1
                        used += 1
                        break
            if chunk is None:
                chunk = formant.synthesize_units(units_from_g2p([ph], rate=rate), sr=self.sr)
                stats.from_formant += 1
            stats.units += 1
            chunk = self._prosody(chunk, i / max(1, len(phones)), rate, pitch,
                                  stressed=ph.stressed)
            pad = ph.pause
            if pad > 0.01:
                out = self._join(out, silence(min(pad, 0.6), self.sr))
            out = self._join(out, chunk)
        return out, stats

    # ------------------------------------------------------------- prosody --
    def _cap_duration(self, chunk: np.ndarray, max_s: float = 0.8,
                      rate: float = 1.0) -> np.ndarray:
        """A recording that drags gets gently time-compressed (pitch preserved)."""
        limit = max_s / max(0.5, rate)
        dur = len(chunk) / self.sr
        if dur <= limit or dur <= 0.05:
            return chunk
        factor = max(0.62, limit / dur)
        return time_stretch(chunk, self.sr, factor)

    def _prosody(self, chunk: np.ndarray, pos: float, rate: float, pitch: float,
                 stressed: bool = False) -> np.ndarray:
        semis = 12.0 * math.log2(max(0.25, pitch))
        semis -= 1.6 * pos                      # sentence declination
        if stressed:
            semis += 1.1
        if abs(semis) > 0.05:
            chunk = pitch_shift(chunk, self.sr, semis)
        if abs(rate - 1.0) > 0.01:
            chunk = time_stretch(chunk, self.sr, 1.0 / max(0.5, rate))
        return normalize(chunk, 0.92)

    def _join(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not a.size:
            return b
        if not b.size:
            return a
        n = int(self.sr * 0.008)
        return crossfade(a, b, n)

    @staticmethod
    def _pause_for(tail: str) -> float:
        if not tail:
            return 0.028
        if any(c in tail for c in ".!?…"):
            return 0.30
        if any(c in tail for c in ",;:־"):
            return 0.15
        return 0.06
