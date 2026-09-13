"""JARVIS speech recognition — built from zero, nothing downloaded.

    WAV bytes → VAD trim → MFCC (+Δ) → banded DTW against a template bank
              → label, text, confidence

Where the templates come from
  JARVIS enrols *itself*: every command phrase is synthesised with our own TTS
  (concat engine from the recorded voicebank, plus the formant fallback), at two
  speaking rates, and stored as MFCC trajectories. The bank therefore exists on a
  fresh machine with no data download. A human can add their own voice on top of
  it (``enroll_wav`` / ``tools/enroll_voice.py``), which raises accuracy a lot —
  template matching is speaker-sensitive by nature.

Honest scope
  This is a **fixed-vocabulary command recogniser with a wake word**, not
  open-vocabulary dictation. Open dictation needs a large acoustic model plus a
  language model; neither can be built from nothing on this machine, and we do
  not pretend otherwise. What this does do — reliably, offline, in real time —
  is hear "JARVIS", then hear one of ~25 commands, and it says so when it is
  not sure instead of guessing.

Self-calibration
  While building the bank we measure two distances: how far a phrase is from its
  own variants (``self_dist``) and how far it is from *other* phrases
  (``cross_dist``). Confidence is where a new utterance lands between those two
  numbers, so the threshold adapts to the bank instead of being a magic constant.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.bus import BUS, T  # noqa: E402
from voice.dsp import VAD, dtw_distance, mfcc, normalize, resample, to_mono  # noqa: E402

SR = 16000              # recognition sample rate (telephone-grade is enough for MFCC)
N_MFCC = 13
HOP_MS = 20.0           # 20ms frames: half the DTW cost of 10ms, no loss for commands
MAX_FRAMES = 220
BANK_DIR = ROOT / "voice" / "stt" / "bank"

# Bump this whenever anything changes that would make an existing bank disagree
# with the voice that has to match it — the DSP, the voicebank, the augmentation
# grid, or the confidence ramp. The bank is gitignored and rebuilt on first use,
# but `ensure_bank` only builds when one is *missing*, so a listener upgrading in
# place would otherwise keep templates synthesised by the old voice and recognise
# nothing.
#   2 — fixed `time_stretch` and `pitch_shift`, which changed every waveform the
#       bank is made of, and moved the confidence ramp to the means.
#   3 — the voicebank was re-aligned. The old builder zipped tokens to VAD
#       segments positionally, so `word_אדוני.wav` held 104 ms and `word_את.wav`
#       held 1.44 s; templates built from that voice do not match the corrected
#       one, and the two must not be mixed.
#   4 — `trim_silence` now cuts relative to the utterance's own speech level,
#       read from the frames that carry energy so a sparse recording is not
#       mistaken for silence. Every template and every query passes through it,
#       so both feature trajectories changed shape.
BANK_FORMAT = 4

# Used only when a bank carries no measurement of its own. Every bank built here
# measures the gap between correct and incorrect recognitions and stores the
# midpoint, so this constant is a floor for an uncalibrated bank, not a policy.
DEFAULT_MIN_CONFIDENCE = 0.30

# Phrases nobody enrolled, spoken in JARVIS's own voice. These are the negatives
# the acceptance threshold is measured against: if one of them is accepted, the
# threshold is too low and JARVIS acts on something nobody said.
#
# Deliberately conversational rather than random. "מה נשמע אצלך" scored 0.512
# against a threshold of 0.469 and was taken as `memory` ("מה זכרת") — a near-miss
# on ordinary Hebrew speech, which is exactly the population that has to stay
# rejected. Bare "שלום" is here too: `hello` is enrolled as "שלום ג'רוויס", so a
# greeting on its own must not fire a command.
#
# Every phrase here was checked against the vocabulary before being included.
# "תודה רבה לך" was removed because `thanks` is enrolled as "תודה"/"תודה רבה" and
# matched it at 1.000 — that is a correct recognition, and leaving it in pinned
# the threshold to its 0.90 ceiling and rejected real commands.
NEGATIVE_PHRASES = ("בלה בלה בלה בלה", "לא שמעתי כלום", "קפה ותה בבקשה",
                    "היום יש גשם בחוץ", "אני צריך לקנות חלב",
                    "מה נשמע אצלך", "אחד שניים שלוש", "שלום",
                    "אין לי מושג מה זה", "הכל בסדר גמור", "למה זה קורה לי")


def default_bank_dir() -> Path:
    """Where the template bank lives, overridable with ``JARVIS_STT_BANK``.

    Enrolled voiceprints are personal data: they should not have to sit inside
    the source tree, and a test run must never write into the shipped bank.
    """
    env = (os.environ.get("JARVIS_STT_BANK") or "").strip()
    return Path(env).expanduser() if env else BANK_DIR
MANIFEST = BANK_DIR / "manifest.json"
FEATURES = BANK_DIR / "templates.npz"


# ══════════════════════════════ command vocabulary ══════════════════════════
# label -> what JARVIS should *hear*, and the text it hands to the reasoning
# engine (which already understands these phrasings).
COMMANDS: Dict[str, Dict[str, Any]] = {
    "wake":        {"text": "ג'רוויס", "say": ["ג'רוויס", "jarvis"], "wake": True},
    "time":        {"text": "מה השעה", "say": ["מה השעה", "מה השעה עכשיו"]},
    "status":      {"text": "מה מצב המחשב", "say": ["מה מצב המחשב", "מצב מערכת"]},
    "security":    {"text": "מה מצב האבטחה", "say": ["מה מצב האבטחה", "הרשאות"]},
    "memory":      {"text": "מה זכרת", "say": ["מה זכרת", "מה אתה זוכר"]},
    "files":       {"text": "סקור קבצים", "say": ["סקור קבצים", "מה יש בתיקייה"]},
    "code":        {"text": "כתוב קוד", "say": ["כתוב קוד", "כתוב פונקציה"]},
    "explain":     {"text": "הסבר את הקוד", "say": ["הסבר את הקוד", "תסביר את הקוד"]},
    "joke":        {"text": "ספר לי בדיחה", "say": ["ספר לי בדיחה", "תצחיק אותי"]},
    "help":        {"text": "מה אתה יודע לעשות", "say": ["מה אתה יודע לעשות", "עזרה"]},
    "thanks":      {"text": "תודה", "say": ["תודה", "תודה רבה"]},
    "hello":       {"text": "שלום", "say": ["שלום ג'רוויס", "בוקר טוב"]},
    "stop":        {"text": "עצור", "say": ["עצור", "תפסיק"]},
    "silence":     {"text": "השתק", "say": ["השתק", "שקט"]},
    "speak":       {"text": "דבר", "say": ["דבר", "תגיד את זה שוב"]},
    "screenshot":  {"text": "צלם מסך", "say": ["צלם מסך", "צילום מסך"]},
    "browser":     {"text": "פתח את הדפדפן", "say": ["פתח את הדפדפן", "פתח דפדפן"]},
    "kill":        {"text": "מתג חירום", "say": ["מתג חירום", "עצור הכול"]},
    "revive":      {"text": "הפעל מחדש", "say": ["הפעל מחדש", "חזור לפעולה"]},
    "play":        {"text": "נגן", "say": ["נגן", "נגן מוזיקה"]},
    "pause":       {"text": "השהה", "say": ["השהה", "עצור נגינה"]},
    "volume_up":   {"text": "הגבר עוצמה", "say": ["הגבר עוצמה", "תגביר"]},
    "volume_down": {"text": "הנמך עוצמה", "say": ["הנמך עוצמה", "תנמיך"]},
    "yes":         {"text": "כן", "say": ["כן", "כן בבקשה"]},
    "no":          {"text": "לא", "say": ["לא", "לא תודה"]},
}


@dataclass
class SttResult:
    ok: bool
    label: str = ""
    text: str = ""
    confidence: float = 0.0
    distance: float = float("inf")
    runner_up: str = ""
    runner_up_distance: float = float("inf")
    seconds: float = 0.0
    ms: float = 0.0
    wake: bool = False
    error: str = ""
    top: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "label": self.label, "text": self.text,
                "confidence": round(self.confidence, 4), "distance": round(self.distance, 4),
                "runner_up": self.runner_up, "runner_up_distance": round(self.runner_up_distance, 4),
                "seconds": round(self.seconds, 3), "ms": round(self.ms, 1), "wake": self.wake,
                "error": self.error, "top": self.top[:5]}


# ══════════════════════════════ WAV input ══════════════════════════════
def read_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode a WAV payload. soundfile if present, otherwise our own RIFF reader."""
    try:
        import soundfile as sf
        import io
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return np.asarray(x, dtype=np.float32), int(sr)
    except Exception:
        pass
    return _riff_decode(data)


def _riff_decode(data: bytes) -> Tuple[np.ndarray, int]:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE payload")
    pos = 12
    fmt = None
    body = b""
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", chunk[:16])
        elif cid == b"data":
            body = chunk
            break
        pos += 8 + size + (size & 1)
    if fmt is None or not body:
        raise ValueError("WAV has no fmt/data chunk")
    audio_fmt, channels, sr, _byterate, _align, bits = fmt
    if audio_fmt == 3 and bits == 32:
        x = np.frombuffer(body, dtype="<f4")
    elif bits == 16:
        x = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 32:
        x = np.frombuffer(body, dtype="<i4").astype(np.float32) / 2147483648.0
    elif bits == 8:
        x = (np.frombuffer(body, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV bit depth {bits}")
    if channels > 1:
        n = (len(x) // channels) * channels
        x = x[:n].reshape(-1, channels).mean(axis=1)
    return x.astype(np.float32), int(sr)


# ══════════════════════════════ features ══════════════════════════════
def hop_seconds() -> float:
    return HOP_MS / 1000.0


def trim_silence(x: np.ndarray, sr: int, pad_ms: float = 60.0) -> np.ndarray:
    """Cut the near-silence off both ends of an utterance.

    Recognition is sensitive to this far more than it looks. The endpointer hands
    over a span that includes the quiet run which triggered it, so the same word
    arrives as 0.478 s of speech when synthesised clean and 0.530 s with ~170 ms
    of trailing near-silence when it comes off a microphone. Those extra frames
    are matched against the template like any other and cost real distance:
    measured on `עצור`, DTW went 2.62 -> 4.62 and confidence 1.00 -> 0.68, which
    is below the acceptance threshold. A command that works on a clean clip and
    fails on a real one is the whole problem enrolment is supposed to solve, so
    the trim has to be measured against the utterance's own level rather than
    against an absolute gate.

    The VAD span is used first, then refined: any 10 ms frame whose RMS is under
    18% of the utterance's own speech level is treated as silence, and only a
    short pad is kept so plosive onsets are not clipped.
    """
    x = to_mono(np.asarray(x, dtype=np.float32))
    if len(x) < int(sr * 0.05):
        return x

    a, b = 0, len(x)
    try:
        segs = VAD(sr).segments(x, min_ms=90.0, pad_ms=pad_ms)
    except Exception:
        segs = []
    if segs:
        a, b = int(segs[0][0]), int(segs[-1][1])

    win = max(1, int(sr * 0.010))
    n = (b - a) // win
    if n < 3:
        return x[a:b] if b > a else x
    seg = x[a:a + n * win]
    e = np.sqrt((seg.reshape(n, win).astype(np.float64) ** 2).mean(axis=1))
    # The speech level has to be taken from frames that actually carry energy.
    # A plain 70th percentile of *all* frames reads as zero whenever the
    # utterance is mostly silence — the formant engine renders sparse audio with
    # exact-zero gaps between units, measured 26 voiced frames out of 141 — and
    # the trim then returned an empty array, so a 1.24 s recording was rejected
    # as "too short" and could not be taught to the bank.
    nz = e[e > 1e-7]
    if nz.size == 0:
        return x[:0]                          # nothing but digital silence
    ref = float(np.percentile(nz, 70))
    if ref <= 1e-7:
        return x[:0]
    thr = max(ref * 0.18, float(e.max()) * 0.02, 1e-5)
    voiced = np.flatnonzero(e > thr)
    if voiced.size == 0:
        return x[:0]
    pad = max(1, int(round(min(pad_ms, 30.0) / 10.0)))      # frames, capped at 30 ms
    lo = max(0, int(voiced[0]) - pad)
    hi = min(n, int(voiced[-1]) + 1 + pad)
    return x[a + lo * win: a + hi * win]


def features(x: np.ndarray, sr: int = SR, hop_ms: float = HOP_MS) -> np.ndarray:
    """MFCC + Δ, trimmed, peak-normalised, cepstral-mean and variance normalised.

    The per-utterance normalisation is what lets a template recorded from one
    voice still match another: absolute cepstral magnitudes are discarded and
    only the *shape* of the trajectory is compared.
    """
    x = to_mono(np.asarray(x, dtype=np.float32))
    if sr != SR:
        x = resample(x, sr, SR)
    x = trim_silence(x, SR)
    # below this peak there is no speech at all — log(0) energies would only
    # produce a meaningless trajectory that could match a template by accident
    if len(x) < int(SR * 0.12) or float(np.abs(x).max() if len(x) else 0.0) < 1e-4:
        return np.zeros((0, N_MFCC * 2), dtype=np.float32)
    x = normalize(x, 0.95)
    if len(x) > int(SR * MAX_FRAMES * hop_ms / 1000):
        x = x[: int(SR * MAX_FRAMES * hop_ms / 1000)]
    f = mfcc(x, SR, n_mfcc=N_MFCC, delta=True, hop_ms=hop_ms)
    if f.ndim != 2 or f.shape[0] < 4:
        return np.zeros((0, N_MFCC * 2), dtype=np.float32)
    f = np.concatenate([f[:, :N_MFCC], f[:, N_MFCC:2 * N_MFCC]], axis=1)
    f = f - f.mean(axis=0, keepdims=True)
    f = f / (f.std(axis=0, keepdims=True) + 1e-6)
    return np.ascontiguousarray(f, dtype=np.float32)


# ══════════════════════════════ engine ══════════════════════════════
class SttEngine:
    """Template-bank command recogniser with a wake word."""

    def __init__(self, bank_dir: Optional[Path | str] = None, sr: int = SR,
                 min_confidence: Optional[float] = None, band: int = 14,
                 coarse_step: int = 3, coarse_band: int = 6,
                 refine_labels: int = 5, refine_per_label: int = 2) -> None:
        self.bank_dir = Path(bank_dir) if bank_dir else default_bank_dir()
        self.sr = int(sr)
        # None means "whatever this bank measured for itself" — see load(). An
        # explicit number always wins, which is how the tests pin a known point.
        self._min_conf_override = None if min_confidence is None else float(min_confidence)
        self.min_confidence = self._min_conf_override if self._min_conf_override is not None \
            else DEFAULT_MIN_CONFIDENCE
        self.band = int(band)
        self.coarse_step = int(coarse_step)
        self.coarse_band = int(coarse_band)
        self.refine_labels = int(refine_labels)
        self.refine_per_label = int(refine_per_label)
        self.labels: List[str] = []
        self.texts: Dict[str, str] = {}
        self._feat: Optional[np.ndarray] = None
        self._offs: Optional[np.ndarray] = None
        self._labels_arr: Optional[np.ndarray] = None
        self.calib: Dict[str, float] = {}
        self.enrolled: List[Dict[str, Any]] = []
        self._loaded = False

    # ------------------------------------------------------------- loading --
    @property
    def available(self) -> bool:
        if not self._loaded:
            self.load()
        return self._feat is not None and len(self.labels) > 0

    def load(self) -> bool:
        self._loaded = True
        if not (self.bank_dir / "templates.npz").exists():
            return False
        try:
            with np.load(self.bank_dir / "templates.npz", allow_pickle=False) as z:
                self._feat = z["feat"]
                self._offs = z["offs"]
                self._labels_arr = z["labels"]
            manifest = json.loads((self.bank_dir / "manifest.json").read_text(encoding="utf-8"))
            if int(manifest.get("format", 0) or 0) != BANK_FORMAT:
                # Built by a different voice. Report it as unavailable so
                # `ensure_bank` rebuilds instead of recognising nothing.
                BUS.emit(T.ERROR, {"where": "stt.load",
                                   "error": f"bank format {manifest.get('format')} != {BANK_FORMAT}, rebuilding"},
                         source="stt")
                self._feat = None
                self.labels = []
                return False
            self.texts = {c["label"]: c["text"] for c in manifest.get("commands", [])}
            self.calib = manifest.get("calibration", {})
            self.enrolled = manifest.get("enrolled", [])
            self.labels = sorted(set(self._labels_arr.tolist()))
            if self._min_conf_override is not None:
                self.min_confidence = self._min_conf_override
            else:
                # Take the point this bank measured for itself. The ramp between
                # self_mean and cross_mean moves whenever the voice or the
                # augmentation moves, so a threshold written down once goes stale:
                # 0.30 was calibrated against a bank whose pitch variants were
                # byte-identical copies, and after `pitch_shift` was fixed it sat
                # *below* the worst false accept (0.337), letting "בלה בלה בלה בלה"
                # through as volume_up.
                try:
                    measured = float(self.calib.get("min_confidence", 0) or 0)
                except (TypeError, ValueError):
                    measured = 0.0
                self.min_confidence = measured if 0.05 <= measured <= 0.95 else DEFAULT_MIN_CONFIDENCE
            return True
        except Exception as exc:
            BUS.emit(T.ERROR, {"where": "stt.load", "error": str(exc)}, source="stt")
            self._feat = None
            return False

    def ensure_bank(self, rebuild: bool = False, verbose: bool = False) -> bool:
        """Build the template bank on first use (JARVIS enrols itself)."""
        if self.available and not rebuild:
            return True
        build_bank(bank_dir=self.bank_dir, verbose=verbose)
        self._loaded = False
        return self.load()

    # ------------------------------------------------------------ templates --
    def _templates(self, label: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
        out = []
        assert self._feat is not None and self._offs is not None and self._labels_arr is not None
        for i in range(len(self._offs) - 1):
            lab = str(self._labels_arr[i])
            if label and lab != label:
                continue
            out.append((lab, self._feat[self._offs[i]:self._offs[i + 1]]))
        return out

    # ----------------------------------------------------------- recognition --
    def recognize(self, x: np.ndarray, sr: int = SR, top_k: int = 3) -> SttResult:
        t0 = time.perf_counter()
        if not self.available:
            return SttResult(ok=False, error="template bank missing — run tools/build_stt_templates.py",
                             ms=(time.perf_counter() - t0) * 1000)
        f = features(x, sr)
        seconds = len(f) * hop_seconds()
        if f.shape[0] < 6:
            return SttResult(ok=False, error="utterance too short or silent", seconds=seconds,
                             ms=(time.perf_counter() - t0) * 1000)

        best = self._match(f)
        if not best:
            return SttResult(ok=False, error="empty template bank", seconds=seconds,
                             ms=(time.perf_counter() - t0) * 1000)

        ranked = sorted(best.items(), key=lambda kv: kv[1])
        label, dist = ranked[0]
        second_label, second_dist = ranked[1] if len(ranked) > 1 else ("", float("inf"))
        conf = self._confidence(dist, second_dist)
        res = SttResult(
            ok=conf >= self.min_confidence,
            label=label if conf >= self.min_confidence else "",
            text=self.texts.get(label, COMMANDS.get(label, {}).get("text", "")) if conf >= self.min_confidence else "",
            confidence=conf, distance=dist,
            runner_up=second_label, runner_up_distance=second_dist,
            seconds=seconds, wake=bool(COMMANDS.get(label, {}).get("wake")) and conf >= self.min_confidence,
            ms=(time.perf_counter() - t0) * 1000,
            top=[{"label": k, "distance": round(v, 4)} for k, v in ranked[:top_k]],
        )
        if not res.ok:
            res.error = f"no confident match (best {label} at confidence {conf:.2f})"
        BUS.emit(T.LISTEN_END, {"label": res.label, "confidence": round(conf, 3),
                                "ok": res.ok, "ms": round(res.ms, 1)}, source="stt")
        return res

    def _match(self, f: np.ndarray) -> Dict[str, float]:
        """Two-stage search: cheap decimated DTW over the whole bank, then a full
        DTW only on the plausible candidates.

        Full DTW is ~5ms per template; a 100-template bank would cost half a
        second per utterance. Decimating by ``coarse_step`` makes the first stage
        ~10x cheaper and reliably keeps the true label in the shortlist, so the
        expensive pass runs on a dozen templates instead of a hundred.
        """
        n_in = f.shape[0]
        coarse_in = f[:: self.coarse_step]
        scored: List[Tuple[float, str, np.ndarray]] = []
        for lab, tpl in self._templates():
            if tpl.shape[0] < 6:
                continue
            if abs(tpl.shape[0] - n_in) > 0.5 * max(tpl.shape[0], n_in):
                continue                      # DTW cannot rescue a gross length mismatch
            d = dtw_distance(coarse_in, tpl[:: self.coarse_step], band=self.coarse_band)
            scored.append((d, lab, tpl))
        if not scored:
            return {}
        scored.sort(key=lambda t: t[0])

        per_label: Dict[str, int] = {}
        shortlist: List[Tuple[str, np.ndarray]] = []
        for _d, lab, tpl in scored:
            if per_label.get(lab, 0) >= self.refine_per_label:
                continue
            if len({l for l, _ in shortlist}) >= self.refine_labels and per_label.get(lab, 0) == 0:
                continue
            per_label[lab] = per_label.get(lab, 0) + 1
            shortlist.append((lab, tpl))

        best: Dict[str, float] = {}
        for lab, tpl in shortlist:
            d = dtw_distance(f, tpl, band=self.band)
            if d < best.get(lab, float("inf")):
                best[lab] = d
        return best

    def _confidence(self, dist: float, second: Optional[float] = None) -> float:
        """Two factors, each in [0,1], averaged:

        * absolute — where the distance sits between the typical cost of a
          correct match (``self_mean``) and the typical cost of a wrong one
          (``cross_mean``);
        * margin — how much better the winner is than the runner-up. A tight race
          between two commands is not a confident recognition, even if both
          distances look small.

        The ramp used to run from ``p90(self)`` to ``p25(cross)``. Those are the
        two tails that face each other, so as soon as the populations overlap at
        all the ramp collapses: measured on a bank whose pitch augmentation
        genuinely varied, they sat 0.168 apart. Confidence then became a cliff —
        a correct match scored 1.0, and anything landing in that 0.168-wide band
        scored 0.5-0.6 and sailed over a 0.3 threshold. 20.8% of known-wrong
        pairs were accepted. Anchoring on the two means keeps the ramp wide
        (2.28 measured) and drops that to 4.2%.
        """
        lo = float(self.calib.get("self_mean") or self.calib.get("self_ref", 1.6))
        hi = float(self.calib.get("cross_mean") or self.calib.get("cross_ref", lo + 4.0))
        if hi - lo < 0.25:
            # A bank too small or too uniform to measure a real spread: widen the
            # ramp around what we have rather than dividing by nearly zero.
            mid = 0.5 * (hi + lo)
            lo, hi = mid - 0.625, mid + 0.625
        if not np.isfinite(dist):
            return 0.0
        absolute = max(0.0, min(1.0, (hi - dist) / (hi - lo)))
        if second is None or not np.isfinite(second):
            return float(absolute)
        margin = (second - dist) / max(second, 1e-6)
        margin = max(0.0, min(1.0, margin / 0.45))       # 45% separation = full marks
        return float(0.55 * absolute + 0.45 * margin)

    def wake_score(self, x: np.ndarray, sr: int = SR) -> float:
        """Confidence that this chunk contains the wake word."""
        if not self.available:
            return 0.0
        f = features(x, sr)
        if f.shape[0] < 6:
            return 0.0
        best = float("inf")
        for _lab, tpl in self._templates("wake"):
            if tpl.shape[0] < 6:
                continue
            best = min(best, dtw_distance(f, tpl, band=self.band))
        return 0.0 if not np.isfinite(best) else self._confidence(best)

    # -------------------------------------------------------------- enroling --
    def enroll_wav(self, label: str, wav_path: Path | str, text: Optional[str] = None,
                   source: str = "user") -> Dict[str, Any]:
        """Add a human recording to the bank (the single biggest accuracy win)."""
        label = str(label)
        if label not in COMMANDS and not text:
            raise ValueError(f"unknown label {label!r} and no text given")
        wav_path = Path(wav_path)
        x, sr = read_wav_bytes(wav_path.read_bytes())
        f = features(x, sr)
        if f.shape[0] < 6:
            return {"ok": False, "label": label, "error": "recording too short"}
        with np.load(self.bank_dir / "templates.npz", allow_pickle=False) as data:
            prev_feat = data["feat"]
            offs = [int(v) for v in data["offs"]]
            labels = [str(v) for v in data["labels"]]
            phrases = ([str(v) for v in data["phrases"]] if "phrases" in data.files
                       else list(labels))
        feat = np.concatenate([prev_feat, f], axis=0)
        offs.append(offs[-1] + int(f.shape[0]))
        labels.append(label)
        phrases.append(f"{wav_path.stem}|{source}|user")
        np.savez_compressed(self.bank_dir / "templates.npz",
                            feat=feat, offs=np.asarray(offs, dtype=np.int64),
                            labels=np.asarray(labels, dtype="U32"),
                            phrases=np.asarray(phrases, dtype="U64"))
        mpath = self.bank_dir / "manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"commands": []}
        entry = {"label": label, "text": text or self.texts.get(label, label),
                 "source": source, "file": wav_path.name, "frames": int(f.shape[0]),
                 "ts": time.time()}
        manifest.setdefault("enrolled", []).append(entry)
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._loaded = False
        self.load()
        return {"ok": True, "label": label, "frames": int(f.shape[0]), "bank": len(labels)}

    # ------------------------------------------------------------------ info --
    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "commands": len(self.labels),
            "templates": int(len(self._offs) - 1) if self._offs is not None else 0,
            "sample_rate": self.sr,
            "min_confidence": self.min_confidence,
            "calibration": self.calib,
            "enrolled": len(self.enrolled),
            "bank_dir": str(self.bank_dir),
        }

    def vocabulary(self) -> List[Dict[str, Any]]:
        return [{"label": k, "text": v.get("text", k), "say": v.get("say", []),
                 "wake": bool(v.get("wake"))} for k, v in COMMANDS.items()]


# ══════════════════════════════ bank builder ══════════════════════════════
def calibrate(bank_dir: Optional[Path | str] = None, self_target: int = 60,
              cross_target: int = 300, seed: int = 7) -> Dict[str, float]:
    """Measure how far apart templates are, so confidence is not a magic number.

    Two populations are sampled from the bank itself:
      * ``self``  — the *same spoken phrase* synthesised differently (rate/engine).
                     This is the distance a correct match typically produces.
      * ``cross`` — two *different phrases*. This is what a wrong match costs.

    A new utterance is scored by where its distance falls between the two.
    """
    bank_dir = Path(bank_dir) if bank_dir else default_bank_dir()
    with np.load(bank_dir / "templates.npz", allow_pickle=False) as z:
        feat = z["feat"]
        offs = [int(v) for v in z["offs"]]
        labels = [str(v) for v in z["labels"]]
        phrases = [str(v) for v in z["phrases"]] if "phrases" in z.files else list(labels)

    def tpl(i: int) -> np.ndarray:
        return feat[offs[i]:offs[i + 1]]

    def phrase_key(i: int) -> str:
        return phrases[i].split("|")[0]

    rng = np.random.default_rng(seed)
    self_d: List[float] = []
    cross_d: List[float] = []
    tries = 0
    n = len(labels)
    while (len(self_d) < self_target or len(cross_d) < cross_target) and tries < 40000:
        tries += 1
        i = int(rng.integers(n))
        j = int(rng.integers(n))
        if i == j:
            continue
        a, b = tpl(i), tpl(j)
        if a.shape[0] < 6 or b.shape[0] < 6:
            continue
        same_phrase = phrase_key(i) == phrase_key(j)
        if same_phrase and len(self_d) >= self_target:
            continue
        if not same_phrase and len(cross_d) >= cross_target:
            continue
        d = dtw_distance(a, b, band=14)
        if not np.isfinite(d):
            continue
        (self_d if same_phrase else cross_d).append(d)

    calib = {
        # a correct match lands near self_ref; a wrong one near cross_ref
        "self_ref": float(np.percentile(self_d, 90)) if self_d else 1.6,
        "self_mean": float(np.mean(self_d)) if self_d else 1.0,
        "cross_ref": float(np.percentile(cross_d, 25)) if cross_d else 5.5,
        "cross_mean": float(np.mean(cross_d)) if cross_d else 6.5,
        "samples": {"self": len(self_d), "cross": len(cross_d)},
    }
    mpath = bank_dir / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
            manifest["calibration"] = calib
            mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    BUS.emit("stt.calibrated", calib, source="stt")
    return calib


def _add_noise(x: np.ndarray, sr: int, snr_db: float, seed: int = 0) -> np.ndarray:
    """Mix white noise in at a target SNR — a cheap stand-in for a real room."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, 1.0, len(x)).astype(np.float32)
    sig = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) or 1e-6
    nrm = float(np.sqrt(np.mean(np.square(n, dtype=np.float64)))) or 1e-6
    return (x + n * (sig / (nrm * (10 ** (snr_db / 20.0))))).astype(np.float32)


def build_bank(bank_dir: Optional[Path | str] = None,
               rates: Sequence[float] = (0.94, 1.06),
               engines: Sequence[str] = ("concat",),
               pitches: Sequence[float] = (-1.5, 0.0, 1.5),
               noises: Sequence[Optional[float]] = (None, 20.0),
               verbose: bool = False) -> Dict[str, Any]:
    """Synthesise every command with our own TTS and store its MFCC trajectory.

    A bank built from one voice recognises one voice. Measured: our concat voice
    scores 1.000 on the wake word while the *same words* in a different timbre
    score 0.000 — so a bank with a single speaker is useless to anyone else.

    Every phrase is therefore synthesised once per (engine, rate) and then varied
    in the waveform domain, which costs milliseconds instead of another TTS call:
    pitch shifts stand in for different speakers' vocal ranges and noise levels
    stand in for a real room.

    The pitch range is +/-1.5 semitones because that is the range the voice
    actually produces. Measured over a real sentence, the declination contour in
    `concat._prosody` spans -1.44 to +0.57 semitones. The range used to be +/-3,
    but that measurement — and the "held-out +7/-6 st scores 1.00" line that used
    to sit here — was taken while `dsp.pitch_shift` was silently a no-op, so
    every "pitch variant" in the bank was a byte-identical copy and the numbers
    described nothing. Once the shift really works, +/-3 moves the formants of a
    one-syllable command far enough to blur `שלום` into `נגן`: it scored 0.394
    and was recognised as `play`. Covering prosody the voice never produces buys
    nothing and costs the short commands.

    Do NOT add the formant engine to the bank: its timbre is so far from the
    recorded voice that within-label distance (10.1) exceeds between-label
    distance (7.1), the calibration inverts and every score collapses to zero.
    Speaker coverage comes from pitch/noise augmentation plus `enroll_wav`,
    which adds the *listener's own* recordings — the only thing that truly works.
    """
    from voice.dsp import pitch_shift
    from voice.tts import get_voice

    bank_dir = Path(bank_dir) if bank_dir else default_bank_dir()
    bank_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    feats: List[np.ndarray] = []
    labels: List[str] = []
    phrases: List[str] = []
    offs: List[int] = [0]
    per_command: Dict[str, int] = {}

    voice = get_voice()
    for label, spec in COMMANDS.items():
        for phrase in spec.get("say", [spec["text"]]):
            for engine in engines:
                for rate in rates:
                    try:
                        res = voice.synthesize(phrase, rate=rate, engine=engine)
                    except Exception as exc:
                        if verbose:
                            print(f"  [skip] {label}/{engine}/{rate}: {exc}")
                        continue
                    base = to_mono(np.asarray(res.samples, dtype=np.float32))
                    sr_in = int(res.sample_rate)
                    for st in pitches:
                        try:
                            x = pitch_shift(base, sr_in, float(st)) if st else base
                        except Exception:
                            x = base
                        for k, snr in enumerate(noises):
                            y = _add_noise(x, sr_in, float(snr), seed=k * 31 + int(abs(st) * 7)) \
                                if snr else x
                            f = features(y, sr_in)
                            if f.shape[0] < 6:
                                continue
                            feats.append(f)
                            labels.append(label)
                            phrases.append(f"{phrase}|{engine}|{rate}|{st:+.0f}st|"
                                           f"{'clean' if not snr else f'snr{int(snr)}'}")
                            offs.append(offs[-1] + f.shape[0])
                            per_command[label] = per_command.get(label, 0) + 1
                    if verbose:
                        print(f"  [+] {label:<12} {engine:<7} rate={rate} "
                              f"variants={len(pitches) * len(noises)}")

    if not feats:
        raise RuntimeError("could not synthesise any command template")

    all_feat = np.concatenate(feats, axis=0)
    np.savez_compressed(bank_dir / "templates.npz",
                        feat=all_feat, offs=np.asarray(offs, dtype=np.int64),
                        labels=np.asarray(labels, dtype="U32"),
                        phrases=np.asarray(phrases, dtype="U64"))

    calib = calibrate(bank_dir)

    manifest = {
        "format": BANK_FORMAT,
        "built_at": time.time(),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "sample_rate": SR,
        "n_mfcc": N_MFCC,
        "rates": list(rates),
        "engines": list(engines),
        "pitches": list(pitches),
        "noises": [n for n in noises],
        "templates": len(labels),
        "commands": [{"label": k, "text": v["text"], "say": v.get("say", []),
                      "wake": bool(v.get("wake")), "templates": per_command.get(k, 0)}
                     for k, v in COMMANDS.items()],
        "calibration": calib,
        "enrolled": [],
    }
    (bank_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    # The manifest has to exist before the threshold can be measured, because the
    # recogniser reads its command texts and calibration from it.
    thresh = derive_threshold(bank_dir)
    if thresh:
        calib.update(thresh)
        manifest["calibration"] = calib
        manifest["build_seconds"] = round(time.perf_counter() - t0, 2)
        (bank_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8")

    BUS.emit("stt.bank.built", {"templates": len(labels), "commands": len(per_command),
                                "seconds": manifest["build_seconds"], "calibration": calib},
             source="stt")
    return manifest


def derive_threshold(bank_dir: Optional[Path | str] = None, n_true: Optional[int] = None,
                     seed: int = 7) -> Dict[str, Any]:
    """Measure where correct and incorrect recognitions actually land, then put
    the acceptance threshold in the gap between them.

    ``calibrate`` measures template-to-template distances, which says how the bank
    is shaped but not what a recogniser does with real speech — the confidence
    also carries a margin term that template distances never see. So this speaks
    the enrolled phrases back through the recogniser, speaks phrases nobody
    enrolled, and records both confidence distributions.

    ``n_true=None`` speaks *every* enrolled phrase, which is the point. A 14-phrase
    sample measured a true minimum of 0.939 and set the threshold at 0.638; the
    full vocabulary contains `דבר`, a one-syllable command that scores 0.600, so
    the threshold derived from the sample rejected a command the bank knows. The
    hardest phrase is the one that decides where the threshold may sit, and a
    sample usually misses it. Costs about thirteen seconds of a ~45 s build.

    Where the two populations overlap there is no threshold that is right, and
    the tie is broken towards silence: JARVIS controls a computer, so missing a
    command is recoverable and acting on one nobody said is not.
    """
    bank_dir = Path(bank_dir) if bank_dir else default_bank_dir()
    if not (bank_dir / "manifest.json").exists():
        return {}
    try:
        manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return {}

    eng = SttEngine(bank_dir=bank_dir)
    if not eng.available:
        return {}

    from voice.tts import get_voice
    voice = get_voice()

    def pcm(text: str) -> np.ndarray:
        res = voice.synthesize(text)
        x = resample(np.asarray(res.samples, dtype=np.float32), res.sample_rate, SR)
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    phrases: List[Tuple[str, str]] = []
    for cmd in manifest.get("commands", []):
        for said in (cmd.get("say") or [cmd.get("text", "")]):
            if said:
                phrases.append((cmd["label"], said))
    if not phrases:
        return {}

    rng = np.random.default_rng(seed)
    if n_true is not None and len(phrases) > int(n_true):
        take = sorted(int(i) for i in rng.choice(len(phrases), size=int(n_true), replace=False))
        phrases = [phrases[i] for i in take]

    true_conf: List[float] = []
    mislabelled = 0
    for label, text in phrases:
        r = eng.recognize(pcm(text), sr=SR)
        if r.label == label:
            true_conf.append(float(r.confidence))
        else:
            mislabelled += 1
            true_conf.append(0.0)
    false_conf = [float(eng.recognize(pcm(t), sr=SR).confidence) for t in NEGATIVE_PHRASES]
    if not true_conf or not false_conf:
        return {}

    t_min = min(true_conf)
    f_max = max(false_conf)
    if t_min > f_max:
        thr = 0.5 * (t_min + f_max)
    else:
        thr = f_max + 0.02                    # overlap: reject the negatives first
    thr = float(min(0.90, max(0.35, thr)))
    out: Dict[str, Any] = {
        "min_confidence": round(thr, 3),
        "true_min": round(t_min, 3),
        "false_max": round(f_max, 3),
        "true_samples": len(true_conf),
        "true_mislabelled": mislabelled,
        "false_samples": len(false_conf),
    }
    BUS.emit("stt.threshold.measured", out, source="stt")
    return out


# ══════════════════════════════ wake-word listener ══════════════════════════
class WakeListener:
    """Always-on listening: a ring buffer that watches for the wake word.

    Feed it microphone chunks (``push``); when the wake word fires it returns the
    buffered audio so the caller can record the command that follows.
    """

    def __init__(self, engine: Optional[SttEngine] = None, window_s: float = 1.8,
                 threshold: float = 0.55, cooldown_s: float = 2.0, sr: int = SR) -> None:
        self.engine = engine or SttEngine()
        self.window = int(window_s * sr)
        self.threshold = float(threshold)
        self.cooldown = float(cooldown_s)
        self.sr = sr
        self._buf = np.zeros(0, dtype=np.float32)
        self._since = 0.0
        self._last_fire = 0.0
        self.armed = True
        # Exposed so a UI can show *why* it did not fire — a listener who says the
        # wake word and gets nothing deserves to see "score 0.12 of 0.55 needed"
        # rather than guess whether the microphone, the room or the bank is wrong.
        self.last_score = 0.0
        self.last_score_ts = 0.0

    def push(self, x: np.ndarray, sr: Optional[int] = None) -> Optional[Dict[str, Any]]:
        x = to_mono(np.asarray(x, dtype=np.float32))
        sr = sr or self.sr
        if sr != self.sr:
            x = resample(x, sr, self.sr)
        self._buf = np.concatenate([self._buf, x])
        if len(self._buf) > self.window * 4:
            self._buf = self._buf[-self.window * 4:]
        now = time.time()
        self._since += len(x) / self.sr
        if not self.armed or self._since < 0.45 or now - self._last_fire < self.cooldown:
            return None
        self._since = 0.0
        tail = self._buf[-self.window:]
        score = self.engine.wake_score(tail, self.sr)
        self.last_score = float(score)
        self.last_score_ts = now
        if score >= self.threshold:
            self._last_fire = now
            BUS.emit(T.WAKE, {"confidence": round(score, 3)}, source="stt")
            return {"confidence": score, "audio": tail, "sample_rate": self.sr, "ts": now}
        return None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._since = 0.0
        self._last_fire = 0.0
        self.last_score = 0.0
        self.last_score_ts = 0.0


# ══════════════════════════════ module API ══════════════════════════════
_ENGINE: Optional[SttEngine] = None
_LISTENER: Optional[WakeListener] = None


def get_engine() -> SttEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SttEngine()
        if not _ENGINE.available:
            _ENGINE.ensure_bank()
    return _ENGINE


def get_listener() -> WakeListener:
    global _LISTENER
    if _LISTENER is None:
        _LISTENER = WakeListener(get_engine())
    return _LISTENER


def recognize_bytes(payload: bytes, min_confidence: Optional[float] = None) -> Dict[str, Any]:
    """Entry point used by the server: WAV bytes in, verdict out."""
    eng = get_engine()
    if min_confidence is not None:
        eng.min_confidence = float(min_confidence)
    try:
        x, sr = read_wav_bytes(payload)
    except Exception as exc:
        return {"ok": False, "error": f"could not read the WAV payload: {exc}", "text": "",
                "label": "", "confidence": 0.0}
    BUS.emit(T.LISTEN_START, {"seconds": round(len(x) / max(1, sr), 2)}, source="stt")
    res = eng.recognize(x, sr)
    out = res.to_dict()
    out["command"] = COMMANDS.get(res.label, {}) if res.label else {}
    return out


def recognize_file(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    return recognize_bytes(p.read_bytes())


def stats() -> Dict[str, Any]:
    return get_engine().stats()


def vocabulary() -> List[Dict[str, Any]]:
    return get_engine().vocabulary()


if __name__ == "__main__":
    print("building the STT template bank (JARVIS enrols itself)…")
    m = build_bank(verbose=True)
    print(f"\nbank: {m['templates']} templates / {len(m['commands'])} commands "
          f"in {m['build_seconds']}s")
    print("calibration:", json.dumps(m["calibration"], ensure_ascii=False))
