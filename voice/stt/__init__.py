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
    x = to_mono(np.asarray(x, dtype=np.float32))
    if len(x) < int(sr * 0.05):
        return x
    try:
        segs = VAD(sr).segments(x, min_ms=90.0, pad_ms=pad_ms)
    except Exception:
        segs = []
    if not segs:
        # fall back to a plain energy gate
        win = max(1, int(sr * 0.02))
        n = len(x) // win
        if n == 0:
            return x
        e = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1))
        thr = max(float(e.max()) * 0.08, 1e-4)
        idx = np.flatnonzero(e > thr)
        if len(idx) == 0:
            return x[:0]                      # nothing voiced: hand back silence-free empty
        return x[idx[0] * win: (idx[-1] + 1) * win]
    return x[segs[0][0]:segs[-1][1]]


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

    def __init__(self, bank_dir: Path | str = BANK_DIR, sr: int = SR,
                 min_confidence: float = 0.30, band: int = 14,
                 coarse_step: int = 3, coarse_band: int = 6,
                 refine_labels: int = 5, refine_per_label: int = 2) -> None:
        self.bank_dir = Path(bank_dir)
        self.sr = int(sr)
        self.min_confidence = float(min_confidence)
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
            self.texts = {c["label"]: c["text"] for c in manifest.get("commands", [])}
            self.calib = manifest.get("calibration", {})
            self.enrolled = manifest.get("enrolled", [])
            self.labels = sorted(set(self._labels_arr.tolist()))
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

        * absolute — where the distance sits between a known-good match
          (``self_ref``) and a known-wrong one (``cross_ref``);
        * margin — how much better the winner is than the runner-up. A tight race
          between two commands is not a confident recognition, even if both
          distances look small.
        """
        lo = float(self.calib.get("self_ref", 1.6))
        hi = float(self.calib.get("cross_ref", lo + 4.0))
        if hi <= lo or not np.isfinite(dist):
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
def calibrate(bank_dir: Path | str = BANK_DIR, self_target: int = 60,
              cross_target: int = 300, seed: int = 7) -> Dict[str, float]:
    """Measure how far apart templates are, so confidence is not a magic number.

    Two populations are sampled from the bank itself:
      * ``self``  — the *same spoken phrase* synthesised differently (rate/engine).
                     This is the distance a correct match typically produces.
      * ``cross`` — two *different phrases*. This is what a wrong match costs.

    A new utterance is scored by where its distance falls between the two.
    """
    bank_dir = Path(bank_dir)
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


def build_bank(bank_dir: Path | str = BANK_DIR, rates: Sequence[float] = (0.94, 1.06),
               engines: Sequence[str] = ("concat",), verbose: bool = False) -> Dict[str, Any]:
    """Synthesise every command with our own TTS and store its MFCC trajectory."""
    from voice.tts import get_voice

    bank_dir = Path(bank_dir)
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
                    f = features(res.samples, res.sample_rate)
                    if f.shape[0] < 6:
                        continue
                    feats.append(f)
                    labels.append(label)
                    phrases.append(f"{phrase}|{engine}|{rate}")
                    offs.append(offs[-1] + f.shape[0])
                    per_command[label] = per_command.get(label, 0) + 1
                    if verbose:
                        print(f"  [+] {label:<12} {engine:<7} rate={rate} frames={f.shape[0]}")

    if not feats:
        raise RuntimeError("could not synthesise any command template")

    all_feat = np.concatenate(feats, axis=0)
    np.savez_compressed(bank_dir / "templates.npz",
                        feat=all_feat, offs=np.asarray(offs, dtype=np.int64),
                        labels=np.asarray(labels, dtype="U32"),
                        phrases=np.asarray(phrases, dtype="U64"))

    calib = calibrate(bank_dir)

    manifest = {
        "built_at": time.time(),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "sample_rate": SR,
        "n_mfcc": N_MFCC,
        "rates": list(rates),
        "engines": list(engines),
        "templates": len(labels),
        "commands": [{"label": k, "text": v["text"], "say": v.get("say", []),
                      "wake": bool(v.get("wake")), "templates": per_command.get(k, 0)}
                     for k, v in COMMANDS.items()],
        "calibration": calib,
        "enrolled": [],
    }
    (bank_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    BUS.emit("stt.bank.built", {"templates": len(labels), "commands": len(per_command),
                                "seconds": manifest["build_seconds"], "calibration": calib},
             source="stt")
    return manifest


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
        if score >= self.threshold:
            self._last_fire = now
            BUS.emit(T.WAKE, {"confidence": round(score, 3)}, source="stt")
            return {"confidence": score, "audio": tail, "sample_rate": self.sr, "ts": now}
        return None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._since = 0.0
        self._last_fire = 0.0


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
