"""JARVIS formant synthesiser — speech from first principles.

No downloaded voice, no cloud TTS, no external engine. This file turns a
phoneme sequence into a waveform using the classic source-filter model:

    glottal source (pulse train + aspiration + frication noise)
        -> time-varying vocal-tract filter (cascaded 2nd-order resonators)
        -> radiation + normalisation

Every parameter below (formant tables, bandwidths, durations, jitter, shimmer,
pitch contours) is hand-tuned here. It is deliberately *not* trying to sound
like a human recording — it is tuned to sound like a precise, calm machine
voice, which is exactly what JARVIS should sound like.

Usage:
    from voice.tts.formant import synthesize
    wav = synthesize([("S", False), ("a", True), ("l", False), ...], sr=24000)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import lfilter
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

    def lfilter(b, a, x, zi=None):  # minimal fallback
        b = np.asarray(b, np.float64); a = np.asarray(a, np.float64)
        a = a / a[0]
        y = np.zeros_like(x, dtype=np.float64)
        order = max(len(a), len(b)) - 1
        z = np.zeros(order)
        for n in range(len(x)):
            acc = b[0] * x[n] + z[0]
            y[n] = acc
            for k in range(order):
                zb = b[k + 1] * x[n] if k + 1 < len(b) else 0.0
                za = -a[k + 1] * acc if k + 1 < len(a) else 0.0
                z[k] = (z[k + 1] if k + 1 < order else 0.0) + zb + za
        return y


SR_DEFAULT = 24000

# ------------------------------------------------------------- phoneme data --
# Vowels: F1, F2, F3 (Hz), duration (s), amplitude
VOWELS: Dict[str, Tuple[float, float, float, float, float]] = {
    "a": (730, 1150, 2450, 0.110, 1.00),
    "e": (500, 1850, 2500, 0.100, 0.95),
    "i": (300, 2150, 2900, 0.095, 0.90),
    "o": (450,  850, 2450, 0.105, 0.95),
    "u": (330,  750, 2400, 0.100, 0.90),
}

# Consonant classes
STOPS_V = {"b": (170, 700, 2000), "d": (190, 1750, 2600), "g": (200, 1500, 2500)}
STOPS_UV = {"p": (480, 1000, 2200), "t": (1800, 3200, 4500), "k": (1500, 2000, 3200)}
FRIC: Dict[str, Tuple[float, float, float, float]] = {
    # peak Hz, bandwidth, amplitude, voicing amount
    "f": (5000, 2500, 0.30, 0.0),
    "v": (4200, 2200, 0.34, 0.45),
    "s": (6000, 2200, 0.55, 0.0),
    "z": (5600, 2200, 0.48, 0.40),
    "S": (3000, 1900, 0.50, 0.0),
    "Z": (2700, 1800, 0.46, 0.40),
    "x": (1450, 1200, 0.52, 0.05),   # chet / khaf — pharyngeal-uvular
    "X": (1450, 1200, 0.52, 0.05),
    "h": (0, 0, 0.20, 0.0),          # aspiration shaped by the next vowel
    "T": (5200, 2400, 0.45, 0.0),    # tsadi — affricate-like sibilant
}
NASALS: Dict[str, Tuple[float, float, float]] = {"m": (230, 1050, 2200), "n": (250, 1650, 2600)}
LIQUIDS: Dict[str, Tuple[float, float, float]] = {
    "l": (360, 1150, 2750), "r": (330, 1050, 1600),
    "y": (280, 2100, 2900), "w": (300, 700, 2300),
}
GLOTTAL = {"A", "aA"}
AFFRICATES = {"J": ("d", "Z"), "CH": ("t", "S"), "T": ("t", "s")}

DURATIONS: Dict[str, float] = {
    "stop": 0.075, "fric": 0.115, "nasal": 0.070, "liquid": 0.065,
    "glottal": 0.040, "h": 0.055, "affricate": 0.105,
}

# Baseline male voice
BASE_F0 = 105.0
F0_RANGE = 26.0        # Hz of declination swing
JITTER = 0.006         # relative pitch noise -> liveness
SHIMMER = 0.020        # relative amplitude noise
BREATH = 0.030         # aspiration mixed into voiced sounds


@dataclass
class Unit:
    sym: str
    stressed: bool = False
    pause_before: float = 0.0
    rate: float = 1.0


# --------------------------------------------------------------- resonator --
def resonator_coeffs(f: float, bw: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """2nd-order resonator (Klatt). Returns (b, a)."""
    f = float(np.clip(f, 60.0, sr * 0.45))
    bw = float(np.clip(bw, 40.0, sr * 0.4))
    r = math.exp(-math.pi * bw / sr)
    theta = 2 * math.pi * f / sr
    a = np.array([1.0, -2 * r * math.cos(theta), r * r])
    b = np.array([(1 - r) * (1 - r)])
    return b, a


def frame_varying_filter(x: np.ndarray, f_traj: np.ndarray, bw_traj: np.ndarray,
                        sr: int, hop: int = 32) -> np.ndarray:
    """Apply a slowly time-varying resonator: filter in overlapping blocks whose
    coefficients come from the trajectory centre. Cheap and artefact-free enough
    for formant synthesis."""
    out = np.zeros_like(x)
    n = len(x)
    for start in range(0, n, hop):
        stop = min(n, start + hop)
        mid = (start + stop) // 2
        f = f_traj[min(mid, len(f_traj) - 1)]
        bw = bw_traj[min(mid, len(bw_traj) - 1)]
        b, a = resonator_coeffs(f, bw, sr)
        seg = x[start:stop]
        pad = np.concatenate([x[max(0, start - 64):start], seg])
        y = lfilter(b, a, pad)[-len(seg):]
        out[start:stop] = y
    return out


# ------------------------------------------------------------------ source --
def glottal_source(n: int, f0_traj: np.ndarray, sr: int, seed: int = 7) -> np.ndarray:
    """Rosenberg-style glottal flow pulse train following a pitch contour."""
    rng = np.random.default_rng(seed)
    phase = 0.0
    out = np.zeros(n)
    for i in range(n):
        f0 = float(f0_traj[min(i, len(f0_traj) - 1)])
        if f0 <= 1.0:
            phase = 0.0
            continue
        t = phase
        if t < 0.4:
            out[i] = 0.5 * (1 - math.cos(math.pi * t / 0.4))
        elif t < 0.6:
            out[i] = math.cos(math.pi * (t - 0.4) / 0.4)
        else:
            out[i] = 0.0
        step = f0 / sr
        step *= 1.0 + rng.normal(0, JITTER)
        phase += step
        if phase >= 1.0:
            phase -= 1.0
    return out


def noise(n: int, sr: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float64)


def band_noise(n: int, sr: int, center: float, bw: float, seed: int = 11) -> np.ndarray:
    if center <= 0:
        return noise(n, sr, seed)
    x = noise(n, sr, seed)
    b, a = resonator_coeffs(center, max(120.0, bw), sr)
    y = lfilter(b, a, x)
    peak = np.max(np.abs(y)) or 1.0
    return y / peak


def envelope(n: int, attack: float = 0.15, release: float = 0.30) -> np.ndarray:
    env = np.ones(n)
    na = int(n * attack)
    nr = int(n * release)
    if na:
        env[:na] = np.linspace(0, 1, na)
    if nr:
        env[-nr:] = np.linspace(1, 0, nr)
    return env


def fade(x: np.ndarray, ms: float = 4.0, sr: int = SR_DEFAULT) -> np.ndarray:
    n = max(1, int(sr * ms / 1000))
    if len(x) <= 2 * n:
        return x * np.linspace(0, 1, len(x)) * np.linspace(1, 0, len(x))
    x = x.copy()
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


# -------------------------------------------------------------- trajectory --
def _formant_targets(units: Sequence[Unit]) -> List[Tuple[float, float, float, float, float]]:
    """Return [(F1, F2, F3, dur, amp)] per unit with context-aware coarticulation."""
    out: List[Tuple[float, float, float, float, float]] = []
    for i, u in enumerate(units):
        s = u.sym
        prev = units[i - 1].sym if i else ""
        nxt = units[i + 1].sym if i + 1 < len(units) else ""

        def vowel_ctx(sym: str) -> Tuple[float, float, float]:
            for other in (nxt, prev):
                if other in VOWELS:
                    f = VOWELS[other]
                    return (f[0], f[1], f[2])
            return (500, 1500, 2500)

        if s in VOWELS:
            f1, f2, f3, dur, amp = VOWELS[s]
            dur *= (1.28 if u.stressed else 1.0) / max(0.4, getattr(u, "rate", 1.0))
            out.append((f1, f2, f3, dur, amp))
        elif s in STOPS_V or s in STOPS_UV:
            burst = STOPS_V.get(s) or STOPS_UV.get(s)
            ctx = vowel_ctx(s)
            out.append(((burst[0] + ctx[0]) / 2, (burst[1] + ctx[1]) / 2,
                        (burst[2] + ctx[2]) / 2, DURATIONS["stop"] / max(0.4, getattr(u, "rate", 1.0)), 0.32))
        elif s in NASALS:
            f = NASALS[s]
            ctx = vowel_ctx(s)
            out.append((f[0], (f[1] + ctx[1]) / 2, (f[2] + ctx[2]) / 2,
                        DURATIONS["nasal"] / max(0.4, getattr(u, "rate", 1.0)), 0.45))
        elif s in LIQUIDS:
            f = LIQUIDS[s]
            ctx = vowel_ctx(s)
            out.append((f[0], (f[1] * 2 + ctx[1]) / 3, (f[2] * 2 + ctx[2]) / 3,
                        DURATIONS["liquid"] / max(0.4, getattr(u, "rate", 1.0)), 0.72))
        elif s in FRIC:
            peak, bw, amp, _voiced = FRIC[s]
            ctx = vowel_ctx(s)
            f1 = ctx[0] if peak <= 0 else max(300.0, peak * 0.35)
            out.append((f1, peak if peak > 0 else ctx[1], ctx[2],
                        (DURATIONS["h"] if s == "h" else DURATIONS["fric"]) / max(0.4, getattr(u, "rate", 1.0)), amp))
        elif s in GLOTTAL:
            ctx = vowel_ctx(s)
            out.append((ctx[0], ctx[1], ctx[2], DURATIONS["glottal"] / max(0.4, getattr(u, "rate", 1.0)), 0.10))
        else:
            ctx = vowel_ctx(s)
            out.append((ctx[0], ctx[1], ctx[2], 0.045 / max(0.4, getattr(u, "rate", 1.0)), 0.4))
    return out


# ---------------------------------------------------------------- synthesis --
def synthesize_units(units: Sequence[Unit], sr: int = SR_DEFAULT, f0: float = BASE_F0,
                     rate: float = 1.0, seed: int = 7,
                     collect: Optional[List[Tuple[str, int, int]]] = None) -> np.ndarray:
    """Render a phoneme sequence into a float waveform in [-1, 1].

    If ``collect`` is given it is filled with ``(sym, start_sample, end_sample)``
    for every unit placed, including ``("#", ...)`` entries for the pauses between
    them. This is how the ASR gets training labels: the renderer already knows
    exactly where each phone lands, so the alignment is exact and free rather
    than estimated.

    Deliberately a hook on *this* function instead of a second copy of the
    timing loop elsewhere. Two implementations of the same timeline drift apart,
    and a recogniser trained against a timeline the synthesiser no longer uses
    fails silently — it keeps recognising audio nobody produces.
    """
    units = [u for u in units if u.sym and u.sym != "_"]
    if not units:
        return np.zeros(int(sr * 0.05))

    targets = _formant_targets(units)
    rate = max(0.5, min(2.0, rate))
    total = sum(t[3] for t in targets) / rate + sum(u.pause_before for u in units)
    n_total = max(int(sr * total), int(sr * 0.05))

    # ---- build per-sample formant + f0 trajectories
    f1 = np.zeros(n_total); f2 = np.zeros(n_total); f3 = np.zeros(n_total)
    amp = np.zeros(n_total); voiced = np.zeros(n_total); noise_amt = np.zeros(n_total)
    f0_traj = np.zeros(n_total)

    pos = 0.0
    for u, (t1, t2, t3, dur, a) in zip(units, targets):
        if u.pause_before > 0:
            p0 = int(pos * sr)
            p1 = int(min(n_total, (pos + u.pause_before) * sr))
            if p1 > p0:
                amp[p0:p1] = 0.0
                voiced[p0:p1] = 0.0
                f0_traj[p0:p1] = 0.0
                if collect is not None:
                    collect.append(("_sil_", p0, p1))
            pos += u.pause_before
        start = int(pos * sr)
        end = int(min(n_total, (pos + dur / rate) * sr))
        if end <= start:
            end = min(n_total, start + 1)
        if collect is not None:
            collect.append((u.sym, start, end))
        seg = slice(start, end)
        L = max(1, end - start)

        # blend toward the neighbouring targets for coarticulation
        f1[seg] = t1; f2[seg] = t2; f3[seg] = t3; amp[seg] = a
        s = u.sym
        if s in VOWELS or s in LIQUIDS or s in NASALS or s in STOPS_V or s in GLOTTAL or s in ("v", "z", "Z"):
            voiced[seg] = 1.0
        elif s in ("b", "d", "g"):
            voiced[seg] = 1.0
            noise_amt[seg] = 0.25
        elif s in FRIC:
            v = FRIC[s][3]
            voiced[seg] = v
            noise_amt[seg] = 1.0
        elif s in STOPS_UV:
            voiced[seg] = 0.0
            noise_amt[seg] = 0.9
        elif s in AFFRICATES:
            voiced[seg] = 0.25
            noise_amt[seg] = 0.9
        else:
            voiced[seg] = 0.5

        # pitch: declination + stress rise
        base = f0 - (pos / max(total, 1e-6)) * F0_RANGE
        if u.stressed:
            base += 12.0
        f0_traj[seg] = np.linspace(base + 6, base - 4, L) * np.where(voiced[seg] > 0.05, 1.0, 0.0)
        pos += dur / rate

    # smooth trajectories so the filter never jumps
    kernel = np.ones(max(3, int(sr * 0.012))) / max(3, int(sr * 0.012))
    for arr in (f1, f2, f3, amp, f0_traj, voiced):
        arr[:] = np.convolve(arr, kernel, mode="same")

    # ---- sources
    glottal = glottal_source(n_total, f0_traj, sr, seed=seed)
    # radiation: emphasise high frequencies slightly (6 dB/octave)
    glottal = np.concatenate([[0.0], np.diff(glottal)]) * 0.6 + glottal * 0.7
    breath = noise(n_total, sr, seed=seed + 1) * BREATH

    # ---- vocal tract: three cascaded resonators + a parallel frication path
    y = np.zeros(n_total)
    y = frame_varying_filter(glottal * voiced + breath * voiced, f1, np.maximum(70, f1 * 0.12), sr)
    y = frame_varying_filter(y, f2, np.maximum(90, f2 * 0.10), sr)
    y = frame_varying_filter(y, f3, np.maximum(110, f3 * 0.10), sr)

    # frication noise through the same tract shape, then mixed by noise_amt
    fric = band_noise(n_total, sr, 0, 0, seed=seed + 2)
    fric = frame_varying_filter(fric, np.maximum(f2, 900), np.maximum(1400, f2 * 0.5), sr)
    y = y + fric * noise_amt * 0.55

    # shimmer + amplitude envelope
    rng = np.random.default_rng(seed + 3)
    shim = 1.0 + rng.normal(0, SHIMMER, n_total)
    y = y * amp * shim

    # ---- output shaping
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y)) or 1.0)
    y = y / peak * 0.88
    # gentle soft clip so loud bursts never crackle
    y = np.tanh(y * 1.05) * 0.92
    return fade(y.astype(np.float32), ms=6, sr=sr)


def units_from_g2p(phonemes: Iterable, rate: float = 1.0) -> List[Unit]:
    """Adapt ``voice.tts.g2p.Phoneme`` objects into synthesiser units.

    Pause markers (``_``) are folded into the *following* unit as a leading
    silence, which is what the renderer expects.
    """
    units: List[Unit] = []
    pending_pause = 0.0
    for ph in phonemes:
        # NOTE: getattr's default is evaluated eagerly, so never subscript here.
        if hasattr(ph, "sym"):
            sym, stressed, pause = ph.sym, bool(ph.stressed), float(ph.pause or 0.0)
        elif isinstance(ph, str):
            sym, stressed, pause = ph, False, 0.0
        else:
            sym = ph[0]
            stressed = bool(ph[1]) if len(ph) > 1 else False
            pause = float(ph[2]) if len(ph) > 2 else 0.0
        if sym == "_" or not sym:
            pending_pause = max(pending_pause, pause or 0.18)
            continue
        units.append(Unit(sym, stressed=stressed, pause_before=max(pause, pending_pause),
                          rate=rate))
        pending_pause = 0.0
    return units


def synthesize(phonemes: Sequence[Tuple[str, bool]], sr: int = SR_DEFAULT,
               f0: float = BASE_F0, rate: float = 1.0, pauses: Optional[Sequence[float]] = None) -> np.ndarray:
    """Convenience wrapper: [(symbol, stressed), ...] -> waveform."""
    units = []
    for i, item in enumerate(phonemes):
        if isinstance(item, str):
            sym, stressed = item, False
        else:
            sym, stressed = item[0], bool(item[1]) if len(item) > 1 else False
        pause = float(pauses[i]) if pauses and i < len(pauses) else 0.0
        if sym == "_":
            if units:
                units[-1].pause_before = 0.0
            units.append(Unit("_", pause_before=max(pause, 0.18)))
            continue
        units.append(Unit(sym, stressed=stressed, pause_before=pause))
    return synthesize_units(units, sr=sr, f0=f0, rate=rate)


def save_wav(path: str, wav: np.ndarray, sr: int = SR_DEFAULT) -> str:
    """Write a 16-bit mono WAV without depending on soundfile."""
    import struct
    import wave
    data = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm)
    return path


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from voice.tts.g2p import convert  # noqa: E402

    text = " ".join(sys.argv[1:]) or "שלום אדוני. כל המערכות פעילות. אני ג'רוויס, ואני מוכן לעבודה."
    phs = convert(text)
    print(f"text     : {text}")
    print(f"phonemes : {' '.join(p.sym for p in phs)}")
    wav = synthesize_units(units_from_g2p(phs))
    out = sys.argv[0].replace("formant.py", "") + "formant_demo.wav"
    save_wav(out, wav)
    print(f"wrote    : {out} ({len(wav)/SR_DEFAULT:.2f}s, peak={np.max(np.abs(wav)):.3f})")
