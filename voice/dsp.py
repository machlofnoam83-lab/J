"""JARVIS DSP toolkit — written from first principles.

Everything the voice engine needs, in one dependency-light module:

  framing / windowing / power spectrum / mel filterbank / MFCC / DCT-II
  energy, zero-crossing rate, spectral flux
  voice activity detection (VAD)
  resampling, PSOLA pitch shifting, time stretching, crossfade, normalisation

Used by BOTH directions:
  * TTS  — unit selection, prosody modification, smoothing
  * STT  — features for the wake word and the command recogniser

No librosa. No torchaudio. Just numpy + scipy.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import lfilter, resample_poly, butter, sosfilt
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


# ------------------------------------------------------------------ basics --
def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    x = to_mono(x)
    m = float(np.max(np.abs(x)) or 1.0)
    return (x / m * peak).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return to_mono(x).astype(np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    if HAVE_SCIPY:
        return resample_poly(to_mono(x), sr_out // g, sr_in // g).astype(np.float32)
    # linear-interpolation fallback
    n_out = int(len(x) * sr_out / sr_in)
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), to_mono(x)).astype(np.float32)


def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Join two signals with an equal-power crossfade of ``n`` samples."""
    a, b = to_mono(a), to_mono(b)
    n = max(0, min(int(n), len(a), len(b)))
    if n == 0:
        return np.concatenate([a, b]).astype(np.float32)
    ramp = np.linspace(0, 1, n)
    ramp = np.sin(ramp * math.pi / 2)          # equal-power curve
    joint = a[-n:] * (1 - ramp) + b[:n] * ramp
    return np.concatenate([a[:-n], joint, b[n:]]).astype(np.float32)


def fade_in_out(x: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    n = max(1, int(sr * ms / 1000))
    x = to_mono(x).copy()
    if len(x) < 2 * n:
        return (x * np.hanning(len(x))).astype(np.float32)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x.astype(np.float32)


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ----------------------------------------------------------------- framing --
def frames(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0
           ) -> Tuple[np.ndarray, int]:
    win = max(8, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    out = x[idx] * np.hamming(win)[None, :]
    return out.astype(np.float64), hop


def power_spectrum(fr: np.ndarray, nfft: int) -> np.ndarray:
    spec = np.fft.rfft(fr, n=nfft)
    return (np.abs(spec) ** 2) / nfft


def hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, nfft: int, n_filters: int = 26,
                   fmin: float = 60.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz = np.array([mel_to_hz(m) for m in mels])
    bins = np.floor((nfft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_filters, nfft // 2 + 1))
    for i in range(n_filters):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid == lo:
            mid = lo + 1
        if hi == mid:
            hi = mid + 1
        for k in range(lo, min(mid, fb.shape[1])):
            fb[i, k] = (k - lo) / max(1, (mid - lo))
        for k in range(mid, min(hi, fb.shape[1])):
            fb[i, k] = (hi - k) / max(1, (hi - mid))
    return fb


def dct2(x: np.ndarray, n: int) -> np.ndarray:
    """Type-II DCT along the last axis (our own implementation)."""
    N = x.shape[-1]
    k = np.arange(n).reshape(-1, 1)
    idx = np.arange(N).reshape(1, -1)
    basis = np.cos(math.pi * k * (2 * idx + 1) / (2 * N))
    return x @ basis.T


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = 13, n_filters: int = 26,
         nfft: int = 512, frame_ms: float = 25.0, hop_ms: float = 10.0,
         lifter: float = 22.0, delta: bool = False) -> np.ndarray:
    """Mel-frequency cepstral coefficients. Returns (T, n_mfcc[, *3 with deltas])."""
    x = to_mono(x)
    fr, _hop = frames(x, sr, frame_ms, hop_ms)
    ps = power_spectrum(fr, nfft)
    fb = mel_filterbank(sr, nfft, n_filters)
    energies = ps @ fb.T
    log_e = np.log(np.maximum(energies, 1e-10))
    cep = dct2(log_e, n_mfcc)
    if lifter:
        n = np.arange(n_mfcc)
        cep *= (1 + (lifter / 2) * np.sin(math.pi * n / lifter))
    if not delta:
        return cep.astype(np.float32)
    d1 = _delta(cep)
    d2 = _delta(d1)
    return np.hstack([cep, d1, d2]).astype(np.float32)


def _delta(feat: np.ndarray, w: int = 2) -> np.ndarray:
    T, D = feat.shape
    out = np.zeros_like(feat)
    denom = 2 * sum(i * i for i in range(1, w + 1)) or 1
    for t in range(T):
        num = np.zeros(D)
        for i in range(1, w + 1):
            a = feat[min(T - 1, t + i)]
            b = feat[max(0, t - i)]
            num += i * (a - b)
        out[t] = num / denom
    return out


# ------------------------------------------------------------- prosody feats --
def rms_energy(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000))
    n = max(1, len(x) // hop)
    seg = x[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(seg ** 2, axis=1) + 1e-12)


def zero_crossing_rate(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000))
    n = max(1, len(x) // hop)
    seg = x[: n * hop].reshape(n, hop)
    return np.mean(np.abs(np.diff(np.sign(seg), axis=1)) > 0, axis=1)


def spectral_flux(x: np.ndarray, sr: int, nfft: int = 512, hop_ms: float = 10.0) -> np.ndarray:
    fr, _ = frames(x, sr, 25.0, hop_ms)
    ps = np.sqrt(power_spectrum(fr, nfft) + 1e-12)
    ps = ps / np.linalg.norm(ps, axis=1, keepdims=True)
    return np.concatenate([[0.0], np.sqrt(np.sum(np.diff(ps, axis=0) ** 2, axis=1))])


def f0_autocorr(x: np.ndarray, sr: int, fmin: float = 65.0, fmax: float = 380.0,
                frame_ms: float = 40.0, hop_ms: float = 10.0) -> np.ndarray:
    """Pitch track by normalised autocorrelation — used for unit alignment
    and for PSOLA pitch shifting."""
    win = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    lo, hi = int(sr / fmax), int(sr / fmin)
    out = []
    for start in range(0, max(1, len(x) - win), hop):
        seg = x[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        seg = seg - seg.mean()
        e = float(np.dot(seg, seg))
        if e < 1e-6:
            out.append(0.0)
            continue
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        ac = ac[: min(len(ac), hi + 1)] / e
        if len(ac) <= lo:
            out.append(0.0)
            continue
        lag = lo + int(np.argmax(ac[lo:]))
        out.append(sr / lag if ac[lag] > 0.3 else 0.0)
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------- time/pitch edit --
def pitch_shift(x: np.ndarray, sr: int, semitones: float, frame_ms: float = 40.0) -> np.ndarray:
    """Cheap resample-and-restore pitch shift (good enough for unit TTS)."""
    if abs(semitones) < 0.01:
        return to_mono(x).astype(np.float32)
    factor = 2 ** (semitones / 12.0)
    n_out = int(len(x) / factor)
    if n_out < 8:
        return to_mono(x).astype(np.float32)
    shifted = resample(to_mono(x), sr, int(sr / factor))
    return resample(shifted, int(sr / factor), sr).astype(np.float32)


def time_stretch(x: np.ndarray, sr: int, factor: float, win_ms: float = 25.0) -> np.ndarray:
    """WSOLA-ish overlap-add time stretch that preserves pitch."""
    if abs(factor - 1.0) < 0.01:
        return to_mono(x).astype(np.float32)
    x = to_mono(x)
    win = max(64, int(sr * win_ms / 1000))
    hop_in = win // 2
    hop_out = int(hop_in * factor)
    if hop_out < 1:
        return x.astype(np.float32)
    n_out = int(len(x) * factor) + win
    out = np.zeros(n_out)
    wsum = np.zeros(n_out)
    w = np.hanning(win)
    pos_in, pos_out = 0, 0
    while pos_in + win <= len(x) and pos_out + win <= n_out:
        out[pos_out:pos_out + win] += x[pos_in:pos_in + win] * w
        wsum[pos_out:pos_out + win] += w
        pos_in += hop_in
        pos_out += hop_out
    out = np.divide(out, np.maximum(wsum, 1e-6))
    return out.astype(np.float32)


# --------------------------------------------------------------------- VAD --
class VAD:
    """Energy + flux voice activity detection with an adaptive noise floor."""

    def __init__(self, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
                 hangover: int = 18, threshold_scale: float = 3.2) -> None:
        self.sr = sr
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.hangover = hangover
        self.threshold_scale = threshold_scale
        self.noise_floor = 1e-4
        self.flux_floor = 1e-3

    def mask(self, x: np.ndarray) -> np.ndarray:
        x = to_mono(x)
        e = rms_energy(x, self.sr, self.hop_ms)
        f = spectral_flux(x, self.sr, hop_ms=self.hop_ms)
        z = zero_crossing_rate(x, self.sr, self.hop_ms)
        # the three feature streams can differ in length by a frame or two
        # (framing vs. diff) — align them before combining
        n = min(len(e), len(f), len(z))
        if n == 0:
            return np.zeros(0, dtype=bool)
        e, f, z = e[:n], f[:n], z[:n]
        floor_e = float(np.quantile(e, 0.10)) or 1e-5
        floor_f = float(np.quantile(f, 0.25)) or 1e-5
        active = (e > floor_e * self.threshold_scale) & (f > floor_f * 1.35) & (z < 0.62)
        # hangover smoothing so word tails are not clipped
        out = np.zeros_like(active)
        run = 0
        for i, a in enumerate(active):
            if a:
                run = self.hangover
            elif run > 0:
                run -= 1
            out[i] = run > 0 or a
        return out

    def segments(self, x: np.ndarray, min_ms: float = 120.0, pad_ms: float = 40.0
                 ) -> List[Tuple[int, int]]:
        """Return sample-index (start, end) pairs of voiced regions."""
        mask = self.mask(x)
        hop = int(self.sr * self.hop_ms / 1000)
        segs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i, a in enumerate(mask):
            if a and start is None:
                start = i
            elif not a and start is not None:
                segs.append((start, i))
                start = None
        if start is not None:
            segs.append((start, len(mask)))
        pad = int(pad_ms / 1000 * self.sr)
        min_len = int(min_ms / 1000 * self.sr)
        out = []
        for a, b in segs:
            s = max(0, a * hop - pad)
            e = min(len(x), b * hop + pad)
            if e - s >= min_len:
                out.append((s, e))
        return out

    def trim(self, x: np.ndarray) -> np.ndarray:
        x = to_mono(x)
        segs = self.segments(x)
        if not segs:
            return x.astype(np.float32)
        return x[segs[0][0]: segs[-1][1]].astype(np.float32)


def split_on_silence(x: np.ndarray, sr: int, silence_ms: float = 140.0,
                     min_len_ms: float = 60.0) -> List[np.ndarray]:
    """Segment a recording into chunks using our VAD."""
    vad = VAD(sr)
    segs = vad.segments(x, min_ms=min_len_ms)
    if not segs:
        return [to_mono(x).astype(np.float32)]
    return [to_mono(x)[s:e].astype(np.float32) for s, e in segs]


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 12) -> float:
    """Banded DTW between two (T, D) feature matrices. Our own implementation."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    INF = float("inf")
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m] / max(1, (n + m) / 2))
