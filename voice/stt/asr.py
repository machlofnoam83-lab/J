#!/usr/bin/env python3
"""Open-vocabulary Hebrew speech recognition — phones, not phrases.

Why this module exists
──────────────────────
``voice/stt/__init__.py`` matches a whole utterance against a bank of enrolled
command templates. It works, and it is well calibrated (50/50 accepted, 0/11
negatives). But it has a hard ceiling: **25 phrases**. Anything outside those
25 is not "poorly recognised", it is *unhearable*. A template bank cannot be
extended to open vocabulary because whole-utterance DTW has no way to generalise
across word boundaries — you would need one template per sentence.

So this recognises **phones** instead. Phones generalise: 26 of them recombine
into every Hebrew word. The pipeline is

    audio ─▶ MFCC+Δ frames ─▶ per-phone Gaussian posteriors
          ─▶ Viterbi (bigram + duration constrained) ─▶ phone string
          ─▶ lexicon DP over a pronunciation trie ─▶ words + confidence

The acoustic model is trained on the 386 labelled phone units already sitting in
``brain/voicebank/`` — the same units TTS synthesises from, so no new recording
is needed. Pronunciations come from ``voice/tts/g2p.py``, which already converts
Hebrew text to phones for synthesis; running it over a harvested word list gives
a lexicon for free, and the two sides cannot drift because they share one G2P.

THIS DOES NOT WORK YET — measured, not assumed
──────────────────────────────────────────────
Do not wire this into the product. The honest numbers, on 20 voicebank words held
out entirely from training (no frame of theirs used for alignment or fitting):

    oracle-boundary frame accuracy   0.280   (chance 0.037)
    held-out word recall             0/20
    mean normalised edit distance    1.63    (acceptance threshold 0.34)

The first line is the one that decides it. Given the *true* phone boundaries, the
acoustic model still picks the correct phone on only 28% of frames. Open-
vocabulary decoding needs roughly 70-90%; every stage downstream compounds that
error, so 0/20 recall is the arithmetic consequence, not a decoder bug.

The cause is data volume, not modelling. The voicebank holds ~147 s of one
speaker, which is ~600 frames per phone spread over 26 MFCC+Δ dimensions. No GMM
configuration, duration prior or smoothing schedule separates Hebrew phones
reliably at that density. Two things were tried and neither moved it:

  * augmentation (gain, resample, noise) — 0.141 -> 0.150 held-out frame accuracy
  * connected-speech bootstrap via forced alignment (below) — 0/20 -> 0/20 recall

What is worth keeping
─────────────────────
Not everything here failed, and the parts that work are reusable:

  * ``forced_align`` — constrained Viterbi over a *known* phone sequence. Aligned
    100/100 voicebank words and produced sane segment counts. This is the tool
    that would label new recording data if more is ever collected.
  * ``DurationModel`` — P(d | phone) measured from the bank instead of a global
    MIN/MAX guess; the bank's own medians run 4 frames (A, Q) to 40 (k), so one
    global bound is wrong for most phones.
  * ``Lexicon`` — 2,910 Hebrew words pronounced through the *same* G2P that TTS
    speaks with, so the two cannot drift apart.

A bug found and fixed on the way is worth recording because it faked a much worse
result than reality: the duration-model rewrite of ``viterbi`` dropped the step
that collapses the per-frame backtrace into segments, so the hypothesis carried
one "phone" per frame — 86 phones for an 86-frame utterance. Normalised edit
distance read 21.76 against a 0.34 threshold, which looked like hopeless
acoustics. It was a bookkeeping error; fixing it moved NED to 1.87. The acoustics
are still too weak, but they are not *that* weak, and the difference matters if
this is ever revisited.

What would actually make it work
────────────────────────────────
More real speech, not better code. The enrollment flow already in the HUD
(🎯 אימון קול) is the right channel: a few minutes per speaker would put the
phone models in the density range where 28% could climb toward usable. Until
that data exists, the 25-command template bank in ``voice/stt/__init__.py``
remains the recogniser that ships — it is measured at 50/50 accepted and 0/11
negatives, and it is honest about its own ceiling.

Nothing here calls the network. No weights are downloaded. numpy only.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice.dsp import mfcc, resample, to_mono  # noqa: E402

SR = 16_000
N_MFCC = 13
DIM = N_MFCC * 2          # static + delta
HOP_MS = 10.0
FRAME_MS = 25.0

# ── decoding constants ──────────────────────────────────────────────────────
# A Hebrew phone is 30-420 ms (measured by the voicebank builder's plausibility
# guard). At a 10 ms hop that is 3-42 frames, and these bounds are that range
# pulled in slightly: hard limits produce fewer silly 1-frame phones than a
# free Viterbi does, without forbidding anything the data actually contains.
MIN_PHONE_FRAMES = 2
MAX_PHONE_FRAMES = 46

# Reward for staying in the same phone. Too low and the decoder chatters between
# phones; too high and it collapses everything into one long phone. This value is
# a log-domain self-loop probability and is tuned in `_tune_self_loop`.
SELF_LOOP_LOGP = math.log(0.86)

# Silence gets its own model. Without it, non-speech is forced through the phone
# inventory and comes out as *some* word, which is the worst failure mode a
# recogniser has: confidently hearing speech that is not there.
SILENCE = "#"

# Longest phone segment the decoders consider, in frames. Measured voicebank
# units run to 1.68 s, but that is an isolated phone with its own onset and
# release; in connected speech nothing legitimately holds that long, and allowing
# it lets the decoder cover a whole utterance with two segments. 60 frames
# (600 ms) is above every real connected-speech phone and below the point where
# collapsing becomes attractive.
MAX_DUR = 60

ASR_FORMAT = 1
MODEL_NAME = "asr_phones.npz"
LEXICON_NAME = "asr_lexicon.json"


def default_model_dir() -> Path:
    """Where ASR models live. Overridable with ``JARVIS_ASR_DIR``.

    Kept out of the repo tree by default for the same reason the STT bank is:
    an acoustic model trained on *your* voice is personal data, and it must not
    be committed or shipped to somebody else.
    """
    env = os.environ.get("JARVIS_ASR_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "models" / "asr"


# ══════════════════════════════════════════════════════════════════════════
#  features
# ══════════════════════════════════════════════════════════════════════════
def raw_features(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """MFCC + Δ with **no** per-utterance normalisation.

    This is deliberately different from ``voice.stt.features``. That one applies
    cepstral mean *and variance* normalisation per utterance, which is exactly
    right for whole-utterance template matching — the trajectory's shape is all
    that is compared. It is wrong here: frame-level phone classification needs
    one fixed coordinate system shared by training and inference, otherwise a
    phone cut from a short unit and the same phone inside a long sentence land
    in different places and the Gaussians cannot cover both.

    Normalisation is applied globally instead, from statistics measured over the
    whole training pool (see ``PhoneModel.fit``), and the same stored mean/std is
    used at inference. Consistency beats cleverness.
    """
    x = to_mono(np.asarray(x, dtype=np.float32))
    if x.size == 0:
        return np.zeros((0, DIM), dtype=np.float32)
    if sr != SR:
        x = resample(x, sr, SR)
    peak = float(np.abs(x).max())
    if peak > 1e-6:
        x = x * (0.95 / peak)
    f = mfcc(x, SR, n_mfcc=N_MFCC, delta=True, frame_ms=FRAME_MS, hop_ms=HOP_MS)
    if f.ndim != 2 or f.shape[0] < 2:
        return np.zeros((0, DIM), dtype=np.float32)
    # mfcc(delta=True) returns static | delta | accel; keep static + delta only
    if f.shape[1] >= DIM:
        f = np.concatenate([f[:, :N_MFCC], f[:, N_MFCC:DIM]], axis=1)
    return np.ascontiguousarray(f, dtype=np.float32)


def frame_energy(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Per-frame RMS, aligned to ``raw_features`` frame indexing."""
    x = to_mono(np.asarray(x, dtype=np.float32))
    if sr != SR:
        x = resample(x, sr, SR)
    hop = int(SR * HOP_MS / 1000.0)
    win = int(SR * FRAME_MS / 1000.0)
    if x.size < win:
        return np.zeros(0, dtype=np.float32)
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    return np.sqrt(np.maximum((fr ** 2).mean(axis=1), 1e-12)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  a very small k-means — numpy, deterministic, no sklearn
# ══════════════════════════════════════════════════════════════════════════
def _kmeans(data: np.ndarray, k: int, seed: int = 0, iters: int = 25
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centres, assignments). k-means++ seeding, Lloyd iterations.

    Deterministic under ``seed``: an acoustic model must rebuild identically, or
    a confidence threshold calibrated against one build is meaningless for the
    next.
    """
    n = data.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    centres = np.empty((k, data.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, n))
    centres[0] = data[first]
    closest = ((data - centres[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 1e-12:                      # all points identical
            centres[i] = data[int(rng.integers(0, n))]
        else:
            centres[i] = data[int(rng.choice(n, p=closest / total))]
        closest = np.minimum(closest, ((data - centres[i]) ** 2).sum(axis=1))

    assign = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d = ((data[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1).astype(np.int32)
        if np.array_equal(new, assign) and _ > 0:
            assign = new
            break
        assign = new
        for i in range(k):
            m = assign == i
            if m.any():
                centres[i] = data[m].mean(axis=0)
    return centres, assign


@dataclass
class GMM:
    """Diagonal-covariance Gaussian mixture over MFCC+Δ frames."""

    means: np.ndarray                 # (k, DIM)
    vars: np.ndarray                  # (k, DIM)
    weights: np.ndarray               # (k,)
    log_norm: np.ndarray = field(default=None, repr=False)   # (k,)

    def __post_init__(self) -> None:
        self.log_norm = -0.5 * (self.vars.sum(axis=1)
                                + DIM * math.log(2.0 * math.pi))

    @staticmethod
    def fit(data: np.ndarray, k: int, seed: int = 0, floor: float = 1e-3) -> "GMM":
        if data.shape[0] == 0:
            raise ValueError("cannot fit a GMM on zero frames")
        k = max(1, min(k, data.shape[0] // 8 or 1))
        if k == 1:
            means = data.mean(axis=0, keepdims=True)
            assign = np.zeros(data.shape[0], dtype=np.int32)
        else:
            means, assign = _kmeans(data, k, seed=seed)
        vs, ws = [], []
        for i in range(means.shape[0]):
            m = assign == i
            cnt = int(m.sum())
            if cnt < 2:                         # starved component: drop it
                continue
            vs.append(data[m].var(axis=0) + floor)
            ws.append(cnt)
        if not vs:                              # degenerate — one broad blob
            means = data.mean(axis=0, keepdims=True)
            vs = [data.var(axis=0) + floor]
            ws = [data.shape[0]]
        return GMM(means=np.asarray(means, dtype=np.float64),
                   vars=np.asarray(vs, dtype=np.float64),
                   weights=np.asarray(ws, dtype=np.float64) / float(sum(ws)))

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """(T, DIM) -> (T,) log p(x). Log-sum-exp over components."""
        if x.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        k = self.means.shape[0]
        out = np.empty((x.shape[0], k), dtype=np.float64)
        inv = 1.0 / self.vars
        for i in range(k):
            diff = x - self.means[i]
            out[:, i] = (self.log_norm[i]
                         - 0.5 * ((diff * diff) * inv[i]).sum(axis=1)
                         + math.log(max(self.weights[i], 1e-12)))
        mx = out.max(axis=1, keepdims=True)
        return (mx[:, 0] + np.log(np.exp(out - mx).sum(axis=1) + 1e-300))


# ══════════════════════════════════════════════════════════════════════════
#  acoustic model
# ══════════════════════════════════════════════════════════════════════════
class PhoneModel:
    """One GMM per phone symbol, plus a silence model and global normalisation.

    The global mean/std is part of the model, not a preprocessing choice: it is
    measured over the training frames and saved, so inference lives in exactly
    the coordinate system training did.
    """

    def __init__(self) -> None:
        self.gmms: Dict[str, GMM] = {}
        self.priors: Dict[str, float] = {}
        self.mean: np.ndarray = np.zeros(DIM, dtype=np.float64)
        self.std: np.ndarray = np.ones(DIM, dtype=np.float64)
        self.n_frames: Dict[str, int] = {}
        self.built_at: float = 0.0
        # decode-time companions, saved in the same file so the model cannot be
        # loaded without the duration prior it was tuned against
        self.dur_logp: Optional[np.ndarray] = None
        self.big_logp: Optional[np.ndarray] = None

    # ── training ────────────────────────────────────────────────────────────
    def fit(self, labelled: Sequence[Tuple[str, np.ndarray]],
            components: int = 4, seed: int = 0) -> "PhoneModel":
        """``labelled`` is a sequence of (phone_symbol, raw_feature_matrix).

        Raw, i.e. *not* normalised — normalisation is computed here from the pool
        so that it reflects all the data at once.
        """
        if not labelled:
            raise ValueError("no training frames")
        pool = np.concatenate([f for _, f in labelled if f.shape[0]], axis=0)
        if pool.shape[0] < 8:
            raise ValueError(f"only {pool.shape[0]} training frames — too few")

        self.mean = pool.mean(axis=0).astype(np.float64)
        self.std = (pool.std(axis=0) + 1e-6).astype(np.float64)
        norm = (pool - self.mean) / self.std

        # re-split the normalised pool back into per-phone buckets
        buckets: Dict[str, List[np.ndarray]] = {}
        off = 0
        for sym, f in labelled:
            n = f.shape[0]
            if n:
                buckets.setdefault(sym, []).append(norm[off:off + n])
            off += n

        total = float(sum(v.shape[0] for b in buckets.values() for v in b))
        for sym, chunks in buckets.items():
            data = np.concatenate(chunks, axis=0)
            self.n_frames[sym] = int(data.shape[0])
            # more components for phones with more evidence: a phone seen on 40
            # frames cannot support 4 Gaussians, and forcing it produces
            # components that model noise.
            k = 1 if data.shape[0] < 24 else (2 if data.shape[0] < 80 else
                                              (3 if data.shape[0] < 260 else components))
            self.gmms[sym] = GMM.fit(data, k, seed=seed)
            self.priors[sym] = max(data.shape[0] / total, 1e-4)
        self.built_at = time.time()
        return self

    # ── scoring ─────────────────────────────────────────────────────────────
    def normalize(self, f: np.ndarray) -> np.ndarray:
        if f.shape[0] == 0:
            return f
        return ((f.astype(np.float64) - self.mean) / self.std)

    def posteriors(self, f: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """(T, DIM) raw features -> ((T, P) log posteriors, symbol order)."""
        syms = sorted(self.gmms)
        nf = self.normalize(f)
        T = nf.shape[0]
        ll = np.empty((T, len(syms)), dtype=np.float64)
        for j, s in enumerate(syms):
            ll[:, j] = self.gmms[s].log_likelihood(nf)
        logp = np.log(np.maximum(np.array([self.priors[s] for s in syms]), 1e-12))
        joint = ll + logp[None, :]
        mx = joint.max(axis=1, keepdims=True)
        denom = mx + np.log(np.exp(joint - mx).sum(axis=1, keepdims=True) + 1e-300)
        return joint - denom, syms

    def frame_loglik(self, f: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Joint (likelihood x prior) in log domain — what Viterbi consumes."""
        syms = sorted(self.gmms)
        nf = self.normalize(f)
        ll = np.empty((nf.shape[0], len(syms)), dtype=np.float64)
        for j, s in enumerate(syms):
            ll[:, j] = self.gmms[s].log_likelihood(nf)
        ll += np.log(np.maximum(np.array([self.priors[s] for s in syms]),
                                1e-12))[None, :]
        return ll, syms

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        syms = sorted(self.gmms)
        arrays: Dict[str, np.ndarray] = {
            "mean": self.mean, "std": self.std,
            "symbols": np.array(syms, dtype=object),
            "priors": np.array([self.priors[s] for s in syms], dtype=np.float64),
            "n_frames": np.array([self.n_frames.get(s, 0) for s in syms],
                                 dtype=np.int64),
            "built_at": np.array([self.built_at], dtype=np.float64),
            "fmt": np.array([ASR_FORMAT], dtype=np.int64),
        }
        for s in syms:
            g = self.gmms[s]
            tag = s.replace("#", "_sil_")
            arrays[f"{tag}::means"] = g.means
            arrays[f"{tag}::vars"] = g.vars
            arrays[f"{tag}::weights"] = g.weights
        if self.dur_logp is not None:
            arrays["dur_logp"] = self.dur_logp
        if self.big_logp is not None:
            arrays["big_logp"] = self.big_logp
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path: Path) -> "PhoneModel":
        z = np.load(path, allow_pickle=True)
        if int(z["fmt"][0]) != ASR_FORMAT:
            raise ValueError(f"ASR model format {int(z['fmt'][0])} != {ASR_FORMAT}")
        m = PhoneModel()
        m.mean = z["mean"].astype(np.float64)
        m.std = z["std"].astype(np.float64)
        m.built_at = float(z["built_at"][0])
        syms = [str(s) for s in z["symbols"]]
        pri = z["priors"].astype(np.float64)
        nf = z["n_frames"].astype(np.int64)
        for i, s in enumerate(syms):
            tag = s.replace("#", "_sil_")
            m.gmms[s] = GMM(means=z[f"{tag}::means"].astype(np.float64),
                            vars=z[f"{tag}::vars"].astype(np.float64),
                            weights=z[f"{tag}::weights"].astype(np.float64))
            m.priors[s] = float(pri[i])
            m.n_frames[s] = int(nf[i])
        if "dur_logp" in z.files:
            m.dur_logp = z["dur_logp"].astype(np.float64)
        if "big_logp" in z.files:
            m.big_logp = z["big_logp"].astype(np.float64)
        return m

    @property
    def symbols(self) -> List[str]:
        return sorted(self.gmms)


# ══════════════════════════════════════════════════════════════════════════
#  lexicon
# ══════════════════════════════════════════════════════════════════════════
HEBREW_RE = re.compile(r"^[\u05d0-\u05ea\u05f3\u05f4'״]+$")


def harvest_words(sources: Iterable[Path], extra: Iterable[str] = (),
                  min_len: int = 1, max_len: int = 14) -> List[str]:
    """Collect Hebrew words from local text sources.

    The vocabulary JARVIS can *hear* should overlap the vocabulary he can *say*,
    so the harvest reads the places Hebrew actually appears in this project: the
    knowledge base, the G2P word tables, the HUD strings, the skill help text.
    A recogniser whose lexicon is unrelated to its own domain is a toy.
    """
    found: Dict[str, None] = {}
    for p in sources:
        p = Path(p)
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for w in re.findall(r"[\u05d0-\u05ea][\u05d0-\u05ea\u05f3\u05f4'״]*", text):
            w = w.strip("'״")
            if min_len <= len(w) <= max_len and HEBREW_RE.match(w):
                found.setdefault(w, None)
    for w in extra:
        w = (w or "").strip("'״ ")
        if w and min_len <= len(w) <= max_len and HEBREW_RE.match(w):
            found.setdefault(w, None)
    return list(found)


class Lexicon:
    """word -> phone sequence, plus a prefix trie for fast DP word search."""

    def __init__(self) -> None:
        self.entries: Dict[str, Tuple[str, ...]] = {}
        self.trie: Dict[Any, Any] = {}

    # ── construction ────────────────────────────────────────────────────────
    def build(self, words: Iterable[str], g2p: Any) -> "Lexicon":
        """Pronounce every word with the *same* G2P that TTS speaks with.

        Sharing the G2P is the point: if the recogniser and the synthesiser
        disagree about how a word is pronounced, JARVIS says something he cannot
        then hear, and the round trip silently breaks.
        """
        skipped = 0
        for w in words:
            if w in self.entries:
                continue
            try:
                phs = g2p.convert(w)
            except Exception:
                skipped += 1
                continue
            seq = tuple(p.sym for p in phs if p.sym and p.sym != SILENCE)
            if len(seq) < 2:            # a single phone is not identifiable
                skipped += 1
                continue
            self.entries[w] = seq
        self._index()
        self.skipped = skipped
        return self

    def _index(self) -> None:
        self.trie = {}
        for w, seq in self.entries.items():
            node = self.trie
            for p in seq:
                node = node.setdefault(p, {})
            node.setdefault("\0", []).append(w)

    # ── queries ─────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, w: str) -> bool:
        return w in self.entries

    def phones(self, w: str) -> Optional[Tuple[str, ...]]:
        return self.entries.get(w)

    def phone_bigram(self) -> Dict[Tuple[str, str], float]:
        """P(phone_b | phone_a) estimated from the lexicon.

        Used as a soft constraint in Viterbi. Without it the decoder happily
        emits sequences no Hebrew word contains; with it, impossible transitions
        cost more but are not forbidden — important, because a mis-heard frame
        should degrade gracefully rather than dead-end the whole utterance.
        """
        counts: Dict[Tuple[str, str], int] = {}
        totals: Dict[str, int] = {}
        for seq in self.entries.values():
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
                totals[a] = totals.get(a, 0) + 1
        return {k: v / float(totals[k[0]]) for k, v in counts.items()}

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"format": ASR_FORMAT, "count": len(self.entries),
             "entries": {k: list(v) for k, v in sorted(self.entries.items())}},
            ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Lexicon":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(d.get("format", 0)) != ASR_FORMAT:
            raise ValueError(f"lexicon format {d.get('format')} != {ASR_FORMAT}")
        lx = Lexicon()
        lx.entries = {k: tuple(v) for k, v in d["entries"].items()}
        lx.skipped = int(d.get("skipped", 0))
        lx._index()
        return lx


# ══════════════════════════════════════════════════════════════════════════
#  forced alignment — bootstrapping connected-speech labels from real voice
# ══════════════════════════════════════════════════════════════════════════
# The 386 voicebank units are *isolated* phones. The decoder never sees isolated
# phones: it sees continuous speech where every phone is bent by its neighbours.
# Training on one domain and decoding the other is a mismatch no amount of GMM
# tuning repairs — measured, held-out frame accuracy was 0.15 against 0.28 on the
# training units themselves.
#
# The raw recordings in voice/recordings/ are a real human voice reading known
# words, so the phone *sequence* of each word is known from the G2P and only the
# boundaries are missing. Constrained Viterbi recovers them: that is forced
# alignment, and it turns ~100 real words into thousands of frames of
# connected-speech phone labels without inventing any audio.
#
# This is not the TTS circularity trap. Nothing here renders synthetic speech and
# then "recognises" it; the labels come from a person, and the only thing the
# model contributes is where inside a known word one phone stops and the next
# starts.

def forced_align(feat: np.ndarray, seq: Sequence[str], syms: Sequence[str],
                 ll_full: np.ndarray, dur: Optional[DurationModel] = None,
                 max_dur: int = MAX_DUR) -> Optional[np.ndarray]:
    """Assign every frame to a phone of a KNOWN sequence, in order.

    Returns an int array of per-frame indices into ``seq``, or None if the frames
    cannot cover the sequence (fewer frames than phones, which happens for very
    short or badly trimmed units).

    Unlike open decoding, the phone order is fixed, so the search is over
    boundaries only: state ``(t, k, d)`` = frame t is the d-th frame of the k-th
    phone. Duration priors keep boundaries off acoustically implausible spots —
    a 1-frame vowel between two consonants is not a real segmentation even if the
    Gaussians momentarily prefer it.
    """
    T = feat.shape[0]
    K = len(seq)
    if T == 0 or K == 0 or T < K:
        return None
    idx = {s: i for i, s in enumerate(syms)}
    cols = [idx.get(p) for p in seq]
    if any(c is None for c in cols):
        return None
    ll = ll_full[:, cols]                                    # (T, K)

    D = min(max_dur, T)
    NEG = -1e15
    # delta[k][d] — best score with the current frame as the d-th of phone k
    delta = np.full((K, D + 1), NEG, dtype=np.float64)
    delta[0, 1] = ll[0, 0]

    bp_k = np.zeros((T, K, D + 1), dtype=np.int16)
    bp_d = np.zeros((T, K, D + 1), dtype=np.int16)

    durp = None
    if dur is not None:
        durp = np.zeros((K, D + 1), dtype=np.float64)
        for k, p in enumerate(seq):
            j = idx.get(p)
            if j is not None:
                m = min(D, dur.logp.shape[1] - 1)
                durp[k, :m + 1] = dur.logp[j, :m + 1]
                if dur.logp.shape[1] - 1 > m:                # fold the tail in
                    durp[k, m] = np.logaddexp(durp[k, m],
                                              dur.logp[j, m + 1:].max()
                                              if dur.logp[j, m + 1:].size else NEG)

    for t in range(1, T):
        new = np.full((K, D + 1), NEG, dtype=np.float64)
        # stay in the same phone
        new[:, 2:] = delta[:, 1:D] + ll[t][:, None]
        bp_k[t][:, 2:] = np.arange(K, dtype=np.int16)[:, None]
        bp_d[t][:, 2:] = np.arange(1, D, dtype=np.int16)[None, :]
        # advance to the next phone in the known sequence
        adv = delta[:-1].max(axis=1)                # best over durations, phone k-1
        adv_arg = delta[:-1].argmax(axis=1).astype(np.int16)
        cand = adv + ll[t, 1:]
        better = cand > new[1:, 1]
        if better.any():
            kk = np.arange(1, K)[better]
            new[kk, 1] = cand[better]
            bp_k[t][kk, 1] = kk - 1
            bp_d[t][kk, 1] = adv_arg[kk - 1]
        delta = new
        if delta.max() <= NEG / 2:
            return None

    # must finish on the LAST phone, otherwise the sequence was not consumed
    final = delta[K - 1].copy()
    if durp is not None:
        final = final + durp[K - 1]
    final[0] = NEG
    if final.max() <= NEG / 2:
        return None
    d = int(final.argmax())

    labels = np.zeros(T, dtype=np.int32)
    k, dd, t = K - 1, d, T - 1
    while t >= 0:
        labels[t] = k
        if t == 0:
            break
        pk, pd = int(bp_k[t][k, dd]), int(bp_d[t][k, dd])
        k, dd = pk, pd
        t -= 1
    return labels


# ══════════════════════════════════════════════════════════════════════════
#  duration model — measured, not guessed
# ══════════════════════════════════════════════════════════════════════════
# The voicebank records how long every phone unit actually is, and those
# durations differ a lot: A and Q median 4 frames, k medians 40. A single global
# MIN/MAX_PHONE_FRAMES therefore either forbids real short phones or permits
# absurd long ones. This builds P(d | phone) from the measurements instead.


class DurationModel:
    """Per-phone distribution over segment length in frames.

    Smoothed two ways, because both failure modes are real: add-k so a duration
    never observed for a rare phone is unlikely rather than impossible, and a
    geometric tail so connected speech — where phones stretch across a word
    boundary — is not cut off at the longest *isolated* unit the bank happens to
    contain.
    """

    def __init__(self, n_phones: int, max_dur: int = MAX_DUR) -> None:
        self.max_dur = max_dur
        self.logp = np.full((n_phones, max_dur + 1), -np.inf, dtype=np.float64)
        self.mean = np.zeros(n_phones, dtype=np.float64)

    @staticmethod
    def build(order: Sequence[str], durations: Dict[str, Sequence[float]],
              hop_ms: float = HOP_MS, max_dur: int = MAX_DUR,
              silence_sym: str = SILENCE, alpha: float = 0.35) -> "DurationModel":
        dm = DurationModel(len(order), max_dur)
        for i, sym in enumerate(order):
            secs = list(durations.get(sym, []))
            counts = np.zeros(max_dur + 2, dtype=np.float64)
            for v in secs:
                d = int(round(max(float(v), 0.0) * 1000.0 / hop_ms))
                d = min(max(d, 1), max_dur + 1)
                counts[d] += 1.0
            if counts.sum() == 0:
                # no measurements for this symbol — fall back to a broad prior
                # centred on the population median rather than a spike.
                counts[3:max_dur // 2] = 1e-3
            counts = counts + alpha                        # add-k smoothing
            # geometric tail beyond max_dur: real speech overruns isolated units
            tail = counts[max_dur + 1]
            counts = counts[:max_dur + 1]
            if tail > 0:
                decay = 0.82
                bonus = np.array([tail * (decay ** (d - max_dur))
                                  for d in range(1, max_dur + 1)])
                counts[1:] += bonus
            # a phone cannot occupy zero frames
            counts[0] = 1e-9
            # silence is allowed to run long: it carries the pauses between words
            if sym == silence_sym:
                counts[max(2, max_dur // 3):] *= 6.0
            pmf = counts / counts.sum()
            dm.logp[i] = np.log(np.maximum(pmf, 1e-12))
            dd = np.arange(max_dur + 1, dtype=np.float64)
            dm.mean[i] = float((pmf * dd).sum())
        return dm

    def log_prob(self, phone_idx: int, d: int) -> float:
        d = int(min(max(d, 0), self.max_dur))
        return float(self.logp[phone_idx, d])


def smooth_bigram(order: Sequence[str], pairs: Dict[Tuple[str, str], float],
                  alpha: float = 0.6) -> np.ndarray:
    """P(q | p) over the phone inventory, add-k smoothed.

    The lexicon yields only the transitions that occur inside real words, so a
    raw estimate assigns *zero* to everything else. Zero in log domain is -inf,
    and one unseen transition then kills the entire utterance — which is what
    made the decoder collapse into a single long phone rather than risk a switch.
    Smoothing keeps unseen transitions improbable but survivable.
    """
    P = len(order)
    idx = {s: i for i, s in enumerate(order)}
    counts = np.full((P, P), alpha, dtype=np.float64)
    for (a, b), pr in (pairs or {}).items():
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += max(float(pr), 0.0) * 20.0
    sil = idx.get(SILENCE)
    if sil is not None:
        # silence brackets every word, so it connects to everything freely
        counts[sil, :] += 4.0
        counts[:, sil] += 4.0
        counts[sil, sil] += 40.0
    return np.log(counts / counts.sum(axis=1, keepdims=True))


# ══════════════════════════════════════════════════════════════════════════
#  Viterbi with an explicit duration model
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class PhoneHyp:
    symbols: List[str]
    starts: List[int]
    ends: List[int]
    logp: float

    def as_string(self) -> str:
        return "".join(" " if s == SILENCE else s for s in self.symbols).strip()

    @property
    def n_frames(self) -> int:
        return self.ends[-1] if self.ends else 0


def viterbi(ll: np.ndarray, syms: Sequence[str],
            dur: DurationModel,
            bigram_logp: np.ndarray,
            max_dur: int = MAX_DUR) -> PhoneHyp:
    """Exact Viterbi over ``(phone, frames_spent_in_that_phone)`` states.

    The duration prior is charged when a segment *completes*, not per frame. That
    is what makes it a real duration model instead of a self-loop hack: a phone
    running 40 frames pays once for being 40 frames long, so the decoder compares
    "one long k" against "several short phones" on equal terms.

    Recursion, for frame t:
        extend   delta[t][p][d] = delta[t-1][p][d-1] + ll[t][p]
        complete comp[q]        = max_d ( delta[t-1][q][d] + logP(d|q) )
        switch   delta[t][p][1] = max_q ( comp[q] + logP(p|q) ) + ll[t][p]

    Cost is O(T x (P x D + P^2)), which for 27 phones and 60 durations is trivial.
    """
    T, P = ll.shape
    if T == 0 or P == 0 or dur is None:
        return PhoneHyp([], [], [], -math.inf)
    D = min(max_dur, max(T, 1))
    NEG = -1e15

    idx = {s: i for i, s in enumerate(syms)}
    sil = idx.get(SILENCE)
    dur_logp = dur.logp[:, :D + 1]

    delta = np.full((P, D + 1), NEG, dtype=np.float64)
    # every phone may open the utterance; charge nothing for the first segment
    for p in range(P):
        delta[p, 1] = ll[0, p]
    if sil is not None:
        delta[:] = NEG
        delta[sil, 1] = ll[0, sil]
        for p in range(P):
            if p != sil:
                delta[p, 1] = ll[0, p]

    # backpointers: which phone, and how long it had already run
    bp_p = np.zeros((T, P, D + 1), dtype=np.int16)
    bp_d = np.zeros((T, P, D + 1), dtype=np.int16)

    for t in range(1, T):
        new = np.full((P, D + 1), NEG, dtype=np.float64)

        # ── extend: same phone, one more frame ──
        new[:, 2:] = delta[:, 1:D] + ll[t][:, None]
        bp_p[t][:, 2:] = np.arange(P, dtype=np.int16)[:, None]
        bp_d[t][:, 2:] = np.arange(1, D, dtype=np.int16)[None, :]

        # ── complete + switch ──
        comp = (delta + dur_logp).max(axis=1)              # (P,)
        comp_arg = (delta + dur_logp).argmax(axis=1).astype(np.int16)
        if comp.max() > NEG / 2:
            trans = comp[:, None] + bigram_logp[:P, :P]    # (P_prev, P_next)
            src = trans.argmax(axis=0)                     # best previous phone
            score = trans[src, np.arange(P)] + ll[t]
            better = score > new[:, 1]
            if better.any():
                new[better, 1] = score[better]
                bp_p[t][better, 1] = src[better]
                bp_d[t][better, 1] = comp_arg[src[better]]
        delta = new

        if delta.max() <= NEG / 2:                           # all paths died
            delta = np.full((P, D + 1), NEG, dtype=np.float64)
            p0 = sil if sil is not None else 0
            delta[p0, 1] = ll[t, p0]

    # ── terminate: charge the final segment its duration ──
    final = delta + dur_logp
    flat = int(final.argmax())
    p, d = np.unravel_index(flat, final.shape)
    p, d = int(p), int(d)
    total = float(final[p, d])

    syms_out: List[str] = []
    ends_out: List[int] = []
    starts_out: List[int] = []
    t = T - 1
    while t >= 0:
        syms_out.append(syms[p])
        ends_out.append(t + 1)
        starts_out.append(max(0, t + 1 - d))
        if t == 0:
            break
        pp, pd = int(bp_p[t][p, d]), int(bp_d[t][p, d])
        p, d = pp, pd
        t -= 1

    syms_out.reverse(); starts_out.reverse(); ends_out.reverse()

    # Collapse the per-frame backtrace into segments. Without this the hypothesis
    # carries one entry per frame — 86 "phones" for an 86-frame utterance instead
    # of ~10 — and every downstream consumer silently misreads it: the phone
    # string is 8x too long, so its edit distance against a real pronunciation
    # explodes and no word can ever match. Measured, this alone took held-out
    # normalised edit distance to 21.8 against a 0.34 acceptance threshold.
    seg_s: List[str] = []
    seg_a: List[int] = []
    seg_b: List[int] = []
    for sym, a, b in zip(syms_out, starts_out, ends_out):
        if seg_s and seg_s[-1] == sym:
            seg_b[-1] = max(seg_b[-1], b)
            seg_a[-1] = min(seg_a[-1], a)
        else:
            seg_s.append(sym); seg_a.append(a); seg_b.append(b)
    return PhoneHyp(seg_s, seg_a, seg_b, total)


# ══════════════════════════════════════════════════════════════════════════
#  phone string -> words
# ══════════════════════════════════════════════════════════════════════════
# Edit costs for matching a decoded phone run against a lexicon pronunciation.
# Substitution is cheaper than indel: the common ASR error here is a vowel
# collapsing (a/e/A are acoustically close in this inventory), which is one
# substitution, whereas an indel shifts everything after it.
SUB_COST = 1.0
INS_COST = 1.3
DEL_COST = 1.3

# A word is acceptable if its normalised phone edit distance is below this.
# Above it the decoder heard something that is not in the lexicon, and guessing
# the nearest word is worse than reporting nothing.
MAX_WORD_NED = 0.34


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Weighted Levenshtein between two phone sequences."""
    la, lb = len(a), len(b)
    if la == 0:
        return INS_COST * lb
    if lb == 0:
        return DEL_COST * la
    prev = [INS_COST * j for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [DEL_COST * i] + [0.0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + INS_COST,
                         cur[j - 1] + DEL_COST,
                         prev[j - 1] + (0.0 if ai == b[j - 1] else SUB_COST))
        prev = cur
    return prev[lb]


def match_words(seq: Sequence[str], lex: Lexicon,
                max_ned: float = MAX_WORD_NED) -> List[Dict[str, Any]]:
    """Segment a phone sequence into lexicon words.

    DP over phone positions. At each position, walk the lexicon trie forward and
    score every word that could end there, allowing bounded edit errors so a
    single mis-decoded phone does not lose the whole word. Cost is normalised by
    word length, so long words are not penalised for having more phones in which
    to make one mistake.

    The trie walk keeps this linear in the number of *plausible* continuations
    rather than the whole lexicon, which matters once the vocabulary is in the
    thousands.
    """
    n = len(seq)
    if n == 0 or len(lex) == 0:
        return []

    best: List[float] = [0.0] + [math.inf] * n
    back: List[Optional[Tuple[int, str, float]]] = [None] * (n + 1)
    budget = 2                     # max edit errors allowed inside one word

    for i in range(n):
        if best[i] == math.inf:
            continue
        node = lex.trie
        j = i
        errs = 0
        while j < n:
            p = seq[j]
            nxt = node.get(p)
            if nxt is None:
                break
            node = nxt
            j += 1
            words = node.get("\0")
            if words:
                span = j - i
                for w in words:
                    ref = lex.entries[w]
                    d = _edit_distance(list(seq[i:j]), list(ref))
                    ned = d / max(len(ref), 1)
                    if ned > max_ned:
                        continue
                    cost = best[i] + d + 0.06 * span      # mild length prior
                    if cost < best[j]:
                        best[j] = cost
                        back[j] = (i, w, d)
            if j - i > 16:
                break
        # also allow skipping one un-decodable phone (noise burst, foreign sound)
        if i + 1 <= n and best[i] + SUB_COST * 1.6 < best[i + 1]:
            best[i + 1] = best[i] + SUB_COST * 1.6
            back[i + 1] = (i, "", SUB_COST * 1.6)

    if best[n] == math.inf:
        # no full segmentation — fall back to the longest reachable prefix so a
        # partial answer is still better than silence
        reach = max((k for k in range(n + 1) if best[k] < math.inf), default=0)
        if reach < 2:
            return []
        n = reach

    out: List[Dict[str, Any]] = []
    k = n
    while k > 0 and back[k] is not None:
        i, w, d = back[k]
        if w:
            out.append({"word": w, "start_phone": i, "end_phone": k, "edit": float(d)})
        k = i
    out.reverse()
    return out


# ══════════════════════════════════════════════════════════════════════════
#  bootstrap: real recordings -> connected-speech phone labels
# ══════════════════════════════════════════════════════════════════════════
def _g2p_instance():
    from voice.tts.g2p import HebrewG2P
    return HebrewG2P()


def bootstrap_labels(model: "PhoneModel", words: Dict[str, Path],
                     g2p: Any, base: Sequence[Tuple[str, np.ndarray]] = (),
                     rounds: int = 2,
                     verbose: bool = False) -> Tuple[List[Tuple[str, np.ndarray]], "PhoneModel"]:
    """Viterbi training on real recorded words.

    Each voicebank word file is a real human saying a known word. Its phone
    sequence comes from the same G2P the lexicon uses, so only the boundaries are
    unknown — and ``forced_align`` recovers them. The model then retrains on those
    connected-speech frames and re-aligns, which is textbook Viterbi training:
    alignment improves the model, the better model improves the alignment.

    ``base`` is the isolated-unit evidence the model started from. It is kept in
    every retrain on purpose: without it the model drifts onto the aligned frames
    alone, and since those frames were labelled *by the model*, each round would
    confirm its own previous guess. Keeping the independent recording data
    anchored in is what stops the loop from grading its own homework.

    Two rounds is enough here. More rounds on ~100 words stops adding information
    and starts letting the model confirm its own mistakes.

    Returns ``(labelled_frames, retrained_model)``.
    """
    import soundfile as sf

    base = list(base)
    seqs: List[Tuple[str, np.ndarray, List[str]]] = []
    for w, path in words.items():
        try:
            x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            continue
        feat = raw_features(x, sr)
        if feat.shape[0] < 4:
            continue
        try:
            phs = [p.sym for p in g2p.convert(w)
                   if p.sym and p.sym not in ("_", SILENCE)]
        except Exception:
            continue
        if len(phs) < 2 or feat.shape[0] < len(phs):
            continue
        seqs.append((w, feat, phs))

    if not seqs:
        return [], model

    out: List[Tuple[str, np.ndarray]] = []
    cur = model
    for r in range(max(1, rounds)):
        out = []
        aligned = 0
        for w, feat, phs in seqs:
            ll, syms = cur.frame_loglik(feat)
            lab = forced_align(feat, phs, syms, ll)
            if lab is None:
                continue
            aligned += 1
            # Store RAW frames, never cur.normalize(feat). `fit()` computes its own
            # mean/std over the whole pool and normalises internally, and `base`
            # holds raw frames too — passing pre-normalised ones here would
            # normalise them twice and put the two sets in different coordinate
            # systems, which would silently corrupt exactly the connected-speech
            # evidence this loop exists to add.
            for k, p_ in enumerate(phs):
                m = lab == k
                if m.sum() >= 1:
                    out.append((p_, feat[m]))
        if verbose:
            print(f"[asr]   round {r+1}: aligned {aligned}/{len(seqs)} words, "
                  f"{sum(f.shape[0] for _, f in out)} connected-speech frames")
        if r + 1 < rounds and out:
            # Both `base` and `out` hold raw frames, so fit() computes one mean/std
            # over their union. That matters: the coordinate system has to describe
            # all the data the model will ever see, not just the part it was first
            # built from.
            cur = PhoneModel().fit(base + out, seed=1337)
    return out, cur


# ══════════════════════════════════════════════════════════════════════════
#  engine
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class AsrResult:
    text: str
    words: List[Dict[str, Any]]
    phones: str
    confidence: float
    seconds: float
    n_frames: int
    silence_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "words": self.words, "phones": self.phones,
                "confidence": round(float(self.confidence), 4),
                "seconds": round(float(self.seconds), 3),
                "frames": int(self.n_frames),
                "silence_ratio": round(float(self.silence_ratio), 3)}


# Raised by anything that would put this recogniser in front of a user. The
# measurements in the module docstring are the reason it exists.
class AsrNotReadyError(RuntimeError):
    """The acoustic model is below usable accuracy; see the module docstring."""


EXPERIMENTAL = True


class AsrEngine:
    """Open-vocabulary Hebrew recogniser built on the voicebank's phone units.

    Experimental and **not** product-ready: 0/20 held-out word recall. See the
    module docstring for the measurements and for what would fix it.
    """

    def __init__(self, model_dir: Optional[Path | str] = None) -> None:
        self.dir = Path(model_dir) if model_dir else default_model_dir()
        self.model: Optional[PhoneModel] = None
        self.lex: Optional[Lexicon] = None
        self.bigram: Dict[Tuple[str, str], float] = {}
        self.dur: Optional[DurationModel] = None
        self.big_logp: Optional[np.ndarray] = None
        self.bootstrap_words: Dict[str, Path] = {}
        self.connected_frames: int = 0

    # ── availability ────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return (self.model is not None and self.lex is not None
                and len(self.lex) > 0 and self.dur is not None
                and self.big_logp is not None)

    def load(self) -> bool:
        mp, lp = self.dir / MODEL_NAME, self.dir / LEXICON_NAME
        if not (mp.exists() and lp.exists()):
            return False
        try:
            self.model = PhoneModel.load(mp)
            self.lex = Lexicon.load(lp)
            self.bigram = self.lex.phone_bigram()
            # rebuild the decode matrices from what the model file carries, so a
            # loaded engine behaves exactly like the one that was built
            order = self.model.symbols
            self.dur = DurationModel(len(order))
            if self.model.dur_logp is not None:
                self.dur.logp = self.model.dur_logp
                dd = np.arange(self.dur.logp.shape[1], dtype=np.float64)
                pr = np.exp(self.dur.logp)
                self.dur.mean = (pr * dd[None, :]).sum(axis=1)
            else:
                self.dur = DurationModel.build(order, {})
                self.model.dur_logp = self.dur.logp
            self.big_logp = (self.model.big_logp
                             if self.model.big_logp is not None
                             else smooth_bigram(order, self.bigram))
            return True
        except Exception as exc:                       # corrupt/foreign build
            sys.stderr.write(f"[asr] load failed: {exc}\n")
            self.model = self.lex = None
            return False

    def ensure(self, rebuild: bool = False, verbose: bool = False) -> bool:
        if not rebuild and self.load():
            return True
        return self.build(verbose=verbose)

    # ── building ────────────────────────────────────────────────────────────
    def build(self, voicebank: Optional[Path] = None, verbose: bool = False,
              bootstrap: bool = True, bootstrap_rounds: int = 2,
              exclude_words: Optional[Iterable[str]] = None) -> bool:
        t0 = time.time()
        vb = Path(voicebank) if voicebank else ROOT / "brain" / "voicebank"
        idx_path = vb / "index.json"
        if not idx_path.exists():
            sys.stderr.write(f"[asr] no voicebank index at {idx_path}\n")
            return False
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        sr_bank = int(idx.get("sample_rate", 24_000))

        labelled: List[Tuple[str, np.ndarray]] = []
        skipped = 0
        for key, meta in (idx.get("phones") or {}).items():
            f = vb / meta["file"]
            if not f.exists():
                skipped += 1
                continue
            try:
                import soundfile as sf
                x, sr = sf.read(str(f), dtype="float32", always_2d=False)
            except Exception:
                skipped += 1
                continue
            feat = raw_features(x, sr)
            if feat.shape[0] < 3:
                skipped += 1
                continue
            labelled.append((str(meta["sym"]), feat))

        # Silence model: frames below the energy floor of the recordings. Without
        # it every pause decodes as a phone and pauses become words.
        sil_frames = _collect_silence(vb, sr_bank)
        if sil_frames.shape[0] >= 40:
            labelled.append((SILENCE, sil_frames))

        if len(labelled) < 20:
            sys.stderr.write(f"[asr] only {len(labelled)} usable phone units\n")
            return False

        self.model = PhoneModel().fit(labelled, seed=1337)
        self.lex = _build_lexicon()
        self.bigram = self.lex.phone_bigram()

        # Second pass: the voicebank word files are a real person saying known
        # words, so forced alignment converts them into connected-speech phone
        # labels. Without this the model only ever sees isolated phones and the
        # decoder meets a domain it was not trained on.
        self.bootstrap_words = _bootstrap_word_files(vb, exclude=exclude_words)
        if self.bootstrap_words and bootstrap:
            conn, self.model = bootstrap_labels(
                self.model, self.bootstrap_words, _g2p_instance(),
                base=labelled, rounds=bootstrap_rounds, verbose=verbose)
            self.connected_frames = sum(f.shape[0] for _, f in conn)
        else:
            self.connected_frames = 0

        # Duration prior straight from the voicebank's own measurements: every
        # phone unit records how long it is, so P(d | phone) is observed rather
        # than assumed. Silence has no units, so it gets a broad prior.
        durs: Dict[str, List[float]] = {}
        for meta in (idx.get("phones") or {}).values():
            try:
                durs.setdefault(str(meta["sym"]), []).append(float(meta["duration"]))
            except (TypeError, ValueError):
                continue
        order = self.model.symbols
        self.dur = DurationModel.build(order, durs)
        self.model.dur_logp = self.dur.logp
        self.model.big_logp = smooth_bigram(order, self.bigram)

        self.dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.dir / MODEL_NAME)
        self.lex.save(self.dir / LEXICON_NAME)
        if verbose:
            print(f"[asr] built {len(self.model.symbols)} phone models "
                  f"({sum(self.model.n_frames.values())} frames), "
                  f"lexicon {len(self.lex)} words, "
                  f"skipped {skipped} units, {time.time() - t0:.1f}s")
        return True

    # ── recognition ─────────────────────────────────────────────────────────
    def transcribe(self, x: np.ndarray, sr: int = SR,
                   max_words: int = 24) -> AsrResult:
        if not self.available:
            return AsrResult("", [], "", 0.0, 0.0, 0, 1.0)
        x = to_mono(np.asarray(x, dtype=np.float32))
        seconds = x.size / float(sr or SR)
        feat = raw_features(x, sr)
        if feat.shape[0] < 4:
            return AsrResult("", [], "", 0.0, seconds, int(feat.shape[0]), 1.0)

        en = frame_energy(x, sr)
        if en.size:
            thr = np.percentile(en[en > 0], 18) if (en > 0).any() else 0.0
            sil_ratio = float((en < thr).mean())
        else:
            sil_ratio = 1.0

        ll, syms = self.model.frame_loglik(feat)
        hyp = viterbi(ll, syms, self.dur, self.big_logp)
        if not hyp.symbols:
            return AsrResult("", [], "", 0.0, seconds, feat.shape[0], sil_ratio)

        # drop silence and decode words on the speech phones only
        speech = [(s, a, b) for s, a, b in
                  zip(hyp.symbols, hyp.starts, hyp.ends) if s != SILENCE]
        if not speech:
            return AsrResult("", [], "", 0.0, seconds, feat.shape[0], sil_ratio)
        pseq = [s for s, _, _ in speech]

        hits = match_words(pseq, self.lex)[:max_words]
        text = " ".join(h["word"] for h in hits)

        # ── confidence ───────────────────────────────────────────────────────
        # Two independent signals, both needed:
        #   * acoustic — mean frame log-posterior of the chosen path. Low when the
        #     audio does not look like any trained phone at all.
        #   * lexical — how much of the phone string was actually consumed by
        #     real words, and how many edits that took.
        # A phone string can be acoustically clean and lexically empty (babble),
        # or lexically plausible and acoustically poor (noise). Reporting either
        # alone is how a recogniser ends up confidently wrong.
        idx = {s: i for i, s in enumerate(syms)}
        acoustic = _acoustic_score(ll, idx, hyp)
        covered = sum(h["end_phone"] - h["start_phone"] for h in hits)
        coverage = covered / max(len(pseq), 1)
        edits = sum(h["edit"] for h in hits)
        lex_q = max(0.0, 1.0 - edits / max(covered, 1))
        conf = float(np.clip(0.55 * _sigmoid(acoustic) + 0.45 * (coverage * lex_q),
                             0.0, 1.0))
        if not hits:
            conf = min(conf, 0.12)

        return AsrResult(text=text, words=hits,
                         phones="".join(s if s != SILENCE else " " for s in hyp.symbols).strip(),
                         confidence=conf, seconds=seconds,
                         n_frames=int(feat.shape[0]), silence_ratio=sil_ratio)

    def transcribe_file(self, path: Path | str) -> AsrResult:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return self.transcribe(x, sr)

    # ── introspection ───────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}
        return {
            "available": True,
            "phones": len(self.model.symbols),
            "phone_symbols": self.model.symbols,
            "training_frames": int(sum(self.model.n_frames.values())),
            "connected_speech_frames": int(self.connected_frames),
            "bootstrap_words": len(self.bootstrap_words),
            "components": {s: int(self.model.gmms[s].means.shape[0])
                           for s in self.model.symbols},
            "lexicon_words": len(self.lex),
            "bigram_pairs": len(self.bigram),
            "mean_phone_frames": {s: round(float(self.dur.mean[i]), 2)
                                  for i, s in enumerate(self.model.symbols)}
                                 if self.dur is not None else {},
            "has_silence": SILENCE in self.model.gmms,
            "model_dir": str(self.dir),
            "built_at": self.model.built_at,
            "format": ASR_FORMAT,
        }


# ══════════════════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════════════════
def _sigmoid(v: float) -> float:
    """Map a mean frame log-posterior to 0..1.

    Calibrated so that -log(P) ~= 3.4 (uniform over 26-30 phones is -3.4) maps to
    0.5: a decoder that is no better than guessing scores exactly chance, and
    only real acoustic evidence moves the number up.
    """
    return 1.0 / (1.0 + math.exp(-(v + 3.4) * 1.6))


def _acoustic_score(ll: np.ndarray, idx: Dict[str, int], hyp: PhoneHyp) -> float:
    """Mean log-likelihood of the chosen phones on their own frames."""
    tot, n = 0.0, 0
    for s, a, b in zip(hyp.symbols, hyp.starts, hyp.ends):
        j = idx.get(s)
        if j is None:
            continue
        lo, hi = max(0, a), min(ll.shape[0], b)
        if hi > lo:
            tot += float(ll[lo:hi, j].sum())
            n += hi - lo
    return tot / n if n else -20.0


def _collect_silence(vb: Path, sr_bank: int) -> np.ndarray:
    """Harvest low-energy frames from the voicebank recordings as silence data.

    Using the leading/trailing edges of real units is better than synthetic
    zeros: it is the actual room tone and microphone noise the recogniser will
    meet, so the silence model rejects the real thing rather than an idealisation.
    """
    out: List[np.ndarray] = []
    try:
        import soundfile as sf
    except Exception:
        return np.zeros((0, DIM), dtype=np.float32)
    for f in sorted(vb.glob("word_*.wav"))[:60]:
        try:
            x, sr = sf.read(str(f), dtype="float32")
        except Exception:
            continue
        feat = raw_features(x, sr)
        en = frame_energy(x, sr)
        if feat.shape[0] == 0 or en.size == 0:
            continue
        m = min(feat.shape[0], en.size)
        pos = en[:m] > 0
        if not pos.any():
            continue
        thr = float(np.percentile(en[:m][pos], 15))
        sel = en[:m] < thr
        if sel.sum() >= 4:
            out.append(feat[:m][sel])
    if not out:
        return np.zeros((0, DIM), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def _bootstrap_word_files(vb: Path, exclude: Optional[Iterable[str]] = None
                          ) -> Dict[str, Path]:
    """Map voicebank word -> its recording, for forced alignment.

    Syllable-drill and letter-name entries are excluded from the *lexicon* word
    list elsewhere; here they are still usable as acoustic evidence, because the
    point is phone boundaries in real speech rather than vocabulary. Only the
    ``exclude`` set — held-out words during evaluation — is kept out.
    """
    idx_path = vb / "index.json"
    if not idx_path.exists():
        return {}
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    drop = set(exclude or ())
    out: Dict[str, Path] = {}
    for w, meta in (idx.get("words") or {}).items():
        if w in drop:
            continue
        f = vb / meta["file"]
        if f.exists():
            out[w] = f
    return out


def _build_lexicon() -> Lexicon:
    """Pronounce a harvested Hebrew vocabulary with the shared TTS G2P."""
    from voice.tts.g2p import HebrewG2P
    g2p = HebrewG2P()

    sources = [
        ROOT / "brain" / "knowledge_base.yaml",
        ROOT / "voice" / "tts" / "g2p.py",
        ROOT / "ui" / "index.html",
        ROOT / "ui" / "assets" / "app.js",
        ROOT / "voice" / "stt" / "__init__.py",
        ROOT / "README.md",
    ]
    sources += sorted((ROOT / "skills").glob("*.py"))
    sources += sorted((ROOT / "brain").glob("*.py"))
    sources += sorted((ROOT / "docs").glob("*.md"))

    # Core vocabulary that must be hearable whatever the harvest finds: the
    # enrolled commands, the digits, and the words JARVIS uses about himself.
    core = [
        "ג'רוויס", "אדוני", "שלום", "תודה", "בבקשה", "כן", "לא", "עצור", "המשך",
        "שקט", "דבר", "נגן", "השהה", "זמן", "שעה", "תאריך", "סטטוס", "מערכת",
        "זיכרון", "קובץ", "קבצים", "תיקיה", "קוד", "הסבר", "בדיחה", "עזרה",
        "צילום", "מסך", "דפדפן", "פתח", "סגור", "הפעל", "כבה", "הרג", "חיה",
        "אחד", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע",
        "עשר", "אפס", "מאה", "אלף", "אחד עשר", "שתים עשרה", "עשרים", "שלושים",
        "מה", "איך", "למה", "מי", "מתי", "איפה", "כמה", "האם", "זה", "זו",
        "אני", "אתה", "את", "הוא", "היא", "אנחנו", "הם", "של", "על", "עם",
        "שלום אדוני", "מה השעה", "מה נשמע", "תודה רבה", "אין בעד מה",
        "פתח את הדפדפן", "צלם מסך", "כבה את המערכת", "הפעל מוזיקה",
        "בית", "עבודה", "מחשב", "טלפון", "חלון", "תוכנה", "הודעה", "מייל",
        "טמפרטורה", "מזג אוויר", "חדשות", "מוזיקה", "סרט", "תמונה",
        "זכור", "שכח", "חפש", "מצא", "כתוב", "קרא", "שמור", "מחק", "העתק",
        "הדבק", "גלול", "למעלה", "למטה", "ימינה", "שמאלה", "קדימה", "אחורה",
        "היום", "מחר", "אתמול", "בוקר", "ערב", "לילה", "צהריים", "שבוע",
        "טוב", "רע", "גדול", "קטן", "מהיר", "איטי", "נכון", "שגוי",
        "חשב", "תכנן", "בדוק", "אמת", "בצע", "המתן", "התחל", "סיים",
        "פייתון", "ג'אווהסקריפט", "פונקציה", "משתנה", "מחלקה", "שגיאה",
        "בדיקה", "שרת", "לקוח", "רשת", "אינטרנט", "דף", "אתר",
        "פנים", "זיהוי", "מצלמה", "מיקרופון", "רמקול", "קול", "דיבור",
        "אבטחה", "הרשאה", "סיסמה", "מפתח", "גישה", "חסימה", "אישור",
        "מח", "רשת עצבית", "מודל", "אימון", "למידה", "חישוב", "נוסחה",
        "קושי", "בעיה", "פתרון", "שאלה", "תשובה", "הסבר לי", "אני לא יודע",
        "אי אפשר", "מסוכן", "אסור", "מותר", "בטוח", "סכנה",
    ]
    words = harvest_words(sources, extra=core)
    lx = Lexicon().build(words, g2p)
    return lx


# ══════════════════════════════════════════════════════════════════════════
#  evaluation — the part that decides whether any of this is real
# ══════════════════════════════════════════════════════════════════════════
def self_report(eng: AsrEngine, limit: int = 60) -> Dict[str, Any]:
    """Decode voicebank word files.

    Partly circular — those recordings supplied the phone units the acoustic
    model trained on — so this measures whether the decoder is *wired* correctly,
    not whether it generalises. Reported separately for that reason.
    """
    vb = ROOT / "brain" / "voicebank"
    idx = json.loads((vb / "index.json").read_text(encoding="utf-8"))
    words = idx.get("words") or {}
    hit = tot = 0
    confs: List[float] = []
    for w, meta in list(words.items())[:limit]:
        f = vb / meta["file"]
        if not f.exists():
            continue
        r = eng.transcribe_file(f)
        tot += 1
        confs.append(r.confidence)
        got = set(r.text.replace("'", "").replace("״", "").split())
        if w in got:
            hit += 1
    return {"decoded": tot, "exact_word_found": hit,
            "word_recall": round(hit / tot, 4) if tot else 0.0,
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0}


def held_out_report(eng: AsrEngine, n: int = 40) -> Dict[str, Any]:
    """Decode TTS renders of words that are NOT in the voicebank.

    The audio is synthesised from the same phone units, so it is not a different
    speaker — but the *word sequences* are unseen, which is what generalisation
    to new vocabulary actually requires, and it is the honest ceiling of what can
    be measured without a second human recording. Say so wherever this number is
    quoted.
    """
    from voice.tts import speak
    vb = ROOT / "brain" / "voicebank"
    known = set((json.loads((vb / "index.json").read_text(encoding="utf-8"))
                 .get("words") or {}))
    cand = [w for w in eng.lex.entries if w not in known and len(w) >= 3]
    rng = np.random.default_rng(7)
    rng.shuffle(cand)
    cand = cand[:n]

    hit = tot = 0
    confs: List[float] = []
    samples: List[Dict[str, Any]] = []
    for w in cand:
        try:
            res = speak(w)
        except Exception as exc:
            samples.append({"word": w, "error": str(exc)[:60]})
            continue
        x = np.asarray(getattr(res, "samples", []), dtype=np.float32)
        sr = int(getattr(res, "sample_rate", 24_000))
        if x.size == 0:
            continue
        r = eng.transcribe(x, sr)
        tot += 1
        confs.append(r.confidence)
        got = set(r.text.replace("'", "").replace("״", "").split())
        ok = w in got
        hit += int(ok)
        if len(samples) < 12:
            samples.append({"word": w, "heard": r.text, "phones": r.phones,
                            "conf": round(r.confidence, 3), "ok": bool(ok)})
    return {"decoded": tot, "exact_word_found": hit,
            "word_recall": round(hit / tot, 4) if tot else 0.0,
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            "samples": samples}


def silence_report(eng: AsrEngine) -> Dict[str, Any]:
    """Feed non-speech and require that nothing is heard.

    The failure this guards against is the one that destroys trust fastest:
    JARVIS answering a room nobody spoke in. Correct behaviour is an empty
    transcript, not a low-confidence guess.
    """
    rng = np.random.default_rng(3)
    cases: Dict[str, np.ndarray] = {
        "digital_silence": np.zeros(SR, dtype=np.float32),
        "low_noise": (rng.standard_normal(SR) * 1e-4).astype(np.float32),
        "loud_noise": (rng.standard_normal(SR) * 0.2).astype(np.float32),
        "sine_1k": (0.1 * np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)).astype(np.float32),
    }
    out = {}
    for name, x in cases.items():
        r = eng.transcribe(x, SR)
        out[name] = {"text": r.text, "words": len(r.words),
                     "conf": round(r.confidence, 3),
                     "clean": (r.text == "" and len(r.words) == 0)}
    out["all_clean"] = all(v["clean"] for k, v in out.items() if isinstance(v, dict))
    return out


# ══════════════════════════════════════════════════════════════════════════
_engine: Optional[AsrEngine] = None


def get_engine(rebuild: bool = False) -> AsrEngine:
    global _engine
    if _engine is None:
        _engine = AsrEngine()
        _engine.ensure(rebuild=rebuild)
    elif rebuild:
        _engine.build()
    return _engine


def transcribe_bytes(payload: bytes) -> Dict[str, Any]:
    """Recognise a WAV payload. Mirrors ``voice.stt.recognize_bytes``.

    Raises unless ``JARVIS_ASR_EXPERIMENTAL=1``. This is a guard, not decoration:
    the function signature is exactly what a server route would call, and the
    recogniser behind it is measured at 0/20 held-out word recall. Shipping it
    would mean JARVIS hearing words nobody said, which is worse than not hearing.
    """
    if EXPERIMENTAL and os.environ.get("JARVIS_ASR_EXPERIMENTAL") != "1":
        raise AsrNotReadyError(
            "open-vocabulary ASR is below usable accuracy (0/20 held-out words); "
            "use voice.stt.recognize_bytes, or set JARVIS_ASR_EXPERIMENTAL=1")
    from voice.stt import read_wav_bytes
    x, sr = read_wav_bytes(payload)
    return get_engine().transcribe(x, sr).to_dict()


def transcribe_file(path: Path | str) -> Dict[str, Any]:
    """See ``transcribe_bytes`` — same guard, same reason."""
    if EXPERIMENTAL and os.environ.get("JARVIS_ASR_EXPERIMENTAL") != "1":
        raise AsrNotReadyError(
            "open-vocabulary ASR is below usable accuracy (0/20 held-out words); "
            "set JARVIS_ASR_EXPERIMENTAL=1 to use it anyway")
    return get_engine().transcribe_file(path).to_dict()


# ══════════════════════════════════════════════════════════════════════════
def _main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="JARVIS open-vocabulary Hebrew ASR")
    ap.add_argument("--build", action="store_true", help="rebuild model + lexicon")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="decode voicebank words (partly circular — wiring check)")
    ap.add_argument("--held-out", action="store_true",
                    help="decode unseen words via TTS (the number that matters)")
    ap.add_argument("--silence", action="store_true")
    ap.add_argument("--text", help="synthesise TEXT then decode it (round trip)")
    ap.add_argument("--wav", help="decode a WAV file")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(list(argv))

    eng = AsrEngine()
    if a.build or not eng.load():
        if not eng.build(verbose=True):
            print("[asr] build FAILED")
            return 1
    if a.stats:
        print(json.dumps(eng.stats(), ensure_ascii=False, indent=1))
    if a.text:
        from voice.tts import speak
        res = speak(a.text)
        x = np.asarray(res.samples, dtype=np.float32)
        r = eng.transcribe(x, int(res.sample_rate))
        print(f"  said  : {a.text}")
        print(f"  heard : {r.text}")
        print(f"  phones: {r.phones}")
        print(f"  conf  : {r.confidence:.3f}  frames={r.n_frames}")
    if a.wav:
        r = eng.transcribe_file(a.wav)
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=1))
    if a.self_test:
        print("self (circular):", json.dumps(self_report(eng), ensure_ascii=False))
    if a.held_out:
        print("held out      :", json.dumps(held_out_report(eng), ensure_ascii=False,
                                            indent=1))
    if a.silence:
        print("silence       :", json.dumps(silence_report(eng), ensure_ascii=False,
                                            indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
