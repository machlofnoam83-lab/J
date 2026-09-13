#!/usr/bin/env python3
"""Build the JARVIS voicebank from raw recordings — no external tools.

Pipeline
    1. load raw WAVs (recorded locally, listed in voicebank_manifest.json)
    2. VAD-segment each recording into spoken chunks
    3. align the chunk count to the known token list (merge/split heuristics)
    4. force-align each token to its phonemes with MFCC + banded DTW against a
       reference synthesised by OUR OWN formant engine
    5. write one WAV per unit:
           word  units : brain/voicebank/word_<name>.wav
           phone units : brain/voicebank/ph_<sym>_<left>_<right>.wav
       plus index.json describing coverage

Run:  python tools/build_voicebank.py --raw ~/voicebank_raw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.dsp import (VAD, crossfade, dtw_distance, fade_in_out, mfcc,  # noqa: E402
                       normalize, resample, silence, to_mono)
from voice.tts.formant import synthesize_units, units_from_g2p, SR_DEFAULT  # noqa: E402
from voice.tts.g2p import convert, Phoneme  # noqa: E402

# The six source recordings ship with the repo so the voicebank can be rebuilt
# anywhere; a personal ~/voicebank_raw (e.g. your own re-recorded voice) wins.
_REPO_RAW = Path(__file__).resolve().parent.parent / "voice" / "recordings"
DEFAULT_RAW = (Path.home() / "voicebank_raw") if (Path.home() / "voicebank_raw").exists() else _REPO_RAW
OUT_DIR = ROOT / "brain" / "voicebank"
MANIFEST = ROOT / "voice/tts/voicebank_manifest.json"


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        sr = fh.getframerate()
        n = fh.getnframes()
        ch = fh.getnchannels()
        width = fh.getsampwidth()
        raw = fh.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data.astype(np.float32), sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(to_mono(x), -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm)


def vad_trim(x: np.ndarray, sr: int) -> np.ndarray:
    """Cut leading/trailing silence from a unit so word durations are natural."""
    vad = VAD(sr, hangover=4, threshold_scale=2.6)
    segs = vad.segments(x, min_ms=40.0, pad_ms=12.0)
    if not segs:
        return x
    return x[segs[0][0]: segs[-1][1]]


def slug(text: str) -> str:
    out = re.sub(r"[^\w\u05d0-\u05ea]+", "_", text.strip()).strip("_")
    return out or "unit"


# ---------------------------------------------------------------- segmentation
def _token_weight(token: str) -> float:
    """How long a token should take to say, in phones. Phones, not letters: `את`
    is two letters and one syllable, `הוא` is three letters and two."""
    try:
        n = len([p for p in convert(token) if p.sym and p.sym != "_"])
    except Exception:
        n = 0
    return float(max(1, n or len(token.strip()) or 1))


def segment_recording(x: np.ndarray, sr: int, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Force-align the declared tokens to the recording.

    This used to VAD-segment and then reconcile the *count*: merge the pairs with
    the shortest gaps until the number matched, or split the longest segment at
    its energy minimum. Getting the count right says nothing about getting the
    boundaries right, and the caller then zipped tokens to segments positionally,
    so a single wrong merge shifted every token after it onto somebody else's
    audio — silently, because nothing checked the result.

    Measured on the shipped bank: correlation between a word's duration and its
    length was +0.076, i.e. none. `word_התיקון.wav` held 64 ms of audio for a
    six-letter word, `word_את.wav` held 1.44 s for a two-letter one. JARVIS was
    concatenating the wrong fragments of Hebrew speech, which is what made him
    sound like he was speaking a foreign language.

    The replacement searches for the assignment instead of forcing a count. Cut
    points are offered at energy minima and VAD edges; dynamic programming picks
    N-1 of them so each span's duration best matches the token's expected share
    of the recording, scored by log-ratio so a too-short and a too-long span cost
    the same. Cutting through voiced audio is penalised. One wrong pause can no
    longer shift the rest, because every boundary is chosen against the whole
    duration profile at once.
    """
    n_tok = len(tokens)
    if n_tok == 0 or x.size == 0:
        return []

    win = max(16, int(sr * 0.010))
    n_env = x.size // win
    if n_env < n_tok + 1:
        return []
    env = np.sqrt(np.mean(x[:n_env * win].reshape(n_env, win).astype(np.float64) ** 2, axis=1))
    peak = float(env.max()) or 1e-9

    voiced = env > peak * 0.08
    where = np.flatnonzero(voiced)
    if where.size == 0:
        return []
    s0, s1 = int(where[0]), int(where[-1]) + 1
    if s1 - s0 < n_tok:
        return []

    # Candidate cut points: quiet local minima plus the edges VAD already found.
    cands = {s0, s1}
    for i in range(1, n_env - 1):
        if env[i] <= env[i - 1] and env[i] <= env[i + 1] and env[i] < peak * 0.35:
            cands.add(i)
    try:
        for a, b in VAD(sr).segments(x, min_ms=70.0, pad_ms=15.0):
            cands.add(min(max(a // win, s0), s1))
            cands.add(min(max(b // win, s0), s1))
    except Exception:
        pass
    cands.discard(s1) if s1 >= n_env else None
    order = sorted(c for c in cands if s0 <= c <= min(s1, n_env - 1))
    if len(order) < n_tok + 1:
        return []
    # Keep the quietest candidates so the DP stays cheap on long recordings.
    if len(order) > 700:
        keep = sorted(order, key=lambda i: env[min(i, n_env - 1)])[:700]
        order = sorted(set(keep) | {s0, min(s1, n_env - 1)})
        if len(order) < n_tok + 1:
            return []

    c = np.asarray(order, dtype=np.int64)
    C = c.size
    weights = np.array([_token_weight(t) for t in tokens], dtype=np.float64)
    span_total = float(c[-1] - c[0]) * win
    expect = span_total * weights / max(1e-9, weights.sum())

    # dur[i, j] = samples between candidate i and candidate j
    dur = (c[None, :] - c[:, None]).astype(np.float64) * win
    with np.errstate(divide="ignore", invalid="ignore"):
        logratio = np.abs(np.log(np.where(dur > 0, dur, np.nan) / expect[:, None, None]))
    INF = float("inf")
    dp = np.full((n_tok + 1, C), INF)
    back = np.zeros((n_tok + 1, C), dtype=np.int64)
    dp[0, 0] = 0.0
    voiced_pen = np.where(env[np.clip(c, 0, n_env - 1)] > peak * 0.35, 0.6, 0.0)
    for k in range(1, n_tok + 1):
        cost = logratio[k - 1]                       # [i, j]
        prev = dp[k - 1][:, None]                    # [i, 1]
        tot = prev + cost + voiced_pen[None, :]
        tot = np.where(np.isfinite(tot), tot, INF)
        j_idx = np.arange(C)
        tot = np.where(j_idx[None, :] > j_idx[:, None], tot, INF)
        best = np.argmin(tot, axis=0)
        dp[k] = tot[best, j_idx]
        back[k] = best

    j = int(np.argmin(dp[n_tok]))
    if not np.isfinite(dp[n_tok, j]):
        return []
    out: List[Tuple[int, int]] = []
    k, jj = n_tok, j
    while k > 0:
        i = int(back[k, jj])
        a = int(c[i]) * win
        b = int(c[jj]) * win
        out.append((a, min(x.size, max(b, a + 1))))
        k, jj = k - 1, i
    out.reverse()
    return out


#: a word whose recording is outside this many seconds-per-phone is not that word
PLAUSIBLE_SPP = (0.030, 0.420)


def _quietest_point(x: np.ndarray, sr: int) -> int:
    win = max(32, int(sr * 0.02))
    if len(x) <= 2 * win:
        return len(x) // 2
    lo, hi = win, len(x) - win
    if hi <= lo:
        return len(x) // 2
    e = np.array([np.sqrt(np.mean(x[i:i + win] ** 2)) for i in range(lo, hi, win // 2)])
    return lo + int(np.argmin(e)) * (win // 2)


# ------------------------------------------------------------------ alignment
def phone_boundaries(x: np.ndarray, sr: int, phones: Sequence[Phoneme]) -> List[Tuple[int, int]]:
    """Force-align a recording to its phone sequence using MFCC + banded DTW.

    The reference is synthesised by our own formant engine, so this is fully
    self-supervised: no external aligner, no downloaded model.
    """
    if not phones:
        return []
    units = units_from_g2p(phones)
    if not units:
        return []
    ref = synthesize_units(units, sr=sr)
    feat_real = mfcc(x, sr, n_mfcc=13, delta=False)
    feat_ref = mfcc(ref, sr, n_mfcc=13, delta=False)
    if len(feat_real) == 0 or len(feat_ref) == 0:
        return [(0, len(x))] * len(units)

    path = _dtw_path(feat_real, feat_ref, band=18)
    # durations of each unit in the reference (frames)
    hop = int(sr * 0.010)
    ref_frames = len(feat_ref)
    bounds: List[Tuple[int, int]] = []
    # map reference frames -> unit index by cumulative energy-free duration model
    unit_frames = _unit_frame_counts(units, ref_frames)
    cursor = 0
    spans = []
    for k, nf in enumerate(unit_frames):
        spans.append((cursor, min(ref_frames, cursor + nf)))
        cursor += nf

    real_len = len(x)
    for a, b in spans:
        ra = _map_frame(path, a, real_len)
        rb = _map_frame(path, max(a + 1, b), real_len)
        rb = max(rb, ra + int(sr * 0.02))
        bounds.append((int(ra), int(min(rb, real_len))))
    return bounds


def _unit_frame_counts(units: Sequence[Any], total_frames: int) -> List[int]:
    from voice.tts.formant import DURATIONS, FRIC, GLOTTAL, LIQUIDS, NASALS, STOPS_UV, STOPS_V, VOWELS
    durs: List[float] = []
    for u in units:
        s = u.sym
        if s in VOWELS:
            durs.append(VOWELS[s][3] * (1.28 if u.stressed else 1.0) + u.pause_before)
        elif s in STOPS_V or s in STOPS_UV:
            durs.append(DURATIONS["stop"] + u.pause_before)
        elif s in NASALS:
            durs.append(DURATIONS["nasal"] + u.pause_before)
        elif s in LIQUIDS:
            durs.append(DURATIONS["liquid"] + u.pause_before)
        elif s in FRIC:
            durs.append((DURATIONS["h"] if s == "h" else DURATIONS["fric"]) + u.pause_before)
        elif s in GLOTTAL:
            durs.append(DURATIONS["glottal"] + u.pause_before)
        else:
            durs.append(0.045 + u.pause_before)
    total = sum(durs) or 1.0
    counts = [max(1, int(round(d / total * total_frames))) for d in durs]
    drift = total_frames - sum(counts)
    for i in range(abs(drift)):
        counts[i % len(counts)] += 1 if drift > 0 else -1
    return counts


def _dtw_path(a: np.ndarray, b: np.ndarray, band: int = 18) -> List[Tuple[int, int]]:
    n, m = len(a), len(b)
    INF = float("inf")
    cost = np.full((n + 1, m + 1), INF)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, int(i * m / max(n, 1)) - band)
        hi = min(m, int(i * m / max(n, 1)) + band)
        for j in range(lo, hi + 1):
            d = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # backtrack
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        opts = [(cost[i - 1, j - 1], i - 1, j - 1), (cost[i - 1, j], i - 1, j), (cost[i, j - 1], i, j - 1)]
        _c, i, j = min(opts)
    path.reverse()
    return path


def _map_frame(path: List[Tuple[int, int]], ref_frame: int, real_len: int) -> int:
    """Given a reference frame index, return the corresponding real-sample index."""
    if not path:
        return 0
    ref_frame = max(0, min(ref_frame, path[-1][1]))
    for ri, rj in path:
        if rj >= ref_frame:
            return ri * int(0.010 * 24000)
    return real_len


# ------------------------------------------------------------------- build ---
def build(raw_dir: Path, out_dir: Path, sr: int = 24000, verbose: bool = True) -> Dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {"sample_rate": sr, "words": {}, "phones": {}, "coverage": {}}
    phone_acc: Dict[str, List[np.ndarray]] = defaultdict(list)
    stats = {"recordings": 0, "segments": 0, "words": 0, "phones": 0, "skipped": []}

    for rec in manifest["recordings"]:
        path = raw_dir / rec["file"]
        if not path.exists():
            stats["skipped"].append(f"{rec['file']}: missing")
            continue
        x, orig_sr = read_wav(path)
        x = resample(x, orig_sr, sr)
        tokens = list(rec["tokens"])
        segs = segment_recording(x, sr, tokens)
        stats["recordings"] += 1
        stats["segments"] += len(segs)
        kind = rec.get("kind", "word")
        if verbose:
            print(f"[voicebank] {rec['file']}: aligned {len(segs)}/{len(tokens)} tokens "
                  f"({len(x)/sr:.1f}s audio, kind={kind})")
        if len(segs) != len(tokens):
            # Refuse the recording rather than guess. The old code truncated
            # whichever list was longer and zipped the rest positionally, which is
            # how misaligned audio got published as the wrong words.
            stats["skipped"].append(f"{rec['file']}: aligned {len(segs)} of {len(tokens)} tokens")
            continue
        # A drill of consonant+vowel syllables (`בו גו דו הו וו זו…` read with the
        # vowel u, i.e. "bu gu du hu wu zu…") is good raw material for phone
        # units, but those tokens are not the words they spell: Hebrew `בו` is
        # "bo" and `או` is "o". Registering them as whole words put Mandarin-like
        # syllables into the top of the concat cascade, where a whole recorded
        # word beats any phone-level rendering.
        emit_words = kind != "syllables"

        for token, (s, e) in zip(tokens, segs):
            chunk = fade_in_out(normalize(vad_trim(x[s:e], sr), 0.92), sr, ms=4)
            if len(chunk) < int(sr * 0.05):
                continue
            for word in re.split(r"\s+", token.strip()):
                if not word:
                    continue

                # phone-level force alignment — runs for every kind, including the
                # syllable drill, which exists precisely to yield clean phones
                phones = convert(word)
                phones = [p for p in phones if p.sym and p.sym != "_"]

                if emit_words and phones:
                    spp = (len(chunk) / sr) / len(phones)
                    if PLAUSIBLE_SPP[0] <= spp <= PLAUSIBLE_SPP[1]:
                        key = slug(word)
                        wpath = out_dir / f"word_{key}.wav"
                        write_wav(wpath, chunk, sr)
                        index["words"][word] = {
                            "file": wpath.name, "samples": int(len(chunk)),
                            "duration": round(len(chunk) / sr, 4), "kind": kind,
                            "spp": round(spp, 4),
                        }
                        stats["words"] += 1
                    else:
                        # The span the aligner chose cannot contain this word.
                        # Publishing it would put somebody else's audio in the
                        # bank under this word's name — the exact failure this
                        # builder is being fixed for — so report it instead.
                        stats.setdefault("implausible", []).append(
                            f"{rec['file']}:{word} {spp:.3f}s/phone")

                if not phones:
                    continue
                bounds = phone_boundaries(chunk, sr, phones)
                if len(bounds) != len(phones):
                    continue
                for i, (ps, pe) in enumerate(bounds):
                    if pe - ps < int(sr * 0.012):
                        continue
                    sym = phones[i].sym
                    left = phones[i - 1].sym if i > 0 else "#"
                    right = phones[i + 1].sym if i + 1 < len(phones) else "#"
                    unit = normalize(chunk[ps:pe], 0.92)
                    tag = f"{sym}_{left}_{right}"
                    phone_acc[tag].append(unit)
                    stats["phones"] += 1

    # average duplicates per phone context (reduces recording noise), keep the best
    for tag, chunks in phone_acc.items():
        chunks = [c for c in chunks if len(c) >= int(sr * 0.012)]
        if not chunks:
            continue
        chunks.sort(key=len)
        keep = chunks[len(chunks) // 2]              # median-length take
        sym, left, right = tag.split("_")
        fname = f"ph_{slug(sym)}_{slug(left)}_{slug(right)}.wav"
        write_wav(out_dir / fname, keep, sr)
        index["phones"][tag] = {
            "file": fname, "sym": sym, "left": left, "right": right,
            "takes": len(chunks), "duration": round(len(keep) / sr, 4),
        }

    # coverage report
    from voice.tts.g2p import CONSONANTS, VOWELS
    covered = {v["sym"] for v in index["phones"].values()}
    index["coverage"] = {
        "phones_needed": len(CONSONANTS) + len(VOWELS),
        "phones_covered": len(covered & (set(CONSONANTS) | set(VOWELS))),
        "words": len(index["words"]),
        "phone_units": len(index["phones"]),
        "missing_phones": sorted((set(CONSONANTS) | set(VOWELS)) - covered),
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["coverage"] = index["coverage"]
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser()
    if not raw.exists():
        print(f"[voicebank] raw dir not found: {raw}")
        print("[voicebank] the formant engine still works without it — run with --raw <dir>")
        return 1
    stats = build(raw, Path(args.out), args.sr, verbose=not args.quiet)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
