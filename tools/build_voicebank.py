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

DEFAULT_RAW = Path.home() / "voicebank_raw"
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
def segment_recording(x: np.ndarray, sr: int, expected: int) -> List[Tuple[int, int]]:
    """VAD-segment, then reconcile the count with the expected token count."""
    vad = VAD(sr)
    segs = vad.segments(x, min_ms=70.0, pad_ms=15.0)
    if not segs:
        return []

    # merge segments separated by a very short gap (a single token split in two)
    gap_merge = int(sr * 0.055)
    merged: List[List[int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] < gap_merge:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # if we still have too many, merge the shortest-gap pairs until we match
    while len(merged) > expected and len(merged) > 1:
        gaps = [(merged[i + 1][0] - merged[i][1], i) for i in range(len(merged) - 1)]
        gaps.sort()
        _g, i = gaps[0]
        merged[i][1] = merged[i + 1][1]
        del merged[i + 1]

    # if we have too few, split the longest segments at their energy minimum
    while len(merged) < expected:
        lengths = [(e - s, i) for i, (s, e) in enumerate(merged)]
        lengths.sort(reverse=True)
        if not lengths:
            break
        _L, i = lengths[0]
        s, e = merged[i]
        if e - s < int(sr * 0.16):
            break
        mid = _quietest_point(x[s:e], sr) + s
        merged[i] = [s, mid]
        merged.insert(i + 1, [mid, e])

    return [(s, e) for s, e in merged[:expected]] if len(merged) >= expected else [(s, e) for s, e in merged]


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
        tokens = rec["tokens"]
        segs = segment_recording(x, sr, len(tokens))
        stats["recordings"] += 1
        stats["segments"] += len(segs)
        if verbose:
            print(f"[voicebank] {rec['file']}: {len(segs)}/{len(tokens)} segments "
                  f"({len(x)/sr:.1f}s audio)")
        if len(segs) != len(tokens):
            stats["skipped"].append(f"{rec['file']}: aligned {len(segs)} of {len(tokens)} tokens")
            if len(segs) < len(tokens):
                tokens = tokens[:len(segs)]
            else:
                segs = segs[:len(tokens)]

        for token, (s, e) in zip(tokens, segs):
            chunk = fade_in_out(normalize(vad_trim(x[s:e], sr), 0.92), sr, ms=4)
            if len(chunk) < int(sr * 0.05):
                continue
            for word in re.split(r"\s+", token.strip()):
                if not word:
                    continue
                key = slug(word)
                wpath = out_dir / f"word_{key}.wav"
                write_wav(wpath, chunk, sr)
                index["words"][word] = {
                    "file": wpath.name, "samples": int(len(chunk)),
                    "duration": round(len(chunk) / sr, 4), "kind": rec.get("kind", "word"),
                }
                stats["words"] += 1

                # phone-level force alignment
                phones = convert(word)
                phones = [p for p in phones if p.sym and p.sym != "_"]
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
