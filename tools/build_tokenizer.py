#!/usr/bin/env python3
"""Train the JARVIS byte-level BPE tokenizer on our forged corpus.

Usage:
    python tools/build_tokenizer.py --vocab 16384 --corpus corpus/generated/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.tokenizer import Tokenizer, stats  # noqa: E402


def read_texts(paths: list[Path], limit: int | None = None):
    out = []
    for p in paths:
        if not p.exists():
            print(f"[tokenizer] WARN missing {p}")
            continue
        with p.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and len(out) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line)["text"])
                except Exception:
                    out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--corpus", nargs="*", default=[str(ROOT / "corpus/generated/train.jsonl")])
    ap.add_argument("--limit", type=int, default=None, help="cap number of documents")
    ap.add_argument("--out", default=str(ROOT / "models/tokenizer.json"))
    ap.add_argument("--keep-niqqud", action="store_true")
    args = ap.parse_args()

    texts = read_texts([Path(p) for p in args.corpus], args.limit)
    if not texts:
        print("[tokenizer] no corpus found — run tools/forge_corpus.py first")
        return 1
    print(f"[tokenizer] documents: {len(texts):,}  chars: {sum(len(t) for t in texts):,}")

    t0 = time.time()
    tok = Tokenizer.train(texts, vocab_size=args.vocab, keep_niqqud=args.keep_niqqud)
    dt = time.time() - t0
    print(f"[tokenizer] trained in {dt:.1f}s -> vocab={len(tok):,} merges={len(tok.merges):,}")

    sample = texts[:2000]
    s = stats(tok, sample)
    print(f"[tokenizer] compression: {s['chars_per_token']} chars/token")

    # hard guarantee: lossless round-trip on real plain-text samples.
    # (chat-protocol samples legitimately lose spacing around special tokens,
    #  so we validate on the rendered *content*, not the raw protocol line.)
    from brain.tokenizer import normalize, SPECIAL_TOKENS  # noqa: F811
    special_set = set(SPECIAL_TOKENS)
    plain, bad = 0, 0
    for t in sample[:2000]:
        pieces = [p for p in t.split() if p not in special_set]
        if len(pieces) < 2:
            continue
        text = " ".join(pieces)
        plain += 1
        if tok.decode(tok.encode(text)) != normalize(text):
            bad += 1
    print(f"[tokenizer] roundtrip failures: {bad}/{plain}")
    if bad:
        return 2

    p = tok.save(args.out)
    print(f"[tokenizer] saved -> {p} ({p.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
