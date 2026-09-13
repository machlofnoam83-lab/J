#!/usr/bin/env python3
"""Train the JARVIS neural core from zero.

This is the real training loop: document packing, AdamW, warmup + cosine decay,
gradient clipping, periodic eval on held-out data, checkpointing with resume,
and live telemetry pushed onto the event bus so the HUD can draw the loss curve
while the brain is learning.

Usage:
    python tools/train.py --size nano --epochs 1 --batch 8
    python tools/train.py --size core --device cuda --epochs 3      # with a GPU
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from brain.model import JarvisModel, ModelConfig, SIZES, estimate_flops  # noqa: E402
from brain.tokenizer import EOS_ID, Tokenizer  # noqa: E402
from core.bus import BUS, T  # noqa: E402


def load_jsonl(path: Path) -> List[str]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line)["text"])
            except Exception:
                out.append(line)
    return out


class Corpus:
    """Tokenised, packed corpus."""

    def __init__(self, texts: List[str], tok: Tokenizer, block_size: int) -> None:
        self.block_size = block_size
        ids: List[int] = []
        for t in texts:
            ids.extend(tok.encode(t))
            ids.append(EOS_ID)
        self.data = torch.tensor(ids, dtype=torch.long)
        print(f"[train] tokens: {len(ids):,}  (~{len(ids)/1e6:.2f}M)")

    def __len__(self) -> int:
        return max(1, (len(self.data) - 1) // self.block_size)

    def batch(self, idx: torch.Tensor, device: str):
        hi = max(1, len(self.data) - self.block_size - 2)
        starts = torch.randint(0, hi, (len(idx),), dtype=torch.long)
        x = torch.stack([self.data[s: s + self.block_size] for s in starts]).to(device)
        y = torch.stack([self.data[s + 1: s + 1 + self.block_size] for s in starts]).to(device)
        return x, y


def lr_at(step: int, total: int, base: float, warmup: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


@torch.no_grad()
def evaluate(model: JarvisModel, corpus: Corpus, device: str, batches: int = 8) -> dict:
    model.eval()
    losses = []
    for i in range(batches):
        idx = torch.arange(batches) + i * batches
        x, y = corpus.batch(idx, device)
        losses.append(float(model.compute_loss(x, y).item()))
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {"loss": round(loss, 4), "ppl": round(math.exp(min(20, loss)), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="nano", choices=list(SIZES))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from epochs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokenizer", default=str(ROOT / "models/tokenizer.json"))
    ap.add_argument("--train", default=str(ROOT / "corpus/generated/train.jsonl"))
    ap.add_argument("--dev", default=str(ROOT / "corpus/generated/dev.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import os
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:
        pass
    print(f"[train] device={device} threads={torch.get_num_threads()} size={args.size}")

    tok = Tokenizer.load(args.tokenizer) if Path(args.tokenizer).exists() else Tokenizer.train(
        load_jsonl(Path(args.train)) or ["שלום עולם hello world"], vocab_size=4096, progress=False)
    cfg = ModelConfig.for_size(args.size, vocab_size=len(tok) + 16)
    model = JarvisModel(cfg).to(device)
    nparams = model.num_params()
    print(f"[train] params: {nparams/1e6:.2f}M  ctx={cfg.block_size}  layers={cfg.n_layer}")

    texts = load_jsonl(Path(args.train))
    dev_texts = load_jsonl(Path(args.dev))
    if not texts:
        print("[train] no corpus — run: python tools/forge_corpus.py --size medium")
        return 1
    corpus = Corpus(texts, tok, cfg.block_size)
    dev = Corpus(dev_texts, tok, cfg.block_size) if dev_texts else None

    total_steps = args.max_steps or max(50, int(len(corpus) * args.epochs / max(1, args.batch)))
    flops = estimate_flops(cfg, total_steps * args.batch * cfg.block_size)
    print(f"[train] steps={total_steps:,}  est. FLOPs={flops/1e9:.1f} GFLOP")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    start_step = 0
    if args.resume and Path(args.resume).exists():
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"], strict=False)
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        start_step = blob.get("meta", {}).get("step", 0)
        print(f"[train] resumed from {args.resume} @ step {start_step}")

    out_path = Path(args.out) if args.out else (ROOT / "models" / f"jarvis_{args.size}.pt")
    model.train()
    t0, running, log_every = time.time(), 0.0, 20
    history: List[dict] = []

    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, args.lr, args.warmup)
        x, y = corpus.batch(torch.arange(args.batch), device)
        loss = model.compute_loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running = running * 0.95 + float(loss.item()) * 0.05
        BUS.emit("train.step", {"step": step, "loss": round(running, 4),
                                "lr": round(opt.param_groups[0]["lr"], 6)}, source="trainer")

        if not args.quiet and step % log_every == 0:
            dt = time.time() - t0
            tps = (step - start_step + 1) * args.batch * cfg.block_size / max(dt, 1e-6)
            print(f"  step {step:6d}/{total_steps}  loss {running:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {tps:,.0f} tok/s  {dt:.0f}s")

        if dev and step and step % args.eval_every == 0:
            m = evaluate(model, dev, device)
            history.append({"step": step, "train_loss": round(running, 4), **m})
            print(f"  [eval] step {step} dev_loss={m['loss']} ppl={m['ppl']}")
            BUS.emit("train.eval", history[-1], source="trainer")

        if step and step % args.save_every == 0:
            model.save(out_path, {"step": step, "loss": running, "size": args.size,
                                  "tokenizer": str(args.tokenizer), "params": nparams})

    final = evaluate(model, dev, device) if dev else {"loss": running, "ppl": None}
    model.save(out_path, {"step": total_steps, "loss": final["loss"], "ppl": final["ppl"],
                          "size": args.size, "tokenizer": str(args.tokenizer), "params": nparams,
                          "history": history})
    dt = time.time() - t0
    print(f"\n[train] DONE in {dt/60:.1f} min | final dev loss={final['loss']} ppl={final['ppl']}")
    print(f"[train] weights -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    (ROOT / "models" / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
