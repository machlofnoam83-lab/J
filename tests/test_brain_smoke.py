"""Smoke test for the neural core: tokenizer round-trip + model fwd/bwd/gen.

Run: python tests/test_brain_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch is an optional dependency at runtime — NeuralCore falls back to a
# retrieval path without it — but this file tests the neural core itself, so it
# cannot run without one. A machine without torch therefore gets an explicit
# SKIP and a zero-failure result, not a traceback: crashing the whole suite over
# an absent optional extra would read as a regression when nothing regressed,
# and silently claiming the neural core passed would be worse.
try:
    import torch  # noqa: E402
    from brain.model import JarvisModel, ModelConfig  # noqa: E402
    _TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    JarvisModel = ModelConfig = None  # type: ignore[assignment]
    _TORCH = False

from brain.tokenizer import Tokenizer, normalize, stats  # noqa: E402

CORPUS = [
    "שלום אדוני, כל המערכות פעילות וממתינות להוראות שלך.",
    "JARVIS online. All systems operational.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "כדי לחשב את השורש הריבועי של 144 נשתמש בפונקציה sqrt.",
    "מה השעה עכשיו? השעה היא אחת עשרה ושלושים וחמש דקות.",
    "פתח את המחשבון, סגור את הדפדפן, הצג את מצב הזיכרון.",
    "אני כותב קוד פייתון שבודק אם מספר הוא ראשוני ומחזיר אמת או שקר.",
    "The quick brown fox jumps over the lazy dog. 0123456789 !@#$%^&*()",
]


def main() -> int:
    ok = 0
    fail = 0

    def check(name: str, cond: bool, info: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name} {info}")
        else:
            fail += 1
            print(f"  FAIL  {name} {info}")

    if not _TORCH:
        print("  SKIP  torch is not installed on this machine.")
        print("        The neural core is an optional extra: JARVIS runs on its")
        print("        retrieval fallback without it, and this file tests the core")
        print("        itself, so there is nothing honest left to assert here.")
        print("        Install torch (see requirements.txt) to run these checks.")
        print(f"\nRESULT: {ok} passed, {fail} failed")
        return 0

    print("== tokenizer ==")
    t0 = time.time()
    tok = Tokenizer.train(CORPUS * 80, vocab_size=1500, progress=False)
    dt = time.time() - t0
    print(f"  trained in {dt:.2f}s -> vocab={len(tok)} merges={len(tok.merges)}")
    # vocab is corpus-bounded: base(10 specials + 256 bytes) + learned merges
    check("vocab > base 266 (merges learned)", len(tok) > 266, f"({len(tok)})")

    s = stats(tok, CORPUS)
    print(f"  compression: {s['chars_per_token']} chars/token")
    check("compression < 3 chars/token", s["chars_per_token"] < 3.0)

    for text in CORPUS:
        rt = tok.decode(tok.encode(text))
        check(f"roundtrip {text[:18]!r}...", rt == normalize(text))

    he = "שלום עולם, ג'רוויס כאן."
    check("hebrew+geresh roundtrip", tok.decode(tok.encode(he)) == normalize(he),
          f"-> {tok.decode(tok.encode(he))!r}")

    print("== model ==")
    cfg = ModelConfig.for_size("nano", vocab_size=len(tok))
    model = JarvisModel(cfg)
    params = model.num_params() / 1e6
    print(f"  NANO params: {params:.2f}M")
    check("params in sane range", 1.0 < params < 60.0, f"({params:.2f}M)")

    seq = 64
    batch = torch.stack(
        [torch.tensor((tok.encode(c) * 8)[:seq], dtype=torch.long) for c in CORPUS]
    )
    targets = torch.roll(batch, -1, dims=1)
    loss = model.compute_loss(batch, targets)
    check("forward+loss finite", bool(torch.isfinite(loss).item()), f"(loss={float(loss.detach()):.4f})")

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("backward produced grads", len(grads) > 0 and all(torch.isfinite(g).all() for g in grads))

    model.zero_grad(set_to_none=True)
    t0 = time.time()
    out = model.generate(batch[:1], max_new_tokens=32, temperature=0.9, eos_id=None)
    dt = time.time() - t0
    print(f"  generated {out.shape[1] - seq} tokens in {dt:.2f}s ({(out.shape[1]-seq)/dt:.1f} tok/s)")
    check("generate produced tokens", out.shape[1] > seq)
    print("  sample:", repr(tok.decode(out[0].tolist())[:90]))

    # cacheless decode must equal cached decode (correctness of KV cache)
    torch.manual_seed(7)
    a = model.generate(batch[:1], max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0,
                       repetition_penalty=1.0, eos_id=None, use_cache=True)
    torch.manual_seed(7)
    b = model.generate(batch[:1], max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0,
                       repetition_penalty=1.0, eos_id=None, use_cache=False)
    check("KV-cache == no-cache output", torch.equal(a, b))

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
