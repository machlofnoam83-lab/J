#!/usr/bin/env python3
"""The size table — checked without torch.

``brain/model.py`` imports torch at module level, so on a machine without it
(this sandbox: the PyTorch index is unreachable and torch cannot be installed)
``SIZES`` cannot be imported at all. Rather than let the size table go
unverified, this file parses it out of the source with ``ast`` and checks the
arithmetic against the module structure the sizes are actually built from:

    Attention  qkv  = n_embd * (n_head + 2*n_kv_head) * head_dim      bias=False
    Attention  proj = n_head * head_dim * n_embd                      bias=False
    SwiGLU     hidden = align64(int(ffn_mult * n_embd))
               gate/up/down = 3 * n_embd * hidden                     bias=False
    Block      ln1 + ln2 = 2 * n_embd
    Embedding  vocab * n_embd, tied to the output projection

Honest limit: this verifies the *table*, not a constructed model. The real
``numel()`` check lives in ``JarvisModel.num_params`` and needs torch, so on a
machine that has one this is a weaker guarantee than it would be. What it does
guarantee is that the doubling the table claims is the doubling the architecture
would produce — which is the claim that is easy to get wrong by hand.

Run: python tests/test_model_sizes.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "brain" / "model.py"
VOCAB = 8057          # tokenizer.json, measured

ok = 0
fail = 0


def check(label: str, cond: bool, detail: str = "") -> bool:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}" + (f"  {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def _sizes() -> dict:
    """Pull SIZES out of the source without importing torch.

    ``dict(n_embd=256, ...)`` has to become ``{"n_embd": 256, ...}`` — with the
    keys quoted, or ``literal_eval`` rejects them as bare names.
    """
    import re
    src = MODEL.read_text(encoding="utf-8")
    m = re.search(r"^SIZES: Dict\[str, Dict\[str, Any\]\] = (\{.*?^\})", src,
                  re.S | re.M)
    if not m:
        raise AssertionError("could not locate SIZES in brain/model.py")

    def to_dict_lit(mm: "re.Match") -> str:
        inner = mm.group(1)
        pairs = []
        for part in inner.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            pairs.append(f'"{k.strip()}": {v.strip()}')
        return "{" + ", ".join(pairs) + "}"

    body = re.sub(r"dict\(([^)]*)\)", to_dict_lit, m.group(1))
    return ast.literal_eval(body)


def align64(h: int) -> int:
    """SwiGLU rounds hidden up to a multiple of 64 to keep matmuls aligned."""
    return ((h + 63) // 64) * 64


def params(d: int, L: int, nh: int, kv: int, ff: float, vocab: int = VOCAB,
           tie: bool = True) -> int:
    hd = d // nh
    qkv = d * (nh + 2 * kv) * hd
    proj = nh * hd * d
    hidden = align64(int(ff * d))
    ffn = 3 * d * hidden
    per_layer = qkv + proj + ffn + 2 * d
    emb = vocab * d
    return emb + L * per_layer + (0 if tie else emb)


def p(cfg: dict) -> int:
    return params(cfg["n_embd"], cfg["n_layer"], cfg["n_head"],
                  cfg["n_kv_head"], cfg["ffn_mult"])


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — model size table")
    print("═" * 68)

    S = _sizes()
    print(f"\n— the table —")
    for name, cfg in S.items():
        print(f"    {name:7} d={cfg['n_embd']:5} L={cfg['n_layer']:3} "
              f"nh={cfg['n_head']:2} kv={cfg['n_kv_head']} ff={cfg['ffn_mult']} "
              f"-> {p(cfg)/1e6:8.3f}M")

    print(f"\n— structural validity —")
    for name, cfg in S.items():
        d, nh, kv = cfg["n_embd"], cfg["n_head"], cfg["n_kv_head"]
        check(f"{name}: n_head divides n_embd", d % nh == 0, f"hd={d//nh}")
        check(f"{name}: n_kv_head divides n_head", nh % kv == 0)
        check(f"{name}: head_dim is at least 32", d // nh >= 32, f"hd={d//nh}")
        check(f"{name}: block_size is positive", cfg["block_size"] > 0)

    print(f"\n— the doubling —")
    check("nano is present", "nano" in S)
    check("nano2x is present", "nano2x" in S)
    if "nano" in S and "nano2x" in S:
        base, doubled = p(S["nano"]), p(S["nano2x"])
        ratio = doubled / base
        print(f"    nano   = {base:,} params")
        print(f"    nano2x = {doubled:,} params")
        print(f"    ratio  = {ratio:.6f}x")
        check("nano2x is exactly double nano", abs(ratio - 2.0) < 0.0005,
              f"{ratio:.6f}x")
        check("the doubling did not change depth",
              S["nano2x"]["n_layer"] == S["nano"]["n_layer"],
              f"L={S['nano2x']['n_layer']}")
        check("the doubling did not change context",
              S["nano2x"]["block_size"] == S["nano"]["block_size"],
              f"ctx={S['nano2x']['block_size']}")
        check("nano2x keeps a healthy head_dim",
              S["nano2x"]["n_embd"] // S["nano2x"]["n_head"] >= 64,
              f"hd={S['nano2x']['n_embd'] // S['nano2x']['n_head']}")

    print(f"\n— the 64-alignment the arithmetic must respect —")
    for name, cfg in S.items():
        raw = int(cfg["ffn_mult"] * cfg["n_embd"])
        check(f"{name}: hidden already 64-aligned (no silent rounding)",
              raw == align64(raw), f"{raw} -> {align64(raw)}")

    print(f"\n— the config default must name a real size —")
    cfg_src = (ROOT / "core" / "config.py").read_text(encoding="utf-8")
    import re as _re
    m = _re.search(r'^\s*size: str = "([^"]+)"', cfg_src, _re.M)
    default = m.group(1) if m else None
    check("core/config.py declares a default size", default is not None,
          f"default={default}")
    check("the default names a size that exists", default in S,
          f"{default} in {sorted(S)}")

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
