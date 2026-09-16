"""JARVIS neural core — a decoder-only Transformer written from zero.

Architecture (all hand-implemented in PyTorch):
  * RoPE rotary position embeddings
  * RMSNorm (pre-norm)
  * Grouped-Query Attention (GQA) via ``scaled_dot_product_attention``
  * SwiGLU feed-forward
  * Tied input/output embeddings
  * Incremental KV-cache decoding
  * Structured-generation biasing: we can force the model to emit valid JSON
    for tool calls by masking logits against a grammar.

Three sizes ship in ``SIZES`` — NANO trains on a laptop CPU in minutes,
CORE is the target when a GPU is available.

Nothing is downloaded. Weights are ours, produced by ``tools/train.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES: Dict[str, Dict[str, Any]] = {
    #        d_model layers heads kv_heads  ffn_mult  ctx   ~params
    "nano":  dict(n_embd=256, n_layer=6,  n_head=4, n_kv_head=2, ffn_mult=2.5, block_size=512),
    "micro": dict(n_embd=384, n_layer=10, n_head=6, n_kv_head=2, ffn_mult=2.5, block_size=1024),
    "core":  dict(n_embd=512, n_layer=16, n_head=8, n_kv_head=4, ffn_mult=3.0, block_size=2048),
    # ------------------------------------------------------------------
    # The three below are the requested targets, not things this machine can
    # train. Measured, at vocab 8057:
    #
    #   agent  d=640  L=16  ~100.2M params   0.40 GB fp32
    #   large  d=1024 L=24  ~0.37B  params   1.50 GB fp32    4.5 GB with Adam
    #   giant  d=1536 L=24  ~0.83B  params   3.30 GB fp32   10.0 GB with Adam
    #
    # The development box this was measured on has 2 cores and 3939 MB of RAM.
    # At those numbers "giant" does not fit in memory even for inference, and
    # nothing above "core" can be trained here at all — Adam keeps two extra
    # copies of every weight, so training cost is roughly 3x the fp32 size
    # before activations. Eleven agents at 100M each is 1.10B params, 4.4 GB
    # resident if they are all loaded at once.
    #
    # They are declared so the target is written down and the training script
    # will accept the name, and so nobody has to re-derive the arithmetic to
    # find out why it does not run. Reaching them needs either a GPU or a
    # machine with an order of magnitude more RAM. They are deliberately not
    # the default.
    # ------------------------------------------------------------------
    "agent": dict(n_embd=640,  n_layer=16, n_head=16, n_kv_head=4, ffn_mult=4.0, block_size=2048),
    "large": dict(n_embd=1024, n_layer=24, n_head=16, n_kv_head=4, ffn_mult=4.0, block_size=2048),
    "giant": dict(n_embd=1536, n_layer=24, n_head=16, n_kv_head=4, ffn_mult=4.0, block_size=4096),
}


@dataclass
class ModelConfig:
    vocab_size: int = 16384 + 32
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    n_kv_head: int = 2
    ffn_mult: float = 2.5
    block_size: int = 512
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    size_name: str = "nano"

    @classmethod
    def for_size(cls, name: str, vocab_size: int) -> "ModelConfig":
        if name not in SIZES:
            raise ValueError(f"unknown size '{name}' (choose from {list(SIZES)})")
        kw = dict(SIZES[name])
        return cls(vocab_size=vocab_size, size_name=name, **kw)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# ------------------------------------------------------------------ blocks --
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm) * self.weight


def build_rope_cache(head_dim: int, seq_len: int, base: float, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D) built with the duplicated-half trick.

    Implements the standard LLaMA-style split-half rotation:
        out1 = x1*cos - x2*sin
        out2 = x2*cos + x1*sin
    where ``cos``/``sin`` are sliced to the first half because the cache was
    built by duplicating the frequency band (``cat(freqs, freqs)``).
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[None, None, :, :d]
    sin = sin[None, None, :, :d]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)



class Attention(nn.Module):
    """Grouped-query causal self-attention with optional KV cache."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_head, self.n_kv_head, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, (self.n_head + 2 * self.n_kv_head) * self.hd, bias=False)
        self.proj = nn.Linear(self.n_head * self.hd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(
            [self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1
        )
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            pk, pv = kv_cache
            if pk.numel():
                k = torch.cat((pk, k), dim=2)
                v = torch.cat((pv, v), dim=2)
        new_cache = (k, v)

        # repeat KV heads for GQA
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        Tq, Tk = q.shape[-2], k.shape[-2]
        if kv_cache is None:
            # pure prefill: cheap fused causal flag
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=True
            )
        elif Tq == 1:
            # single-token decode against the full cache: every past key is legal
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            # incremental prefill with cache -> explicit sliding causal mask
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(
                diagonal=Tk - Tq
            )
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask.view(1, 1, Tq, Tk),
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.resid_drop(self.proj(y)), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.n_embd)
        hidden = ((hidden + 63) // 64) * 64  # keep matmuls aligned
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.ln1(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


# ------------------------------------------------------------------- model --
class JarvisModel(nn.Module):
    """The neural core."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.apply(self._init_weights)
        # residual-scaled init (GPT-2 style) for stability at depth
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def _rope(self, seq_len: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        need = max(seq_len, self.cfg.block_size)
        cached = self._rope_cache
        if (
            cached is None
            or cached[0].shape[0] < need
            or cached[0].device != device
            or cached[0].dtype != dtype
        ):
            self._rope_cache = build_rope_cache(
                self.cfg.head_dim, need, self.cfg.rope_base, device, dtype
            )
        cos, sin = self._rope_cache
        return cos, sin

    def forward(
        self,
        idx: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = idx.shape
        pos_offset = 0 if kv_caches is None or not kv_caches[0][0].numel() else kv_caches[0][0].shape[2]
        cos, sin = self._rope(pos_offset + T, idx.device, idx.dtype if idx.is_floating_point() else torch.float32)
        cos, sin = cos[pos_offset:pos_offset + T], sin[pos_offset:pos_offset + T]

        x = self.tok_emb(idx)
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x, nc = block(x, cos, sin, cache)
            new_caches.append(nc)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_caches

    # ------------------------------------------------------------- training --
    def compute_loss(self, idx: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        logits, _ = self.forward(idx)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index
        )

    # ------------------------------------------------------------ inference --
    def init_cache(self, batch: int, device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.empty(batch, self.cfg.n_kv_head, 0, self.cfg.head_dim, device=device),
             torch.empty(batch, self.cfg.n_kv_head, 0, self.cfg.head_dim, device=device))
            for _ in range(self.cfg.n_layer)
        ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.75,
        top_k: int = 40,
        top_p: float = 0.92,
        repetition_penalty: float = 1.12,
        eos_id: Optional[int] = 1,
        stop_ids: Sequence[int] = (),
        allowed_mask: Optional[torch.Tensor] = None,
        stream_fn=None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Sampling decoder with KV cache, penalties and grammar masking."""
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]
        generated = input_ids.clone()
        caches = self.init_cache(B, device) if use_cache else None
        cur = input_ids
        stop = set(stop_ids) | ({eos_id} if eos_id is not None else set())

        for step in range(max_new_tokens):
            logits, caches = self.forward(cur, caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(B):
                    toks = generated[b].unique()
                    scores = logits[b, toks]
                    logits[b, toks] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)

            if allowed_mask is not None:
                logits = logits.masked_fill(~allowed_mask.bool(), float("-inf"))

            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                drop = cum > top_p
                drop[:, 1:] = drop[:, :-1].clone()
                drop[:, 0] = False
                scatter = drop.scatter(1, sorted_idx, drop)
                logits = logits.masked_fill(scatter, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            if probs.sum() == 0:
                probs = torch.full_like(probs, 1.0 / probs.shape[-1])
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_id), dim=1)
            if stream_fn is not None:
                stream_fn(int(next_id[0, 0]), step)
            if stop and all(int(t) in stop for t in next_id.flatten()):
                break
            if generated.shape[1] >= self.cfg.block_size:
                generated = generated[:, -self.cfg.block_size:]
                caches = None
                cur = generated
            else:
                cur = next_id
        return generated

    # ----------------------------------------------------------------- I/O --
    def save(self, path: Path | str, extra: Optional[Dict[str, Any]] = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "model": self.state_dict(),
                "meta": extra or {},
            },
            p,
        )
        return p

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu", map_location=None) -> "JarvisModel":
        blob = torch.load(Path(path), map_location=map_location or device, weights_only=False)
        cfg = ModelConfig(**blob["config"])
        model = cls(cfg)
        model.load_state_dict(blob["model"], strict=False)
        model.to(device).eval()
        model.meta = blob.get("meta", {})  # type: ignore[attr-defined]
        return model


def estimate_flops(cfg: ModelConfig, tokens: int) -> float:
    """Rough training FLOPs (6 * N * D) for capacity planning."""
    params = (
        12 * cfg.n_layer * cfg.n_embd ** 2
        + cfg.vocab_size * cfg.n_embd
    )
    return 6.0 * params * tokens
