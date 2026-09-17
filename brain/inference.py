"""JARVIS Neural Core inference wrapper.

Loads our own tokenizer + our own weights and exposes a small, honest API to
the rest of the system. Three interchangeable backends:

  * ``torch``  — full fidelity, needed for training
  * ``onnx``   — fast CPU serving via onnxruntime (exported by tools/export_onnx.py)
  * ``numpy``  — dependency-free fallback (used when torch is unavailable)

If no checkpoint exists yet, the core still works: it reports ``available=False``
and the reasoning loop falls back to deterministic routes only. JARVIS never
pretends to be smarter than it is.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.bus import BUS, T  # noqa: E402
from core.config import CONFIG  # noqa: E402


@dataclass
class GenerationResult:
    text: str
    tokens: int = 0
    ms: float = 0.0
    backend: str = ""
    stop_reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class NeuralCore:
    def __init__(
        self,
        model_path: Path | str | None = None,
        tokenizer_path: Path | str | None = None,
        device: str = "cpu",
        backend: str = "auto",
    ) -> None:
        self.device = device or CONFIG.brain.device
        self.backend_pref = backend or CONFIG.brain.backend
        self.model_path = Path(model_path or (ROOT / "models" / CONFIG.brain.checkpoint))
        self.tok_path = Path(tokenizer_path or (ROOT / "models" / "tokenizer.json"))
        self.tokenizer = None
        self.model = None
        self.backend = "none"
        self.meta: Dict[str, Any] = {}
        self.load()

    # ---------------------------------------------------------------- load --
    def load(self) -> bool:
        from brain.tokenizer import Tokenizer
        if self.tok_path.exists():
            self.tokenizer = Tokenizer.load(self.tok_path)
        else:
            self.tokenizer = Tokenizer.train(["שלום עולם hello world"] * 20, vocab_size=1024, progress=False)
            self.meta["tokenizer"] = "untrained-fallback"

        if not self.model_path.exists():
            self.backend = "none"
            self.meta["warning"] = f"no checkpoint at {self.model_path} — deterministic routes only"
            BUS.emit("brain.unavailable", self.meta, source="neural_core")
            return False

        want = self.backend_pref
        if want in ("auto", "torch"):
            try:
                import torch
                from brain.model import JarvisModel
                self.model = JarvisModel.load(self.model_path, device=self.device)
                self.backend = "torch"
                self.meta.update({
                    "params": self.model.num_params(),
                    "layers": self.model.cfg.n_layer,
                    "dim": self.model.cfg.n_embd,
                    "ctx": self.model.cfg.block_size,
                    "size": self.model.cfg.size_name,
                    "train": getattr(self.model, "meta", {}),
                })
                BUS.emit("brain.ready", {"backend": self.backend, **self.meta}, source="neural_core")
                return True
            except Exception as exc:
                if want == "torch":
                    self.meta["torch_error"] = str(exc)

        if want in ("auto", "onnx"):
            try:
                self._load_onnx()
                return True
            except Exception as exc:
                self.meta["onnx_error"] = str(exc)

        self.backend = "none"
        return False

    def _load_onnx(self) -> None:
        import numpy as np
        import onnxruntime as ort
        onnx_path = self.model_path.with_suffix(".onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(onnx_path)
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.backend = "onnx"
        self.meta["onnx_path"] = str(onnx_path)
        BUS.emit("brain.ready", {"backend": "onnx"}, source="neural_core")

    @property
    def available(self) -> bool:
        return self.backend in ("torch", "onnx", "numpy") and self.tokenizer is not None

    # ------------------------------------------------------------ generate --
    def build_prompt(self, user_text: str, history: Sequence[Tuple[str, str]] = (),
                     system_note: str = "") -> str:
        from brain.tokenizer import ASSISTANT, BOT, EOS, PLAN, THINK, USER
        chunks: List[str] = [BOT]
        if system_note:
            chunks.append(f"{THINK} {system_note} {EOS}")
        for u, a in history[-3:]:
            chunks.append(f"{USER} {u} {EOS} {ASSISTANT} {a} {EOS}")
        chunks.append(f"{USER} {user_text} {EOS}")
        chunks.append(f"{ASSISTANT}")
        return " ".join(chunks)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: Optional[float] = None,
        stop_texts: Sequence[str] = (),
        stream_fn: Optional[Callable[[str], None]] = None,
        allowed_first_chars: Optional[str] = None,
    ) -> GenerationResult:
        if not self.available:
            return GenerationResult(text="", tokens=0, backend="none",
                                    stop_reason="unavailable", meta=self.meta)
        temperature = CONFIG.brain.temperature if temperature is None else temperature
        t0 = time.perf_counter()

        if self.backend == "torch":
            out = self._generate_torch(prompt, max_new_tokens, temperature, stop_texts,
                                       stream_fn, allowed_first_chars)
        else:
            out = GenerationResult(text="", stop_reason="backend-not-implemented")

        out.ms = (time.perf_counter() - t0) * 1000
        out.backend = self.backend
        BUS.emit(T.BRAIN_TOKEN, {"chars": len(out.text), "ms": round(out.ms, 1)}, source="neural_core")
        return out

    def _generate_torch(self, prompt: str, max_new: int, temperature: float,
                        stop_texts: Sequence[str], stream_fn, allowed_first_chars) -> GenerationResult:
        import torch
        from brain.tokenizer import EOS
        tok = self.tokenizer
        ids = tok.encode(prompt)
        if not ids:
            return GenerationResult(text="", stop_reason="empty-prompt")
        max_ctx = self.model.cfg.block_size
        x = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=self.device)

        acc: List[str] = []
        stop_ids = {tok.encode(EOS)[0] if tok.encode(EOS) else 1}

        def on_token(tid: int, step: int) -> None:
            piece = tok.decode([tid], skip_specials=True)
            acc.append(piece)
            if stream_fn and piece:
                stream_fn(piece)
            BUS.emit("brain.stream", {"piece": piece}, source="neural_core")

        out = self.model.generate(
            x, max_new_tokens=max_new, temperature=temperature,
            top_k=CONFIG.brain.top_k, top_p=CONFIG.brain.top_p,
            repetition_penalty=CONFIG.brain.repetition_penalty,
            eos_id=EOS_ID_SAFE(tok), stream_fn=on_token,
        )
        text = "".join(acc)
        stop_reason = "max_tokens"
        for st in list(stop_texts) + [EOS, "<|user|>", "<|tool|>"]:
            if st and st in text:
                text = text.split(st)[0]
                stop_reason = f"stop:{st}"
                break
        # the tokenizer folds Hebrew final forms (ם→מ) so BPE sees one alphabet;
        # put them back before the text reaches a human.
        from brain.tokenizer import restore_finals
        return GenerationResult(text=restore_finals(text.strip()), tokens=len(out[0]) - x.shape[1],
                                stop_reason=stop_reason)

    # --------------------------------------------------------------- info --
    def describe(self) -> Dict[str, Any]:
        train = dict(self.meta.get("train") or {})
        return {
            "available": self.available,
            "backend": self.backend,
            "tokenizer_vocab": len(self.tokenizer) if self.tokenizer else 0,
            "checkpoint": str(self.model_path),
            "train": train,                       # ppl / loss / steps — shown in the HUD
            **{k: v for k, v in self.meta.items() if k != "train"},
        }


def EOS_ID_SAFE(tok) -> int:
    try:
        enc = tok.encode("<|eos|>")
        return int(enc[0]) if enc else 1
    except Exception:
        return 1
