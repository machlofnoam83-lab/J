from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "outputs/jai-3b-he"
    train_file: str = "data/train.jsonl"
    eval_file: str | None = "data/eval.jsonl"
    training_mode: str = "assistant_only"
    max_length: int = 512
    max_steps: int = 500
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False

    def validate(self) -> None:
        if self.training_mode not in {"assistant_only", "all_tokens"}:
            raise ValueError("training_mode must be 'assistant_only' or 'all_tokens'")
        if self.max_length < 64:
            raise ValueError("max_length must be at least 64")
        if self.lora_rank <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("LoRA rank and gradient accumulation must be positive")


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    valid = {field.name for field in fields(TrainConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    config = TrainConfig(**raw)
    config.validate()
    return config
