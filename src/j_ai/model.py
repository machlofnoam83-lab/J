from __future__ import annotations

from pathlib import Path
from typing import Any


def compute_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    *,
    load_in_4bit: bool = True,
    trainable: bool = False,
    use_flash_attention: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available() and load_in_4bit:
        raise RuntimeError("4-bit loading requires a CUDA GPU. Set load_in_4bit: false for CPU.")
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": compute_dtype(),
        "trust_remote_code": False,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not trainable:
        model.eval()
    return model


def load_for_inference(model_or_adapter: str, load_in_4bit: bool = True):
    """Load either a full model or a PEFT adapter directory."""
    from peft import PeftConfig, PeftModel

    path = Path(model_or_adapter)
    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        peft_config = PeftConfig.from_pretrained(str(path))
        base = load_base_model(peft_config.base_model_name_or_path, load_in_4bit=load_in_4bit)
        model = PeftModel.from_pretrained(base, str(path))
        tokenizer = load_tokenizer(str(path) if (path / "tokenizer_config.json").exists() else peft_config.base_model_name_or_path)
    else:
        model = load_base_model(model_or_adapter, load_in_4bit=load_in_4bit)
        tokenizer = load_tokenizer(model_or_adapter)
    model.eval()
    return model, tokenizer
