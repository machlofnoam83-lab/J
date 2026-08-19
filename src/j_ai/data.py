from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VALID_ROLES = {"system", "user", "assistant"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            validate_record(record, source=f"{path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No usable records found in {path}")
    return records


def validate_record(record: dict[str, Any], source: str = "record") -> None:
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{source}: messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError(f"{source}: each message must be an object")
            if message.get("role") not in VALID_ROLES:
                raise ValueError(f"{source}: invalid role {message.get('role')!r}")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"{source}: message content must be non-empty text")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError(f"{source}: conversation has no assistant answer")
    elif "text" in record:
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"{source}: text must be non-empty")
    else:
        raise ValueError(f"{source}: expected a 'messages' or 'text' field")


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for record in records:
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if digest not in hashes:
            hashes.add(digest)
            unique.append(record)
    return unique


def tokenize_record(record: dict[str, Any], tokenizer: Any, max_length: int, mode: str) -> dict:
    """Tokenize one record, masking non-assistant chat tokens when requested."""
    if "text" in record:
        encoded = tokenizer(
            record["text"], truncation=True, max_length=max_length, add_special_tokens=True
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    messages = record["messages"]
    if mode == "all_tokens":
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}

    # Build cumulative prefixes. Chat templates such as Qwen's are prefix-stable, so
    # each newly appended message maps to one contiguous token span.
    input_ids: list[int] = []
    labels: list[int] = []
    previous: list[int] = []
    for index, message in enumerate(messages):
        current = tokenizer.apply_chat_template(
            messages[: index + 1], tokenize=True, add_generation_prompt=False
        )
        common = 0
        while common < min(len(previous), len(current)) and previous[common] == current[common]:
            common += 1
        # In the unlikely event a template rewrites its suffix, preserve correct text and masking.
        if common < len(previous):
            input_ids = input_ids[:common]
            labels = labels[:common]
        added = current[common:]
        input_ids.extend(added)
        labels.extend(added if message["role"] == "assistant" else [-100] * len(added))
        previous = current

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError("Record has no assistant tokens after truncation; shorten the prompt")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class CausalLMCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        inputs, masks, labels = [], [], []
        for item in features:
            padding = max_len - len(item["input_ids"])
            inputs.append(item["input_ids"] + [pad_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
