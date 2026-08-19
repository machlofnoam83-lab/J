from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import deduplicate, read_jsonl


def chunk_text(text: str, chunk_chars: int) -> list[dict[str, str]]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[dict[str, str]] = []
    buffer = ""
    for paragraph in paragraphs:
        if buffer and len(buffer) + len(paragraph) + 2 > chunk_chars:
            chunks.append({"text": buffer})
            buffer = ""
        if len(paragraph) > chunk_chars:
            for start in range(0, len(paragraph), chunk_chars):
                piece = paragraph[start : start + chunk_chars].strip()
                if piece:
                    chunks.append({"text": piece})
        else:
            buffer = f"{buffer}\n\n{paragraph}".strip()
    if buffer:
        chunks.append({"text": buffer})
    return chunks


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, deduplicate, and split JAI data")
    parser.add_argument("inputs", nargs="+", help="JSONL chat files or UTF-8 .txt files")
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--eval-out", default="data/eval.jsonl")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records: list[dict] = []
    for filename in args.inputs:
        path = Path(filename)
        if path.suffix.lower() == ".txt":
            records.extend(chunk_text(path.read_text(encoding="utf-8"), args.chunk_chars))
        else:
            records.extend(read_jsonl(path))
    records = deduplicate(records)
    if len(records) < 2:
        raise ValueError("At least 2 unique records are required to make train/eval splits")
    random.Random(args.seed).shuffle(records)
    eval_size = max(1, round(len(records) * args.eval_ratio))
    eval_size = min(eval_size, len(records) - 1)
    write_jsonl(Path(args.train_out), records[eval_size:])
    write_jsonl(Path(args.eval_out), records[:eval_size])
    print(f"Prepared {len(records) - eval_size} train and {eval_size} evaluation records")


if __name__ == "__main__":
    main()
