from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure held-out loss and generate test answers")
    parser.add_argument("--model", default="outputs/jai-3b-he")
    parser.add_argument("--data", default="data/eval.jsonl")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from .chat import generate_reply
    from .data import CausalLMCollator, read_jsonl, tokenize_record
    from .model import load_for_inference

    model, tokenizer = load_for_inference(args.model, load_in_4bit=not args.no_4bit)
    records = read_jsonl(args.data)
    encoded = [tokenize_record(item, tokenizer, args.max_length, "assistant_only") for item in records]
    loader = DataLoader(encoded, batch_size=1, collate_fn=CausalLMCollator(tokenizer))
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            losses.append(model(**batch).loss.float().item())
    mean_loss = sum(losses) / len(losses)

    generations = []
    for record in records[:20]:
        if "messages" not in record:
            continue
        prompt_messages = record["messages"]
        while prompt_messages and prompt_messages[-1]["role"] == "assistant":
            prompt_messages = prompt_messages[:-1]
        generations.append({
            "prompt": prompt_messages,
            "answer": generate_reply(model, tokenizer, prompt_messages, temperature=0.0),
        })
    report = {
        "model": args.model,
        "examples": len(records),
        "assistant_token_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "generations": generations,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "generations"}, indent=2))
    print(f"Full report: {destination}")


if __name__ == "__main__":
    main()
