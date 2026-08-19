from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a JAI LoRA adapter into its base model")
    parser.add_argument("--adapter", default="outputs/jai-3b-he")
    parser.add_argument("--output", default="outputs/jai-3b-merged")
    args = parser.parse_args()

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    print("Merging on CPU; about 12-16 GB of system RAM may be required.")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter, device_map="cpu", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
