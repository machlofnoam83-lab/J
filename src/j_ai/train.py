from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA-train the JAI 3B model")
    parser.add_argument("--config", default="configs/train_4060.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint directory, or 'latest'")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    from .config import load_config
    from .data import CausalLMCollator, read_jsonl, tokenize_record
    from .model import load_base_model, load_tokenizer

    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Training a 3B model requires an NVIDIA CUDA GPU")
    set_seed(config.seed)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base model: {config.model_name} | mode: {config.training_mode}")

    tokenizer = load_tokenizer(config.model_name)
    model = load_base_model(
        config.model_name,
        load_in_4bit=config.load_in_4bit,
        trainable=True,
        use_flash_attention=config.use_flash_attention,
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def build_dataset(filename: str) -> Dataset:
        raw = Dataset.from_list(read_jsonl(filename))
        return raw.map(
            lambda record: tokenize_record(
                record, tokenizer, config.max_length, config.training_mode
            ),
            remove_columns=raw.column_names,
            desc=f"Tokenizing {filename}",
        )

    train_dataset = build_dataset(config.train_file)
    eval_dataset = build_dataset(config.eval_file) if config.eval_file else None
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    training_args = TrainingArguments(
        output_dir=str(output),
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=2,
        optim="paged_adamw_8bit" if config.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    checkpoint = True if args.resume == "latest" else args.resume
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    trainer.save_metrics("train", result.metrics)
    print(f"Adapter saved to {output}")


if __name__ == "__main__":
    main()
