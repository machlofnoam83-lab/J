from __future__ import annotations

import argparse

DEFAULT_SYSTEM = "אתה JAI, עוזר אישי חכם, מדויק, מועיל וזהיר. אם אינך יודע, אמור זאת."


def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int = 384, temperature: float = 0.7) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=0.9 if do_sample else None,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with JAI in the terminal")
    parser.add_argument("--model", default="outputs/jai-3b-he")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    from .model import load_for_inference

    model, tokenizer = load_for_inference(args.model, load_in_4bit=not args.no_4bit)
    messages = [{"role": "system", "content": args.system}]
    print("JAI מוכן. כתוב /clear לניקוי השיחה או /exit ליציאה.")
    while True:
        try:
            text = input("אתה: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/exit":
            break
        if text == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("השיחה נוקתה.")
            continue
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        reply = generate_reply(model, tokenizer, messages, args.max_new_tokens, args.temperature)
        print(f"JAI: {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
