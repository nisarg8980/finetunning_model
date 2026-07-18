#!/usr/bin/env python
"""
Chat with your fine-tuned model to test it by hand.

Loads the base Mistral in 4-bit with your LoRA adapter and gives you an interactive
prompt. Type a question, read the answer. Commands: 'reset' clears the conversation,
'exit' or 'quit' leaves.

Usage:
    set HF_TOKEN=...        (Windows)   /   export HF_TOKEN=...   (Linux/Mac)
    python chat.py --adapter_dir ./cyber-qa-out/adapter
    # to feel the difference, load the plain model with no fine-tuning:
    python chat.py --base_only
"""

import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = "You are a helpful cybersecurity assistant."


def fold_system(messages: list) -> list:
    """Merge the system message into the first user turn (Mistral has no system role)."""
    system_text = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"].strip() + "\n\n"
            continue
        rest.append(m)
    if system_text and rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": system_text + rest[0]["content"]}
    elif system_text:
        rest.insert(0, {"role": "user", "content": system_text.strip()})
    return rest


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive chat test for the fine-tuned model.")
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--adapter_dir", default="./cyber-qa-out/adapter")
    ap.add_argument("--base_only", action="store_true", help="Load base model without the adapter.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")

    use_adapter = not args.base_only and os.path.isdir(args.adapter_dir)
    tok_src = args.adapter_dir if use_adapter else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading model (first load can take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", token=token
    )
    if use_adapter:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print(f"Loaded fine-tuned adapter: {args.adapter_dir}")
    else:
        print("Loaded BASE model (no adapter).")
    model.eval()

    history = []
    print("\nType a question. Commands: 'reset' clears chat, 'exit' quits.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        if user.lower() == "reset":
            history = []
            print("(conversation cleared)\n")
            continue

        history.append({"role": "user", "content": user})
        messages = fold_system([{"role": "system", "content": SYSTEM_PROMPT}] + history)
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0][inputs.size(1):], skip_special_tokens=True).strip()
        print(f"Assistant: {answer}\n")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
