#!/usr/bin/env python
"""
Measure severity-classification accuracy of the fine-tuned model, optionally against
the base model, on a held-out triage set. This produces the headline POC number:
"base model X% accuracy vs fine-tuned Y%".

Usage:
    export HF_TOKEN=...
    python eval_triage_accuracy.py \
        --adapter_dir ../mistral7b-qlora-out/adapter \
        --eval_file triage_eval.jsonl \
        --compare_base
"""

import argparse
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LABELS = ["Critical", "High", "Medium", "Low", "Informational"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triage classification accuracy eval.")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--compare_base", action="store_true")
    p.add_argument("--report", default="poc_triage_report.md")
    return p.parse_args()


def load_eval(path: str):
    """Return list of (user_prompt, gold_label)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
            items.append((user, extract_label(assistant)))
    return items


def extract_label(text: str):
    """Find the severity label in a model output or gold answer."""
    m = re.search(r"severity\s*[:=]\s*(critical|high|medium|low|informational)", text, re.I)
    if m:
        return m.group(1).capitalize() if m.group(1).lower() != "informational" else "Informational"
    # fall back to first label mentioned anywhere
    for label in LABELS:
        if re.search(rf"\b{label}\b", text, re.I):
            return label
    return None


def load_model(base_model: str, adapter_dir, token):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", token=token
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, tokenizer, items):
    correct = 0
    per_class = {label: [0, 0] for label in LABELS}  # [correct, total]
    for user, gold in items:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        pred = extract_label(tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True))
        if gold in per_class:
            per_class[gold][1] += 1
            if pred == gold:
                per_class[gold][0] += 1
                correct += 1
    accuracy = correct / len(items) if items else 0.0
    return accuracy, per_class


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")

    tok_src = args.adapter_dir if os.path.isdir(args.adapter_dir) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = load_eval(args.eval_file)
    lines = ["# Triage POC: severity classification accuracy\n",
             f"Held-out examples: {len(items)}\n"]

    print("Evaluating fine-tuned model...")
    ft_model = load_model(args.base_model, args.adapter_dir, token)
    ft_acc, ft_per = evaluate(ft_model, tokenizer, items)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_acc, base_per = None, None
    if args.compare_base:
        print("Evaluating base model...")
        base_model = load_model(args.base_model, None, token)
        base_acc, base_per = evaluate(base_model, tokenizer, items)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lines.append("## Overall accuracy\n")
    lines.append(f"- Fine-tuned: **{ft_acc:.1%}**")
    if base_acc is not None:
        lines.append(f"- Base: {base_acc:.1%}")
        lines.append(f"- Improvement: {ft_acc - base_acc:+.1%}")
    lines.append("\n## Per-class accuracy (fine-tuned)\n")
    for label in LABELS:
        c, t = ft_per[label]
        lines.append(f"- {label}: {c}/{t}" + (f" ({c / t:.0%})" if t else ""))

    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
