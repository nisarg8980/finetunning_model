import argparse
import json
import math
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from env_setup import load_env

load_env()  # read HF_TOKEN from .env so it need not be set on the command line


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned Mistral model.")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter_dir", required=True, help="Path to the trained LoRA adapter.")
    p.add_argument("--eval_file", default=None, help="Held-out JSONL for perplexity.")
    p.add_argument("--prompts_file", default=None, help="JSONL/txt of test prompts.")
    p.add_argument("--compare_base", action="store_true", help="Also evaluate the base model.")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--report", default="eval_report.md")
    return p.parse_args()


def normalize_to_messages(example: dict) -> list:
    if example.get("messages"):
        return example["messages"]
    instruction = (example.get("instruction") or "").strip()
    extra = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    user = f"{instruction}\n\n{extra}".strip() if extra else instruction
    msgs = [{"role": "user", "content": user}]
    if output:
        msgs.append({"role": "assistant", "content": output})
    return msgs


def load_eval_texts(path: str, tokenizer) -> list:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = normalize_to_messages(json.loads(line))
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    return texts


def load_prompts(path: str) -> list:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                prompts.append(line)  # treat as raw text prompt
                continue
            if "prompt" in obj:
                prompts.append(obj["prompt"])
            elif obj.get("messages"):
                first_user = next((m["content"] for m in obj["messages"] if m["role"] == "user"), "")
                prompts.append(first_user)
            else:
                instruction = (obj.get("instruction") or "").strip()
                extra = (obj.get("input") or "").strip()
                prompts.append(f"{instruction}\n\n{extra}".strip() if extra else instruction)
    return prompts


def load_model(base_model: str, adapter_dir: str | None, token):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", token=token
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


@torch.no_grad()
def perplexity(model, tokenizer, texts: list, max_len: int) -> float:
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(model.device)
        if input_ids.size(1) < 2:
            continue
        out = model(input_ids, labels=input_ids)
        n_tokens = input_ids.size(1) - 1  # shifted targets
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def generate(model, tokenizer, prompts: list, max_new_tokens: int) -> list:
    answers = []
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        answers.append(tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip())
    return answers


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")

    tok_src = args.adapter_dir if os.path.isdir(args.adapter_dir) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_texts = load_eval_texts(args.eval_file, tokenizer) if args.eval_file else []
    prompts = load_prompts(args.prompts_file) if args.prompts_file else []

    lines = ["# Fine-tuning evaluation report\n"]

    # ---- Fine-tuned model ----
    print("Loading fine-tuned model...")
    ft_model = load_model(args.base_model, args.adapter_dir, token)
    ft_ppl = perplexity(ft_model, tokenizer, eval_texts, args.max_seq_len) if eval_texts else None
    ft_gens = generate(ft_model, tokenizer, prompts, args.max_new_tokens) if prompts else []
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Base model (optional) ----
    base_ppl, base_gens = None, []
    if args.compare_base:
        print("Loading base model for comparison...")
        base_model = load_model(args.base_model, None, token)
        base_ppl = perplexity(base_model, tokenizer, eval_texts, args.max_seq_len) if eval_texts else None
        base_gens = generate(base_model, tokenizer, prompts, args.max_new_tokens) if prompts else []
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Perplexity section ----
    if eval_texts:
        lines.append("## Perplexity on held-out set (lower is better)\n")
        lines.append(f"- Fine-tuned: **{ft_ppl:.3f}**")
        if base_ppl is not None:
            delta = base_ppl - ft_ppl
            verdict = "improved" if delta > 0 else "no improvement / worse"
            lines.append(f"- Base: {base_ppl:.3f}")
            lines.append(f"- Change: {delta:+.3f} ({verdict})")
        lines.append("")

    # ---- Generations section ----
    if prompts:
        lines.append("## Sample generations\n")
        for i, prompt in enumerate(prompts):
            lines.append(f"### Prompt {i + 1}\n")
            lines.append(f"> {prompt}\n")
            lines.append(f"**Fine-tuned:**\n\n{ft_gens[i]}\n")
            if base_gens:
                lines.append(f"**Base:**\n\n{base_gens[i]}\n")

    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nReport written to: {args.report}")

    _save_perplexity_chart(args.report, ft_ppl, base_ppl, len(eval_texts))


def _save_perplexity_chart(report_path, ft_ppl, base_ppl, n_items):
    """Save a perplexity bar chart next to the report. Never fatal."""
    if ft_ppl is None:
        return  # no eval set -> nothing to plot
    try:
        from viz_utils import grouped_bar, chart_path_for, COLOR_BASE, COLOR_FT

        if base_ppl is not None:
            series = {"Base (before)": [base_ppl], "Fine-tuned (after)": [ft_ppl]}
        else:
            series = {"Fine-tuned": [ft_ppl]}
        out = chart_path_for(report_path)
        grouped_bar(
            out,
            title="Perplexity on held-out set",
            group_labels=["Perplexity"],
            series=series,
            colors=[COLOR_BASE, COLOR_FT],
            ylabel="perplexity",
            value_fmt="{:.2f}",
            footer=f"{n_items} held-out examples  -  LOWER is better",
        )
        print(f"Chart written to: {out}")
    except Exception as exc:
        print(f"(chart skipped: {exc})")


if __name__ == "__main__":
    main()
