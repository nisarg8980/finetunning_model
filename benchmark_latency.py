#!/usr/bin/env python
"""
Benchmark inference latency and throughput for the fine-tuned Mistral QLoRA model.

Reports, per prompt and averaged:
  - Time to first token (TTFT): how long you wait before anything appears. This is
    what dominates the "feels slow" perception for short answers.
  - Tokens/sec after the first token: steady-state decode speed, which dominates
    total wait time for long answers.

Use this to decide whether the model is fast enough to deploy as-is, or whether you
need a smaller max_new_tokens, a merged fp16 model instead of 4-bit, or a stronger
GPU. It does not judge answer quality; see evaluate_model.py and
hallucination_check.py for that.

Usage:
    export HF_TOKEN=...
    python benchmark_latency.py \
        --adapter_dir ./cyber-qa-out/adapter \
        --prompts_file data/test_prompts.sample.jsonl \
        --compare_base
"""

import argparse
import json
import os
import statistics
import threading
import time

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

WARMUP_PROMPT = "Say hello in one short sentence."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark latency/throughput of a fine-tuned Mistral model.")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter_dir", default=None, help="Path to the trained LoRA adapter.")
    p.add_argument("--prompts_file", required=True, help="JSONL/txt of prompts to benchmark with.")
    p.add_argument("--compare_base", action="store_true", help="Also benchmark the base model.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_warmup", type=int, default=2, help="Untimed warmup generations before measuring.")
    p.add_argument("--report", default="latency_report.md")
    return p.parse_args()


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


def load_model(base_model: str, adapter_dir, token):
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


def time_one_generation(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """Run one generation, streaming tokens so we can time TTFT separately from decode speed."""
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    first_token_time = None
    text = ""
    for chunk in streamer:
        if first_token_time is None and chunk:
            first_token_time = time.perf_counter()
        text += chunk
    thread.join()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    ttft = (first_token_time - start) if first_token_time else None
    decode_time = (end - first_token_time) if first_token_time else 0.0
    tokens_per_sec = (n_tokens - 1) / decode_time if n_tokens > 1 and decode_time > 0 else None

    return {
        "prompt": prompt,
        "ttft_s": ttft,
        "total_s": end - start,
        "n_tokens": n_tokens,
        "tokens_per_sec": tokens_per_sec,
    }


def benchmark(model, tokenizer, prompts: list, max_new_tokens: int, num_warmup: int) -> list:
    for _ in range(num_warmup):
        time_one_generation(model, tokenizer, WARMUP_PROMPT, max_new_tokens=32)
    return [time_one_generation(model, tokenizer, p, max_new_tokens) for p in prompts]


def summarize(results: list) -> dict:
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    tps = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] is not None]
    return {
        "mean_ttft_s": statistics.mean(ttfts) if ttfts else None,
        "median_ttft_s": statistics.median(ttfts) if ttfts else None,
        "mean_tokens_per_sec": statistics.mean(tps) if tps else None,
        "median_tokens_per_sec": statistics.median(tps) if tps else None,
    }


def format_section(title: str, results: list) -> list:
    summary = summarize(results)
    lines = [f"## {title}\n"]
    if summary["mean_ttft_s"] is not None:
        lines.append(f"- Mean time to first token: **{summary['mean_ttft_s']:.3f} s** "
                     f"(median {summary['median_ttft_s']:.3f} s)")
    if summary["mean_tokens_per_sec"] is not None:
        lines.append(f"- Mean decode speed: **{summary['mean_tokens_per_sec']:.1f} tok/s** "
                     f"(median {summary['median_tokens_per_sec']:.1f} tok/s)")
    lines.append("")
    lines.append("| Prompt | TTFT (s) | Tokens | Tok/s | Total (s) |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        ttft = f"{r['ttft_s']:.3f}" if r["ttft_s"] is not None else "n/a"
        tps = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] is not None else "n/a"
        prompt_preview = r["prompt"][:60].replace("|", "/") + ("..." if len(r["prompt"]) > 60 else "")
        lines.append(f"| {prompt_preview} | {ttft} | {r['n_tokens']} | {tps} | {r['total_s']:.3f} |")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")

    tok_src = args.adapter_dir if args.adapter_dir and os.path.isdir(args.adapter_dir) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = load_prompts(args.prompts_file)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts_file}")

    device_note = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no CUDA GPU detected)"
    lines = ["# Latency benchmark report\n", f"Device: {device_note}\n",
             f"Prompts: {len(prompts)}, max_new_tokens: {args.max_new_tokens}, "
             f"warmup runs: {args.num_warmup}\n"]

    print("Loading fine-tuned model...")
    ft_model = load_model(args.base_model, args.adapter_dir, token)
    ft_results = benchmark(ft_model, tokenizer, prompts, args.max_new_tokens, args.num_warmup)
    lines += format_section("Fine-tuned model", ft_results)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.compare_base:
        print("Loading base model for comparison...")
        base_model = load_model(args.base_model, None, token)
        base_results = benchmark(base_model, tokenizer, prompts, args.max_new_tokens, args.num_warmup)
        lines += format_section("Base model", base_results)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lines.append("## How to read this\n")
    lines.append("- TTFT is what you notice as \"lag\" before a response starts.")
    lines.append("- Tokens/sec after the first token is what dominates wait time for long answers.")
    lines.append("- The LoRA adapter adds a small amount of compute per layer, so expect the "
                 "fine-tuned model to be slightly slower than base, not faster.")

    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
