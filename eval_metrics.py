#!/usr/bin/env python
"""
Before/after evaluation of the fine-tune: Accuracy, F1, and Response Quality.

Runs the SAME held-out questions through the base model ("before") and the
fine-tuned model ("after"), then reports, for each:

  - Accuracy: fraction of answers whose token-F1 against the reference is >=
    --threshold. Free-form QA has no single gold string, so we treat an answer as
    "correct" when it overlaps the reference strongly enough. Tune --threshold to
    taste (0.5 is a reasonable default).
  - F1: mean token-level F1 against the reference answer (word overlap, order-free).
  - Response Quality: mean ROUGE-L (longest-common-subsequence overlap, rewards
    right content in the right order). If sentence-transformers is installed, we
    also report mean semantic cosine similarity, which credits correct answers
    phrased differently from the reference.

The point of running the base model too is to show the delta the fine-tune actually
bought you. Higher is better on every metric here.

This measures closeness to your reference answers; it does not by itself prove facts
are correct. Pair it with hallucination_check.py (fabrication probe) and read a few
answers yourself.

Usage:
    export HF_TOKEN=...
    python eval_metrics.py \
        --adapter_dir ./cyber-qa-out/adapter \
        --val_file data/cybersecurity_qa_val.jsonl
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Reuse the exact metric + data helpers the hallucination check already uses,
# so "F1" means the same thing across both reports.
from hallucination_check import token_f1, rouge_l, fold_system, load_val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Before/after Accuracy, F1, Response Quality.")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter_dir", required=True, help="Path to the trained LoRA adapter.")
    p.add_argument("--val_file", required=True, help="Held-out JSONL ({'messages': [...]}).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="An answer counts as correct when token-F1 >= this (accuracy).")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--skip_base", action="store_true",
                   help="Only evaluate the fine-tuned model (no before/after comparison).")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N validation items (faster smoke test).")
    p.add_argument("--report", default="metrics_report.md")
    return p.parse_args()


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


@torch.no_grad()
def generate_all(model, tokenizer, items, max_new_tokens: int) -> list:
    """Generate one answer per (messages, reference) item."""
    answers = []
    for messages, _reference in items:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        answers.append(tokenizer.decode(out[0][inputs.size(1):], skip_special_tokens=True).strip())
    return answers


def try_semantic_scores(answers: list, references: list):
    """Mean cosine similarity via sentence-transformers, or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return None
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    ans_emb = embedder.encode(answers, convert_to_tensor=True, show_progress_bar=False)
    ref_emb = embedder.encode(references, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(ans_emb, ref_emb).diagonal()
    return [float(s) for s in sims]


def score(answers: list, references: list, threshold: float) -> dict:
    f1s = [token_f1(a, r) for a, r in zip(answers, references)]
    rls = [rouge_l(a, r) for a, r in zip(answers, references)]
    n = len(answers)
    correct = sum(1 for f in f1s if f >= threshold)
    return {
        "n": n,
        "accuracy": correct / n if n else float("nan"),
        "mean_f1": sum(f1s) / n if n else float("nan"),
        "mean_rouge_l": sum(rls) / n if n else float("nan"),
        "f1s": f1s,
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report(ft_scores: dict, base_scores, ft_mean_sem, base_mean_sem, threshold: float) -> str:
    lines = ["# Before/after metrics report\n",
             f"Validation items: {ft_scores['n']} | accuracy threshold (token-F1): "
             f"{threshold:.2f}\n"]

    header = "| Metric | Before (base) | After (fine-tuned) | Change |"
    sep = "|---|---|---|---|"
    if base_scores is None:
        header = "| Metric | After (fine-tuned) |"
        sep = "|---|---|"
    lines.append(header)
    lines.append(sep)

    def row(name: str, after: float, before, is_pct: bool):
        f = fmt_pct if is_pct else (lambda v: f"{v:.3f}")
        if before is None:
            return f"| {name} | {f(after)} |"
        delta = after - before
        arrow = "improved" if delta > 0 else ("no change" if delta == 0 else "worse")
        dfmt = f"{delta * 100:+.1f} pts" if is_pct else f"{delta:+.3f}"
        return f"| {name} | {f(before)} | {f(after)} | {dfmt} ({arrow}) |"

    lines.append(row("Accuracy (F1 >= threshold)", ft_scores["accuracy"],
                     base_scores["accuracy"] if base_scores else None, is_pct=True))
    lines.append(row("Mean token-F1", ft_scores["mean_f1"],
                     base_scores["mean_f1"] if base_scores else None, is_pct=False))
    lines.append(row("Response quality (ROUGE-L)", ft_scores["mean_rouge_l"],
                     base_scores["mean_rouge_l"] if base_scores else None, is_pct=False))
    if ft_mean_sem is not None:
        lines.append(row("Response quality (semantic cos)", ft_mean_sem,
                         base_mean_sem, is_pct=False))
    else:
        lines.append("")
        lines.append("_Semantic similarity skipped (install `sentence-transformers` to enable)._")
    lines.append("")

    lines.append("## How to read this\n")
    lines.append("- Accuracy/F1/ROUGE-L all measure overlap with your reference answers; higher is better.")
    lines.append("- A positive 'Change' means the fine-tune improved over the base model on that metric.")
    lines.append("- These reward matching the reference wording. For whether facts are actually correct, "
                 "also run hallucination_check.py and read a sample of answers.")
    return "\n".join(lines)


def run_before_after_eval(base_model: str, adapter_dir: str, val_file: str, token,
                          threshold: float = 0.5, max_new_tokens: int = 256,
                          skip_base: bool = False, limit=None,
                          report_path: str = "metrics_report.md") -> str:
    """Generate answers with base + fine-tuned models, score them, write the report.

    Importable so the training script can call it as a post-training step. Loads each
    7B model one at a time and frees it before the next, to fit a 12 GB GPU.
    """
    tok_src = adapter_dir if os.path.isdir(adapter_dir) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = load_val(val_file)
    if limit:
        items = items[:limit]
    references = [ref for _msgs, ref in items]
    print(f"Loaded {len(items)} validation items.")

    # ---- After (fine-tuned) ----
    print("Loading fine-tuned model...")
    ft_model = load_model(base_model, adapter_dir, token)
    ft_answers = generate_all(ft_model, tokenizer, items, max_new_tokens)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Before (base) ----
    base_answers = None
    if not skip_base:
        print("Loading base model for before/after comparison...")
        base = load_model(base_model, None, token)
        base_answers = generate_all(base, tokenizer, items, max_new_tokens)
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Metrics (models are freed; embedder loads alone if present) ----
    ft_scores = score(ft_answers, references, threshold)
    base_scores = score(base_answers, references, threshold) if base_answers else None

    ft_sem = try_semantic_scores(ft_answers, references)
    base_sem = try_semantic_scores(base_answers, references) if base_answers else None
    ft_mean_sem = sum(ft_sem) / len(ft_sem) if ft_sem else None
    base_mean_sem = sum(base_sem) / len(base_sem) if base_sem else None

    report = build_report(ft_scores, base_scores, ft_mean_sem, base_mean_sem, threshold)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nReport written to: {report_path}")
    return report


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    run_before_after_eval(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        val_file=args.val_file,
        token=token,
        threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
        skip_base=args.skip_base,
        limit=args.limit,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
