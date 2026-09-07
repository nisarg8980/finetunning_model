#!/usr/bin/env python
"""
Run the full fine-tuning pipeline on ANY machine (Linux / macOS / Windows).

    python run_pipeline.py

Steps: preflight -> train -> hallucination check -> latency benchmark. Training
also auto-produces the before/after metrics report and the training curves, so
they are not run as separate steps here.

Portable by design: it re-invokes the SAME Python interpreter that launched it
(sys.executable), so whatever virtual environment you run this with is the one
used for every step. Your Hugging Face token is read from .env automatically
(see .env.example); no shell-specific setup or .bat file needed.

Examples:
    python run_pipeline.py                        # full run, 1 epoch
    python run_pipeline.py --epochs 2             # train longer
    python run_pipeline.py --skip_train           # just re-evaluate an existing adapter
"""

import argparse
import subprocess
import sys


def run(step: str, script_args: list) -> None:
    print("\n" + "=" * 64)
    print(f"  {step}")
    print("=" * 64)
    # sys.executable -> use the current interpreter/venv, so this works anywhere.
    result = subprocess.run([sys.executable] + script_args)
    if result.returncode != 0:
        sys.exit(f"\nPipeline stopped: '{step}' failed (exit {result.returncode}). "
                 "Fix the error above and re-run.")


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full QLoRA fine-tuning pipeline.")
    p.add_argument("--train_file", default="data/cybersecurity_qa_train.jsonl")
    p.add_argument("--val_file", default="data/cybersecurity_qa_val.jsonl")
    p.add_argument("--prompts_file", default="data/test_prompts.sample.jsonl")
    p.add_argument("--output_dir", default="./cyber-qa-out")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--skip_train", action="store_true",
                   help="Evaluate an existing adapter without retraining.")
    args = p.parse_args()

    adapter = f"{args.output_dir}/adapter"

    run("Step 1/4: Preflight check (GPU, libraries, token)",
        ["preflight_check.py"])

    if not args.skip_train:
        run("Step 2/4: Fine-tune with QLoRA (also writes curves + before/after metrics)",
            ["finetune_mistral_qlora.py",
             "--train_file", args.train_file,
             "--eval_file", args.val_file,
             "--output_dir", args.output_dir,
             "--epochs", str(args.epochs)])
    else:
        print("\n(skipping training; evaluating existing adapter)")

    run("Step 3/4: Hallucination check (groundedness + fake-CVE probe)",
        ["hallucination_check.py",
         "--adapter_dir", adapter,
         "--val_file", args.val_file,
         "--report", f"{args.output_dir}/hallucination_report.md"])

    run("Step 4/4: Latency benchmark (tokens/sec, time-to-first-token)",
        ["benchmark_latency.py",
         "--adapter_dir", adapter,
         "--prompts_file", args.prompts_file,
         "--report", f"{args.output_dir}/latency_report.md"])

    print("\n" + "=" * 64)
    print(f"  Pipeline complete. Everything is in: {args.output_dir}")
    print("    adapter/                     - your fine-tuned model")
    print("    training_curves.png          - training/validation loss + LR schedule")
    print("    metrics_report.md / .png     - before vs after: F1, ROUGE-L, close-match")
    print("    hallucination_report.md/.png - groundedness + fabrication rate")
    print("    latency_report.md / .png     - inference speed on your GPU")
    print("=" * 64)


if __name__ == "__main__":
    main()
