#!/usr/bin/env python
"""
Fine-tune Mistral-7B-Instruct-v0.3 on chat / instruction data with QLoRA, with an
optional push of the result to the Hugging Face Hub.

Why QLoRA and not a full fine-tune:
    A full fine-tune of a 7B model needs roughly 80 GB+ of VRAM (weights +
    gradients + Adam optimizer states). An RTX 3060 has 12 GB, so a full
    fine-tune cannot run on it. QLoRA loads the base model in 4-bit and trains
    small LoRA adapters, which fits comfortably in 12 GB and produces the same
    practical result for instruction tuning.

Data format (JSONL, one example per line). All of these are accepted:
    {"messages": [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}]}
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    {"instruction": "...", "input": "...", "output": "..."}
A system message is folded into the first user turn, because the Mistral chat
template does not accept a standalone system role.

Usage:
    export HF_TOKEN=...            # required: Mistral models are gated on the Hub
    python finetune_mistral_qlora.py \
        --train_file data/cybersecurity_qa_train.jsonl \
        --eval_file data/cybersecurity_qa_val.jsonl \
        --output_dir ./cyber-qa-out \
        --push_repo your-username/mistral7b-cyber-qa --push_private

Security note (NOW-ISMS-AI-001): never put real customer or personal data in the
training set. Use anonymized or synthetic examples. The HF token is read from the
environment, never hard-coded. Pushing publishes the model, so use --push_private
unless the data is confirmed safe to release.
"""

import argparse
import gc
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Mistral-7B-Instruct.")
    # Model / data
    p.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--train_file", required=True, help="Path to training JSONL.")
    p.add_argument("--eval_file", default=None, help="Optional eval JSONL.")
    p.add_argument("--output_dir", default="./mistral7b-qlora-out")

    # Training hyperparameters (defaults tuned for a 12 GB RTX 3060)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument(
        "--eval_steps",
        type=int,
        default=20,
        help="Run validation every N steps (only used when --eval_file is given). "
        "Smaller = more points on the validation-loss curve.",
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also log to TensorBoard (live curves). Needs the 'tensorboard' package; "
        "view with: tensorboard --logdir <output_dir>/runs",
    )
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop early if validation loss has not improved for this many evals in a "
        "row (needs --eval_file). Set 0 to disable and always train the full --epochs.",
    )
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Post-training
    p.add_argument(
        "--skip_post_eval",
        action="store_true",
        help="Skip the automatic before/after metrics (Accuracy, F1, Response Quality) "
        "that otherwise run on --eval_file after training.",
    )
    p.add_argument(
        "--metrics_threshold",
        type=float,
        default=0.5,
        help="Post-eval accuracy threshold: an answer counts as correct when its "
        "token-F1 vs the reference is >= this value.",
    )
    p.add_argument(
        "--merge_after",
        action="store_true",
        help="After training, reload the base model in fp16, merge the adapter, "
        "and save a standalone model. Needs ~16 GB RAM/VRAM; runs on CPU if needed.",
    )
    # Hugging Face push
    p.add_argument("--push_repo", default=None, help="HF repo id to push to, e.g. user/name.")
    p.add_argument("--push_private", action="store_true", help="Create the HF repo as private.")
    p.add_argument(
        "--push_merged",
        action="store_true",
        help="Push the standalone merged model instead of the small adapter "
        "(implies --merge_after; ~15 GB upload).",
    )
    return p.parse_args()


def fold_system(messages: list) -> list:
    """Merge any system message into the first user turn (Mistral has no system role)."""
    system_text = ""
    rest = []
    for m in messages:
        if m.get("role") == "system":
            system_text += (m.get("content", "").strip() + "\n\n")
            continue
        rest.append(m)
    if system_text and rest and rest[0].get("role") == "user":
        rest[0] = {"role": "user", "content": system_text + rest[0]["content"]}
    elif system_text:
        rest.insert(0, {"role": "user", "content": system_text.strip()})
    return rest


def normalize_to_messages(example: dict) -> dict:
    """Convert any supported schema into a uniform {"messages": [...]} record."""
    if example.get("messages"):
        return {"messages": fold_system(example["messages"])}

    instruction = (example.get("instruction") or "").strip()
    extra_input = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    user_content = f"{instruction}\n\n{extra_input}".strip() if extra_input else instruction
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def build_format_fn(tokenizer):
    """Render each record into a single training string via the chat template."""

    def format_fn(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return format_fn


def load_and_prepare(path: str, tokenizer):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(normalize_to_messages, remove_columns=ds.column_names)
    ds = ds.map(build_format_fn(tokenizer), remove_columns=["messages"])
    return ds


def push_to_hf(src_dir: str, repo_id: str, private: bool, token):
    """Upload a folder (adapter or merged model) to the Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo

    if not token:
        raise SystemExit("HF_TOKEN must be a WRITE token to push to the Hub.")
    print(f"Pushing {src_dir} to https://huggingface.co/{repo_id} ...")
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")
    HfApi(token=token).upload_folder(folder_path=src_dir, repo_id=repo_id, repo_type="model")
    print(f"Pushed: https://huggingface.co/{repo_id}")


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        print("Warning: HF_TOKEN not set. Gated Mistral weights may fail to download.")

    # 4-bit (NF4) quantization config for the frozen base model.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA on attention + MLP projections (standard target set for Mistral).
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    train_ds = load_and_prepare(args.train_file, tokenizer)
    eval_ds = load_and_prepare(args.eval_file, tokenizer) if args.eval_file else None

    # When we have an eval set, keep the checkpoint with the LOWEST validation loss
    # instead of the last one. This is what prevents the "trained too long, saved the
    # overfit final weights" trap. It requires a save at every eval step, so we align
    # save_steps to eval_steps here.
    keep_best = eval_ds is not None
    save_steps = args.eval_steps if keep_best else args.save_steps

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="paged_adamw_8bit",  # 8-bit paged optimizer keeps VRAM low
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_steps=args.logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        max_length=args.max_seq_len,
        packing=False,
        dataset_text_field="text",
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        load_best_model_at_end=keep_best,
        metric_for_best_model="eval_loss" if keep_best else None,
        greater_is_better=False if keep_best else None,
        seed=args.seed,
        report_to="tensorboard" if args.tensorboard else "none",
    )

    callbacks = []
    if keep_best and args.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        # Stop if validation loss has not improved for this many evals in a row.
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    # Save the raw metric log and draw training curves (loss / val loss / LR).
    save_training_curves(trainer.state.log_history, args.output_dir)

    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    # Before/after metrics (Accuracy, F1, Response Quality) on the held-out set.
    # Runs by default when an eval file is given; --skip_post_eval turns it off.
    if args.eval_file and not args.skip_post_eval:
        run_post_training_eval(trainer, model, args, adapter_dir, hf_token)
    else:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged_dir = None
    if args.merge_after or args.push_merged:
        merged_dir = merge_adapter(args.model_name, adapter_dir, args.output_dir, hf_token)

    if args.push_repo:
        src = merged_dir if args.push_merged else adapter_dir
        push_to_hf(src, args.push_repo, args.push_private, hf_token)


def save_training_curves(log_history: list, output_dir: str) -> None:
    """Write the trainer log to JSON and plot loss / val loss / LR schedule to a PNG.

    The Hugging Face Trainer records one dict per logging event in
    trainer.state.log_history. Training-step logs carry 'loss' and 'learning_rate';
    evaluation logs carry 'eval_loss'. We pull those out by step and plot them.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "trainer_log_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)
    print(f"Training log saved to: {log_path}")

    train_steps, train_loss = [], []
    lr_steps, lr_values = [], []
    eval_steps, eval_loss = [], []
    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry and step is not None:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "learning_rate" in entry and step is not None:
            lr_steps.append(step)
            lr_values.append(entry["learning_rate"])
        if "eval_loss" in entry and step is not None:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed; write straight to file
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping curves. `pip install matplotlib` to enable. "
              f"Raw numbers are still in {log_path}.")
        return

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    if train_loss:
        ax_loss.plot(train_steps, train_loss, label="training loss", color="tab:blue")
    if eval_loss:
        ax_loss.plot(eval_steps, eval_loss, label="validation loss",
                     color="tab:orange", marker="o")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training and validation loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    if lr_values:
        ax_lr.plot(lr_steps, lr_values, label="learning rate", color="tab:green")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("training step")
    ax_lr.set_title("Learning-rate schedule")
    ax_lr.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Training curves saved to: {png_path}")


def run_post_training_eval(trainer, model, args, adapter_dir: str, hf_token) -> None:
    """Free the training model, then run the before/after metrics on the eval set.

    The trainer's model, optimizer states, and gradients are released first so the
    fresh base + fine-tuned models the eval loads can fit in 12 GB. If anything here
    fails (e.g. OOM), we warn but do not fail the run: the adapter is already saved,
    and eval_metrics.py can be run separately.
    """
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        from eval_metrics import run_before_after_eval

        report_path = os.path.join(args.output_dir, "metrics_report.md")
        print("\nRunning before/after metrics (Accuracy, F1, Response Quality)...")
        run_before_after_eval(
            base_model=args.model_name,
            adapter_dir=adapter_dir,
            val_file=args.eval_file,
            token=hf_token,
            threshold=args.metrics_threshold,
            report_path=report_path,
        )
    except Exception as exc:  # never let eval sink an otherwise-good training run
        print(f"WARN: post-training metrics failed ({exc}). The adapter is saved; "
              f"run `python eval_metrics.py --adapter_dir {adapter_dir} "
              f"--val_file {args.eval_file}` separately.")


def merge_adapter(base_model_name: str, adapter_dir: str, output_dir: str, hf_token) -> str:
    """Reload the base model in fp16 and merge the LoRA adapter into a standalone model.

    A 4-bit base cannot be merged cleanly, so we reload in fp16 here. If GPU VRAM
    is insufficient, this falls back to CPU automatically. Returns the merged dir.
    """
    from peft import PeftModel

    print("Merging adapter into base weights (fp16)...")
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        token=hf_token,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()

    merged_dir = os.path.join(output_dir, "merged")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")
    return merged_dir


if __name__ == "__main__":
    main()
