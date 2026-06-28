#!/usr/bin/env python
"""
Preflight check for local QLoRA training (Windows + RTX 3060 12GB).

Run this BEFORE the long training run to catch environment problems in seconds:
    python preflight_check.py

It verifies CUDA PyTorch, the GPU and its VRAM, bitsandbytes, the core libraries,
and the HF token. It does not download the model, so it is fast and safe.
"""

import importlib
import os
import sys


def try_version(name: str):
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "ok")
    except Exception:
        return None


def main() -> int:
    problems = []
    warnings = []
    print("=== Preflight check ===")

    # PyTorch + CUDA
    try:
        import torch
        print(f"torch: {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"CUDA available: {cuda}")
        if not cuda:
            problems.append(
                "PyTorch cannot see CUDA. Install the CUDA build of torch "
                "(pip install torch --index-url https://download.pytorch.org/whl/cu121). "
                "On CPU this training is unusably slow."
            )
        else:
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {vram:.1f} GB")
            print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
            if vram < 11.5:
                warnings.append(
                    f"Only {vram:.1f} GB VRAM detected. Use --max_seq_len 512 (or lower) "
                    "and keep --per_device_batch_size 1."
                )
    except Exception as exc:
        problems.append(f"torch not installed or broken: {exc}")

    # bitsandbytes (required for QLoRA 4-bit + paged 8-bit optimizer)
    bnb = try_version("bitsandbytes")
    print(f"bitsandbytes: {bnb or 'MISSING'}")
    if not bnb:
        problems.append(
            "bitsandbytes missing. QLoRA needs it. pip install bitsandbytes "
            "(>=0.43 has Windows wheels)."
        )

    # core libraries
    for lib in ["transformers", "peft", "trl", "datasets", "huggingface_hub"]:
        version = try_version(lib)
        print(f"{lib}: {version or 'MISSING'}")
        if not version:
            problems.append(f"{lib} missing. Run: pip install -r requirements.txt")

    # Hugging Face token (Mistral is gated)
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN: set")
    else:
        warnings.append(
            "HF_TOKEN not set. Mistral weights are gated; export your token "
            "(and accept the license on the model page) or the download fails."
        )

    # summary
    print("\n=== Summary ===")
    for w in warnings:
        print("WARN:", w)
    for p in problems:
        print("FAIL:", p)
    if not problems:
        print("PASS: environment looks ready for QLoRA training on your 3060.")
        return 0
    print(f"\n{len(problems)} blocking issue(s) found. Fix these before training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
