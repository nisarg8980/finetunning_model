#!/usr/bin/env python
"""
Upload a folder (merged model or a folder of GGUF files) to the Hugging Face Hub.

Usage:
    export HF_TOKEN=...      # a WRITE token from https://huggingface.co/settings/tokens
    # push the merged fp16 model:
    python push_to_hub.py --src ./mistral7b-qlora-out/merged --repo_id your-username/mistral7b-myfinetune
    # or push only the GGUF folder (smaller, this is what Ollama pulls):
    python push_to_hub.py --src ./gguf --repo_id your-username/mistral7b-myfinetune-gguf

Security note (NOW-ISMS-AI-001): the token is read from the environment, never
hard-coded. Use --private if the fine-tune was trained on sensitive material.
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


def main() -> None:
    ap = argparse.ArgumentParser(description="Push a model/GGUF folder to the HF Hub.")
    ap.add_argument("--src", required=True, help="Local folder to upload.")
    ap.add_argument("--repo_id", required=True, help="e.g. your-username/mistral7b-myfinetune")
    ap.add_argument("--private", action="store_true", help="Create the repo as private.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export a WRITE token first.")

    if not os.path.isdir(args.src):
        raise SystemExit(f"Source folder not found: {args.src}")

    create_repo(args.repo_id, token=token, private=args.private, exist_ok=True, repo_type="model")
    HfApi(token=token).upload_folder(
        folder_path=args.src,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned Mistral-7B",
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
