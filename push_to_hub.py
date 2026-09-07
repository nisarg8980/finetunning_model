#!/usr/bin/env python
import argparse
import os

from huggingface_hub import HfApi, create_repo

from env_setup import load_env

load_env()  # read HF_TOKEN from .env so it need not be set on the command line


def main() -> None:
    ap = argparse.ArgumentParser(description="Push a model/GGUF folder to the HF Hub.")
    ap.add_argument("--src", required=True, help="Local folder to upload.")
    ap.add_argument("--repo_id", required=True, help="e.g. your-username/mistral7b-myfinetune")
    ap.add_argument("--private", action="store_true", help="Create the repo as private.")
    args = ap.parse_args()

    # Prefer the dedicated write token; fall back to HF_TOKEN for older setups.
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("No token found. Set HF_WRITE_TOKEN (a WRITE token) in .env.")

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
