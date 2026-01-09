#!/usr/bin/env python3
"""
Download a Hugging Face model repo into STORAGE_PATH (or a provided directory).

Examples:
  export STORAGE_PATH="/path/to/storage"
  python scripts/download_hf_model.py --repo-id "Qwen/Qwen3-4B-Base"

  # Download a finetuned model you've uploaded:
  python scripts/download_hf_model.py --repo-id "yourname/qwen3-4b_solver_v3"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def default_dest(storage_path: str, repo_id: str) -> str:
    # Keep it stable and filesystem-safe.
    repo_dirname = repo_id.replace("/", "_")
    return str(Path(storage_path) / "models" / repo_dirname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3-4B-Base")
    parser.add_argument("--revision", default=None, help="Optional HF revision (branch/tag/commit)")
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory. Default: $STORAGE_PATH/models/<repo_id_with_underscores>/",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        help="HF token if required (or set HF_TOKEN env var).",
    )
    args = parser.parse_args()

    storage_path = os.getenv("STORAGE_PATH")
    if not storage_path and not args.dest:
        raise SystemExit("STORAGE_PATH is not set and --dest was not provided.")

    dest_dir = args.dest or default_dest(storage_path, args.repo_id)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading {args.repo_id} -> {dest_dir}")
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        token=args.token,
    )
    print(f"Done. Local path: {local_dir}")


if __name__ == "__main__":
    main()

