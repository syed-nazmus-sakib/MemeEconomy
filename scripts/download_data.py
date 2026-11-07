#!/usr/bin/env python3
"""
download_data.py
================
Downloads the MemeEconomy dataset from Hugging Face into the repo root:
memes/<split>/ (memes, megathreads, news articles) and env_patch/ (the
paper's environment, restored by `environment/setup_environment.py --exact`).

The dataset is gated because Tier-5 memes are intentionally harmful (see
CONTENT_DISCLAIMER.md). Accept the terms on the dataset page first, then log in:
    huggingface-cli login            # or: export HF_TOKEN=hf_...

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --split paper98     # only the 98 paper instances
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT            = Path(__file__).resolve().parents[1]
HF_DATASET_REPO = "SyedNazmusSakib/MemeEconomy"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the MemeEconomy dataset from Hugging Face")
    ap.add_argument("--repo", default=HF_DATASET_REPO, help="Hugging Face dataset id")
    ap.add_argument("--split", choices=("paper98", "extension", "all"), default="all")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    if args.split == "all":
        patterns = ["memes/**", "env_patch/**"]
    else:
        patterns = [f"memes/{args.split}/**", "env_patch/**"]

    path = snapshot_download(repo_id=args.repo, repo_type="dataset",
                             local_dir=ROOT, allow_patterns=patterns)
    n = sum(1 for p in (Path(path) / "memes").glob("*/*") if p.is_dir())
    print(f"Downloaded {n} instance folders → {ROOT / 'memes'}")


if __name__ == "__main__":
    main()
