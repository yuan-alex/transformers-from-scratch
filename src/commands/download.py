"""Download a model repo to the local cache."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="Download a model repo (config, tokenizer, weights) to the local cache."
    )
    p.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo id, e.g. 'HuggingFaceTB/SmolLM2-135M'",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download every file in the repo (default: only config, tokenizer, "
        "and safetensors weights needed for inference).",
    )
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def main():
    from utils.hf import download_repo

    args = parse_args()
    print(f"Downloading {args.repo}...")
    path = download_repo(
        args.repo,
        cache_dir=args.cache_dir,
        allow_patterns=["*"] if args.all else None,
    )
    print(f"{args.repo} ready at {path}")


if __name__ == "__main__":
    main()
