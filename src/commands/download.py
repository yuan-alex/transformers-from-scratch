"""Download model weights and tokenizer to the local cache."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Download model weights and tokenizer.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--arch",
        choices=("gpt2", "smollm2", "all"),
        help="Which architecture to download, or 'all' for both",
    )
    group.add_argument(
        "--repo",
        type=str,
        help="Arbitrary HuggingFace repo id, e.g. 'erwanf/gpt2-mini'",
    )
    p.add_argument(
        "--filename",
        type=str,
        default="model.safetensors",
        help="Weights filename within the repo (default: model.safetensors)",
    )
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def _download_gpt2(cache_dir):
    from utils.hf import load_gpt2_tokenizer, load_gpt2_weights

    print("Downloading GPT2 tokenizer...")
    load_gpt2_tokenizer(cache_dir=cache_dir)
    print("Downloading GPT2 weights...")
    load_gpt2_weights(cache_dir=cache_dir)
    print("GPT2 ready.")


def _download_smollm2(cache_dir):
    from utils.hf import load_smollm2_tokenizer, load_smollm2_weights

    print("Downloading SmolLM2 tokenizer...")
    load_smollm2_tokenizer(cache_dir=cache_dir)
    print("Downloading SmolLM2 weights...")
    load_smollm2_weights(cache_dir=cache_dir)
    print("SmolLM2 ready.")


def _download_repo(repo_id, filename, cache_dir):
    from utils.hf import load_repo_tokenizer, load_repo_weights

    print(f"Downloading tokenizer for {repo_id}...")
    load_repo_tokenizer(repo_id, cache_dir=cache_dir)
    print(f"Downloading weights for {repo_id} ({filename})...")
    load_repo_weights(repo_id, cache_dir=cache_dir, filename=filename)
    print(f"{repo_id} ready.")


def main():
    args = parse_args()
    cache_dir = args.cache_dir

    if args.repo:
        _download_repo(args.repo, args.filename, cache_dir)
        return

    if args.arch in ("gpt2", "all"):
        _download_gpt2(cache_dir)
    if args.arch in ("smollm2", "all"):
        _download_smollm2(cache_dir)


if __name__ == "__main__":
    main()
