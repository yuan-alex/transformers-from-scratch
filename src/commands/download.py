"""Download model weights and tokenizer to the local cache."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Download model weights and tokenizer.")
    p.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo id, e.g. 'erwanf/gpt2-mini'",
    )
    p.add_argument(
        "--arch",
        choices=("gpt2", "smollm2"),
        default=None,
        help="Architecture-specific loader to use (e.g. for bfloat16 conversion). "
        "If omitted, the generic loader is used.",
    )
    p.add_argument(
        "--filename",
        type=str,
        default="model.safetensors",
        help="Weights filename within the repo (default: model.safetensors). "
        "Only used when --arch is not given.",
    )
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def _download_gpt2(repo_id, cache_dir):
    from utils.hf import load_gpt2_tokenizer, load_gpt2_weights

    print(f"Downloading GPT2 tokenizer for {repo_id}...")
    load_gpt2_tokenizer(repo_id=repo_id, cache_dir=cache_dir)
    print(f"Downloading GPT2 weights for {repo_id}...")
    load_gpt2_weights(repo_id=repo_id, cache_dir=cache_dir)
    print(f"{repo_id} ready.")


def _download_smollm2(repo_id, cache_dir):
    from utils.hf import load_smollm2_tokenizer, load_smollm2_weights

    print(f"Downloading SmolLM2 tokenizer for {repo_id}...")
    load_smollm2_tokenizer(repo_id=repo_id, cache_dir=cache_dir)
    print(f"Downloading SmolLM2 weights for {repo_id}...")
    load_smollm2_weights(repo_id=repo_id, cache_dir=cache_dir)
    print(f"{repo_id} ready.")


def _download_repo(repo_id, filename, cache_dir):
    from utils.hf import load_repo_tokenizer, load_repo_weights

    print(f"Downloading tokenizer for {repo_id}...")
    load_repo_tokenizer(repo_id, cache_dir=cache_dir)
    print(f"Downloading weights for {repo_id} ({filename})...")
    load_repo_weights(repo_id, cache_dir=cache_dir, filename=filename)
    print(f"{repo_id} ready.")


def main():
    args = parse_args()
    repo_id = args.repo
    cache_dir = args.cache_dir

    if args.arch == "gpt2":
        _download_gpt2(repo_id, cache_dir)
    elif args.arch == "smollm2":
        _download_smollm2(repo_id, cache_dir)
    else:
        _download_repo(repo_id, args.filename, cache_dir)


if __name__ == "__main__":
    main()
