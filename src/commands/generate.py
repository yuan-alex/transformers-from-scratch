"""Generate text from a prompt using GPT2 or Llama."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a prompt.")
    p.add_argument(
        "--arch",
        choices=("gpt2", "llama"),
        required=True,
        help="Architecture to use for inference",
    )
    p.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo id for weights/tokenizer "
        "(must be compatible with --arch)",
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir
    repo_id = args.repo

    if args.arch == "gpt2":
        from models.gpt2 import GPT2

        model = GPT2(cache_dir=cache_dir, repo_id=repo_id)
        prompt = model.tokenizer.encode(args.prompt)
    else:
        from models.llama import Llama

        model = Llama(cache_dir=cache_dir, repo_id=repo_id)
        prompt = model.tokenizer.encode(args.prompt)

    result = model.generate(prompt, max_tokens=args.max_tokens)
    print(model.tokenizer.decode(result, skip_special_tokens=True))


if __name__ == "__main__":
    main()
