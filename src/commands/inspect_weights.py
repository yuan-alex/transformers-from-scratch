"""Print model weight keys and shapes."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Inspect model weight keys and shapes.")
    p.add_argument("--arch", choices=("gpt2", "llama"), required=True)
    p.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo id for weights/tokenizer",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Max number of keys to print"
    )
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.arch == "gpt2":
        from models.gpt2 import GPT2

        model = GPT2(repo_id=args.repo, cache_dir=args.cache_dir)
    else:
        from models.llama import Llama

        model = Llama(repo_id=args.repo, cache_dir=args.cache_dir)

    keys = list(model.model_weights.keys())
    if args.limit is not None:
        keys = keys[: args.limit]

    for k in keys:
        print(k, model.model_weights[k].shape)


if __name__ == "__main__":
    main()
