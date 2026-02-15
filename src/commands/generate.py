"""Generate text from a prompt using GPT2 or SmolLM2."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a prompt.")
    p.add_argument("--model", choices=("gpt2", "smollm2"), required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir

    if args.model == "gpt2":
        from models.gpt2 import GPT2

        model = GPT2(cache_dir=cache_dir)
        prompt = model.tokenizer.encode(args.prompt)
    else:
        from models.smollm2 import SmolLM2

        model = SmolLM2(cache_dir=cache_dir)
        prompt = model.tokenizer.encode(args.prompt)

    result = model.generate(prompt, max_tokens=args.max_tokens)
    print(model.tokenizer.decode(result, skip_special_tokens=True))


if __name__ == "__main__":
    main()
