#!/usr/bin/env python3
"""Command router for generation and weight inspection."""

import sys
import os

# Ensure src is on path when running as script
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))


def main():
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [options]", file=sys.stderr)
        print("Commands: generate, inspect-weights", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    argv = sys.argv[2:]

    if cmd == "generate":
        from commands.generate import main as generate_main

        # Restore argv so argparse sees only generate args
        sys.argv = ["generate"] + argv
        generate_main()
    elif cmd == "inspect-weights":
        from commands.inspect_weights import main as inspect_main

        sys.argv = ["inspect_weights"] + argv
        inspect_main()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Commands: generate, inspect-weights", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
