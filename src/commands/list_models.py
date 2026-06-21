"""List models downloaded in the local HuggingFace cache."""

import argparse
import json
from pathlib import Path

from huggingface_hub import scan_cache_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="List downloaded models with their architecture and HF id."
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: standard HF cache)",
    )
    return p.parse_args()


def _arch_from_config(snapshot_path: Path) -> str:
    """Read architecture from config.json in a snapshot, if present."""
    config_file = snapshot_path / "config.json"
    if not config_file.is_file():
        return "unknown"
    try:
        cfg = json.loads(config_file.read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    archs = cfg.get("architectures")
    if isinstance(archs, list) and archs:
        return ", ".join(archs)
    model_type = cfg.get("model_type")
    if model_type:
        return model_type
    return "unknown"


def main():
    args = parse_args()
    info = scan_cache_dir(cache_dir=args.cache_dir)

    repos = sorted(info.repos, key=lambda r: r.repo_id)
    if not repos:
        print("No downloaded models found.")
        return

    rows = [("HF ID", "ARCHITECTURE", "SIZE")]
    for repo in repos:
        arch = "unknown"
        revisions = sorted(repo.revisions, key=lambda r: r.commit_hash)
        if revisions:
            arch = _arch_from_config(revisions[-1].snapshot_path)
        rows.append((repo.repo_id, arch, repo.size_on_disk_str))

    name_w = max(len(r[0]) for r in rows)
    arch_w = max(len(r[1]) for r in rows)
    size_w = max(len(r[2]) for r in rows)

    header = f"{rows[0][0]:<{name_w}}  {rows[0][1]:<{arch_w}}  {rows[0][2]:>{size_w}}"
    print(header)
    print("-" * len(header))
    for repo_id, arch, size in rows[1:]:
        print(f"{repo_id:<{name_w}}  {arch:<{arch_w}}  {size:>{size_w}}")


if __name__ == "__main__":
    main()
