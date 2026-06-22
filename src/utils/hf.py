"""HuggingFace download and load helpers."""

import json

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.numpy import load_file
from transformers import AutoConfig, AutoTokenizer


# --- GPT2 ---

GPT2_FILENAME = "model.safetensors"


def load_gpt2_weights(repo_id: str, cache_dir: str | None = None):
    """Download and load GPT2 weights as numpy dict (safetensors)."""
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=GPT2_FILENAME,
        cache_dir=cache_dir,
    )
    return load_file(model_file)


def load_gpt2_tokenizer(repo_id: str, cache_dir: str | None = None):
    """Load GPT2 tokenizer."""
    return AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)


# --- Llama (Llama 2/3, SmolLM2, and other LlamaForCausalLM models) ---


def _safetensors_shards(repo_id: str, cache_dir: str | None = None) -> list[str]:
    """Return the safetensors filename(s) for a repo, handling sharding.

    Large checkpoints are split across multiple files with a
    `model.safetensors.index.json` mapping each tensor to its shard. Small
    models ship a single `model.safetensors`.
    """
    try:
        index_path = hf_hub_download(
            repo_id, "model.safetensors.index.json", cache_dir=cache_dir
        )
        weight_map = json.load(open(index_path))["weight_map"]
        return sorted(set(weight_map.values()))
    except Exception:
        return ["model.safetensors"]


def load_llama_weights(repo_id: str, cache_dir: str | None = None):
    """Download and load Llama-family weights as JAX-friendly dict (bfloat16 -> float32).

    Discovers the weight file(s) via the safetensors index, so sharded models
    load too — no assumed filename.
    """
    weights = {}
    for shard in _safetensors_shards(repo_id, cache_dir=cache_dir):
        shard_path = hf_hub_download(repo_id, shard, cache_dir=cache_dir)
        with safe_open(shard_path, framework="jax") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if hasattr(tensor, "dtype") and str(tensor.dtype) == "bfloat16":
                    tensor = tensor.astype(np.float32)
                weights[key] = tensor
    return weights


def load_llama_tokenizer(repo_id: str, cache_dir: str | None = None):
    """Load a Llama-family tokenizer."""
    return AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)


# --- Generic ---

DEFAULT_WEIGHTS_FILENAME = "model.safetensors"


def load_repo_weights(
    repo_id: str, cache_dir: str | None = None, filename: str = DEFAULT_WEIGHTS_FILENAME
):
    """Download and load weights for an arbitrary HF repo as a numpy dict."""
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )
    return load_file(model_file)


def load_repo_tokenizer(repo_id: str, cache_dir: str | None = None):
    """Load tokenizer for an arbitrary HF repo."""
    return AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)


# Files needed for inference: config, tokenizer, and safetensors weights.
# Excludes duplicate .bin weights, training_args.bin, optimizer state, etc.
INFERENCE_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]


def download_repo(
    repo_id: str,
    cache_dir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download a model repo to the local cache and return its local path.

    Uses huggingface_hub.snapshot_download (the same mechanism transformers'
    from_pretrained uses), which queries the repo and fetches its files —
    handling sharded weights — instead of assuming a weights filename.
    """
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns if allow_patterns is not None else INFERENCE_PATTERNS,
    )


def load_config(repo_id: str, cache_dir: str | None = None) -> dict:
    """Load the HF model config as a dict (with sensible defaults)."""
    try:
        cfg = AutoConfig.from_pretrained(repo_id, cache_dir=cache_dir)
        return cfg.to_dict()
    except Exception:
        return {}


def attention_dims(config: dict, weights: dict) -> tuple[int, int, int]:
    """Infer (hidden_dim, num_heads, num_kv_heads) from config or weights."""
    hidden_dim = config.get("hidden_size")
    num_heads = config.get("num_attention_heads")
    num_kv_heads = config.get("num_key_value_heads") or config.get(
        "num_attention_heads"
    )
    if hidden_dim is None:
        hidden_dim = weights["model.embed_tokens.weight"].shape[-1]
    if num_heads is None:
        num_heads = 1
    if num_kv_heads is None:
        num_kv_heads = num_heads
    return hidden_dim, num_heads, num_kv_heads


def num_layers(config: dict) -> int:
    """Return transformer layer count from config."""
    n = config.get("num_hidden_layers") or config.get("num_layers")
    if n is None:
        raise ValueError("Config missing 'num_hidden_layers'")
    return int(n)
