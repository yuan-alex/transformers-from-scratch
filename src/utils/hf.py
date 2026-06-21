"""HuggingFace download and load helpers."""

import numpy as np
from huggingface_hub import hf_hub_download
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


# --- SmolLM2 ---

SMOLLM2_FILENAME = "model.safetensors"


def load_smollm2_weights(repo_id: str, cache_dir: str | None = None):
    """Download and load SmolLM2 weights as JAX-friendly dict (bfloat16 -> float32)."""
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=SMOLLM2_FILENAME,
        cache_dir=cache_dir,
    )
    weights = {}
    with safe_open(model_file, framework="jax") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if hasattr(tensor, "dtype") and str(tensor.dtype) == "bfloat16":
                tensor = tensor.astype(np.float32)
            weights[key] = tensor
    return weights


def load_smollm2_tokenizer(repo_id: str, cache_dir: str | None = None):
    """Load SmolLM2 tokenizer."""
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
