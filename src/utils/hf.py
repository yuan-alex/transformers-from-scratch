"""HuggingFace download and load helpers."""

import os
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.numpy import load_file
from transformers import AutoTokenizer


def get_cache_dir() -> str:
    """Return cache directory for HuggingFace downloads (env or default)."""
    return os.environ.get("HF_CACHE_DIR", "./data/hf")


# --- GPT2 ---

GPT2_REPO = "erwanf/gpt2-mini"
GPT2_FILENAME = "model.safetensors"


def load_gpt2_weights(cache_dir: str | None = None):
    """Download and load GPT2 weights as numpy dict (safetensors)."""
    cache_dir = cache_dir or get_cache_dir()
    model_file = hf_hub_download(
        repo_id=GPT2_REPO,
        filename=GPT2_FILENAME,
        cache_dir=cache_dir,
    )
    return load_file(model_file)


def load_gpt2_tokenizer(cache_dir: str | None = None):
    """Load GPT2 tokenizer."""
    cache_dir = cache_dir or get_cache_dir()
    return AutoTokenizer.from_pretrained(GPT2_REPO, cache_dir=cache_dir)


# --- SmolLM2 ---

SMOLLM2_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_FILENAME = "model.safetensors"


def load_smollm2_weights(cache_dir: str | None = None):
    """Download and load SmolLM2 weights as JAX-friendly dict (bfloat16 -> float32)."""
    cache_dir = cache_dir or get_cache_dir()
    model_file = hf_hub_download(
        repo_id=SMOLLM2_REPO,
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


def load_smollm2_tokenizer(cache_dir: str | None = None):
    """Load SmolLM2 tokenizer."""
    cache_dir = cache_dir or get_cache_dir()
    return AutoTokenizer.from_pretrained(SMOLLM2_REPO, cache_dir=cache_dir)
