import jax
import jax.numpy as jnp


def compute_inv_freq(dim, base=10000.0, rope_scaling=None):
    """Inverse RoPE frequencies, optionally adjusted by a rope_scaling config.

    Supports the "llama3" scaling type used by Llama 3.1/3.2 (HF
    `_compute_llama3_parameters`). Unknown/absent scaling falls back to the
    default frequencies (correct for SmolLM2 and Llama < 3.1).
    """
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))

    if not rope_scaling:
        return inv_freq

    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
    if rope_type != "llama3":
        # Other scaling types (linear, dynamic, yarn, ...) not implemented.
        return inv_freq

    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * jnp.pi / inv_freq
    # Long wavelengths (low freq): scale down by `factor`.
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # Medium wavelengths: smoothly interpolate between scaled and unscaled.
    smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium, smoothed, inv_freq_llama)
    # Short wavelengths (high freq): left untouched.
    return inv_freq_llama


def rope(x, dim, base=10000.0, rope_scaling=None):
    seq_len = x.shape[-2]

    # Compute frequencies
    inv_freq = compute_inv_freq(dim, base, rope_scaling)

    # Compute angles for each position
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # (seq_len, dim/2)

    # Duplicate frequencies for both halves
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (seq_len, dim)

    cos = jnp.cos(emb)
    sin = jnp.sin(emb)

    # Rotate half
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    x_rotated = jnp.concatenate([-x2, x1], axis=-1)

    # Apply rotation
    return x * cos + x_rotated * sin


class Attention:
    def __init__(self, attn_weights, attn_bias, proj_weights, proj_bias) -> None:
        self.attn_weights = attn_weights
        self.attn_bias = attn_bias
        self.proj_weights = proj_weights
        self.proj_bias = proj_bias

    def __call__(self, x):
        seq_len = x.shape[0]
        qkv = x @ self.attn_weights + self.attn_bias
        q, k, v = jnp.split(qkv, 3, axis=-1)
        scores = q @ k.T
        scores = scores / jnp.sqrt(k.shape[-1])
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(causal_mask == 0, -1e10, scores)
        scores = jax.nn.softmax(scores, axis=-1)
        out = scores @ v
        out = out @ self.proj_weights + self.proj_bias
        return out

    def __str__(self):
        return f"Attention weights: {self.attn_weights.shape}, bias: {self.attn_bias.shape}, proj weights: {self.proj_weights.shape}, proj bias: {self.proj_bias.shape}"


class MultiHeadAttention:
    def __init__(
        self, attn_weights, attn_bias, proj_weights, proj_bias, num_heads=8
    ) -> None:
        self.attn_weights = attn_weights
        self.attn_bias = attn_bias
        self.proj_weights = proj_weights
        self.proj_bias = proj_bias
        self.num_heads = num_heads

    def __call__(self, x):
        seq_len = x.shape[0]
        hidden_dim = x.shape[1]  # Should be 512
        head_dim = hidden_dim // self.num_heads  # 512 // 8 = 64

        # 1. Project input to Q, K, V
        qkv = x @ self.attn_weights + self.attn_bias  # (seq_len, 512*3)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # Each: (seq_len, 512)

        # 2. Split into multiple heads
        # Reshape from (seq_len, 512) to (seq_len, num_heads, head_dim)
        q = q.reshape(seq_len, self.num_heads, head_dim)  # (seq_len, 8, 64)
        k = k.reshape(seq_len, self.num_heads, head_dim)  # (seq_len, 8, 64)
        v = v.reshape(seq_len, self.num_heads, head_dim)  # (seq_len, 8, 64)

        # 3. Transpose to (num_heads, seq_len, head_dim) for batch processing
        q = q.transpose(1, 0, 2)  # (8, seq_len, 64)
        k = k.transpose(1, 0, 2)  # (8, seq_len, 64)
        v = v.transpose(1, 0, 2)  # (8, seq_len, 64)

        # 4. Compute attention scores for all heads
        scores = q @ k.transpose(0, 2, 1)  # (8, seq_len, seq_len)
        scores = scores / jnp.sqrt(head_dim)  # Scale by sqrt(64)

        # 5. Apply causal mask (prevent attending to future tokens)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))  # Lower triangular matrix
        scores = jnp.where(causal_mask == 0, -1e10, scores)  # Mask future positions

        # 6. Apply softmax to get attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)  # (8, seq_len, seq_len)

        # 7. Apply attention weights to values
        output = attn_weights @ v  # (8, seq_len, 64)

        # 8. Concatenate heads back together
        output = output.transpose(1, 0, 2)  # (seq_len, 8, 64)
        output = output.reshape(seq_len, hidden_dim)  # (seq_len, 512)

        # 9. Final projection
        output = output @ self.proj_weights + self.proj_bias

        return output

    def __str__(self):
        return f"MultiHeadAttention weights: {self.attn_weights.shape}, bias: {self.attn_bias.shape}, proj weights: {self.proj_weights.shape}, proj bias: {self.proj_bias.shape}"


class GroupedQueryAttention:
    def __init__(
        self,
        wq,
        wk,
        wv,
        wo,
        num_heads=9,
        num_kv_heads=3,
        hidden_dim=576,
        rope_theta=10000.0,
        rope_scaling=None,
    ) -> None:
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.q_count = num_heads
        self.kv_count = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

    def __call__(self, x):
        seq_len = x.shape[0]

        # project
        q = x @ self.wq.T
        k = x @ self.wk.T
        v = x @ self.wv.T

        # reshape from (seq_len, heads, dim) -> (heads, seq_len, dim)
        q = q.reshape(seq_len, self.q_count, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.kv_count, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.kv_count, self.head_dim).transpose(1, 0, 2)

        # rope
        q = rope(q, self.head_dim, base=self.rope_theta, rope_scaling=self.rope_scaling)
        k = rope(k, self.head_dim, base=self.rope_theta, rope_scaling=self.rope_scaling)

        # expand k and v to match q
        if self.q_count != self.kv_count:
            repeats = self.q_count // self.kv_count
            k = k.repeat(repeats, axis=0)
            v = v.repeat(repeats, axis=0)

        # attention
        attn_scores = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(self.head_dim)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_scores = jnp.where(mask == 0, -1e9, attn_scores)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        output = attn_probs @ v

        # merge heads and project (heads, token, dim) -> (token, heads, dim) -> concatenated (token, dim)
        output = output.transpose(1, 0, 2).reshape(seq_len, self.hidden_dim)

        return output @ self.wo.T
