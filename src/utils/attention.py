import jax
import jax.numpy as jnp

from utils.activations import softmax


def rope(x, dim, base=10000.0):
    seq_len = x.shape[-2]

    # Compute frequencies
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))

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
        qkv = x @ self.attn_weights + self.attn_bias
        q, k, v = jnp.split(qkv, 3, axis=-1)
        scores = q @ k.T
        scores = scores / jnp.sqrt(k.shape[-1])
        scores = softmax(scores)
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
        attn_weights = softmax(scores, axis=-1)  # (8, seq_len, seq_len)

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
    def __init__(self, wq, wk, wv, wo, q_count=9, kv_count=3) -> None:
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.q_count = q_count
        self.kv_count = kv_count
        self.head_dim = 576 // q_count

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
        q, k = rope(q, self.head_dim), rope(k, self.head_dim)

        # expand k and v to match q
        k = k.repeat(3, axis=0)
        v = v.repeat(3, axis=0)

        # attention
        attn_scores = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(self.head_dim)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_scores = jnp.where(mask == 0, -1e9, attn_scores)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        output = attn_probs @ v

        # merge heads and project (heads, token, dim) -> (token, heads, dim) -> concatenated (token, dim)
        output = output.transpose(1, 0, 2).reshape(seq_len, 576)

        return output @ self.wo.T
