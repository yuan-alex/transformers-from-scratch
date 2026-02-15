import jax.numpy as jnp


class LayerNorm:
    def __init__(self, weight, bias, eps=1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        return self.weight * x_norm + self.bias

    def __str__(self):
        return f"LayerNorm weights: {self.weight.shape}, bias: {self.bias.shape}"


class RMSNorm:
    def __init__(self, w, eps=1e-5) -> None:
        self.w = w
        self.eps = eps

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.w
