import jax
import jax.numpy as jnp


def softmax(x, axis=-1):
    return jax.nn.softmax(x, axis=axis)


class GELU:
    def __call__(self, x):
        return jax.nn.gelu(x, approximate=True)

    def __str__(self):
        return "GELU"
