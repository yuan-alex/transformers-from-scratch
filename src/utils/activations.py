import jax


class GELU:
    def __call__(self, x):
        return jax.nn.gelu(x, approximate=True)

    def __str__(self):
        return "GELU"
