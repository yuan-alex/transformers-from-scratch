class Linear:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def __call__(self, x):
        return x @ self.weights + self.bias

    def __str__(self):
        return f"Linear weights: {self.weights.shape}, bias: {self.bias.shape}"
