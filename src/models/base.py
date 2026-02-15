from abc import ABC, abstractmethod

import jax.numpy as jnp


class Model(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def next_token(self, input_tokens: jnp.ndarray) -> int:
        pass

    def generate(self, input_tokens: jnp.ndarray, max_tokens: int = 100) -> jnp.ndarray:
        tokens = input_tokens
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_tokens):
            next_token = self.next_token(tokens)
            tokens = jnp.concatenate([tokens, jnp.array([next_token])], axis=0)
            if next_token == eos_token_id:
                break

        return tokens[len(input_tokens):]
