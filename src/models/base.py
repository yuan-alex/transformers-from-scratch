from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def next_token(self, input_text: str) -> str:
        pass

    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        text = input_text
        eos_token = getattr(self.tokenizer, "eos_token", None)

        for _ in range(max_tokens):
            next_token = self.next_token(text)
            text += next_token
            if eos_token and next_token == eos_token:
                break

        return text[len(input_text):]
