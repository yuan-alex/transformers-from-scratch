import jax.numpy as jnp


def Transformer():
    pass


def GroupedQueryAttetnion():
    pass


def SmolLM2():
    pass


class SmolLM2(Model):
    def __init__(self) -> None:
        self.layers = []

        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/SmolLM2-135M-Instruct",
            cache_dir="./data/hf",
        )
        super().__init__(tokenizer)

        model_file = hf_hub_download(
            repo_id="unsloth/SmolLM2-135M-Instruct",
            filename="model.safetensors",
            cache_dir="./data/hf",
        )

        self.model_weights = {}
        with safe_open(model_file, framework="jax") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if hasattr(tensor, "dtype") and str(tensor.dtype) == "bfloat16":
                    tensor = tensor.astype(np.float32)
                self.model_weights[key] = tensor

        layers_count = 30
        for i in range(layers_count):
            self.layers.append(RMSNorm())
            pass

        # self.model = Transformer(self.model_weights)

    def next_token(self, input_text: str) -> str:
        input_tokens = jnp.array(self.tokenizer.encode(input_text))
        token_embeddings = self.model_weights["model.embed_tokens.weight"][input_tokens]
        print(token_embeddings.shape)
        # next_token_id = self.model(embeddings)
        # return self.tokenizer.decode(next_token_id)
