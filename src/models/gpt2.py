import numpy as np

from models.base import Model
from utils.norms import LayerNorm
from utils.attention import MultiHeadAttention
from utils.layers import Linear
from utils.activations import GELU
from utils.hf import load_gpt2_weights, load_gpt2_tokenizer


class GPT2(Model):
    def __init__(self, cache_dir: str | None = None) -> None:
        tokenizer = load_gpt2_tokenizer(cache_dir=cache_dir)
        super().__init__(tokenizer)

        self.model_weights = load_gpt2_weights(cache_dir=cache_dir)

        self.attention_blocks = []
        self.ffn_blocks = []

        for i in range(4):
            # Attention block (LayerNorm + Attention)
            ln_1_weight = self.model_weights[f"transformer.h.{i}.ln_1.weight"]
            ln_1_bias = self.model_weights[f"transformer.h.{i}.ln_1.bias"]
            attn_c_attn_weight = self.model_weights[f"transformer.h.{i}.attn.c_attn.weight"]
            attn_c_attn_bias = self.model_weights[f"transformer.h.{i}.attn.c_attn.bias"]
            attn_c_proj_weight = self.model_weights[f"transformer.h.{i}.attn.c_proj.weight"]
            attn_c_proj_bias = self.model_weights[f"transformer.h.{i}.attn.c_proj.bias"]

            self.attention_blocks.append(
                {
                    "ln": LayerNorm(ln_1_weight, ln_1_bias),
                    "attn": MultiHeadAttention(
                        attn_c_attn_weight,
                        attn_c_attn_bias,
                        attn_c_proj_weight,
                        attn_c_proj_bias,
                    ),
                }
            )

            # FFN block (LayerNorm + Linear + GELU + Linear)
            ln_2_weight = self.model_weights[f"transformer.h.{i}.ln_2.weight"]
            ln_2_bias = self.model_weights[f"transformer.h.{i}.ln_2.bias"]
            ffn_1_weight = self.model_weights[f"transformer.h.{i}.mlp.c_fc.weight"]
            ffn_1_bias = self.model_weights[f"transformer.h.{i}.mlp.c_fc.bias"]
            ffn_2_weight = self.model_weights[f"transformer.h.{i}.mlp.c_proj.weight"]
            ffn_2_bias = self.model_weights[f"transformer.h.{i}.mlp.c_proj.bias"]

            self.ffn_blocks.append(
                {
                    "ln": LayerNorm(ln_2_weight, ln_2_bias),
                    "linear1": Linear(ffn_1_weight, ffn_1_bias),
                    "gelu": GELU(),
                    "linear2": Linear(ffn_2_weight, ffn_2_bias),
                }
            )

    def next_token(self, input_text):
        input_tokens = self.tokenizer.encode(input_text)
        token_embeddings = self.model_weights["transformer.wte.weight"][input_tokens]
        pos_embeddings = self.model_weights["transformer.wpe.weight"][
            : len(input_tokens)
        ]
        x = token_embeddings + pos_embeddings

        for attn_block, ffn_block in zip(self.attention_blocks, self.ffn_blocks):
            # Attention with residual
            attn_out = attn_block["attn"](attn_block["ln"](x))
            x += attn_out  # ← Residual connection

            # FFN with residual
            ffn_out = ffn_block["linear2"](
                ffn_block["gelu"](ffn_block["linear1"](ffn_block["ln"](x)))
            )
            x += ffn_out  # ← Residual connection

        # final layer norm
        x = LayerNorm(
            self.model_weights["transformer.ln_f.weight"],
            self.model_weights["transformer.ln_f.bias"],
        )(x)

        logits = x @ self.model_weights["transformer.wte.weight"].T
        token_ids = np.argmax(logits, axis=-1)

        next_token_id = int(token_ids[-1])
        return self.tokenizer.decode(next_token_id)
