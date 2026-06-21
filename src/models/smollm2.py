import jax
import jax.numpy as jnp

from models.base import Model
from utils.norms import RMSNorm
from utils.attention import GroupedQueryAttention
from utils.hf import (
    load_smollm2_weights,
    load_smollm2_tokenizer,
    load_config,
    attention_dims,
    num_layers,
)


class SmolLM2MLP:
    def __init__(self, gate_proj, up_proj, down_proj):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def swiglu(self, x):
        # Gate path
        gate = jnp.dot(x, self.gate_proj.T)  # Project to intermediate dim
        gate = jax.nn.silu(gate)  # Apply SiLU activation

        # Value path
        value = jnp.dot(x, self.up_proj.T)  # Project to intermediate dim

        # Combine
        hidden = gate * value  # Element-wise multiply

        # Output
        output = jnp.dot(hidden, self.down_proj.T)  # Project back to d_model

        return output

    def __call__(self, x):
        return self.swiglu(x)


class SmolLM2TransformerBlock:
    def __init__(
        self,
        weights,
        num_heads,
        num_kv_heads,
        hidden_dim,
        rope_theta=10000.0,
        rope_scaling=None,
    ) -> None:
        self.weights = weights
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_dim = hidden_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

    def __call__(self, x):
        ans = x.copy()

        layer_1 = [
            RMSNorm(self.weights["input_layernorm"]),
            GroupedQueryAttention(
                self.weights["self_attn_q_proj"],
                self.weights["self_attn_k_proj"],
                self.weights["self_attn_v_proj"],
                self.weights["self_attn_o_proj"],
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                hidden_dim=self.hidden_dim,
                rope_theta=self.rope_theta,
                rope_scaling=self.rope_scaling,
            ),
        ]
        layer_2 = [
            RMSNorm(self.weights["post_attention_layernorm"]),
            SmolLM2MLP(
                self.weights["mlp_gate_proj"],
                self.weights["mlp_up_proj"],
                self.weights["mlp_down_proj"],
            ),
        ]

        layer_result = ans
        for layer in layer_1:
            layer_result = layer(layer_result)
        ans += layer_result

        layer_result = ans.copy()
        for layer in layer_2:
            layer_result = layer(layer_result)
        ans += layer_result

        return ans


class SmolLM2(Model):
    def __init__(self, repo_id: str, cache_dir: str | None = None) -> None:
        self.layers = []

        tokenizer = load_smollm2_tokenizer(repo_id=repo_id, cache_dir=cache_dir)
        super().__init__(tokenizer)

        self.model_weights = load_smollm2_weights(repo_id=repo_id, cache_dir=cache_dir)

        config = load_config(repo_id, cache_dir=cache_dir)
        hidden_dim, num_heads, num_kv_heads = attention_dims(config, self.model_weights)
        rope_theta = config.get("rope_theta", 10000.0)
        rope_scaling = config.get("rope_scaling")

        layers_count = num_layers(config)
        for i in range(layers_count):
            weights = {
                "input_layernorm": self.model_weights[
                    f"model.layers.{i}.input_layernorm.weight"
                ],
                "mlp_down_proj": self.model_weights[
                    f"model.layers.{i}.mlp.down_proj.weight"
                ],
                "mlp_gate_proj": self.model_weights[
                    f"model.layers.{i}.mlp.gate_proj.weight"
                ],
                "mlp_up_proj": self.model_weights[
                    f"model.layers.{i}.mlp.up_proj.weight"
                ],
                "post_attention_layernorm": self.model_weights[
                    f"model.layers.{i}.post_attention_layernorm.weight"
                ],
                "self_attn_k_proj": self.model_weights[
                    f"model.layers.{i}.self_attn.k_proj.weight"
                ],
                "self_attn_o_proj": self.model_weights[
                    f"model.layers.{i}.self_attn.o_proj.weight"
                ],
                "self_attn_q_proj": self.model_weights[
                    f"model.layers.{i}.self_attn.q_proj.weight"
                ],
                "self_attn_v_proj": self.model_weights[
                    f"model.layers.{i}.self_attn.v_proj.weight"
                ],
            }
            self.layers.append(
                SmolLM2TransformerBlock(
                    weights,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    hidden_dim=hidden_dim,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                )
            )

    def next_token(self, input_tokens: jnp.ndarray) -> int:
        token_embeddings = self.model_weights["model.embed_tokens.weight"][input_tokens]

        x = token_embeddings
        for layer in self.layers:
            x = layer(x)

        x = RMSNorm(self.model_weights["model.norm.weight"])(x)

        # Untied models ship a separate lm_head; tied models reuse the embedding.
        lm_head = self.model_weights.get(
            "lm_head.weight", self.model_weights["model.embed_tokens.weight"]
        )
        logits = x @ lm_head.T
        token_ids = jnp.argmax(logits[-1, :], axis=-1)

        next_token_id = int(token_ids)
        return next_token_id
