# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Quantized Transformers.

Input : (B, T) int32   tokens
Output: (B, T, V) float32 logits

Applies quantization to parts of the network.
"""

import dataclasses

from egg import base
from egg.lib import quantization
import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class NetworkConfig(base.MakeableConfig[nn.Module]):
  """Transformer hyperparameters."""

  vocab_size: int = 10  # Number of tokens in the vocabulary.
  sequence_length: int = 32  # max context length
  embed_dim: int = 64  # Embedding dimension
  num_heads: int = 1  # Number of attention heads in each block
  num_layers: int = 3  # Number of transformer blocks
  ff_dim: int = 128  # Feed-forward dimension
  bias: bool = False  # Whether to use bias in Dense layers.
  quantize_config: quantization.QuantizeConfig = dataclasses.field(
      default_factory=quantization.QuantizeConfig
  )  # Quantization config for the network.

  def make(self) -> "Transformer":
    return Transformer(self)


class Transformer(nn.Module):
  """Decoder-only Transformer with causal attention."""

  config: NetworkConfig

  def setup(self):
    self.quantizer = self.config.quantize_config.make()

  @nn.compact
  def __call__(self, tokens: jax.Array) -> jax.Array:
    """Forward pass: [B, T] int32 tokens -> [B, T, V] float32 logits."""

    # Pre-split RNG keys for the entire forward pass
    base_rng = self.make_rng("noise")
    num_keys = 2 + self.config.num_layers  # embed + output + num_layers
    rngs = jax.random.split(base_rng, num_keys)

    x, pad_mask = _embed(tokens, self.config, self.quantizer, rngs[0])
    causal_mask = nn.make_causal_mask(tokens)  # (B, 1, T, T)
    padding = nn.make_attention_mask(pad_mask[..., 0], pad_mask[..., 0])
    mask = nn.combine_masks(causal_mask, padding)  # (B, 1, T, T)

    for i in range(self.config.num_layers):
      x = TransformerBlock(self.config, name=f"layer_{i}")(
          x, mask, self.quantizer, rngs[i + 1]
      )

    output_layer = nn.Dense(
        self.config.vocab_size, use_bias=self.config.bias, name="output"
    )
    output = output_layer(x)
    return self.quantizer(output, rngs[-1])  # (B, T, V)


class TransformerBlock(nn.Module):
  """Single transformer block: pre-norm attention + feed-forward."""

  config: NetworkConfig

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      mask: jax.Array,
      quantizer: quantization.Quantizer,
      rng: jax.Array,
  ) -> jax.Array:
    # Split RNG keys for each quantization step
    rngs = jax.random.split(rng, 6)

    # Pre-norm self-attention
    h = nn.LayerNorm(use_bias=self.config.bias, name="ln_attn")(x)
    h = quantizer(h, rngs[0])
    h = nn.MultiHeadDotProductAttention(
        num_heads=self.config.num_heads,
        use_bias=self.config.bias,
        name="self_attn",
    )(inputs_q=h, inputs_k=h, inputs_v=h, mask=mask)
    h = quantizer(h, rngs[1])
    x = x + h  # Residual

    # Pre-norm feed-forward
    h = nn.LayerNorm(use_bias=self.config.bias, name="ln_ff")(x)
    h = quantizer(h, rngs[2])
    h = nn.Dense(self.config.ff_dim, use_bias=self.config.bias, name="ff_up")(h)
    h = nn.relu(h)
    h = quantizer(h, rngs[3])
    h = nn.Dense(
        self.config.embed_dim, use_bias=self.config.bias, name="ff_down"
    )(h)
    h = quantizer(h, rngs[4])
    x = x + h  # Residual

    return quantizer(x, rngs[5])


def _embed(
    tokens: jax.Array,
    config: NetworkConfig,
    quantizer: quantization.Quantizer,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Token + learned positional embeddings."""
  token_key, pos_key, add_key = jax.random.split(rng, 3)

  pad_mask = (tokens != base.PAD_TOKEN)[..., None]  # (B, T, 1)

  # Token embeddings
  # Substitute PAD with 0 to keep indices in range.
  safe_ids = jnp.where(tokens == base.PAD_TOKEN, 0, tokens)
  tok_emb = nn.Embed(config.vocab_size, config.embed_dim, name="token_emb")(
      safe_ids
  )  # (B, T, D)
  tok_emb = quantizer(tok_emb, token_key)

  # Positional embeddings
  pos_indices = jnp.arange(tokens.shape[1])  # (T,)
  pos_emb = nn.Embed(config.sequence_length, config.embed_dim, name="pos_emb")(
      pos_indices,
  )  # (T, D)
  pos_emb = quantizer(pos_emb, pos_key)
  pos_emb = jnp.expand_dims(pos_emb, 0)  # (1, T, D)

  x = (tok_emb + pos_emb) * pad_mask  # zero-out PAD rows
  return quantizer(x, add_key), pad_mask  # (B, T, D)
