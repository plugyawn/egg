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

"""Vanilla decoder-only Transformer for Egg experiments (Flax).

Input : (B, T) int32   tokens
Output: (B, T, V) float32 logits
"""

from __future__ import annotations

import dataclasses

from egg import base
import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class NetworkConfig(base.MakeableConfig[nn.Module]):
  """Transformer hyperparameters."""

  vocab_size: int = 2  # Number of tokens in the vocabulary.
  sequence_length: int = 32  # max context length
  embed_dim: int = 64  # Embedding dimension
  num_heads: int = 4  # Number of attention heads in each block
  num_layers: int = 3  # Number of transformer blocks
  ff_dim: int = 128  # Feed-forward dimension
  bias: bool = False  # Whether to use bias in Dense layers.

  def make(self) -> Transformer:
    return Transformer(self)


class Transformer(nn.Module):
  """Decoder-only Transformer with causal attention."""

  config: NetworkConfig

  @nn.compact
  def __call__(self, tokens: jax.Array, **kwargs) -> jax.Array:
    """Forward pass: [B, T] int32 tokens -> [B, T, V] float32 logits."""
    del kwargs  # Unused.

    x, pad_mask = _embed(tokens, self.config)  # (B, T, D), (B, T, 1)
    causal_mask = nn.make_causal_mask(tokens)  # (B, 1, T, T)
    padding = nn.make_attention_mask(pad_mask[..., 0], pad_mask[..., 0])
    mask = nn.combine_masks(causal_mask, padding)  # (B, 1, T, T)

    for i in range(self.config.num_layers):
      x = TransformerBlock(self.config, name=f"layer_{i}")(x, mask)

    output_layer = nn.Dense(
        self.config.vocab_size, use_bias=self.config.bias, name="output"
    )

    return output_layer(x)  # (B, T, V)

  @nn.compact
  def decode_step(self, tokens: jax.Array, **kwargs) -> jax.Array:
    """Single-token cached decode step for autoregressive sampling."""
    del kwargs  # Unused.

    if tokens.ndim != 2 or tokens.shape[1] != 1:
      raise ValueError(
          "decode_step expects tokens with shape (batch_size, 1)."
      )

    cache_index = self.variable(
        "cache",
        "position",
        lambda: jnp.array(0, dtype=jnp.int32),
    )
    x = _embed_decode(tokens, self.config, cache_index.value)
    cache_index.value = cache_index.value + tokens.shape[1]

    for i in range(self.config.num_layers):
      x = TransformerBlock(
          self.config,
          decode=True,
          name=f"layer_{i}",
      )(x, mask=None)

    output_layer = nn.Dense(
        self.config.vocab_size, use_bias=self.config.bias, name="output"
    )
    return output_layer(x)  # (B, 1, V)


class TransformerBlock(nn.Module):
  """Single transformer block: pre-norm attention + feed-forward."""

  config: NetworkConfig
  decode: bool = False

  @nn.compact
  def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
    # Pre-norm self-attention
    h = nn.LayerNorm(use_bias=self.config.bias, name="ln_attn")(x)
    h = nn.MultiHeadDotProductAttention(
        num_heads=self.config.num_heads,
        use_bias=self.config.bias,
        decode=self.decode,
        name="self_attn",
    )(inputs_q=h, inputs_k=h, inputs_v=h, mask=mask)
    x = x + h  # Residual

    # Pre-norm feed-forward
    h = nn.LayerNorm(use_bias=self.config.bias, name="ln_ff")(x)
    h = nn.Dense(self.config.ff_dim, use_bias=self.config.bias, name="ff_up")(h)
    h = nn.relu(h)
    h = nn.Dense(
        self.config.embed_dim, use_bias=self.config.bias, name="ff_down"
    )(h)
    x = x + h  # Residual

    return x


def _embed(
    tokens: jax.Array, cfg: NetworkConfig
) -> tuple[jax.Array, jax.Array]:
  """Learned position embeddings with PAD_TOKEN masking.

  Args:
    tokens: (B, T) int32 tokens
    cfg: NetworkConfig

  Returns:
    x        : (B, T, D) float32   embedded sequence
    pad_mask : (B, T, 1) bool      True on non-PAD positions
  """
  pad_mask = (tokens != base.PAD_TOKEN)[..., None]  # (B, T, 1)

  # Token embeddings – substitute PAD with 0 to keep indices in range.
  safe_ids = jnp.where(tokens == base.PAD_TOKEN, 0, tokens)
  tok_emb = nn.Embed(cfg.vocab_size, cfg.embed_dim, name="token_emb")(
      safe_ids
  )  # (B, T, D)

  # Learned positional embeddings (broadcasted).
  pos_emb = nn.Embed(cfg.sequence_length, cfg.embed_dim, name="pos_emb")(
      jnp.arange(tokens.shape[1])
  )  # (T, D)
  pos_emb = jnp.expand_dims(pos_emb, 0)  # (1, T, D)

  x = (tok_emb + pos_emb) * pad_mask  # zero-out PAD rows
  return x, pad_mask


def _embed_decode(
    tokens: jax.Array,
    cfg: NetworkConfig,
    start_position: jax.Array,
) -> jax.Array:
  """Embed one decode chunk at its absolute position."""
  pad_mask = (tokens != base.PAD_TOKEN)[..., None]  # (B, 1, 1)

  safe_ids = jnp.where(tokens == base.PAD_TOKEN, 0, tokens)
  tok_emb = nn.Embed(cfg.vocab_size, cfg.embed_dim, name="token_emb")(safe_ids)

  pos_idx = start_position + jnp.arange(tokens.shape[1], dtype=jnp.int32)
  pos_emb = nn.Embed(cfg.sequence_length, cfg.embed_dim, name="pos_emb")(pos_idx)
  pos_emb = jnp.expand_dims(pos_emb, 0)  # (1, 1, D)

  return (tok_emb + pos_emb) * pad_mask
