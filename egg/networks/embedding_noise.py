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

"""Network definitions for egg experiments, using Flax.

This is a transformer that adds a noise term to the embeddings.
"""

import dataclasses

from egg import base
import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class NetworkConfig(base.MakeableConfig[nn.Module]):
  """Configuration for the Transformer model."""

  vocab_size: int = 10  # The number of tokens in the vocabulary.
  sequence_length: int = 32  # The maximum length of a sequence.
  embed_dim: int = 64  # Embedding dimension.
  num_heads: int = 1  # The number of attention heads in each Transformer block.
  num_layers: int = 3  # The number of Transformer blocks to stack.
  ff_dim: int = 128  # The feed-forward dimension of each Transformer block.
  bias: bool = False  # Whether to use bias in Dense and Attention layers.
  mismatch_scale: float = 0.01  # The scale of the noise to add to embeddings.
  fixed_noise: bool = False  # Whether to use a fixed noise key.

  def make(self) -> "Transformer":
    """Returns a Transformer model with the given configuration."""
    return Transformer(self)


class Transformer(nn.Module):
  """A minimal, decoder-only Transformer model.

  This model is designed for autoregressive sequence generation tasks.
  It takes a sequence of integer tokens as input and outputs logits
  over the vocabulary for each position.
  """

  config: NetworkConfig

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    """Performs the forward pass of the Transformer model.

    Args:
      x: The input token sequence of shape `(batch_size, sequence_length)`. An
        RNG stream named 'noise' can be passed to the `apply` method to inject
        stochasticity.

    Returns:
      The output logits of shape `(batch_size, sequence_length, vocab_size)`.
    """

    # 1. Embeddings and positional encoding
    pad_mask = (x != base.PAD_TOKEN)[..., None]  # (B, T, 1)

    # Substitute PAD with 0 to keep indices in range.
    safe_ids = jnp.where(x == base.PAD_TOKEN, 0, x)

    token_emb = nn.Embed(
        num_embeddings=self.config.vocab_size,
        features=self.config.embed_dim,
        name="token_embedding",
    )(safe_ids)
    pos_emb = nn.Embed(
        num_embeddings=self.config.sequence_length,
        features=self.config.embed_dim,
        name="positional_embedding",
    )
    positions = jnp.arange(0, x.shape[1])
    x_emb = token_emb + pos_emb(positions)

    x_emb = x_emb * pad_mask  # Zero-out PAD embeddings

    # 2. Optional noise injection (matches original Colab)
    # This requires passing an RNG stream named 'noise' during the `apply` call.
    if self.has_rng("noise") and self.config.mismatch_scale > 0.0:
      if self.config.fixed_noise:
        # Use a fixed key for deterministic, but different, noise
        noise_key = jax.random.PRNGKey(0)
      else:
        # Use the provided 'noise' RNG stream for stochasticity
        noise_key = self.make_rng("noise")
      noise = jax.random.normal(noise_key, x_emb.shape)
      x_emb = x_emb + noise * self.config.mismatch_scale
      x_emb = x_emb * pad_mask  # Ensure PADs remain zero after noise

    # 3. Transformer blocks
    causal_mask = nn.make_causal_mask(x)
    padding_mask = nn.make_attention_mask(pad_mask[..., 0], pad_mask[..., 0])
    mask = nn.combine_masks(causal_mask, padding_mask)  # (B, 1, T, T)

    for i in range(self.config.num_layers):
      x_emb = _TransformerBlock(
          embed_dim=self.config.embed_dim,
          num_heads=self.config.num_heads,
          ff_dim=self.config.ff_dim,
          bias=self.config.bias,
          name=f"transformer_block_{i}",
      )(x_emb, mask=mask)

    # 4. Output logits
    logits = nn.Dense(
        self.config.vocab_size, use_bias=self.config.bias, name="output_logits"
    )(x_emb)
    return logits


class _TransformerBlock(nn.Module):
  """A single, standard Transformer block.

  This block consists of a multi-head self-attention layer followed by a
  feed-forward network. Each is wrapped with a residual connection and
  layer normalization.
  """

  embed_dim: int
  num_heads: int
  ff_dim: int
  bias: bool

  @nn.compact
  def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    """Performs a forward pass through the Transformer block.

    Args:
      x: The input sequence of shape `(batch_size, sequence_length, embed_dim)`.
      mask: The attention mask.

    Returns:
      The output sequence of the same shape as the input.
    """
    # Self-attention sub-layer
    attn_output = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        use_bias=self.bias,
        name="self_attention",
    )(inputs_q=x, inputs_kv=x, mask=mask)

    # Residual connection and layer normalization
    x = nn.LayerNorm(use_bias=self.bias, name="ln1")(attn_output) + x

    # Feed-forward sub-layer
    mlp = nn.Sequential(
        [
            nn.Dense(self.ff_dim, use_bias=self.bias, name="ff_dense1"),
            nn.relu,
            nn.Dense(self.embed_dim, use_bias=self.bias, name="ff_dense2"),
        ],
        name="feed_forward",
    )
    ff_output = mlp(x)

    # Residual connection and layer normalization
    x = nn.LayerNorm(use_bias=self.bias, name="ln2")(ff_output) + x
    return x
