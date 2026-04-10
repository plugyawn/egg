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

"""Simple linear network.

We mainly use this to study the effects of noise and precision on the
training and sampling behavior.

This model allows for several types of numerical perturbations:
1.  Additive Gaussian noise on embeddings (controlled by `noise_scale`).
2.  Computational precision of the Dense layer (controlled by
`use_low_precision`,
    affecting matrix multiplication precision, e.g., on TPUs).
3.  Value quantization of embeddings and logits (controlled by
`quantize_config`,
    simulating reduced bit-width representations).
"""

from __future__ import annotations

import dataclasses

from egg import base
from egg.lib import quantization
import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class NetworkConfig(base.MakeableConfig[nn.Module]):
  """Configuration for the simple linear network."""

  vocab_size: int
  embedding_dim: int
  noise_scale: float = 0.0  # Scale of Gaussian noise added to embeddings.
  use_low_precision: bool = False  # Whether to use low precision in layer.
  num_bits: int = 32  # Number of bits to quantize embeddings and logits.
  deterministic: bool = True  # Whether to quantize deterministically.

  def make(self) -> SimpleModel:
    return SimpleModel(self)


class SimpleModel(nn.Module):
  """Simple linear network."""

  config: NetworkConfig

  def setup(self):
    self.quantizer = quantization.QuantizeConfig(
        num_bits=self.config.num_bits,
        deterministic=self.config.deterministic,
    ).make()
    self.embed = nn.Embed(
        num_embeddings=self.config.vocab_size,
        features=self.config.embedding_dim,
    )

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    # (B, T) -> (B, T, D)
    # Create a mask for non-PAD tokens.
    pad_mask = (x != base.PAD_TOKEN)[..., None]  # (B, T, 1)

    # Substitute PAD_TOKEN with 0 for embedding lookup to avoid index errors.
    safe_x = jnp.where(x == base.PAD_TOKEN, 0, x)
    embedding = self.embed(safe_x)

    # Zero out embeddings for PAD_TOKEN positions.
    embedding = embedding * pad_mask

    # All random operations below use keys split from 'noise'
    if not self.has_rng("noise"):
      raise ValueError("RNG key 'noise' is required for this model.")
    base_rng = self.make_rng("noise")
    k1, k2, k3 = jax.random.split(base_rng, 3)

    # Optional: Add noise
    if self.config.noise_scale > 0:
      noise = jax.random.normal(k1, embedding.shape) * self.config.noise_scale
      embedding = embedding + noise
      embedding = embedding * pad_mask  # Reapply mask after adding noise

    # Optional: Quantize embeddings
    embedding = self.quantizer(embedding, k2)
    embedding = embedding * pad_mask  # Reapply mask after quantization

    # Dense layer with optional low precision computation
    precision = (
        jax.lax.Precision.DEFAULT
        if self.config.use_low_precision
        else jax.lax.Precision.HIGHEST
    )
    # (B, T, D) -> (B, T, V)
    logits = nn.Dense(features=self.config.vocab_size, precision=precision)(
        embedding
    )

    # Optional: Quantize logits
    logits = self.quantizer(logits, k3)
    # Note: We don't typically mask the logits output based on input padding

    return logits
