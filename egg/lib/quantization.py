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

"""Simulating quantization effects in jax."""

import dataclasses
import typing as tp

from egg import base
import jax
import jax.numpy as jnp

T = tp.TypeVar("T")


class Quantizer(tp.Protocol):

  def __call__(self, tree: T, key: jax.Array) -> T:
    """Quantizes a PyTree of arrays."""


def null_quantizer(tree: T, key: jax.Array) -> T:
  """Returns the input tree without quantization."""
  del key  # Unused.
  return tree


@dataclasses.dataclass
class QuantizeConfig(base.MakeableConfig[Quantizer]):
  num_bits: int = 32  # How many bits to quantize to.
  deterministic: bool = True  # Whether to quantize deterministically.

  def make(self) -> Quantizer:
    if self.deterministic:
      return _make_deterministic_quantizer(self.num_bits)
    else:
      return _make_stochastic_quantizer(self.num_bits)


def _deterministic_quantize(arr: jnp.ndarray, num_bits: int) -> jnp.ndarray:
  """Deterministic uniform rounding to `num_bits` precision."""
  if num_bits >= 32 or arr.size == 0:
    return arr

  max_abs = jnp.max(jnp.abs(arr))
  scale = jnp.where(max_abs == 0, 1.0, (2 ** (num_bits - 1) - 1) / max_abs)
  return jnp.round(arr * scale) / scale


def _make_deterministic_quantizer(num_bits: int) -> Quantizer:
  """Returns a deterministic quantizer."""

  def quantizer(tree: T, key: jax.Array) -> T:
    del key  # Unused.
    return jax.tree_util.tree_map(
        lambda arr: _deterministic_quantize(arr, num_bits), tree
    )

  return quantizer


def _stochastic_quantize(
    arr: jnp.ndarray,
    key: jax.Array,
    num_bits: int,
) -> jnp.ndarray:
  """Stochastic rounding to `num_bits` precision."""
  if num_bits >= 32 or arr.size == 0:
    return arr

  max_abs = jnp.max(jnp.abs(arr))
  scale = jnp.where(max_abs == 0, 1.0, (2 ** (num_bits - 1) - 1) / max_abs)
  scaled = arr * scale

  floor_vals = jnp.floor(scaled)
  frac = scaled - floor_vals

  rnd = jax.random.uniform(key, arr.shape, dtype=arr.dtype)
  rounded = jnp.where(rnd < frac, floor_vals + 1.0, floor_vals)
  return rounded / scale


def _make_stochastic_quantizer(num_bits: int) -> Quantizer:
  """Returns a stochastic quantizer."""

  def quantizer(tree: T, key: jax.Array) -> T:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    quantized_leaves = [
        _stochastic_quantize(leaf, k, num_bits) for leaf, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, quantized_leaves)

  return quantizer
