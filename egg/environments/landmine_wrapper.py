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

"""Wrapper for adding a 'landmine' sequence penalty to an environment."""

from __future__ import annotations

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for a wrapper that adds a 'landmine' sequence penalty."""

  inner_env_config: base.MakeableConfig[base.Environment]
  # The specific, rare sequence of tokens that acts as a landmine.
  landmine_sequence: jax.Array | None = None
  # The mean catastrophic reward to apply if the landmine sequence is generated.
  landmine_reward: float = 0.0
  # The standard deviation of the landmine reward.
  landmine_std: float = 0.0

  def make(self) -> "LandmineWrapperEnv":
    """Creates an instance of the LandmineWrapperEnv."""
    return LandmineWrapperEnv(
        inner_env=self.inner_env_config.make(),
        config=self,
        # Pre-convert the sequence to a JAX array for use in JIT functions.
        landmine_sequence=jnp.array(self.landmine_sequence, dtype=jnp.int32),
    )


@dataclasses.dataclass(frozen=True)
class LandmineWrapperEnv(base.Environment):
  """An environment wrapper that penalizes a specific 'landmine' sequence.

  This wrapper takes an existing environment and modifies its reward function.
  If the agent's 'answer' contains the specified `landmine_sequence` as a
  subsequence, it ADDS a stochastic penalty sampled from a Normal distribution
  to the reward from the inner environment.
  """

  inner_env: base.Environment
  config: EnvConfig
  landmine_sequence: jax.Array

  @property
  def spec(self) -> base.EnvSpec:
    """Delegates spec to the inner environment."""
    return self.inner_env.spec

  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Delegates prompt generation to the inner environment."""
    return self.inner_env.get_prompt(key)

  def _is_subsequence(self, answer: jax.Array) -> jnp.bool_:
    """JAX-friendly check if the landmine is a subsequence of the answer."""
    landmine_len = self.landmine_sequence.shape[0]
    # Pad with a sentinel value that won't exist in the vocab
    padded_landmine = jnp.pad(
        self.landmine_sequence, (0, 1), constant_values=-1
    )

    def scan_fn(carry_idx, answer_token):
      # carry_idx is our current position in the landmine sequence
      is_match = answer_token == padded_landmine[carry_idx]
      # Advance our position in the landmine sequence if we get a match
      new_idx = carry_idx + is_match
      return new_idx, None

    # Scan over the agent's answer, trying to match the landmine sequence
    final_idx, _ = jax.lax.scan(scan_fn, 0, answer)
    # If we matched all tokens, the final index will be >= the landmine length
    return final_idx >= landmine_len

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Calculates reward, adding a stochastic penalty if the landmine is found."""
    # Get the reward from the underlying environment first.
    base_reward = self.inner_env.get_reward(prompt, answer, key)

    # Check if the landmine sequence exists in the answer.
    hit_landmine = self._is_subsequence(answer)

    # Sample the penalty amount.
    noise = jax.random.normal(key) * self.config.landmine_std
    penalty_sample = self.config.landmine_reward + noise

    # Determine the penalty amount: sampled penalty if present, 0 otherwise.
    penalty = jnp.where(hit_landmine, penalty_sample, 0.0)

    # Add the penalty to the base reward.
    final_reward = base_reward + penalty.astype(jnp.float32)
    return final_reward
