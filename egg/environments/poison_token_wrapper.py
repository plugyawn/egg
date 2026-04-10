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

"""Wrapper for penalizing a specific token in an environment."""

from __future__ import annotations

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for a wrapper that adds a 'poison token' penalty."""

  inner_env_config: base.MakeableConfig[base.Environment]  # The wrapped env.
  poison_token: int = 0  # The token to penalize.
  poison_reward: float = -10.0  # The reward to apply if the token is found.

  def make(self) -> PoisonTokenWrapperEnv:
    """Creates an instance of the PoisonTokenWrapperEnv."""
    return PoisonTokenWrapperEnv(
        inner_env=self.inner_env_config.make(), config=self
    )


@dataclasses.dataclass(frozen=True)
class PoisonTokenWrapperEnv(base.Environment):
  """An environment wrapper that penalizes a specific 'poison' token.

  This wrapper takes an existing environment and modifies its reward function.
  If the agent's 'answer' sequence contains the specified `poison_token`,
  it returns a large negative `poison_reward`. Otherwise, it returns the
  reward from the original, wrapped environment.
  """

  inner_env: base.Environment
  config: EnvConfig

  @property
  def spec(self) -> base.EnvSpec:
    """Delegates spec to the inner environment."""
    return self.inner_env.spec

  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Delegates prompt generation to the inner environment."""
    return self.inner_env.get_prompt(key)

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Calculates reward, applying a penalty if the poison token is found."""
    # Get the reward from the underlying environment first.
    base_reward = self.inner_env.get_reward(prompt, answer, key)

    # Check if the poison token exists in the answer sequence.
    contains_poison = jnp.any(answer == self.config.poison_token)

    # Determine the penalty amount: poison_reward if present, 0 otherwise.
    penalty = jnp.where(contains_poison, self.config.poison_reward, 0.0)

    # Add the penalty to the base reward.
    final_reward = base_reward + penalty.astype(jnp.float32)

    return final_reward
