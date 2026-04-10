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

"""Wrapper for adding Bernoulli noise to rewards in an environment."""

from __future__ import annotations

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for a wrapper that adds Bernoulli noise to rewards."""

  inner_env_config: base.MakeableConfig[base.Environment]  # The wrapped env.
  p_noise: float = 0.0  # Probability of overriding the reward with noise.
  min_val: float = 0.0  # Minimum value for the noise override.
  max_val: float = 1.0  # Maximum value for the noise override.

  def make(self) -> BernoulliNoiseWrapperEnv:
    """Creates an instance of the BernoulliNoiseWrapperEnv."""
    return BernoulliNoiseWrapperEnv(
        inner_env=self.inner_env_config.make(), config=self
    )


@dataclasses.dataclass(frozen=True)
class BernoulliNoiseWrapperEnv(base.Environment):
  """An environment wrapper that adds Bernoulli noise to the rewards.

  This wrapper takes an existing environment and modifies its reward function.
  With probability `p_noise`, it overrides the base reward with a value
  chosen uniformly from {min_val, max_val}. Otherwise, it returns the base
  reward.
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
    """Calculates reward, adding Bernoulli noise to the inner env reward."""
    # Get the reward from the underlying environment first.
    base_reward = self.inner_env.get_reward(prompt, answer, key)

    key_override, key_value = jax.random.split(key)

    # Determine if we should override the reward.
    should_override = jax.random.bernoulli(key_override, self.config.p_noise)

    # Determine which value to use for overriding (50/50 min_val or max_val).
    override_val = jnp.where(
        jax.random.bernoulli(key_value, 0.5),
        self.config.max_val,
        self.config.min_val,
    )

    # Apply the override or keep the base reward.
    final_reward = jnp.where(should_override, override_val, base_reward).astype(
        jnp.float32
    )

    return final_reward
