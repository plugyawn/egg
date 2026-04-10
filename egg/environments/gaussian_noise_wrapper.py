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

"""Wrapper for adding Gaussian noise to rewards in an environment."""

from __future__ import annotations

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for a wrapper that adds Gaussian noise to rewards."""

  inner_env_config: base.MakeableConfig[base.Environment]  # The wrapped env.
  noise_std: float = 1.0  # The standard deviation of the Gaussian noise.

  def make(self) -> GaussianNoiseWrapperEnv:
    """Creates an instance of the GaussianNoiseWrapperEnv."""
    return GaussianNoiseWrapperEnv(
        inner_env=self.inner_env_config.make(), config=self
    )


@dataclasses.dataclass(frozen=True)
class GaussianNoiseWrapperEnv(base.Environment):
  """An environment wrapper that adds IID Gaussian noise to the rewards.

  This wrapper takes an existing environment and modifies its reward function.
  It adds zero-mean Gaussian noise with the specified standard deviation
  to the reward returned by the wrapped environment.
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
    """Calculates reward, adding Gaussian noise to the inner env reward."""
    # Get the reward from the underlying environment first.
    base_reward = self.inner_env.get_reward(prompt, answer, key)

    # Generate Gaussian noise.
    noise = (
        jax.random.normal(key, shape=base_reward.shape) * self.config.noise_std
    )

    # Add the noise to the base reward.
    final_reward = base_reward + noise.astype(jnp.float32)

    return final_reward
