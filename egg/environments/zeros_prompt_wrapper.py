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

"""Wrapper for making some pct of prompts all zeros.

This is meant as a metaphor for "easy prompts" that are not very challenging
for LLMs. The zero prompts come up more often than others, and then optionally
also have an additional noise_std on the rewards.
"""

from __future__ import annotations

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for a wrapper that sometimes zeros prompts and adds noise."""

  inner_env_config: base.MakeableConfig[base.Environment]  # The wrapped env.
  prob_zero: float = 0.0  # Probability of zeroing out all tokens.
  zero_noise_std: float = 0.0  # Std of Gaussian noise to add to zero rewards.

  def make(self) -> ZerosPromptWrapperEnv:
    """Creates an instance of the ZerosPromptWrapperEnv."""
    if not (0.0 <= self.prob_zero <= 1.0):
      raise ValueError("prob_zero must be in [0, 1].")
    if self.zero_noise_std < 0.0:
      raise ValueError("zero_noise_std must be non-negative.")
    return ZerosPromptWrapperEnv(
        inner_env=self.inner_env_config.make(), config=self
    )


@dataclasses.dataclass(frozen=True)
class ZerosPromptWrapperEnv(base.Environment):
  """An environment wrapper that sometimes makes prompts all zeros.

  If a prompt is zeroed out, IID Gaussian noise is added to the reward.
  """

  inner_env: base.Environment
  config: EnvConfig

  @property
  def spec(self) -> base.EnvSpec:
    """Delegates spec to the inner environment."""
    return self.inner_env.spec

  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Delegates prompt generation and sometimes zeros it out."""
    key_prompt, key_override = jax.random.split(key)
    prompt = self.inner_env.get_prompt(key_prompt)
    zero_prompt = jnp.zeros_like(prompt)
    use_override = jax.random.uniform(key_override) < self.config.prob_zero
    return jnp.where(use_override, zero_prompt, prompt)

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Calculates reward, adds noise if prompt is all zeros."""
    key_reward, key_noise = jax.random.split(key)
    # Get the reward from the underlying environment first.
    base_reward = self.inner_env.get_reward(prompt, answer, key_reward)

    # Check if everything is zero.
    is_all_zero = jnp.all(prompt == 0) & jnp.all(answer == 0)

    # Generate Gaussian noise.
    noise = (
        jax.random.normal(key_noise, shape=base_reward.shape)
        * self.config.zero_noise_std
    )
    add_noise = jnp.where(is_all_zero, noise.astype(jnp.float32), 0.0)

    # Add the noise to the base reward.
    return base_reward + add_noise
