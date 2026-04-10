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

"""Wrapper for restricting an environment to a finite set of prompts."""

from __future__ import annotations

import dataclasses

from egg import base
import jax


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for a wrapper that limits to a finite set of prompts."""

  inner_env_config: base.MakeableConfig[base.Environment]  # The wrapped env.
  num_prompts: int = 10  # The number of unique prompts to sample.
  seed: int = 42  # Seed for generating the finite set of prompts.

  def make(self) -> FinitePromptsWrapperEnv:
    """Creates an instance of the FinitePromptsWrapperEnv."""
    inner_env = self.inner_env_config.make()
    prompt_keys = jax.random.split(
        jax.random.PRNGKey(self.seed), self.num_prompts
    )
    prompts = jax.vmap(inner_env.get_prompt)(prompt_keys)
    return FinitePromptsWrapperEnv(
        inner_env=inner_env, config=self, prompts=prompts
    )


@dataclasses.dataclass(frozen=True)
class FinitePromptsWrapperEnv(base.Environment):
  """An environment wrapper that restricts prompts to a pre-generated finite set."""

  inner_env: base.Environment
  config: EnvConfig
  prompts: jax.Array  # Shape (num_prompts, prompt_length)

  @property
  def spec(self) -> base.EnvSpec:
    """Delegates spec to the inner environment."""
    return self.inner_env.spec

  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Returns a prompt randomly selected from the finite set."""
    prompt_idx = jax.random.randint(
        key, shape=(), minval=0, maxval=self.config.num_prompts
    )
    return self.prompts[prompt_idx]

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Delegates reward calculation to the inner environment."""
    return self.inner_env.get_reward(prompt, answer, key)
