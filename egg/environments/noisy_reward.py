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

"""Environment with noisy rewards for one action."""

from __future__ import annotations

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for the NoisyRewardEnv."""

  mu: float = 1.0  # Mean for action 1
  sigma: float = 1.0  # Standard deviation of action 1
  prompt_length: int = 1  # Length of the prompt (irrelevant)

  @property
  def vocab_size(self) -> int:
    return 2

  def make(self) -> "NoisyRewardEnv":
    return NoisyRewardEnv(self)


class NoisyRewardEnv(base.Environment):
  """An environment where action 0 gives reward 1, and action 1 gives noisy reward.

  The prompt is irrelevant in this environment. The reward only depends on the
  first token of the answer.
  """

  def __init__(self, config: EnvConfig):
    self.config = config

  @property
  def spec(self) -> base.EnvSpec:
    return base.EnvSpec(
        vocab_size=self.config.vocab_size,
        prompt_length=self.config.prompt_length,
        answer_length=1,
    )

  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Returns a dummy prompt."""
    return jnp.zeros((self.config.prompt_length,), dtype=jnp.int32)

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Calculates the reward based on the first action.

    Args:
      prompt: Unused.
      answer: The agent's answer sequence.
      key: JAX PRNG key for random number generation.

    Returns:
      A scalar reward.
    """
    del prompt  # Unused

    action = answer[0]

    # Reward is 1.0 if action is 0
    reward_action_0 = 1.0

    # Reward is sampled from N(mu, sigma^2) if action is 1
    reward_action_1 = (
        jax.random.normal(key, shape=()) * self.config.sigma + self.config.mu
    )

    # Select reward based on the action taken
    reward = jnp.where(action == 0, reward_action_0, reward_action_1)

    return reward.astype(jnp.float32)
