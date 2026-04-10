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

"""Environment for a reverse-copy task.

The agent receives a prompt sequence of integers and must produce the same
tokens **in reverse order** as its response.  Rewards encourage exact
reversal or per-token accuracy, making this a lightweight probe of positional
reasoning in Transformer policies.
"""

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for the ReverseCopyEnv."""

  prompt_length: int = 10
  kappa: float = 1  # Reward shaping k>1 guide, k<1 trap
  vocab_size: int = 2  # Keep vocab_size here for easy config
  reward_to_first_error: bool = True  # Reward only until first error
  target_reverse: bool = True  # If False, target is a forward copy
  target_increment: bool = False  # If True, target is (x + 1) % vocab_size

  def make(self) -> "ReverseCopyEnv":
    if self.kappa > 1:
      raise ValueError("kappa must be less than or equal to 1.")
    return ReverseCopyEnv(self)


class ReverseCopyEnv(base.Environment):
  """An environment for the Reverse Copy Task."""

  def __init__(self, config: EnvConfig):
    self.config = config

  @property
  def spec(self) -> base.EnvSpec:
    return base.EnvSpec(
        vocab_size=self.config.vocab_size,
        prompt_length=self.config.prompt_length,
        answer_length=self.config.prompt_length,
    )

  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Generates a random prompt."""
    return jax.random.randint(
        key=key,
        shape=(self.config.prompt_length,),
        minval=0,
        maxval=self.config.vocab_size,
        dtype=jnp.int32,
    )

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Calculates the reward based on the kappa-parameterized formula."""
    # Ensure the answer is the same length as the prompt for comparison.
    answer = answer[: self.config.prompt_length]
    return reversal_reward(
        prompt=prompt,
        response=answer,
        kappa=self.config.kappa,
        reward_to_first_error=self.config.reward_to_first_error,
        target_reverse=self.config.target_reverse,
        target_increment=self.config.target_increment,
        vocab_size=self.config.vocab_size,
    )


@jax.jit
def reversal_reward(
    prompt: jax.Array,
    response: jax.Array,
    kappa: float,
    reward_to_first_error: bool,
    target_reverse: bool,
    target_increment: bool,
    vocab_size: int,
) -> jax.Array:
  """Calculates the reward for the reversal task.

  The reward is defined as:
    R = kappa * c + (1 - kappa) * I(c=1)
  where c is the fraction of correctly reversed bits, and I(c=1) is an
  indicator for a perfect match. The calculation of c depends on
  reward_to_first_error. The target sequence can be a reverse copy, a forward
  copy, and can optionally have each token incremented by 1 modulo
  vocab_size.

  Args:
    prompt: The original prompt sequence.
    response: The response sequence to be evaluated.
    kappa: The reward shaping parameter, kappa <= 1.
    reward_to_first_error: If True, c is the fraction of correct tokens from the
      start until the first error. Otherwise, c is the total fraction of correct
      tokens.
    target_reverse: If True, the base target is the reverse of the prompt.
    target_increment: If True, the target elements are incremented by 1 mod
      vocab_size.
    vocab_size: The vocabulary size.

  Returns:
    A scalar reward. The maximum possible reward is 1.0.
  """
  length = prompt.shape[0]

  # Construct the target
  base_target = jax.lax.cond(
      target_reverse, lambda: jnp.flip(prompt, axis=0), lambda: prompt
  )

  def increment_fn(x):
    return (x + 1) % vocab_size

  target = jax.lax.cond(
      target_increment, lambda: increment_fn(base_target), lambda: base_target
  )
  correct_mask = response == target

  def true_fun():
    # Count the number of correct elements from the start until the first False.
    return jnp.sum(jnp.cumprod(correct_mask))

  def false_fun():
    # Calculate c, the total fraction of correctly placed bits.
    return jnp.sum(correct_mask)

  num_correct = jax.lax.cond(reward_to_first_error, true_fun, false_fun)

  c = jnp.astype(num_correct, jnp.float32) / jnp.astype(length, jnp.float32)

  # Calculate the completion bonus.
  is_perfect = num_correct == length
  completion_bonus = (1.0 - kappa) * jnp.astype(is_perfect, jnp.float32)

  # Calculate the shaping reward from partial credit.
  shaping_reward = kappa * c

  reward = shaping_reward + completion_bonus
  return reward
