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

"""Environment for a subsequence matching task.

This environment tests an agent's ability to find a prompt sequence within
its own generated output. The agent is given a prompt (a sequence of integers)
and must generate a response sequence of a fixed length. The reward is based
on whether the original prompt is a subsequence of the generated response.

This notion of "subsequence match" allows for gaps in between so that:
  - "ab" is a subsequence of "abcd"
  - "ad" is a subsequence of "abcd"
  - "ba" is not a subsequence of "abcd"
"""

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  prompt_length: int = 2  # Number of integers in the prompt.
  answer_length: int = 32  # Number of integers in the response.
  vocab_size: int = 5  # Integers are sampled from [0, vocab_size - 1].

  def make(self) -> "SubsequenceMatchEnv":
    """Returns a SubsequenceMatchEnv with the given configuration."""
    return SubsequenceMatchEnv(self)


class SubsequenceMatchEnv(base.Environment):
  """Environment for a subsequence matching task.

  The agent is given a prompt sequence of integers. The goal is to generate
  a response sequence of a fixed length such that the original prompt is a
  subsequence of the generated response.

  - Prompt: [prompt_length] A sequence of integers representing the prompt.
  - Answer: [answer_length] A sequence of integers representing the response.
  - Reward: +1 or -1 based on the prompt is a subsequence of the response.
  - State: This environment is stateless, tokens sampled uniformly.
  """

  def __init__(self, config: EnvConfig):
    self.config = config

  @property
  def spec(self) -> base.EnvSpec:
    return base.EnvSpec(
        vocab_size=self.config.vocab_size,
        prompt_length=self.config.prompt_length,
        answer_length=self.config.answer_length,
    )

  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Samples a new prompt to start a new episode."""
    prompt = jax.random.randint(
        key,
        (self.config.prompt_length,),
        minval=0,
        maxval=self.config.vocab_size,
    )
    return prompt

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Evaluates the agent's action and returns a reward."""
    # Actually allow a longer response via chain of thought.
    answer = answer[-self.config.answer_length :]
    return subsequence_reward(prompt, answer)


def _is_subsequence_jax(
    prompt: jax.Array,  # [P]
    response: jax.Array,  # [R]
) -> jnp.bool_:  # scalar bool
  """True iff `prompt` is a subsequence of `response` (JAX-friendly)."""
  prompt_len = prompt.shape[0]

  # Pad with a sentinel that is guaranteed not to collide with any real token.
  sentinel = jnp.asarray(-1, dtype=prompt.dtype)
  prompt_padded = jnp.concatenate([prompt, sentinel[None]], axis=0)  # [P+1]

  def step(matched: jnp.int32, token: jnp.int32) -> tuple[jnp.int32, None]:
    expected_next = prompt_padded[matched]
    matched += jnp.asarray(token == expected_next, jnp.int32)
    return matched, None

  matched_final, _ = jax.lax.scan(step, 0, response)  # carry is scalar
  return matched_final >= prompt_len  # bool scalar


@jax.jit
def subsequence_reward(prompt: jax.Array, response: jax.Array) -> jax.Array:
  """Returns +1 if `prompt` is a subsequence of `response`, else −1."""
  success = _is_subsequence_jax(prompt, response)
  return jnp.where(success, 1.0, -1.0).astype(jnp.float32)
