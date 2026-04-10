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

"""Environment for a parity-bit classification task.

The agent sees a sequence of binary tokens and must output **1** if the sum is
even, **0** if odd.  This task assesses the model's ability to aggregate
information across many positions.
"""

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  prompt_length: int = 10  # Number of bits in the prompt.

  def make(self) -> "ParityEnv":
    return ParityEnv(self)


class ParityEnv(base.Environment):
  """Even-parity detector."""

  def __init__(self, config: EnvConfig):
    self.config = config

  @property
  def spec(self) -> base.EnvSpec:
    return base.EnvSpec(
        vocab_size=2,  # binary tokens
        prompt_length=self.config.prompt_length,
        answer_length=1,
    )

  def get_prompt(self, key: jax.Array) -> jax.Array:
    return jax.random.randint(
        key=key,
        shape=(self.config.prompt_length,),
        minval=0,
        maxval=2,  # binary tokens
        dtype=jnp.int32,
    )

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    return parity_reward(prompt, answer[0])


@jax.jit
def parity_reward(prompt: jax.Array, prediction: jax.Array) -> jax.Array:
  """Returns +1 if `prediction` matches even-parity label, else -1."""
  true_label = jnp.asarray((prompt.sum() & 1) ^ 1, jnp.int32)  # 1 = even
  return jnp.where(prediction == true_label, 1.0, -1.0)
