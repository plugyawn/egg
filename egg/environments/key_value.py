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

"""Environment for a key-value lookup (associative recall) task.

Prompt layout:
    [k1, v1, k2, v2, ..., kn, vn, query]

Where:
    - (ki, vi) are key-value pairs (n = num_pairs)
    - query is the lookup key (always equal to one of the ki)

The agent must output the value vj where kj == query.
This tests content-addressable memory and retrieval in sequence models.
"""

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EnvConfig(base.MakeableConfig[base.Environment]):
  """Configuration for KVLookupEnv."""

  num_pairs: int = 5  # Number of key-value pairs per episode
  vocab_size: int = 10  # Tokens drawn from [0, vocab_size)

  def make(self) -> "KVLookupEnv":
    return KVLookupEnv(self)

  @property
  def prompt_length(self) -> int:
    """Length of the prompt presented to the agent."""
    return 2 * self.num_pairs + 1  # 2 tokens per pair + 1 query token


class KVLookupEnv(base.Environment):
  """Key-value recall environment (stateless).

  Keys and values are independent uniform integers in [0, vocab_size).
  The query key is randomly selected from the generated keys.
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
    """Samples a fresh episode."""
    key1, key2, key3 = jax.random.split(key, 3)

    # Generate keys and values
    keys = jax.random.randint(
        key1,
        shape=(self.config.num_pairs,),
        minval=0,
        maxval=self.config.vocab_size,
        dtype=jnp.int32,
    )
    values = jax.random.randint(
        key2,
        shape=(self.config.num_pairs,),
        minval=0,
        maxval=self.config.vocab_size,
        dtype=jnp.int32,
    )

    # Choose query key from the existing keys
    query_idx = jax.random.randint(
        key3, shape=(), minval=0, maxval=self.config.num_pairs
    )
    query = keys[query_idx]

    # Flatten to: k1, v1, k2, v2, ..., kn, vn, query
    kv_pairs = jnp.column_stack([keys, values]).ravel()
    prompt = jnp.concatenate([kv_pairs, jnp.array([query])], axis=0)

    return prompt

  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Returns +1 if agent's first token is the correct value, else -1."""
    return kv_lookup_reward(prompt, answer[0])


@jax.jit
def kv_lookup_reward(
    prompt: jax.Array,  # Shape: (2*num_pairs + 1,)
    prediction: jax.Array,  # Scalar prediction
) -> jax.Array:
  """Computes the associative recall reward."""
  # Parse prompt: [k1, v1, k2, v2, ..., kn, vn, query]
  keys = prompt[:-1:2]  # Extract keys: indices 0, 2, 4, ...
  values = prompt[1:-1:2]  # Extract values: indices 1, 3, 5, ...
  query = prompt[-1]  # Last element is the query
  num_pairs = keys.shape[0]

  # Get the indices of all matches
  match_indices_padded = jnp.where(
      keys == query, size=num_pairs, fill_value=-1
  )[0]

  # Since the query is always one of the keys, there's at least one match.
  # The valid indices will be >= 0, padded with -1.
  last_match_idx = jnp.max(match_indices_padded)
  target_value = values[last_match_idx]

  return jnp.where(prediction == target_value, 1.0, -1.0).astype(jnp.float32)
