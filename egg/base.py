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

"""Core interfaces and datatypes for Egg experiments."""

from __future__ import annotations

import abc
import typing as tp

from flax.training import train_state as ts
import jax
import pandas as pd

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
PAD_TOKEN: int = -1
Metrics: tp.TypeAlias = dict[str, jax.Array]
Params: tp.TypeAlias = tp.Any  # Pytree of jax.Array
StateT = tp.TypeVar("StateT", bound=ts.TrainState)
T = tp.TypeVar("T")


class MakeableConfig(abc.ABC, tp.Generic[T]):
  """Makeable config interface, should be serializable."""

  @abc.abstractmethod
  def make(self) -> T:
    """Returns an object of type T with the given configuration."""
    ...


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------
class EnvSpec(tp.NamedTuple):
  """Specification for an environment."""

  vocab_size: int  # The size of the vocabulary.
  prompt_length: int  # The expected length of the prompt sequence.
  answer_length: int  # The expected length of the answer sequence.


class Environment(abc.ABC):
  """Stateless oracle: supplies prompts & scores answers."""

  @property
  @abc.abstractmethod
  def spec(self) -> EnvSpec:
    """Returns environment specification."""
    ...

  @abc.abstractmethod
  def get_prompt(self, key: jax.Array) -> jax.Array:
    """Returns a single new prompt."""
    ...

  @abc.abstractmethod
  def get_reward(
      self, prompt: jax.Array, answer: jax.Array, key: jax.Array
  ) -> jax.Array:
    """Returns the reward for a single (prompt, answer) pair."""
    ...


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
class Batch(tp.NamedTuple):
  """Minimal bundle passed from Actor → Learner every step."""

  prompts: jax.Array  # (B, prompt_len)
  answers: jax.Array  # (B, answer_len)
  rewards: jax.Array  # (B,)
  sample_log_probs: jax.Array  # (B, total_seq_len)
  aux: Metrics


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
class Actor(abc.ABC, tp.Generic[StateT]):
  """A component that reads the shared state and emits a Batch."""

  @abc.abstractmethod
  def sample_batch(
      self,
      state: StateT,
      key: jax.Array,
  ) -> tuple[Batch, StateT, Metrics]:
    """Returns (batch, possibly-annotated_state, actor_metrics)."""
    raise NotImplementedError


class LossFn(tp.Protocol[StateT]):
  """Computes a scalar loss and auxiliary metrics.

  The Learner calls jax.value_and_grad on this internally.
  Implementations live in the losses/ directory.
  """

  def __call__(
      self,
      params: Params,
      state: StateT,
      batch: Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, Metrics]:
    """Returns (scalar_loss, auxiliary_metrics)."""
    ...


class Learner(tp.Protocol[StateT]):
  """A learner that updates the policy given a Batch of experience."""

  def init_state(self, key: jax.Array) -> StateT:
    """Initializes the learner state (params + optimizer state)."""
    ...

  def step(
      self,
      state: StateT,
      batch: Batch,
      key: jax.Array,
  ) -> tuple[StateT, Metrics]:
    """Returns (new_state, learner_metrics)."""
    ...


class Trainer(tp.Protocol[StateT]):
  """A callable that runs the Actor → Learner loop."""

  def __call__(
      self,
      actor: Actor[StateT],
      learner: Learner[StateT],
  ) -> pd.DataFrame:
    ...
