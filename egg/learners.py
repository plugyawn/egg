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

"""Learner implementations for Egg."""

from __future__ import annotations

import dataclasses

from egg import base
from flax import linen as nn
from flax.training import train_state as ts
import jax
import jax.numpy as jnp
import optax


@dataclasses.dataclass(frozen=True)
class LearnerConfig(base.MakeableConfig["SgdLearner"]):
  """Makeable config that produces an SgdLearner."""

  model_config: base.MakeableConfig[nn.Module]  # Neural network.
  loss_config: base.MakeableConfig[base.LossFn]  # Loss function.
  prompt_length: int  # Length of prompt portion of each sequence.
  learning_rate: float = 1e-4  # Learning rate for Adam optimizer.

  def make(self) -> SgdLearner:
    return SgdLearner(
        model=self.model_config.make(),
        loss_fn=self.loss_config.make(),
        opt=optax.adam(self.learning_rate),
        prompt_length=self.prompt_length,
    )


@dataclasses.dataclass(frozen=True)
class SgdLearner(base.Learner):
  """A plain-vanilla learner: loss → grads → optimizer update."""

  model: nn.Module  # The neural network.
  loss_fn: base.LossFn  # Scalar loss function.
  opt: optax.GradientTransformation  # Optimizer (e.g. Adam).
  prompt_length: int  # Prompt length for dummy init.

  def init_state(self, key: jax.Array) -> ts.TrainState:
    """Initialises TrainState (params + optimiser state)."""
    dummy_seq = jnp.full(
        (1, self.prompt_length), base.PAD_TOKEN, dtype=jnp.int32
    )
    params_key, noise_key = jax.random.split(key)
    params = self.model.init(
        {"params": params_key, "noise": noise_key}, dummy_seq
    )["params"]
    return ts.TrainState.create(
        apply_fn=self.model.apply,
        params=params,
        tx=self.opt,
    )

  def step(
      self,
      state: ts.TrainState,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[ts.TrainState, base.Metrics]:
    """Runs loss_fn, computes grads, applies update."""
    (loss, loss_metrics), grads = jax.value_and_grad(
        self.loss_fn,
        argnums=0,
        has_aux=True,
    )(state.params, state, batch, key)
    state = state.apply_gradients(grads=grads)
    return state, {
        **loss_metrics,
        "loss": loss,
        "grad_norm": optax.tree.norm(grads),
    }
