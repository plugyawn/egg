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

"""Vanilla synchronous trainer."""

from __future__ import annotations

import dataclasses
import json
import time
import typing as tp

from egg import base
from egg.lib import logging
import jax
import jax.numpy as jnp
import pandas as pd


@dataclasses.dataclass
class TrainerConfig(base.MakeableConfig["VanillaSyncTrainer"]):
  """Configuration for the vanilla sync trainer."""

  steps: int = 1000  # Total number of training steps.
  seed: int = 0  # Random seed for PRNG.
  log_freq: int | None = None  # Log every int steps, or logarithmic if None.
  log_details: bool = False  # Log detailed per-example metrics.

  def make(self) -> VanillaSyncTrainer:
    return VanillaSyncTrainer(config=self)


@dataclasses.dataclass(frozen=True)
class VanillaSyncTrainer(base.Trainer):
  """A simple synchronous actor-learner trainer."""

  config: TrainerConfig

  def _should_log(self, step: int) -> bool:
    if self.config.log_freq is None:
      return logging.logarithmic_logging(step)
    else:
      return step % self.config.log_freq == 0

  def __call__(
      self,
      actor: base.Actor[base.StateT],
      learner: base.Learner[base.StateT],
  ) -> pd.DataFrame:
    # Initialize RNG and learner state
    key = jax.random.PRNGKey(self.config.seed)
    key, init_key = jax.random.split(key)
    state = learner.init_state(init_key)

    @jax.jit
    def train_step(
        state: base.StateT, key: jax.Array
    ) -> tuple[base.StateT, jax.Array, base.Metrics]:
      # Actor: sample a batch
      key, sample_key = jax.random.split(key)
      batch, state, actor_metrics = actor.sample_batch(state, sample_key)

      # Learner: compute grads + update state
      key, grad_key = jax.random.split(key)
      state, learner_metrics = learner.step(state, batch, grad_key)

      metrics = {
          "reward": batch.rewards.mean(),
          "pos_reward": jnp.sum(batch.rewards > 0),
      }
      if self.config.log_details:
        metrics.update({
            **{f"actor/{k}": v for k, v in actor_metrics.items()},
            **{f"learner/{k}": v for k, v in learner_metrics.items()},
        })
      return state, key, metrics

    records: list[dict[str, tp.Any]] = []
    start = time.time()
    cumulative_reward = 0.0
    cumulative_positive = 0.0
    logger = logging.RunningMeanLogger()

    for step in range(1, self.config.steps + 1):
      state, key, metrics = train_step(state, key)

      # Accumulate rewards
      logger.record(metrics)
      cumulative_reward += float(metrics["reward"])
      cumulative_positive += float(metrics["pos_reward"])

      # Logging
      if self._should_log(step) or step == self.config.steps:
        log_metrics = logger.write()
        log_metrics["step"] = step
        log_metrics["time"] = time.time() - start
        log_metrics["cum_reward"] = cumulative_reward
        log_metrics["cum_positive"] = cumulative_positive
        records.append(log_metrics)

        # print a summary
        summary = {
            "step": log_metrics["step"],
            "time": _format_float(log_metrics["time"]),
            "reward": _format_float(log_metrics.get("reward", 0.0)),
            "cum_reward": _format_float(log_metrics["cum_reward"]),
            "cum_positive": _format_float(log_metrics["cum_positive"]),
        }
        print(json.dumps(summary))

    print("--- Training finished ---")
    return pd.DataFrame(records)


def _format_float(f: float) -> float:
  """Formats a float to 3 significant figures."""
  return float(format(f, ".3g"))
