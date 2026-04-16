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

"""Vanilla async trainer.

Here we inject a delay between the actor and the learner.
"""

from __future__ import annotations

import dataclasses
import json
import time
import typing as tp

from egg import base
from egg.lib import logging
from egg.lib import quantization
import jax
import pandas as pd


def _block_tree(tree):
  return jax.tree_util.tree_map(jax.block_until_ready, tree)


@dataclasses.dataclass(frozen=True)
class TrainerConfig(base.MakeableConfig["VanillaAsyncTrainer"]):
  """Configuration for the vanilla async trainer."""

  steps: int = 1000  # Total training steps.
  seed: int = 0  # Random seed.
  log_freq: int | None = None  # Log frequency, or logarithmic if None.
  sampler_delay: int = 0  # Delay of actor to learner.
  uniform_delay: bool = False  # If True, delay is sampled [0, sampler_delay].
  sampler_bits: int = 32  # Quantization bits for sampler params.
  deterministic: bool = True  # Deterministic quantization.
  log_details: bool = False  # Log detailed per-example metrics.
  log_learner_performance: bool = True  # Log learner timing metrics.

  def make(self) -> VanillaAsyncTrainer:
    return VanillaAsyncTrainer(config=self)


@dataclasses.dataclass(frozen=True)
class VanillaAsyncTrainer(base.Trainer):
  """Asynchronous actor-learner trainer with simulated parameter lag."""

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
    quantizer = quantization.QuantizeConfig(
        num_bits=self.config.sampler_bits,
        deterministic=self.config.deterministic,
    ).make()

    # Initialize RNG and learner state
    key = jax.random.PRNGKey(self.config.seed)
    key, init_key = jax.random.split(key)
    state = learner.init_state(init_key)

    # Initialize parameter history buffer
    # Invariant: params_history is ordered (oldest, ..., newest).
    hist_len = self.config.sampler_delay + 1
    params_history: tuple[tp.Any, ...] = tuple([state.params] * hist_len)

    @jax.jit
    def sample_batch_step(
        current_state: base.StateT,
        actor_params: tp.Any,
        key: jax.Array,
    ) -> tuple[base.Batch, base.Metrics]:
      actor_state = current_state.replace(params=actor_params)
      batch, _, actor_metrics = actor.sample_batch(actor_state, key)
      return batch, actor_metrics

    @jax.jit
    def train_on_batch_step(
        current_state: base.StateT,
        batch: base.Batch,
        key: jax.Array,
    ) -> tuple[base.StateT, base.Metrics]:
      return learner.step(current_state, batch, key)

    @jax.jit
    def evaluate_learner_performance(
        current_state: base.StateT,
        key: jax.Array,
    ) -> jax.Array:
      """Samples a batch using the current learner network to evaluate."""
      batch, _, _ = actor.sample_batch(current_state, key)
      return batch.rewards.mean()

    key, warm_quant_key, warm_sample_key, warm_train_key = jax.random.split(
        key, 4
    )
    warm_actor_params = quantizer(state.params, warm_quant_key)
    warm_batch, _ = sample_batch_step(state, warm_actor_params, warm_sample_key)
    warm_state, _ = train_on_batch_step(state, warm_batch, warm_train_key)
    _block_tree(warm_batch.prompts)
    _block_tree(warm_state.params)

    records: list[dict[str, tp.Any]] = []
    start = time.time()
    cumulative_reward = 0.0
    logger = logging.RunningMeanLogger()

    for step in range(1, self.config.steps + 1):
      # RNG for: (quantization, training step, evaluation)
      key, quant_key, sample_key, train_key, eval_key = jax.random.split(
          key, 5
      )

      # Choose which snapshot the actor will see this step.
      # Fixed worst-case delay 0=oldest. Uniform delay sample in [0, hist_len].
      if self.config.uniform_delay:
        # Split to avoid coupling the quantizer randomness with the index draw.
        quant_key, draw_key = jax.random.split(quant_key)
        idx = int(
            jax.random.randint(draw_key, shape=(), minval=0, maxval=hist_len)
        )
        chosen_params = params_history[idx]
      else:
        chosen_params = params_history[0]

      # Quantize the chosen params for the actor.
      actor_params = quantizer(chosen_params, quant_key)

      sample_t0 = time.perf_counter()
      batch, actor_metrics = sample_batch_step(state, actor_params, sample_key)
      _block_tree(batch.prompts)
      sample_time_s = time.perf_counter() - sample_t0

      train_t0 = time.perf_counter()
      state, learner_metrics = train_on_batch_step(state, batch, train_key)
      _block_tree(state.params)
      train_time_s = time.perf_counter() - train_t0

      metrics: base.Metrics = {
          "reward": batch.rewards.mean(),
          "sample_time_s": sample_time_s,
          "train_time_s": train_time_s,
          "total_step_time_s": sample_time_s + train_time_s,
      }
      if self.config.log_details:
        metrics.update({
            **{f"actor/{k}": v for k, v in actor_metrics.items()},
            **{f"learner/{k}": v for k, v in learner_metrics.items()},
        })

      # Update history with the new parameters (drop oldest, append newest)
      params_history = params_history[1:] + (state.params,)

      # Accumulate metrics
      logger.record(metrics)
      cumulative_reward += float(metrics["reward"])

      # Logging
      if self._should_log(step) or step == self.config.steps:
        metrics = logger.write()
        metrics["step"] = step
        metrics["time"] = time.time() - start
        metrics["cum_reward"] = cumulative_reward

        if self.config.log_learner_performance:
          learner_reward = evaluate_learner_performance(state, eval_key)
          metrics["learner_reward"] = float(learner_reward)

        records.append(metrics)

        # print a summary
        summary: dict[str, tp.Any] = {
            "step": metrics["step"],
            "time": _format_float(metrics["time"]),
            "reward": _format_float(metrics.get("reward", 0.0)),
            "cum_reward": _format_float(metrics["cum_reward"]),
        }
        if self.config.log_learner_performance:
          summary["learner_reward"] = _format_float(metrics["learner_reward"])
        print(json.dumps(summary))

    print("--- Training finished ---")
    return pd.DataFrame(records)


def _format_float(f: float) -> float:
  """Formats a float to 3 significant figures."""
  return float(format(f, ".3g"))
