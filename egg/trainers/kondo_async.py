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

"""Async trainer with proper Kondo screening and compacted backward passes."""

from __future__ import annotations

import dataclasses
import json
import time
import typing as tp

from egg import base
from egg.lib import logging
from egg.lib import quantization
from egg.losses import common
import jax
import jax.numpy as jnp
import pandas as pd


def _block_tree(tree):
  return jax.tree_util.tree_map(jax.block_until_ready, tree)


@dataclasses.dataclass(frozen=True)
class TrainerConfig(base.MakeableConfig["KondoAsyncTrainer"]):
  """Configuration for the proper Kondo async trainer."""

  steps: int = 1000
  seed: int = 0
  log_freq: int | None = None
  sampler_delay: int = 0
  uniform_delay: bool = False
  sampler_bits: int = 32
  deterministic: bool = True
  log_details: bool = False
  log_learner_performance: bool = True

  pct_learn: float = 0.5
  priority: str = "delight"
  alpha_additive: float = 0.5
  use_grouped_baseline: bool = True
  num_groups: int | None = None

  def make(self) -> "KondoAsyncTrainer":
    if not (0.0 < self.pct_learn <= 1.0):
      raise ValueError("pct_learn must be in (0, 1].")
    return KondoAsyncTrainer(config=self)


@dataclasses.dataclass(frozen=True)
class KondoAsyncTrainer(base.Trainer):
  """Async trainer that screens a full batch and backprops only on kept rows."""

  config: TrainerConfig

  def _should_log(self, step: int) -> bool:
    if self.config.log_freq is None:
      return logging.logarithmic_logging(step)
    return step % self.config.log_freq == 0

  def _can_reuse_sampler_logprobs_for_screen(
      self,
      actor_cfg: tp.Any,
  ) -> bool:
    """Returns True if sampler log-probs exactly match learner log-probs."""
    if self.config.sampler_delay != 0:
      return False
    if self.config.sampler_bits < 32 or not self.config.deterministic:
      return False
    if getattr(actor_cfg, "epsilon", 0.0) != 0.0:
      return False
    if getattr(actor_cfg, "bug_prob", 0.0) != 0.0:
      return False
    if getattr(actor_cfg, "correct_prob", 0.0) != 0.0:
      return False
    if getattr(actor_cfg, "random_prob", 0.0) != 0.0:
      return False
    if getattr(actor_cfg, "override_token_prob", None) is not None:
      return False
    sampler_net_cfg = getattr(actor_cfg, "sampler_network_config", None)
    if sampler_net_cfg is None:
      return True
    return getattr(sampler_net_cfg, "sigma", 0.0) == 0.0

  def __call__(
      self,
      actor: base.Actor[base.StateT],
      learner: base.Learner[base.StateT],
  ) -> pd.DataFrame:
    actor_cfg = getattr(actor, "config", None)
    if actor_cfg is None:
      raise ValueError("KondoAsyncTrainer expects an actor with a config.")
    batch_size = actor_cfg.prompts_per_batch * actor_cfg.samples_per_prompt
    keep_count = max(1, int(round(self.config.pct_learn * batch_size)))
    reuse_sampler_logprobs_for_screen = (
        self._can_reuse_sampler_logprobs_for_screen(actor_cfg)
    )

    quantizer = quantization.QuantizeConfig(
        num_bits=self.config.sampler_bits,
        deterministic=self.config.deterministic,
    ).make()

    key = jax.random.PRNGKey(self.config.seed)
    key, init_key = jax.random.split(key)
    state = learner.init_state(init_key)

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
    def screen_batch_step(
        current_state: base.StateT,
        batch: base.Batch,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, base.Metrics]:
      if reuse_sampler_logprobs_for_screen:
        signals = common.delight_signals_from_sample_logprobs(
            batch,
            use_grouped_baseline=self.config.use_grouped_baseline,
            num_groups=self.config.num_groups,
            priority=self.config.priority,
            alpha_additive=self.config.alpha_additive,
        )
      else:
        signals = common.delight_signals(
            current_state.params,
            current_state,
            batch,
            key,
            use_grouped_baseline=self.config.use_grouped_baseline,
            num_groups=self.config.num_groups,
            priority=self.config.priority,
            alpha_additive=self.config.alpha_additive,
        )
      token_gate, gate_threshold, k_target = common.topk_token_gate(
          signals.priority_tok,
          signals.fwd.token_mask,
          self.config.pct_learn,
      )
      row_scores = jnp.sum(token_gate * signals.priority_tok, axis=1)
      row_scores = jnp.where(signals.fwd.row_mask > 0.0, row_scores, -jnp.inf)
      top_values, keep_idx = jax.lax.top_k(row_scores, keep_count)
      valid_rows = jnp.sum(signals.fwd.row_mask) + 1e-8
      tok_count = jnp.sum(signals.fwd.token_mask) + 1e-8
      selected_tok_count = jnp.sum(token_gate) + 1e-8
      kept_selected_tok_count = jnp.sum(jnp.take(token_gate, keep_idx, axis=0))
      delight_row = (
          jnp.sum(signals.delight_tok * signals.fwd.token_mask, axis=1)
          / (jnp.sum(signals.fwd.token_mask, axis=1) + 1e-8)
      ) * signals.fwd.row_mask

      metrics: base.Metrics = {
          "rows_total": valid_rows,
          "rows_kept": jnp.asarray(keep_count, jnp.float32),
          "keep_fraction": jnp.asarray(keep_count, jnp.float32) / valid_rows,
          "token_keep_fraction": selected_tok_count / tok_count,
          "gate_threshold": gate_threshold,
          "row_gate_threshold": top_values[-1],
          "token_k_target": jnp.asarray(k_target, jnp.float32),
          "priority_row_mean": jnp.sum(row_scores * signals.fwd.row_mask)
          / valid_rows,
          "priority_row_kept_mean": jnp.mean(top_values),
          "advantage_mean": jnp.sum(signals.advantages * signals.fwd.row_mask)
          / valid_rows,
          "surprisal_token_mean": jnp.sum(
              signals.surprisal_tok * signals.fwd.token_mask
          )
          / tok_count,
          "delight_token_mean": jnp.sum(
              signals.delight_tok * signals.fwd.token_mask
          )
          / tok_count,
          "delight_row_mean": jnp.sum(delight_row) / valid_rows,
          "tokens_selected": selected_tok_count,
          "tokens_selected_in_kept_rows": kept_selected_tok_count,
          "selected_token_recall": kept_selected_tok_count / selected_tok_count,
          "used_sampler_logprobs": jnp.asarray(
              reuse_sampler_logprobs_for_screen, jnp.float32
          ),
      }
      return keep_idx, signals.advantages, token_gate, metrics

    @jax.jit
    def compact_batch_step(
        batch: base.Batch,
        keep_idx: jax.Array,
        advantages: jax.Array,
        token_gate: jax.Array,
    ) -> tuple[base.Batch, base.Metrics]:
      kept_batch = common.compact_batch_rows(batch, keep_idx)
      kept_rewards = jnp.take(advantages, keep_idx, axis=0)
      kept_token_gate = jnp.take(token_gate, keep_idx, axis=0)
      kept_aux = dict(kept_batch.aux)
      kept_aux["row_mask"] = jnp.ones((keep_count,), dtype=jnp.float32)
      kept_aux["parent_row_ids"] = keep_idx
      kept_aux["loss_token_mask_answer"] = kept_token_gate
      kept_aux["loss_normalizer"] = jnp.sum(
          (batch.answers >= 0).astype(jnp.float32)
      )
      kept_batch = kept_batch._replace(rewards=kept_rewards, aux=kept_aux)

      total_tokens = jnp.sum((batch.answers >= 0).astype(jnp.float32))
      kept_tokens = jnp.sum((kept_batch.answers >= 0).astype(jnp.float32))
      selected_tokens = jnp.sum(kept_token_gate)
      metrics: base.Metrics = {
          "tokens_total": total_tokens,
          "tokens_kept_for_backward": kept_tokens,
          "backward_token_fraction": kept_tokens / (total_tokens + 1e-8),
          "tokens_selected_for_loss": selected_tokens,
          "selected_token_fraction_in_kept_rows": selected_tokens
          / (kept_tokens + 1e-8),
      }
      return kept_batch, metrics

    @jax.jit
    def train_on_kept_step(
        current_state: base.StateT,
        kept_batch: base.Batch,
        key: jax.Array,
    ) -> tuple[base.StateT, base.Metrics]:
      return learner.step(current_state, kept_batch, key)

    @jax.jit
    def evaluate_learner_performance(
        current_state: base.StateT,
        key: jax.Array,
    ) -> jax.Array:
      batch, _, _ = actor.sample_batch(current_state, key)
      return batch.rewards.mean()

    # Warm up compiled paths once so timing excludes first-compile cost.
    key, warm_quant_key, warm_sample_key, warm_screen_key, warm_train_key = (
        jax.random.split(key, 5)
    )
    warm_actor_params = quantizer(state.params, warm_quant_key)
    warm_batch, _ = sample_batch_step(state, warm_actor_params, warm_sample_key)
    warm_keep_idx, warm_advantages, warm_token_gate, _ = screen_batch_step(
        state, warm_batch, warm_screen_key
    )
    warm_kept_batch, _ = compact_batch_step(
        warm_batch, warm_keep_idx, warm_advantages, warm_token_gate
    )
    warm_state, _ = train_on_kept_step(state, warm_kept_batch, warm_train_key)
    _block_tree(warm_batch.prompts)
    _block_tree(warm_keep_idx)
    _block_tree(warm_kept_batch.prompts)
    _block_tree(warm_state.params)

    records: list[dict[str, tp.Any]] = []
    start = time.perf_counter()
    cumulative_reward = 0.0
    logger = logging.RunningMeanLogger()

    for step in range(1, self.config.steps + 1):
      key, quant_key, sample_key, screen_key, train_key, eval_key = (
          jax.random.split(key, 6)
      )

      if self.config.uniform_delay:
        quant_key, draw_key = jax.random.split(quant_key)
        idx = int(
            jax.random.randint(draw_key, shape=(), minval=0, maxval=hist_len)
        )
        chosen_params = params_history[idx]
      else:
        chosen_params = params_history[0]
      actor_params = quantizer(chosen_params, quant_key)

      sample_t0 = time.perf_counter()
      batch, actor_metrics = sample_batch_step(state, actor_params, sample_key)
      _block_tree(batch.prompts)
      sample_time_s = time.perf_counter() - sample_t0

      screen_t0 = time.perf_counter()
      keep_idx, advantages, token_gate, screen_metrics = screen_batch_step(
          state, batch, screen_key
      )
      _block_tree(keep_idx)
      screen_time_s = time.perf_counter() - screen_t0

      compact_t0 = time.perf_counter()
      kept_batch, compact_metrics = compact_batch_step(
          batch, keep_idx, advantages, token_gate
      )
      _block_tree(kept_batch.prompts)
      compact_time_s = time.perf_counter() - compact_t0

      train_t0 = time.perf_counter()
      state, learner_metrics = train_on_kept_step(state, kept_batch, train_key)
      _block_tree(state.params)
      train_time_s = time.perf_counter() - train_t0

      params_history = params_history[1:] + (state.params,)

      metrics: base.Metrics = {
          "reward": batch.rewards.mean(),
          "sample_time_s": jnp.asarray(sample_time_s, jnp.float32),
          "screen_time_s": jnp.asarray(screen_time_s, jnp.float32),
          "compact_time_s": jnp.asarray(compact_time_s, jnp.float32),
          "train_time_s": jnp.asarray(train_time_s, jnp.float32),
          "total_step_time_s": jnp.asarray(
              sample_time_s + screen_time_s + compact_time_s + train_time_s,
              jnp.float32,
          ),
      }
      metrics.update({f"screen/{k}": v for k, v in screen_metrics.items()})
      metrics.update({f"compact/{k}": v for k, v in compact_metrics.items()})
      if self.config.log_details:
        metrics.update({
            **{f"actor/{k}": v for k, v in actor_metrics.items()},
            **{f"learner/{k}": v for k, v in learner_metrics.items()},
        })

      logger.record(metrics)
      cumulative_reward += float(metrics["reward"])

      if self._should_log(step) or step == self.config.steps:
        record = logger.write()
        record["step"] = step
        record["time"] = time.perf_counter() - start
        record["cum_reward"] = cumulative_reward

        if self.config.log_learner_performance:
          learner_reward = evaluate_learner_performance(state, eval_key)
          record["learner_reward"] = float(learner_reward)

        records.append(record)

        summary: dict[str, tp.Any] = {
            "step": step,
            "time": _format_float(record["time"]),
            "reward": _format_float(record.get("reward", 0.0)),
            "cum_reward": _format_float(cumulative_reward),
            "keep_fraction": _format_float(
                record.get("screen/keep_fraction", 0.0)
            ),
            "train_time_s": _format_float(record.get("train_time_s", 0.0)),
        }
        if self.config.log_learner_performance:
          summary["learner_reward"] = _format_float(record["learner_reward"])
        print(json.dumps(summary))

    print("--- Training finished ---")
    return pd.DataFrame(records)


def _format_float(f: float) -> float:
  return float(format(f, ".3g"))
