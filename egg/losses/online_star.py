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

"""Online star only learns from positive examples."""

from __future__ import annotations

import dataclasses

from absl import logging
from egg import base
from egg.lib import statistics
from egg.losses import common
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class LossConfig(base.MakeableConfig[base.LossFn]):
  """Makeable config that produces an OnlineSTaR loss."""

  reward_threshold: float = 0.99
  use_grouped_baseline: bool = False
  num_groups: int | None = None

  def make(self) -> OnlineSTaR:
    if self.use_grouped_baseline and self.num_groups is None:
      logging.warning(
          "OnlineSTaR LossConfig: use_grouped_baseline is True but num_groups"
          " is not set."
      )
    return OnlineSTaR(
        reward_threshold=self.reward_threshold,
        use_grouped_baseline=self.use_grouped_baseline,
        num_groups=self.num_groups,
    )


@dataclasses.dataclass(frozen=True)
class OnlineSTaR(base.LossFn):
  """Supervised fine-tuning loss for Online STaR."""

  reward_threshold: float = 0.99
  use_grouped_baseline: bool = False
  num_groups: int | None = None

  def __call__(
      self,
      params: base.Params,
      state: base.StateT,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, base.Metrics]:
    fwd = common.forward_pass(params, state, batch, key)
    lp_samp = common.sampler_answer_logprobs(batch)
    rewards = batch.rewards
    batch_size = batch.prompts.shape[0]

    # Cross entropy on answer tokens
    losses = -jnp.sum(fwd.lp_answer, axis=1)  # (B,)

    # Determine which examples to supervise on
    advantages = None
    if self.use_grouped_baseline:
      group_ids = batch.aux.get("group_ids", None)
      advantages = common.grouped_advantages(
          rewards,
          group_ids,
          self.num_groups,
          fwd.row_mask,
      )
      supervise_mask = advantages > self.reward_threshold
    else:
      supervise_mask = rewards > self.reward_threshold

    # Mask loss if not a "positive" example
    losses = losses * supervise_mask
    num_supervised_examples = jnp.sum(supervise_mask)
    avg_loss = jnp.sum(losses) / (num_supervised_examples + 1e-8)

    # Metrics
    entropy = statistics.entropy_from_logp(fwd.lp_all)
    logp_stats = statistics.logp_stats(
        learner_logp=fwd.lp_answer,
        sampler_logp=lp_samp,
    )
    metrics = {
        "num_examples": jnp.array(batch_size),
        "num_supervised_examples": num_supervised_examples,
        "raw_reward": jnp.mean(rewards),
        **statistics.scalar_stats(entropy, "policy_entropy"),
        **logp_stats,
    }
    if self.use_grouped_baseline:
      metrics["advantage"] = jnp.mean(advantages)

    return avg_loss, metrics
