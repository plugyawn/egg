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

"""Token-level PPO loss with optional KL anchor."""

from __future__ import annotations

import dataclasses

from egg import base
from egg.lib import statistics
from egg.losses import common
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class LossConfig(base.MakeableConfig[base.LossFn]):
  """Makeable config for token-level PPO."""

  clip_epsilon: float = 0.2
  beta_kl: float = 0.0
  use_grouped_baseline: bool = True
  num_groups: int | None = None

  def make(self) -> PpoLoss:
    if self.use_grouped_baseline and self.num_groups is None:
      raise ValueError(
          "num_groups must be set when use_grouped_baseline is True"
      )
    return PpoLoss(
        clip_epsilon=float(self.clip_epsilon),
        beta_kl=float(self.beta_kl),
        use_grouped_baseline=bool(self.use_grouped_baseline),
        num_groups=None if self.num_groups is None else int(self.num_groups),
    )


@dataclasses.dataclass(frozen=True)
class PpoLoss(base.LossFn):
  """PPO clipped surrogate (token-level) + optional KL(π_act || π_θ) anchor."""

  clip_epsilon: float
  beta_kl: float
  use_grouped_baseline: bool
  num_groups: int | None

  def __call__(
      self,
      params: base.Params,
      state: base.StateT,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, base.Metrics]:
    fwd = common.forward_pass(params, state, batch, key)
    lp_act = common.sampler_answer_logprobs(batch)
    rewards = batch.rewards

    # Advantages (sequence-level), broadcast to tokens
    if self.use_grouped_baseline:
      group_ids = batch.aux.get("group_ids")
      advantages = common.grouped_advantages(
          rewards,
          group_ids,
          self.num_groups,
          fwd.row_mask,
      )
    else:
      advantages = rewards

    a_tok = advantages[:, None] * fwd.token_mask  # (B, A)

    # Token-level ratios r_t = exp(log πθ - log π_act)
    log_ratio_tok = (fwd.lp_answer - lp_act) * fwd.token_mask
    ratio_tok = jnp.exp(log_ratio_tok) + 0.0

    # Clipped surrogate per token
    r_clip = jnp.clip(
        ratio_tok, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
    )
    obj_unclipped = ratio_tok * a_tok
    obj_clipped = r_clip * a_tok
    obj_min = jnp.minimum(obj_unclipped, obj_clipped)

    # Surrogate loss
    tok_count = jnp.sum(fwd.token_mask) + 1e-8
    loss_surrogate = -jnp.sum(obj_min) / tok_count

    # KL anchor
    kl_tok = (lp_act - fwd.lp_answer) * fwd.token_mask
    loss_kl = jnp.sum(kl_tok) / tok_count
    loss = loss_surrogate + self.beta_kl * loss_kl

    # Metrics
    ent = statistics.entropy_from_logp(fwd.lp_all)
    clipped = (ratio_tok > (1.0 + self.clip_epsilon)) | (
        ratio_tok < (1.0 - self.clip_epsilon)
    )
    clip_fraction = jnp.sum(clipped * fwd.token_mask) / tok_count
    approx_kl_tok = ((ratio_tok - 1.0) - log_ratio_tok) * fwd.token_mask
    approx_kl = jnp.sum(approx_kl_tok) / tok_count

    b_eff = jnp.sum(fwd.row_mask) + 1e-8
    adv_mean = jnp.sum(advantages * fwd.row_mask) / b_eff
    adv_centered = advantages - adv_mean
    adv_std = jnp.sqrt(jnp.sum((adv_centered**2) * fwd.row_mask) / b_eff)

    metrics: base.Metrics = {
        "loss": loss,
        "loss_surrogate": loss_surrogate,
        "loss_kl": loss_kl,
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "ratio_mean": jnp.sum(ratio_tok * fwd.token_mask) / tok_count,
        "advantage_mean": adv_mean,
        "advantage_std": adv_std,
        **statistics.scalar_stats(ent, "policy_entropy"),
        **statistics.logp_stats(
            learner_logp=fwd.lp_answer,
            sampler_logp=lp_act,
        ),
    }
    if self.use_grouped_baseline and self.num_groups is not None:
      metrics["num_groups"] = jnp.array(self.num_groups, jnp.int32)
    metrics["clip_epsilon"] = jnp.array(self.clip_epsilon, jnp.float32)
    metrics["beta_kl"] = jnp.array(self.beta_kl, jnp.float32)
    return loss, metrics
