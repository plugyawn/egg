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

"""Token-level PMPO loss.

Implements a token-level variant of "Preference Optimization as Probabilistic
Inference" (Abdolmaleki et al., 2024), with per-sequence accept/reject signals
broadcast across the answer tokens.
"""

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
  """Makeable config that produces a token-level PMPO loss."""

  alpha: float = 0.5
  beta: float = 0.1
  gamma: float = 1e-9
  reward_threshold: float = 0.0
  use_grouped_baseline: bool = True
  num_groups: int | None = None

  def make(self) -> PMPOToken:
    if self.use_grouped_baseline and self.num_groups is None:
      logging.info(
          "PMPO-Token: use_baseline=True with no num_groups → global baseline."
      )
    if self.gamma < 0.0:
      raise ValueError("gamma must be >= 0.0")
    return PMPOToken(config=self)


@dataclasses.dataclass(frozen=True)
class PMPOToken(base.LossFn):
  """Token-level PMPO loss."""

  config: LossConfig

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
    ans_tok_mask = (batch.answers >= 0).astype(jnp.float32)

    # --- Preference classification (sequence-level) ---
    if self.config.use_grouped_baseline:
      group_ids = batch.aux.get("group_ids")
      advantages = common.grouped_advantages(
          rewards,
          group_ids,
          self.config.num_groups,
          fwd.row_mask,
      )
      is_pref_seq = advantages > 0.0
    else:
      is_pref_seq = rewards > self.config.reward_threshold

    # Broadcast sequence preference to tokens
    pref_mask = (is_pref_seq.astype(jnp.float32) * fwd.row_mask)[
        :, None
    ] * ans_tok_mask
    rej_mask = ((1.0 - is_pref_seq.astype(jnp.float32)) * fwd.row_mask)[
        :, None
    ] * ans_tok_mask

    # --- Token-level preferred / dis-preferred terms ---
    loss_pref_tok = -fwd.lp_answer
    sum_pref = jnp.sum(loss_pref_tok * pref_mask)
    cnt_pref = jnp.sum(pref_mask) + 1e-8
    avg_loss_preferred = sum_pref / cnt_pref

    log_gamma = (
        jnp.log(self.config.gamma) if self.config.gamma > 0.0 else -jnp.inf
    )
    loss_rej_tok = jnp.logaddexp(log_gamma, fwd.lp_answer)
    sum_rej = jnp.sum(loss_rej_tok * rej_mask)
    cnt_rej = jnp.sum(rej_mask) + 1e-8
    avg_loss_dispreferred = sum_rej / cnt_rej

    # KL regularization
    kl_tok = lp_samp - fwd.lp_answer
    loss_kl = jnp.sum(kl_tok * fwd.token_mask) / (
        jnp.sum(fwd.token_mask) + 1e-8
    )

    # Combine total loss
    total_loss = (
        self.config.alpha * avg_loss_preferred
        + (1.0 - self.config.alpha) * avg_loss_dispreferred
        + self.config.beta * loss_kl
    )

    # --- Metrics ---
    b_eff = jnp.sum(fwd.row_mask) + 1e-8
    frac_preferred = (
        jnp.sum(is_pref_seq.astype(jnp.float32) * fwd.row_mask) / b_eff
    )
    seq_log_prob = jnp.sum(fwd.lp_answer * ans_tok_mask, axis=1)
    metrics: base.Metrics = {
        "loss": total_loss,
        "loss_preferred": avg_loss_preferred,
        "loss_dispreferred": avg_loss_dispreferred,
        "loss_kl": loss_kl,
        "frac_preferred": frac_preferred,
        "seq_logp_mean": jnp.sum(seq_log_prob * fwd.row_mask) / b_eff,
        "tok_pref_frac": jnp.sum(pref_mask) / (jnp.sum(fwd.token_mask) + 1e-8),
        **statistics.logp_stats(
            learner_logp=fwd.lp_answer,
            sampler_logp=lp_samp,
        ),
    }
    return total_loss, metrics
