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

"""Policy-gradient loss for pre-screened compacted batches."""

from __future__ import annotations

import dataclasses

from egg import base
from egg.lib import statistics
from egg.losses import common
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class LossConfig(base.MakeableConfig[base.LossFn]):
  """Loss config for compacted Kondo batches with precomputed advantages."""

  beta_kl: float = 0.0

  def make(self) -> "ScreenedPolicyGradient":
    return ScreenedPolicyGradient(beta_kl=float(self.beta_kl))


@dataclasses.dataclass(frozen=True)
class ScreenedPolicyGradient(base.LossFn):
  """Token-level PG on compacted rows with optional KL anchor."""

  beta_kl: float = 0.0

  def __call__(
      self,
      params: base.Params,
      state: base.StateT,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, base.Metrics]:
    fwd = common.forward_pass(params, state, batch, key)
    lp_samp = common.sampler_answer_logprobs(batch)
    advantages = batch.rewards
    loss_token_mask = batch.aux.get("loss_token_mask_answer")
    if loss_token_mask is None:
      loss_token_mask = fwd.token_mask
    loss_token_mask = loss_token_mask.astype(jnp.float32) * fwd.token_mask
    loss_normalizer = batch.aux.get("loss_normalizer")
    if loss_normalizer is None:
      loss_normalizer = jnp.sum(loss_token_mask) + 1e-8
    else:
      loss_normalizer = jnp.asarray(loss_normalizer, jnp.float32) + 1e-8

    a_tok = advantages[:, None] * loss_token_mask
    per_tok_pg = -a_tok * fwd.lp_answer
    backward_tok_count = jnp.sum(fwd.token_mask) + 1e-8
    selected_tok_count = jnp.sum(loss_token_mask) + 1e-8
    loss_pg = jnp.sum(per_tok_pg) / loss_normalizer

    kl_tok = (lp_samp - fwd.lp_answer) * fwd.token_mask
    loss_kl = jnp.sum(kl_tok) / backward_tok_count
    loss = loss_pg + self.beta_kl * loss_kl

    valid_rows = jnp.sum(fwd.row_mask) + 1e-8
    metrics: base.Metrics = {
        "loss": loss,
        "loss_pg": loss_pg,
        "loss_kl": loss_kl,
        "beta_kl": jnp.asarray(self.beta_kl, jnp.float32),
        "advantage_mean": jnp.sum(advantages * fwd.row_mask) / valid_rows,
        "valid_row_count": valid_rows,
        "valid_token_count": backward_tok_count,
        "selected_token_count": selected_tok_count,
        "loss_normalizer": loss_normalizer,
        **statistics.scalar_stats(
            statistics.entropy_from_logp(fwd.lp_all), "policy_entropy"
        ),
        **statistics.logp_stats(
            learner_logp=fwd.lp_answer,
            sampler_logp=lp_samp,
        ),
    }
    return loss, metrics
