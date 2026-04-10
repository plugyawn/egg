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

"""Basic REINFORCE loss with optional grouped baseline."""

from __future__ import annotations

import dataclasses

from egg import base
from egg.lib import statistics
from egg.losses import common
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class LossConfig(base.MakeableConfig[base.LossFn]):
  """Makeable config that produces a Reinforce loss."""

  use_iw: bool = False
  importance_weight_cap: float = 10.0
  use_grouped_baseline: bool = True
  num_groups: int | None = None
  eps: float = 1e-8

  def make(self) -> Reinforce | ReinforceGroupedBaseline:
    loss = Reinforce(
        use_iw=self.use_iw,
        importance_weight_cap=self.importance_weight_cap,
        eps=self.eps,
    )
    if self.use_grouped_baseline:
      if self.num_groups is None:
        raise ValueError(
            "num_groups must be set when use_grouped_baseline is True"
        )
      return ReinforceGroupedBaseline(
          loss=loss, num_groups=self.num_groups, eps=self.eps
      )
    else:
      return loss


@dataclasses.dataclass(frozen=True)
class Reinforce(base.LossFn):
  """Vectorised REINFORCE respecting row_mask."""

  use_iw: bool = True
  importance_weight_cap: float = 10.0
  eps: float = 1e-8

  def __call__(
      self,
      params: base.Params,
      state: base.StateT,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, base.Metrics]:
    fwd = common.forward_pass(params, state, batch, key)
    lp_samp = common.sampler_answer_logprobs(batch)

    num_valid_rows = jnp.sum(fwd.row_mask)
    safe_denom_rows = num_valid_rows + self.eps
    num_valid_tokens = jnp.sum(fwd.token_mask)
    safe_denom_tokens = num_valid_tokens + self.eps
    ans_tok_mask = (batch.answers >= 0).astype(jnp.float32)

    rewards = batch.rewards

    # --- Importance weights (masked) ---
    lp_samp_answer = None
    if self.use_iw and lp_samp is not None:
      log_iw_per_tok = (fwd.lp_answer - lp_samp) * ans_tok_mask
      log_iw_seq = jnp.sum(log_iw_per_tok, axis=1)
      iw = jnp.exp(
          jnp.clip(log_iw_seq, max=jnp.log(self.importance_weight_cap))
      )
      lp_samp_answer = lp_samp
    else:
      iw = jnp.ones_like(rewards)

    # --- REINFORCE loss (masked mean) ---
    seq_log_prob_sum = jnp.sum(fwd.lp_answer * ans_tok_mask, axis=1)
    target = jax.lax.stop_gradient(rewards * iw)
    losses_per_seq = -seq_log_prob_sum * target
    loss = jnp.sum(losses_per_seq * fwd.row_mask) / safe_denom_rows

    # --- Metrics ---
    mean_iw = jnp.sum(iw * fwd.row_mask) / safe_denom_rows
    entropy_per_tok = statistics.entropy_from_logp(fwd.lp_all)
    mean_entropy = jnp.sum(entropy_per_tok * fwd.token_mask) / safe_denom_tokens

    metrics: base.Metrics = {
        "loss": loss,
        "importance_weight_mean": mean_iw,
        "policy_entropy_mean": mean_entropy,
        "valid_row_count": num_valid_rows,
        "valid_token_count": num_valid_tokens,
    }

    if lp_samp_answer is not None:
      logp_stats = statistics.logp_stats(
          learner_logp=fwd.lp_answer,
          sampler_logp=lp_samp_answer,
      )
      metrics.update(logp_stats)

    return loss, metrics


@dataclasses.dataclass(frozen=True)
class ReinforceGroupedBaseline(base.LossFn):
  """REINFORCE loss wrapper for per-group baseline subtraction."""

  loss: Reinforce
  num_groups: int
  eps: float = 1e-8

  def __call__(
      self,
      params: base.Params,
      state: base.StateT,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, base.Metrics]:
    rewards = batch.rewards
    row_mask = batch.aux.get("row_mask")
    if row_mask is None:
      row_mask = jnp.ones(rewards.shape[0], dtype=jnp.float32)
    row_mask = row_mask.astype(jnp.float32)

    group_ids = (
        batch.aux.get("group_ids", None) if batch.aux is not None else None
    )

    # Compute grouped advantages
    advantage = common.grouped_advantages(
        rewards,
        group_ids,
        self.num_groups,
        row_mask,
        self.eps,
    )

    # Pass advantage as reward, ensuring row_mask is in aux
    new_aux = dict(batch.aux) if batch.aux is not None else {}
    new_aux["row_mask"] = row_mask
    adj_batch = batch._replace(rewards=advantage, aux=new_aux)

    # Delegate to base loss
    loss, metrics = self.loss(params, state, adj_batch, key)

    # Add baseline-specific metrics
    metrics = dict(metrics)
    safe_denom = jnp.sum(row_mask) + self.eps
    metrics["advantage_mean"] = jnp.sum(advantage * row_mask) / safe_denom
    metrics["num_groups"] = jnp.array(self.num_groups)
    baseline = rewards - advantage
    metrics["baseline_mean"] = jnp.sum(baseline * row_mask) / safe_denom

    return loss, metrics
