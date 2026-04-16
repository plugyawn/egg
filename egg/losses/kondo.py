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

"""Token-level Kondo gate: binary keep/skip gating by delight.

The Kondo gate skips the backward pass on uninformative tokens.  For each
token, delight χ = advantage × surprisal is computed from the forward pass.
A target fraction `pct_learn` of tokens are kept (top-k by the chosen priority
signal); the rest are zeroed.  Survivors receive uniform weight.
"""

from __future__ import annotations

import dataclasses
import enum

from absl import logging
from egg import base
from egg.lib import statistics
from egg.losses import common
import jax
import jax.numpy as jnp


class PriorityType(enum.StrEnum):
  DELIGHT = 'delight'
  ADVANTAGE = 'advantage'
  ABS_ADVANTAGE = 'abs_advantage'
  SURPRISAL = 'surprisal'
  UNIFORM = 'uniform'
  ADDITIVE = 'additive'


@dataclasses.dataclass
class LossConfig(base.MakeableConfig[base.LossFn]):
  """Config for token-level Kondo gate loss."""

  pct_learn: float = 1.0
  beta_kl: float = 0.0
  use_grouped_baseline: bool = True
  num_groups: int | None = None

  priority: str = 'delight'
  alpha_additive: float = 0.5

  def make(self) -> 'KondoLoss':
    if not (0.0 < self.pct_learn <= 1.0):
      raise ValueError('pct_learn must be in (0, 1].')
    if self.use_grouped_baseline and self.num_groups is None:
      logging.warning(
          'Kondo: use_grouped_baseline=True but num_groups is None; falling'
          ' back to a global baseline.'
      )
    return KondoLoss(
        pct_learn=float(self.pct_learn),
        beta_kl=float(self.beta_kl),
        use_grouped_baseline=bool(self.use_grouped_baseline),
        num_groups=None if self.num_groups is None else int(self.num_groups),
        priority=PriorityType(self.priority),
        alpha_additive=float(self.alpha_additive),
    )

@dataclasses.dataclass(frozen=True)
class KondoLoss(base.LossFn):
  """Token-level Kondo gate.

  Binary keep/skip, uniform weight among survivors.
  """

  pct_learn: float
  beta_kl: float
  use_grouped_baseline: bool
  num_groups: int | None
  priority: PriorityType
  alpha_additive: float

  def __call__(
      self,
      params: base.Params,
      state: base.StateT,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, base.Metrics]:

    prompts, answers, rewards = batch.prompts, batch.answers, batch.rewards
    batch_size, prompt_len = prompts.shape
    _, answer_len = answers.shape
    if answer_len <= 0:
      raise ValueError('answers must contain at least one token per example.')

    row_mask = batch.aux.get('row_mask')
    if row_mask is None:
      row_mask = jnp.ones((batch_size,), dtype=jnp.float32)
    row_mask = row_mask.astype(jnp.float32)
    ans_tok_mask = (answers >= 0).astype(jnp.float32)
    token_mask = ans_tok_mask * row_mask[:, None]
    total_valid_tokens = jnp.sum(token_mask)

    seqs = jnp.concatenate([prompts, answers], axis=-1)
    seqs_in = jnp.where(seqs < 0, jnp.zeros_like(seqs), seqs)

    logits = state.apply_fn({'params': params}, seqs_in, rngs={'noise': key})

    tgt = seqs_in[:, 1:]
    lp_all = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    lp_pol = jnp.take_along_axis(lp_all, tgt[..., None], axis=-1).squeeze(-1)

    start = prompt_len - 1
    end = prompt_len + answer_len - 1
    lp_pol_answer = lp_pol[:, start:end]
    lp_act_answer = batch.sample_log_probs[
        :, prompt_len : prompt_len + answer_len
    ]

    if self.use_grouped_baseline and self.num_groups is not None:
      group_ids = batch.aux.get('group_ids')
      if group_ids is None:
        logging.warning(
            'Kondo: num_groups provided but group_ids missing; using global'
            ' baseline.'
        )
        denom = jnp.sum(row_mask) + 1e-9
        base_val = jnp.sum(rewards * row_mask) / denom
        baseline = jnp.full_like(rewards, base_val)
      else:
        g = int(self.num_groups)
        sum_r = (
            jnp.zeros((g,), rewards.dtype).at[group_ids].add(rewards * row_mask)
        )
        cnt_r = jnp.zeros((g,), jnp.float32).at[group_ids].add(row_mask)
        base_g = sum_r / (cnt_r + 1e-9)
        baseline = base_g[group_ids]
    else:
      denom = jnp.sum(row_mask) + 1e-9
      base_val = jnp.sum(rewards * row_mask) / denom
      baseline = jnp.full_like(rewards, base_val)

    advantage = rewards - jax.lax.stop_gradient(baseline)
    a_tok = (advantage[:, None]) * token_mask

    surprisal_tok = -lp_pol_answer
    priority_score = common.compute_priority(
        self.priority,
        advantage,
        surprisal_tok,
        alpha=self.alpha_additive,
    )

    k_target = jnp.maximum(
        1, jnp.round(self.pct_learn * total_valid_tokens).astype(jnp.int32)
    )

    vals_flat = jnp.where(token_mask > 0.0, priority_score, -jnp.inf).reshape(
        -1
    )
    sorted_vals = jnp.sort(vals_flat)
    threshold = sorted_vals[vals_flat.size - k_target] - 1e-6
    gate = (priority_score >= threshold).astype(jnp.float32) * token_mask

    gate_sg = jax.lax.stop_gradient(gate)

    per_tok_reinf = -a_tok * lp_pol_answer
    per_tok_loss = gate_sg * per_tok_reinf

    tok_count = total_valid_tokens + 1e-8
    loss_pg = jnp.sum(per_tok_loss) / tok_count

    kl_tok = (lp_act_answer - lp_pol_answer) * token_mask
    loss_kl = jnp.sum(kl_tok) / tok_count
    loss = loss_pg + jnp.asarray(self.beta_kl, jnp.float32) * loss_kl

    b_eff = jnp.sum(row_mask) + 1e-8
    adv_mean = jnp.sum(advantage * row_mask) / b_eff
    adv_centered = advantage - adv_mean
    adv_std = jnp.sqrt(jnp.sum((adv_centered**2) * row_mask) / b_eff)

    gate_nnz = jnp.sum(gate)
    pct_tokens_trained = gate_nnz / (total_valid_tokens + 1e-8)

    delight_tok = advantage[:, None] * surprisal_tok

    metrics: base.Metrics = {
        'loss': loss,
        'loss_pg': loss_pg,
        'loss_kl': loss_kl,
        'pct_learn_target': jnp.asarray(self.pct_learn, jnp.float32),
        'pct_tokens_trained': pct_tokens_trained,
        'gate_nnz': gate_nnz,
        'total_valid_tokens': total_valid_tokens,
        'beta_kl': jnp.asarray(self.beta_kl, jnp.float32),
        'advantage_mean': adv_mean,
        'advantage_std': adv_std,
        'surprisal_token_mean': jnp.sum(surprisal_tok * token_mask) / tok_count,
        'delight_token_mean': jnp.sum(delight_tok * token_mask) / tok_count,
        'priority_mean': jnp.sum(priority_score * token_mask) / tok_count,
        **statistics.scalar_stats(
            statistics.entropy_from_logp(lp_all[:, start:end, :]),
            'policy_entropy',
        ),
        **statistics.logp_stats(
            learner_logp=lp_pol_answer, sampler_logp=lp_act_answer
        ),
    }
    if self.use_grouped_baseline and self.num_groups is not None:
      metrics['num_groups'] = jnp.array(self.num_groups, jnp.int32)

    return loss, metrics
