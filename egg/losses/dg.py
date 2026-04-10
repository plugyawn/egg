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

"""Token-level Delightful Gradient loss with a probabilistic sigmoid gate."""

from __future__ import annotations

import dataclasses

from absl import logging
from egg import base
from egg.lib import statistics
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class LossConfig(base.MakeableConfig[base.LossFn]):
  """Makeable config for token-level Delightful Gradient."""

  eta_learn: float = 1.0  # Temperature η >= 0 for the sigmoid gate.
  lambda_learn: float = 0.0  # Joy threshold λ to shift the gate.
  stochastic_gate: bool = False  # If True, sample gate; else, use mean.
  use_grouped_baseline: bool = True
  num_groups: int | None = None
  beta_kl: float = 0.0  # KL anchor weight (π_act || π_θ); 0 disables.

  # Override delight = (1-override_ucb) * advantage + override_ucb * surprisal
  override_ucb: float | None = None
  delight_noise_std: float = 0.0  # Std of noise added to gate input

  def make(self) -> "DelightSigmoidLoss":
    if float(self.eta_learn) < 0.0:
      raise ValueError("eta_learn (temperature) must be >= 0.")
    if self.use_grouped_baseline and self.num_groups is None:
      logging.warning(
          "Delight: use_grouped_baseline=True but num_groups is None; falling"
          " back to global baseline."
      )
    if self.override_ucb is not None and not (0.0 <= self.override_ucb <= 1.0):
      raise ValueError("override_ucb must be in [0, 1].")
    if float(self.delight_noise_std) < 0.0:
      raise ValueError("delight_noise_std must be non-negative.")
    return DelightSigmoidLoss(
        eta_learn=float(self.eta_learn),
        lambda_learn=float(self.lambda_learn),
        stochastic_gate=bool(self.stochastic_gate),
        use_grouped_baseline=bool(self.use_grouped_baseline),
        num_groups=None if self.num_groups is None else int(self.num_groups),
        override_ucb=None
        if self.override_ucb is None
        else float(self.override_ucb),
        delight_noise_std=float(self.delight_noise_std),
        beta_kl=float(self.beta_kl),
    )


@dataclasses.dataclass(frozen=True)
class DelightSigmoidLoss(base.LossFn):
  """Token-level Delightful Gradient using a probabilistic sigmoid gate.

  The per-token update is scaled by a weight w_t, which is either the
  probability p_t = sigmoid((χ_t - λ) / η) or a sample from Bernoulli(p_t).
  χ_t is the delight (advantage * surprisal), λ is the threshold, and η is the
  temperature. The base PG term is -A * log pi (REINFORCE with baseline).
  If η = 0, the gate becomes an indicator function I(χ_t > λ).
  Includes an optional KL(π_act || π_θ) anchor term.
  """

  eta_learn: float
  lambda_learn: float
  stochastic_gate: bool
  use_grouped_baseline: bool
  num_groups: int | None
  beta_kl: float
  override_ucb: float | None
  delight_noise_std: float

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
      raise ValueError("answers must contain at least one token per example.")

    # Masks
    row_mask = batch.aux.get("row_mask")
    if row_mask is None:
      row_mask = jnp.ones((batch_size,), dtype=jnp.float32)
    row_mask = row_mask.astype(jnp.float32)  # (B,)
    ans_tok_mask = (answers >= 0).astype(jnp.float32)  # (B, A)
    token_mask = ans_tok_mask * row_mask[:, None]  # (B, A)

    # Model log-probs (next-token)
    seqs = jnp.concatenate([prompts, answers], axis=-1)  # (B, P+A)
    seqs_in = jnp.where(seqs < 0, jnp.zeros_like(seqs), seqs)
    model_key, noise_key, gate_key = jax.random.split(key, 3)
    logits = state.apply_fn(
        {"params": params}, seqs_in, rngs={"noise": model_key}
    )
    tgt = seqs_in[:, 1:]  # (B, P+A-1)
    lp_all = jax.nn.log_softmax(logits[:, :-1], axis=-1)  # (B, P+A-1, V)
    lp_pol = jnp.take_along_axis(lp_all, tgt[..., None], axis=-1).squeeze(
        -1
    )  # (B, P+A-1)

    # Answer-token slice
    start = prompt_len - 1
    end = prompt_len + answer_len - 1
    lp_pol_answer = lp_pol[:, start:end]  # (B, A)
    lp_act_answer = batch.sample_log_probs[
        :, prompt_len : prompt_len + answer_len
    ]  # (B, A)

    # Baseline → sequence advantage
    if self.use_grouped_baseline and self.num_groups is not None:
      group_ids = batch.aux.get("group_ids")
      if group_ids is None:
        logging.warning(
            "Delight: num_groups provided but group_ids missing; using global"
            " baseline."
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

    advantage = rewards - jax.lax.stop_gradient(baseline)  # (B,)
    a_tok = (advantage[:, None]) * token_mask  # (B, A)

    # Surprisal per token
    surprisal_tok = -lp_pol_answer  # (B, A)

    # Delight χ_t = A * surprisal
    delight_tok = advantage[:, None] * surprisal_tok  # (B, A)

    if self.override_ucb is None:
      gate_input = delight_tok
    else:
      # Override with the UCB term to test alternative.
      ucb = jnp.asarray(self.override_ucb, dtype=delight_tok.dtype)
      gate_input = (1.0 - ucb) * advantage[:, None] + ucb * surprisal_tok

    if self.delight_noise_std > 0.0:
      noise = (
          jax.random.normal(noise_key, shape=gate_input.shape)
          * self.delight_noise_std
      )
      gate_input = gate_input + noise

    # Sigmoid gate probability: p*_t = sigmoid((gate_input - λ) / η)
    # If eta_learn is 0, this becomes an indicator function: I(χ_t > λ)
    if self.eta_learn == 0.0:
      gate_prob = (gate_input > self.lambda_learn).astype(jnp.float32)
    else:
      gate_prob = jax.nn.sigmoid(
          (gate_input - self.lambda_learn) / self.eta_learn
      )

    # Determine gate weight (either sampled or mean)
    if self.stochastic_gate:
      gate_weight = jax.random.bernoulli(gate_key, p=gate_prob).astype(
          jnp.float32
      )
    else:
      gate_weight = gate_prob

    gate_weight_sg = jax.lax.stop_gradient(gate_weight)

    # Hedonic PG term and delightful scaling
    per_tok_reinf = -a_tok * lp_pol_answer  # (B, A)
    per_tok_loss = gate_weight_sg * per_tok_reinf  # (B, A)

    # Mean loss over valid tokens
    tok_count = jnp.sum(token_mask) + 1e-8
    loss_delight = jnp.sum(per_tok_loss) / tok_count

    # KL anchor: E[log π_act - log πθ] over answer tokens
    kl_tok = (lp_act_answer - lp_pol_answer) * token_mask
    loss_kl = jnp.sum(kl_tok) / tok_count
    loss = loss_delight + self.beta_kl * loss_kl

    # Metrics
    b_eff = jnp.sum(row_mask) + 1e-8
    adv_mean = jnp.sum(advantage * row_mask) / b_eff
    adv_centered = advantage - adv_mean
    adv_std = jnp.sqrt(jnp.sum((adv_centered**2) * row_mask) / b_eff)
    surprisal_seq = jnp.sum(surprisal_tok * ans_tok_mask, axis=1)  # (B,)

    metrics = {
        "loss": loss,
        "eta_learn": jnp.array(self.eta_learn, jnp.float32),
        "loss_delight": loss_delight,
        "loss_kl": loss_kl,
        "beta_kl": jnp.array(self.beta_kl, jnp.float32),
        "lambda_learn": jnp.array(self.lambda_learn, jnp.float32),
        "override_ucb": jnp.asarray(
            self.override_ucb if self.override_ucb is not None else -1.0,
            jnp.float32,
        ),
        "delight_noise_std": jnp.array(self.delight_noise_std, jnp.float32),
        "advantage_mean": adv_mean,
        "advantage_std": adv_std,
        "surprisal_token_mean": jnp.sum(surprisal_tok * token_mask) / tok_count,
        "surprisal_seq_mean": jnp.sum(surprisal_seq * row_mask) / b_eff,
        "gate_prob_mean": jnp.sum(gate_prob * token_mask) / tok_count,
        "gate_weight_mean": jnp.sum(gate_weight * token_mask) / tok_count,
        **statistics.scalar_stats(
            statistics.entropy_from_logp(lp_all[:, start:end, :]),
            "policy_entropy",
        ),
        **statistics.logp_stats(
            learner_logp=lp_pol_answer,
            sampler_logp=lp_act_answer,
        ),
    }
    if self.use_grouped_baseline and self.num_groups is not None:
      metrics["num_groups"] = jnp.array(self.num_groups, jnp.int32)

    return loss, metrics
