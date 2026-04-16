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

"""Shared forward-pass and baseline utilities for all loss functions."""

from __future__ import annotations

import dataclasses

from egg import base
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class PolicyLogProbs:
  """Result of a forward pass, sliced to the answer portion."""

  lp_all: jax.Array  # (B, A, V) — full log-softmax over answer tokens
  lp_answer: jax.Array  # (B, A) — log-prob of the chosen answer tokens
  token_mask: jax.Array  # (B, A) — valid answer tokens × valid rows
  row_mask: jax.Array  # (B,) — valid rows


@dataclasses.dataclass(frozen=True)
class DelightSignals:
  """Learner-side screening signals used by DG and proper Kondo."""

  fwd: PolicyLogProbs
  sampler_lp_answer: jax.Array  # (B, A)
  advantages: jax.Array  # (B,)
  surprisal_tok: jax.Array  # (B, A)
  delight_tok: jax.Array  # (B, A)
  priority_tok: jax.Array  # (B, A)
  priority_row: jax.Array  # (B,)


def forward_pass(
    params: base.Params,
    state: base.StateT,
    batch: base.Batch,
    key: jax.Array,
) -> PolicyLogProbs:
  """Forward pass → answer-token log-probs. Written once, used by all losses."""
  prompts, answers = batch.prompts, batch.answers
  prompt_len, answer_len = prompts.shape[1], answers.shape[1]

  seqs = jnp.concatenate([prompts, answers], axis=-1)
  seqs_in = jnp.where(seqs < 0, 0, seqs)
  logits = state.apply_fn({"params": params}, seqs_in, rngs={"noise": key})

  tgt = seqs_in[:, 1:]
  lp_all = jax.nn.log_softmax(logits[:, :-1], axis=-1)
  lp_pol = jnp.take_along_axis(lp_all, tgt[..., None], axis=-1).squeeze(-1)

  start = prompt_len - 1
  end = prompt_len + answer_len - 1

  ans_tok_mask = (answers >= 0).astype(jnp.float32)
  row_mask = batch.aux.get("row_mask", jnp.ones(prompts.shape[0], jnp.float32))
  row_mask = row_mask.astype(jnp.float32)

  return PolicyLogProbs(
      lp_all=lp_all[:, start:end, :],
      lp_answer=lp_pol[:, start:end],
      token_mask=ans_tok_mask * row_mask[:, None],
      row_mask=row_mask,
  )


def sampler_answer_logprobs(batch: base.Batch) -> jax.Array:
  """Extract sampler log-probs aligned to the answer portion."""
  prompt_len = batch.prompts.shape[1]
  answer_len = batch.answers.shape[1]
  return batch.sample_log_probs[:, prompt_len : prompt_len + answer_len]


def grouped_advantages(
    rewards: jax.Array,
    group_ids: jax.Array | None,
    num_groups: int | None,
    row_mask: jax.Array,
    eps: float = 1e-9,
) -> jax.Array:
  """Per-group baseline-subtracted advantages."""
  safe_denom = jnp.sum(row_mask) + eps
  if group_ids is None or num_groups is None:
    baseline = jnp.sum(rewards * row_mask) / safe_denom
    return rewards - jax.lax.stop_gradient(baseline)
  sum_r = (
      jnp.zeros(num_groups, rewards.dtype).at[group_ids].add(rewards * row_mask)
  )
  cnt_r = jnp.zeros(num_groups, jnp.float32).at[group_ids].add(row_mask)
  baseline_per_group = sum_r / (cnt_r + eps)
  return rewards - jax.lax.stop_gradient(baseline_per_group[group_ids])


def compute_priority(
    priority: str,
    advantage: jax.Array,
    surprisal_tok: jax.Array,
    alpha: float = 0.5,
) -> jax.Array:
  """Per-token Kondo priority score."""
  if priority == "delight":
    return advantage[:, None] * surprisal_tok
  elif priority == "advantage":
    return jnp.broadcast_to(advantage[:, None], surprisal_tok.shape)
  elif priority == "abs_advantage":
    return jnp.broadcast_to(jnp.abs(advantage[:, None]), surprisal_tok.shape)
  elif priority == "surprisal":
    return surprisal_tok
  elif priority == "uniform":
    return jnp.ones_like(surprisal_tok)
  elif priority == "additive":
    return alpha * advantage[:, None] + (1.0 - alpha) * surprisal_tok
  else:
    raise ValueError(f"Unknown priority: {priority}")


def topk_token_gate(
    priority_tok: jax.Array,
    token_mask: jax.Array,
    pct_learn: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Binary top-k token gate matching the dense Kondo masking rule."""
  total_valid_tokens = jnp.sum(token_mask)
  k_target = jnp.maximum(
      1, jnp.round(jnp.asarray(pct_learn, jnp.float32) * total_valid_tokens).astype(jnp.int32)
  )
  vals_flat = jnp.where(token_mask > 0.0, priority_tok, -jnp.inf).reshape(-1)
  sorted_vals = jnp.sort(vals_flat)
  threshold = sorted_vals[vals_flat.size - k_target] - 1e-6
  gate = (priority_tok >= threshold).astype(jnp.float32) * token_mask
  return gate, threshold, k_target


def delight_signals(
    params: base.Params,
    state: base.StateT,
    batch: base.Batch,
    key: jax.Array,
    *,
    use_grouped_baseline: bool,
    num_groups: int | None,
    priority: str = "delight",
    alpha_additive: float = 0.5,
) -> DelightSignals:
  """Computes learner-side delight and row-level screening scores."""
  fwd = forward_pass(params, state, batch, key)
  sampler_lp_answer = sampler_answer_logprobs(batch)
  rewards = batch.rewards

  if use_grouped_baseline:
    group_ids = batch.aux.get("group_ids")
    advantages = grouped_advantages(
        rewards,
        group_ids,
        num_groups,
        fwd.row_mask,
    )
  else:
    advantages = rewards

  surprisal_tok = -fwd.lp_answer
  delight_tok = advantages[:, None] * surprisal_tok
  priority_tok = compute_priority(
      priority,
      advantages,
      surprisal_tok,
      alpha=alpha_additive,
  )

  tok_count_per_row = jnp.sum(fwd.token_mask, axis=1)
  safe_tok_count = tok_count_per_row + 1e-8
  priority_row = (
      jnp.sum(priority_tok * fwd.token_mask, axis=1) / safe_tok_count
  ) * fwd.row_mask

  return DelightSignals(
      fwd=fwd,
      sampler_lp_answer=sampler_lp_answer,
      advantages=advantages,
      surprisal_tok=surprisal_tok,
      delight_tok=delight_tok,
      priority_tok=priority_tok,
      priority_row=priority_row,
  )


def compact_batch_rows(batch: base.Batch, row_indices: jax.Array) -> base.Batch:
  """Selects a fixed subset of rows from a batch and row-shaped aux fields."""
  prompts = jnp.take(batch.prompts, row_indices, axis=0)
  answers = jnp.take(batch.answers, row_indices, axis=0)
  rewards = jnp.take(batch.rewards, row_indices, axis=0)
  sample_log_probs = jnp.take(batch.sample_log_probs, row_indices, axis=0)

  aux: dict[str, jax.Array] = {}
  batch_rows = batch.prompts.shape[0]
  for key, value in batch.aux.items():
    if hasattr(value, "shape") and value.shape and value.shape[0] == batch_rows:
      aux[key] = jnp.take(value, row_indices, axis=0)
    else:
      aux[key] = value

  return base.Batch(
      prompts=prompts,
      answers=answers,
      rewards=rewards,
      sample_log_probs=sample_log_probs,
      aux=aux,
  )
