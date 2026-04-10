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
