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

"""A FixedBPS-style actor that can inject buggy or correct answers.

- With prob correct_prob: inject a perfect prompt-reversal.
- With prob random_prob (if not correct): inject a random answer.
- With prob bug_prob (if not correct and not random): inject a fixed landmine
answer.
- Otherwise: sample from the model (ARSampler).

Under injections, per-token log-probs are recomputed PAD-safely.
"""

from __future__ import annotations

import dataclasses

from egg import base
from egg.lib import ar_sample
import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ActorConfig(base.MakeableConfig[base.Actor]):
  """Configuration for the DreamerActor."""

  env_config: base.MakeableConfig[base.Environment]
  sequence_length: int
  prompts_per_batch: int
  samples_per_prompt: int
  landmine_sequence: jax.Array  # What defines the landmine?
  bug_prob: float = 0.0  # Probability of injecting the landmine answer
  correct_prob: float = 0.0  # Probability of injecting a correct answer
  random_prob: float = 0.0  # Probability of injecting a random answer
  override_token_prob: float | None = None  # Fixed prob for overridden tokens
  sampler_network_config: base.MakeableConfig[nn.Module] | None = None
  epsilon: float = 0.0  # random action prob for ARSampler

  def make(self) -> "DreamerActor":
    # Light sanity checks.
    if not (0.0 <= self.bug_prob <= 1.0):
      raise ValueError("bug_prob must be in [0, 1].")
    if not (0.0 <= self.correct_prob <= 1.0):
      raise ValueError("correct_prob must be in [0, 1].")
    if not (0.0 <= self.random_prob <= 1.0):
      raise ValueError("random_prob must be in [0, 1].")
    if self.override_token_prob is not None and not (
        0.0 < self.override_token_prob <= 1.0
    ):
      raise ValueError("override_token_prob must be in (0, 1].")

    env = self.env_config.make()
    sampler_network = (
        self.sampler_network_config.make()
        if self.sampler_network_config
        else None
    )
    return DreamerActor(
        config=self,
        env=env,
        sampler=ar_sample.ARSampler(
            sequence_length=self.sequence_length,
            epsilon=self.epsilon,
            vocab_size=env.spec.vocab_size,
        ),
        landmine_sequence=jnp.array(self.landmine_sequence, dtype=jnp.int32),
        sampler_network=sampler_network,
    )


@dataclasses.dataclass(frozen=True)
class DreamerActor(base.Actor[base.StateT]):
  """Injects 'correct', 'random', or 'bug' answers, else samples."""

  config: ActorConfig
  env: base.Environment
  sampler: ar_sample.ARSampler
  landmine_sequence: jnp.ndarray  # (a_len,)
  sampler_network: nn.Module | None = None

  def sample_batch(
      self, state: base.StateT, key: jax.Array
  ) -> tuple[base.Batch, base.StateT, base.Metrics]:

    p, s = self.config.prompts_per_batch, self.config.samples_per_prompt
    batch_size = p * s
    k_prompt, k_sample, k_bug, k_correct, k_random, k_reward, k_recomp = (
        jax.random.split(key, 7)
    )

    # 1) Prompts (shape: (p, p_len)) -> repeat to (B, p_len).
    prompts = jax.vmap(self.env.get_prompt)(jax.random.split(k_prompt, p))
    prompts_flat = jnp.repeat(prompts, s, axis=0).astype(jnp.int32)
    p_len = prompts_flat.shape[-1]

    # 2) Sample candidate full sequences + original per-position logps.
    sampler_apply_fn = (
        self.sampler_network.apply if self.sampler_network else state.apply_fn
    )
    seqs, logps_full = self.sampler(
        sampler_apply_fn, state.params, prompts_flat, k_sample
    )

    # Answer length.
    a_len = int(self.config.sequence_length) - int(p_len)
    if a_len < 0:
      raise ValueError(
          f"sequence_length ({self.config.sequence_length}) < prompt_len"
          f" ({p_len})."
      )
    if a_len == 0:
      answers = jnp.zeros((batch_size, 0), dtype=jnp.int32)
      rewards = jax.vmap(self.env.get_reward)(
          prompts_flat, answers, jax.random.split(k_reward, batch_size)
      )
      batch = base.Batch(
          prompts=prompts_flat,
          answers=answers,
          rewards=rewards,
          sample_log_probs=logps_full,
          aux={"group_ids": jnp.repeat(jnp.arange(p, dtype=jnp.int32), s)},
      )
      return (
          batch,
          state,
          {
              "bug_fraction": jnp.array(0.0, jnp.float32),
              "correct_fraction": jnp.array(0.0, jnp.float32),
              "random_fraction": jnp.array(0.0, jnp.float32),
              "sample_fraction": jnp.array(1.0, jnp.float32),
              "prompt_len": jnp.array(p_len, jnp.int32),
              "answer_len": jnp.array(a_len, jnp.int32),
          },
      )

    # 3) Decide type per row (iid): correct > random > bug > sample.
    sampled_answers = seqs[:, p_len:].astype(jnp.int32)  # (B, a_len)

    u_correct = jax.random.uniform(k_correct, (batch_size,))
    u_random = jax.random.uniform(k_random, (batch_size,))
    u_bug = jax.random.uniform(k_bug, (batch_size,))

    is_correct = u_correct < self.config.correct_prob
    is_random = (~is_correct) & (u_random < self.config.random_prob)
    is_bug = (~is_correct) & (~is_random) & (u_bug < self.config.bug_prob)
    is_sample = (~is_correct) & (~is_random) & (~is_bug)

    # 4) Construct injected answers.
    answers = sampled_answers

    # 4a) Correct answers: reverse the prompt; slice/pad to a_len.
    if self.config.correct_prob > 0:
      rev_prompts = jnp.flip(prompts_flat, axis=1)  # (B, p_len)
      if p_len >= a_len:
        correct_answers = rev_prompts[:, :a_len]
      else:
        pad_width = a_len - p_len
        correct_answers = jnp.pad(
            rev_prompts,
            ((0, 0), (0, pad_width)),
            constant_values=base.PAD_TOKEN,
        )
      answers = jnp.where(is_correct[:, None], correct_answers, answers)

    # 4b) Random answers.
    if self.config.random_prob > 0:
      random_answers = jax.random.randint(
          k_random,
          shape=(batch_size, a_len),
          minval=0,
          maxval=self.env.spec.vocab_size,
          dtype=jnp.int32,
      )
      answers = jnp.where(is_random[:, None], random_answers, answers)

    # 4c) Bug / landmine answers.
    if self.config.bug_prob > 0:
      if self.landmine_sequence.shape != (a_len,):
        raise ValueError(
            f"landmine_sequence must have shape {(a_len,)}, got"
            f" {tuple(self.landmine_sequence.shape)}"
        )
      landmine_answers = jnp.tile(
          self.landmine_sequence[None, :], (batch_size, 1)
      )  # (B, a_len)
      answers = jnp.where(is_bug[:, None], landmine_answers, answers)

    # 5) Recompute per-token log-probs for injected rows (PAD-safe).
    final_seqs = jnp.concatenate(
        [prompts_flat, answers], axis=-1
    )  # (B, seq_len)

    if self.config.override_token_prob is not None:
      # Fixed log prob for non-PAD answer tokens.
      override_log_prob = jnp.log(self.config.override_token_prob + 1e-9)
      answer_mask = answers != base.PAD_TOKEN
      fixed_answer_logps = jnp.where(
          answer_mask,
          jnp.full_like(answers, override_log_prob, dtype=jnp.float32),
          0.0,
      )

      # Logps for the prompt part are still 0, as in get_full_logprobs_b_l.
      recomputed_full_logps = jnp.concatenate(
          [
              jnp.zeros((batch_size, p_len), dtype=jnp.float32),
              fixed_answer_logps,
          ],
          axis=-1,
      )
      # Adjust for the shift in get_full_logprobs_b_l (logp at t is for token t)
      # so we need to shift the answer logps one to the right.
      recomputed_full_logps = jnp.pad(
          recomputed_full_logps[:, :-1], ((0, 0), (1, 0)), constant_values=0.0
      )

    else:
      # Compute from model.
      recomputed_full_logps = ar_sample.get_full_logprobs_b_l(
          apply_fn=sampler_apply_fn,
          params=state.params,
          sequences=final_seqs,
          rng=k_recomp,
      )

    # Use recomputed logps if row was injected; otherwise keep original.
    recompute_row = ~is_sample
    sample_log_probs = jnp.where(
        recompute_row[:, None], recomputed_full_logps, logps_full
    )

    # 6) Rewards (env is authoritative).
    rewards = jax.vmap(self.env.get_reward)(
        prompts_flat, answers, jax.random.split(k_reward, batch_size)
    )

    # 7) Aux: group ids, masks, and fractions.
    group_ids = jnp.repeat(jnp.arange(p, dtype=jnp.int32), s)
    answer_mask = (answers != base.PAD_TOKEN).astype(jnp.float32)

    batch = base.Batch(
        prompts=prompts_flat,
        answers=answers,
        rewards=rewards,
        sample_log_probs=sample_log_probs,
        aux={"group_ids": group_ids, "answer_mask": answer_mask},
    )
    metrics = {
        "bug_fraction": jnp.mean(is_bug.astype(jnp.float32)),
        "correct_fraction": jnp.mean(is_correct.astype(jnp.float32)),
        "random_fraction": jnp.mean(is_random.astype(jnp.float32)),
        "sample_fraction": jnp.mean(is_sample.astype(jnp.float32)),
        "injected_fraction": jnp.mean((~is_sample).astype(jnp.float32)),
        "prompt_len": jnp.array(p_len, jnp.int32),
        "answer_len": jnp.array(a_len, jnp.int32),
    }
    return batch, state, metrics
