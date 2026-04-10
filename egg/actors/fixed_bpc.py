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

"""Fixed BPC IID sampling actor.

For each batch of size B = prompts_per_batch * correct_per_prompt.
"""

from __future__ import annotations

import dataclasses
import functools

from egg import base
from egg.lib import ar_sample
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class ActorConfig(base.MakeableConfig[base.Actor]):
  """Configuration for FixedBPCActor."""

  env_config: base.MakeableConfig[base.Environment]  # Config for the env.
  sequence_length: int  # Total sequence length (prompt + answer).
  prompts_per_batch: int  # Number of prompts per batch.
  correct_per_prompt: int  # Number of correct samples per prompt.
  max_samples_per_prompt: int = 128  # Max attempts per prompt.
  reward_threshold: float = 0.99  # What counts as correct.
  sampler_network_config: base.MakeableConfig[nn.Module] | None = None

  def make(self) -> FixedBPCActor:
    sampler_network = None
    if self.sampler_network_config:
      sampler_network = self.sampler_network_config.make()
    return FixedBPCActor(
        config=self,
        env=self.env_config.make(),
        sampler=ar_sample.ARSampler(sequence_length=self.sequence_length),
        sampler_network=sampler_network,
    )


@dataclasses.dataclass(frozen=True)
class FixedBPCActor(base.Actor[base.StateT]):
  """Samples a batch with a fixed number of correct answers per prompt."""

  config: ActorConfig
  env: base.Environment
  sampler: ar_sample.ARSampler
  sampler_network: nn.Module | None = None

  @functools.partial(jax.jit, static_argnames=("self",))
  def sample_batch(
      self, state: base.StateT, key: jax.Array
  ) -> tuple[base.Batch, base.StateT, base.Metrics]:
    """Samples B = prompts_per_batch * correct_per_prompt correct answers."""
    p, c = self.config.prompts_per_batch, self.config.correct_per_prompt
    m = self.config.max_samples_per_prompt
    thresh = self.config.reward_threshold
    seq_len = self.config.sequence_length

    key, k_prompt = jax.random.split(key)
    prompts = jax.vmap(self.env.get_prompt)(jax.random.split(k_prompt, p))
    prompt_len = prompts.shape[1]
    ans_len = seq_len - prompt_len

    # Buffers to store results for each prompt
    prompt_buf = [[None] * c for _ in range(p)]
    ans_buf = [[None] * c for _ in range(p)]
    rew_buf = [[None] * c for _ in range(p)]
    logp_buf = [[None] * c for _ in range(p)]
    mask_buf = np.zeros((p, c), dtype=bool)

    correct_counts = jnp.zeros((p,), dtype=jnp.int32)
    num_generated = 0
    attempt = 0

    # Sample answers for active prompts
    if self.sampler_network:
      sampler_apply_fn = self.sampler_network.apply
    else:
      sampler_apply_fn = state.apply_fn

    # Loop until all prompts have enough correct samples or max attempts reached
    for attempt in range(m):
      if jnp.all(correct_counts >= c):
        break

      # Identify prompts that still need samples
      active_mask = correct_counts < c
      active_indices = jnp.where(active_mask)[0]
      num_active = active_indices.shape[0]

      if num_active == 0:
        break  # Should be caught by the all() check above, but for safety

      key, k_sample, k_reward = jax.random.split(key, 3)
      sample_keys = jax.random.split(k_sample, num_active)
      reward_keys = jax.random.split(k_reward, num_active)

      active_prompts = prompts[active_indices]

      def _sample(prompts: jax.Array, k: jax.Array) -> ar_sample.SampleResult:
        return self.sampler(
            apply_fn=sampler_apply_fn,
            params=state.params,
            prompts=prompts,
            key=k,
        )

      seqs, logps = jax.vmap(_sample)(active_prompts, sample_keys)
      answers = seqs[:, prompt_len:]  # (P x S, answer_len)
      num_generated += num_active

      # Evaluate rewards
      rewards = jax.vmap(self.env.get_reward)(
          active_prompts, answers, reward_keys
      )

      # Filter for correct answers and add to buffers
      is_correct = rewards >= thresh
      for i in range(num_active):
        prompt_idx = active_indices[i]
        current_correct = correct_counts[prompt_idx]
        if is_correct[i] and current_correct < c:
          prompt_buf[prompt_idx][current_correct] = active_prompts[i]
          ans_buf[prompt_idx][current_correct] = answers[i]
          rew_buf[prompt_idx][current_correct] = rewards[i]
          logp_buf[prompt_idx][current_correct] = logps[i]
          mask_buf[prompt_idx, current_correct] = True
          correct_counts = correct_counts.at[prompt_idx].add(1)

    # Flatten and stack, filling in None with padding
    flat_prompts = []
    flat_answers = []
    flat_rewards = []
    flat_logps = []

    dummy_prompt = jnp.zeros_like(prompts[0])
    dummy_answer = jnp.zeros(ans_len, dtype=jnp.int32)
    dummy_logp = jnp.zeros(seq_len, dtype=jnp.float32)

    for i in range(p):
      for j in range(c):
        flat_prompts.append(
            prompt_buf[i][j] if mask_buf[i, j] else dummy_prompt
        )
        flat_answers.append(ans_buf[i][j] if mask_buf[i, j] else dummy_answer)
        flat_rewards.append(rew_buf[i][j] if mask_buf[i, j] else 0.0)
        flat_logps.append(logp_buf[i][j] if mask_buf[i, j] else dummy_logp)

    final_prompts = jnp.stack(flat_prompts)  # (B, prompt_len)
    final_answers = jnp.stack(flat_answers)  # (B, ans_len)
    final_rewards = jnp.stack(flat_rewards)  # (B,)
    final_logps = jnp.stack(flat_logps)  # (B, seq_len)
    final_mask = mask_buf.reshape(-1)  # (B,)

    # Create group_ids
    group_ids = jnp.repeat(jnp.arange(p), c, axis=0)  # (B,)

    batch = base.Batch(
        prompts=final_prompts,
        answers=final_answers,
        rewards=final_rewards,
        sample_log_probs=final_logps,
        aux={"group_ids": group_ids, "mask": final_mask},
    )

    metrics = {
        "num_generated": jnp.array(num_generated),
        "num_correct": jnp.sum(mask_buf),
        "attempts_needed": jnp.array(attempt + 1),
        "padding_fraction": 1.0 - jnp.mean(final_mask.astype(jnp.float32)),
    }
    return batch, state, metrics
