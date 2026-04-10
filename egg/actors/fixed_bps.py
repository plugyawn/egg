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

"""Fixed BPS IID sampling actor.

For each batch of size B = prompts_per_batch * samples_per_prompt.
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
  """Configuration for the FixedBPSActor."""

  env_config: base.MakeableConfig[base.Environment]
  sequence_length: int
  prompts_per_batch: int
  samples_per_prompt: int
  sampler_network_config: base.MakeableConfig[nn.Module] | None = None
  epsilon: float = 0.0  # Probability of random action in sampler

  def make(self) -> "FixedBPSActor":
    sampler_network = None
    if self.sampler_network_config:
      sampler_network = self.sampler_network_config.make()
    env = self.env_config.make()
    return FixedBPSActor(
        config=self,
        env=env,
        sampler=ar_sample.ARSampler(
            sequence_length=self.sequence_length,
            epsilon=self.epsilon,
            vocab_size=env.spec.vocab_size,
        ),
        sampler_network=sampler_network,
    )


@dataclasses.dataclass(frozen=True)
class FixedBPSActor(base.Actor[base.StateT]):
  """Samples a batch with a fixed number of samples per prompt."""

  config: ActorConfig
  env: base.Environment
  sampler: ar_sample.ARSampler
  sampler_network: nn.Module | None = None

  def sample_batch(
      self, state: base.StateT, key: jax.Array
  ) -> tuple[base.Batch, base.StateT, base.Metrics]:

    p, s = self.config.prompts_per_batch, self.config.samples_per_prompt
    _, k_prompt, k_sample, k_reward = jax.random.split(key, 4)

    # 1. prompts ---------------------------------------------------------
    prompts = jax.vmap(self.env.get_prompt)(jax.random.split(k_prompt, p))
    prompts_flat = jnp.repeat(prompts, s, axis=0)  # (B, P_len)
    prompt_len = prompts_flat.shape[-1]

    # 2. batched autoregressive sample ----------------------------------
    if self.sampler_network:
      sampler_apply_fn = self.sampler_network.apply
    else:
      sampler_apply_fn = state.apply_fn

    seqs, logps = self.sampler(
        sampler_apply_fn, state.params, prompts_flat, k_sample
    )
    answers = seqs[:, prompt_len:]  # (B, A_len)

    # 3. rewards ---------------------------------------------------------
    rewards = jax.vmap(self.env.get_reward)(
        prompts_flat, answers, jax.random.split(k_reward, p * s)
    )

    group_ids = jnp.repeat(jnp.arange(p), s)  # (B,)

    batch = base.Batch(
        prompts=prompts_flat,
        answers=answers,
        rewards=rewards,
        sample_log_probs=logps,
        aux={"group_ids": group_ids},
    )
    return batch, state, {}
