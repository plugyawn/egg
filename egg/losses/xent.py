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

"""Standard cross-entropy loss on the answer tokens."""

from __future__ import annotations

import dataclasses

from egg import base
from egg.lib import statistics
from egg.losses import common
import jax
import jax.numpy as jnp


class LossConfig(base.MakeableConfig[base.LossFn]):
  """Makeable config that produces a CrossEntropyLoss."""

  def make(self) -> CrossEntropyLoss:
    return CrossEntropyLoss()


@dataclasses.dataclass(frozen=True)
class CrossEntropyLoss(base.LossFn):
  """Standard cross-entropy loss on the answer tokens."""

  def __call__(
      self,
      params: base.Params,
      state: base.StateT,
      batch: base.Batch,
      key: jax.Array,
  ) -> tuple[jax.Array, base.Metrics]:
    fwd = common.forward_pass(params, state, batch, key)
    lp_samp = common.sampler_answer_logprobs(batch)

    # Cross-entropy = negative log-prob of the correct next token
    loss = -jnp.sum(fwd.lp_answer * fwd.token_mask) / (
        jnp.sum(fwd.token_mask) + 1e-8
    )

    # Metrics
    entropy = statistics.entropy_from_logp(fwd.lp_all)
    logp_stats = statistics.logp_stats(
        learner_logp=fwd.lp_answer,
        sampler_logp=lp_samp,
    )
    metrics = {
        "num_examples": jnp.array(batch.prompts.shape[0]),
        **statistics.scalar_stats(entropy, "policy_entropy"),
        **logp_stats,
    }
    return loss, metrics
