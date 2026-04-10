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

"""Light-weight helpers for computing descriptive statistics on JAX arrays.

All functions are JIT-safe (pure JAX).  Every returned metric is a 0-D
`jnp.ndarray`, so it can flow through existing logging code unchanged.
"""

import typing as tp

import jax
import jax.numpy as jnp
import optax


def scalar_stats(x: jnp.ndarray, prefix: str) -> dict[str, jnp.ndarray]:
  """Computes min/max/mean/std with a prefix-for-key helper."""
  return {
      f"{prefix}_min": jnp.min(x),
      f"{prefix}_max": jnp.max(x),
      f"{prefix}_mean": jnp.mean(x),
      f"{prefix}_std": jnp.std(x),
  }


def correlation(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Pearson r between two 1-D tensors (returns 0 if either is constant)."""
  x_c = x - jnp.mean(x)
  y_c = y - jnp.mean(y)
  denom = jnp.sqrt(jnp.sum(x_c**2) * jnp.sum(y_c**2))
  return jnp.where(denom == 0.0, 0.0, jnp.sum(x_c * y_c) / denom)


def entropy_from_logp(logp: jnp.ndarray) -> jnp.ndarray:
  """Shannon entropy of each row in `logp` (… , V)."""
  p = jnp.exp(logp)
  return -jnp.sum(p * logp, axis=-1)  # (...,)


def logp_stats(
    learner_logp: jnp.ndarray,
    sampler_logp: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
  """Summaries & correlation of per-token log-probs and probs."""
  diff = learner_logp - sampler_logp
  stats = {}
  stats.update(scalar_stats(learner_logp, "logp_learner"))
  stats.update(scalar_stats(sampler_logp, "logp_sampler"))
  stats.update(scalar_stats(diff, "logp_diff"))
  stats["logp_corr"] = correlation(learner_logp, sampler_logp)

  # --- probability space ------------------------------------------------
  learner_prob = jnp.exp(learner_logp)
  sampler_prob = jnp.exp(sampler_logp)
  prob_diff = learner_prob - sampler_prob
  stats.update(scalar_stats(learner_prob, "prob_learner"))
  stats.update(scalar_stats(sampler_prob, "prob_sampler"))
  stats.update(scalar_stats(prob_diff, "prob_diff"))
  stats["prob_corr"] = correlation(learner_prob, sampler_prob)

  return stats


def l2_global_norm(tree1: tp.Any, tree2: tp.Any) -> jnp.ndarray:
  """Computes the global L2 norm of the difference between two trees."""
  diff_tree = jax.tree_util.tree_map(lambda x, y: x - y, tree1, tree2)
  return jnp.asarray(optax.tree.norm(diff_tree))
