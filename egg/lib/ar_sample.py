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

"""Autoregressive sampling."""

from __future__ import annotations

import dataclasses
import typing as tp

from egg import base
import jax
import jax.numpy as jnp


class SampleResult(tp.NamedTuple):
  tokens: jax.Array  # (B, seq_length) or (seq_length,) if B == 1
  log_probs: jax.Array  # same shape as `tokens`


def _extract_logits(
    outs: jax.Array | dict[str, jax.Array],
) -> jax.Array:
  if isinstance(outs, dict):
    return outs["policy_logits"]
  return outs


class _StepKeys(tp.NamedTuple):
  next_key: jax.Array
  sample_key: jax.Array
  explore_key: jax.Array
  model_key: jax.Array | None


def _split_step_keys(
    key: jax.Array,
    *,
    fixed_model_key: bool,
    fixed_model_rng: jax.Array | None,
) -> _StepKeys:
  """Splits one dense-sampler step worth of RNG state."""
  key, sample_key, explore_key = jax.random.split(key, 3)
  if fixed_model_key:
    model_key = fixed_model_rng
  else:
    key, model_key = jax.random.split(key)
  return _StepKeys(
      next_key=key,
      sample_key=sample_key,
      explore_key=explore_key,
      model_key=model_key,
  )


@dataclasses.dataclass(frozen=True)
class ARSampler:
  """Autoregressively sample *a batch* of sequences in parallel."""

  sequence_length: int
  pad_token: int = base.PAD_TOKEN
  fixed_model_key: bool = False  # If True, model key is fixed per episode.
  epsilon: float = 0.0  # Probability of taking a random action.
  vocab_size: int | None = None  # Must be provided if epsilon > 0
  use_decode_cache: bool = True

  def __post_init__(self):
    if self.epsilon > 0 and self.vocab_size is None:
      raise ValueError("vocab_size must be provided if epsilon > 0")

  def __call__(
      self,
      apply_fn: tp.Callable[..., jax.Array | dict[str, jax.Array]],
      params: tp.Any,
      prompts: jax.Array,  # (B, T_p) or (T_p,)
      key: jax.Array,
      model: tp.Any | None = None,
  ) -> SampleResult:
    """Autoregressively sample *a batch* of sequences in parallel."""
    if (
        self.use_decode_cache
        and model is not None
        and hasattr(model, "decode_step")
        and hasattr(model, "init_decode_cache")
    ):
      return self._sample_cached(apply_fn, params, prompts, key, model)
    return self._sample_dense(apply_fn, params, prompts, key)

  def _sample_dense(
      self,
      apply_fn: tp.Callable[..., jax.Array | dict[str, jax.Array]],
      params: tp.Any,
      prompts: jax.Array,
      key: jax.Array,
  ) -> SampleResult:
    """Dense autoregressive sampling without decode cache."""
    squeeze = prompts.ndim == 1
    if squeeze:
      prompts = prompts[None, :]  # (1, T_p)
    batch_size, prompt_len = prompts.shape

    seq = jnp.full(
        (batch_size, self.sequence_length), self.pad_token, prompts.dtype
    )
    seq = seq.at[:, :prompt_len].set(prompts)
    logp = jnp.zeros((batch_size, self.sequence_length), jnp.float32)

    if self.fixed_model_key:
      k_model, key = jax.random.split(key)
    else:
      k_model = None  # Will be generated inside the loop

    init = _Carry(seq=seq, logp=logp, key=key)

    def step(t: int, carry: _Carry) -> _Carry:
      seq, logp, key = carry
      step_keys = _split_step_keys(
          key,
          fixed_model_key=self.fixed_model_key,
          fixed_model_rng=k_model,
      )

      outs = apply_fn({"params": params}, seq, rngs={"noise": step_keys.model_key})
      logits = _extract_logits(outs)
      logits_t = logits[:, t - 1]  # (B, V)

      # Sample from model
      model_tok = jax.random.categorical(step_keys.sample_key, logits_t)  # (B,)

      if self.epsilon > 0:
        # Epsilon-greedy exploration
        explore_action = jax.random.randint(
            step_keys.explore_key,
            shape=(batch_size,),
            minval=0,
            maxval=self.vocab_size,
        )
        explore_cond = (
            jax.random.uniform(step_keys.explore_key, shape=(batch_size,))
            < self.epsilon
        )
        tok = jnp.where(explore_cond, explore_action, model_tok)
      else:
        tok = model_tok

      # Log probabilities of all actions according to the base model policy
      log_probs_model = jax.nn.log_softmax(logits_t)
      # Log probability of the *actually chosen* token 'tok'
      lp_t_model = log_probs_model[jnp.arange(batch_size), tok]

      if self.epsilon > 0:
        # Probability of token 'tok' under the model component
        prob_tok_model = jnp.exp(lp_t_model)
        # Probability of token 'tok' under the uniform random component
        prob_tok_random = 1.0 / self.vocab_size

        # The epsilon-greedy policy's probability for *any* action 'a' is:
        # P(a) = (1 - epsilon) * P(a | model) + epsilon * P(a | random)

        # So, for the specific token 'tok' that was chosen:
        mixed_prob_tok = (
            1.0 - self.epsilon
        ) * prob_tok_model + self.epsilon * prob_tok_random

        # The log probability of the chosen token 'tok' under the mixed policy
        lp_t = jnp.log(mixed_prob_tok + 1e-9)  # Added small value for stability
      else:
        lp_t = lp_t_model

      seq = seq.at[:, t].set(tok)
      logp = logp.at[:, t].set(lp_t)
      return _Carry(seq, logp, step_keys.next_key)

    final = jax.lax.fori_loop(
        lower=prompt_len,
        upper=self.sequence_length,
        body_fun=step,
        init_val=init,
    )

    tokens, log_probs, _ = final
    if squeeze:
      tokens, log_probs = tokens[0], log_probs[0]  # back to 1-D

    return SampleResult(tokens=tokens, log_probs=log_probs)

  def _sample_cached(
      self,
      apply_fn: tp.Callable[..., jax.Array | dict[str, jax.Array]],
      params: tp.Any,
      prompts: jax.Array,
      key: jax.Array,
      model: tp.Any,
  ) -> SampleResult:
    """Autoregressive sampling with prompt prefill and cached decoding."""
    squeeze = prompts.ndim == 1
    if squeeze:
      prompts = prompts[None, :]
    batch_size, prompt_len = prompts.shape
    if prompt_len == 0:
      return self._sample_dense(apply_fn, params, prompts, key)

    seq = jnp.full(
        (batch_size, self.sequence_length), self.pad_token, prompts.dtype
    )
    seq = seq.at[:, :prompt_len].set(prompts)
    logp = jnp.zeros((batch_size, self.sequence_length), jnp.float32)

    if self.fixed_model_key:
      k_model, key = jax.random.split(key)
    else:
      k_model = None

    def decode_apply(
        cache: dict[str, jax.Array] | None,
        token: jax.Array,
        model_key: jax.Array | None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
      variables: dict[str, tp.Any] = {"params": params}
      if cache is not None:
        variables["cache"] = cache
      if model_key is None:
        outs, mutated = apply_fn(
            variables,
            token,
            method=model.decode_step,
            mutable=["cache"],
        )
      else:
        outs, mutated = apply_fn(
            variables,
            token,
            rngs={"noise": model_key},
            method=model.decode_step,
            mutable=["cache"],
        )
      return _extract_logits(outs), mutated["cache"]

    def init_cache_apply(token: jax.Array) -> dict[str, jax.Array]:
      _, mutated = apply_fn(
          {"params": params},
          token,
          method=model.init_decode_cache,
          mutable=["cache"],
      )
      return mutated["cache"]

    def prompt_token_at(i: int) -> jax.Array:
      return jax.lax.dynamic_slice(prompts, (0, i), (batch_size, 1))

    first_step_keys = _split_step_keys(
        key,
        fixed_model_key=self.fixed_model_key,
        fixed_model_rng=k_model,
    )
    cache = init_cache_apply(prompt_token_at(0))
    prefill_model_key = first_step_keys.model_key
    logits_prev, cache = decode_apply(
        cache,
        prompt_token_at(0),
        prefill_model_key,
    )

    def prefill_step(
        i: int,
        carry: tuple[dict[str, jax.Array], jax.Array],
    ) -> tuple[dict[str, jax.Array], jax.Array]:
      current_cache, current_logits = carry
      current_logits, current_cache = decode_apply(
          current_cache,
          prompt_token_at(i),
          prefill_model_key,
      )
      return current_cache, current_logits

    cache, logits_prev = jax.lax.fori_loop(
        1,
        prompt_len,
        prefill_step,
        (cache, logits_prev),
    )

    init = _DecodeCarry(
        seq=seq,
        logp=logp,
        key=first_step_keys.next_key,
        sample_key=first_step_keys.sample_key,
        explore_key=first_step_keys.explore_key,
        cache=cache,
        logits_prev=logits_prev,
    )

    def step(t: int, carry: _DecodeCarry) -> _DecodeCarry:
      seq, logp, key, sample_key, explore_key, cache, logits_prev = carry

      logits_t = logits_prev[:, 0]  # (B, V)
      model_tok = jax.random.categorical(sample_key, logits_t)

      if self.epsilon > 0:
        explore_action = jax.random.randint(
            explore_key,
            shape=(batch_size,),
            minval=0,
            maxval=self.vocab_size,
        )
        explore_cond = (
            jax.random.uniform(explore_key, shape=(batch_size,)) < self.epsilon
        )
        tok = jnp.where(explore_cond, explore_action, model_tok)
      else:
        tok = model_tok

      log_probs_model = jax.nn.log_softmax(logits_t)
      lp_t_model = log_probs_model[jnp.arange(batch_size), tok]

      if self.epsilon > 0:
        prob_tok_model = jnp.exp(lp_t_model)
        prob_tok_random = 1.0 / self.vocab_size
        mixed_prob_tok = (
            1.0 - self.epsilon
        ) * prob_tok_model + self.epsilon * prob_tok_random
        lp_t = jnp.log(mixed_prob_tok + 1e-9)
      else:
        lp_t = lp_t_model

      seq = seq.at[:, t].set(tok)
      logp = logp.at[:, t].set(lp_t)

      def decode_next(
          state: tuple[jax.Array, dict[str, jax.Array]],
      ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array], jax.Array]:
        inner_key, inner_cache = state
        next_step_keys = _split_step_keys(
            inner_key,
            fixed_model_key=self.fixed_model_key,
            fixed_model_rng=k_model,
        )
        next_logits, inner_cache = decode_apply(
            inner_cache,
            tok[:, None],
            next_step_keys.model_key,
        )
        return (
            next_step_keys.next_key,
            next_step_keys.sample_key,
            next_step_keys.explore_key,
            inner_cache,
            next_logits,
        )

      def keep_state(
          state: tuple[jax.Array, dict[str, jax.Array]],
      ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array], jax.Array]:
        inner_key, inner_cache = state
        return inner_key, sample_key, explore_key, inner_cache, logits_prev

      key, sample_key, explore_key, cache, logits_prev = jax.lax.cond(
          t + 1 < self.sequence_length,
          decode_next,
          keep_state,
          (key, cache),
      )
      return _DecodeCarry(
          seq,
          logp,
          key,
          sample_key,
          explore_key,
          cache,
          logits_prev,
      )

    final = jax.lax.fori_loop(
        lower=prompt_len,
        upper=self.sequence_length,
        body_fun=step,
        init_val=init,
    )

    tokens, log_probs, _, _, _, _, _ = final
    if squeeze:
      tokens, log_probs = tokens[0], log_probs[0]

    return SampleResult(tokens=tokens, log_probs=log_probs)


def get_full_logprobs_b_l(
    apply_fn: tp.Callable[..., jax.Array],
    params: tp.Any,
    sequences: jax.Array,  # (B, L), int32 (may include PAD_TOKEN=-1)
    rng: jax.Array | None = None,
) -> jax.Array:
  """Return per-token log p(token_t | seq_<t) for whole sequence, shape (B, L)."""
  batch_size, seq_len = sequences.shape
  if seq_len == 0:
    return jnp.zeros((batch_size, 0), dtype=jnp.float32)

  # Feed PADs as token 0 into the model.
  seq_in = jnp.where(sequences < 0, 0, sequences)

  # Forward pass (optionally with 'noise' RNG).
  if rng is None:
    logits = apply_fn({"params": params}, seq_in)  # (B, L, V)
  else:
    logits = apply_fn({"params": params}, seq_in, rngs={"noise": rng})

  if seq_len == 1:
    return jnp.zeros((batch_size, 1), dtype=jnp.float32)

  # Log-softmax over vocabulary for t=1..L-1 (first position has no context).
  tok_logp_all = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)

  # Targets are the *input* sequence shifted by 1 (PADs already mapped to 0).
  targets = seq_in[:, 1:]  # (B, L-1)
  tok_logps = jnp.take_along_axis(
      tok_logp_all, targets[..., None], axis=-1
  ).squeeze(-1)

  # Mask out locations where the original (pre-mapped) target was PAD.
  valid_mask = (sequences[:, 1:] >= 0).astype(tok_logps.dtype)  # (B, L-1)
  tok_logps = tok_logps * valid_mask

  # Pad a leading zero so output aligns to length L.
  return jnp.pad(tok_logps, ((0, 0), (1, 0)), constant_values=0.0).astype(
      jnp.float32
  )


class _Carry(tp.NamedTuple):
  seq: jax.Array  # (B, seq_length)
  logp: jax.Array  # (B, seq_length)
  key: jax.Array


class _DecodeCarry(tp.NamedTuple):
  seq: jax.Array  # (B, seq_length)
  logp: jax.Array  # (B, seq_length)
  key: jax.Array
  sample_key: jax.Array
  explore_key: jax.Array
  cache: dict[str, jax.Array]
  logits_prev: jax.Array  # (B, 1, vocab_size)
