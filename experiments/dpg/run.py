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

"""Run the RL experiment with async updates and configurable actor network bias."""

import dataclasses
import typing as tp

from absl import app
from egg import base
from egg import learners
from egg.actors import dreamer_bps
from egg.environments import bernoulli_noise_wrapper
from egg.environments import gaussian_noise_wrapper
from egg.environments import landmine_wrapper
from egg.environments import reverse_copy
from egg.environments import zeros_prompt_wrapper
from egg.lib import logging
from egg.losses import catalog
from egg.losses import dg
from egg.networks import logit_noise
from egg.networks import transformers
from egg.trainers import vanilla_async
import fancyflags as ff
import jax.numpy as jnp
import pandas as pd


@dataclasses.dataclass
class SweepConfig:
  """Configuration options to be overridden by flags."""

  # Environment parameters
  prompt_length: int = 10
  vocab_size: int = 2
  kappa: float = 1  # For reverse_copy reward
  reward_to_first_error: bool = False  # For reverse_copy reward
  target_reverse: bool = True  # For reverse_copy reward
  target_increment: bool = False  # For reverse_copy reward

  # Noise settings
  noise_std: float = 0.0  # Std dev of Gaussian noise to add to rewards
  p_noise: float = 0.0  # Probability of overriding the reward unif({0, 1})
  prob_zero: float = 0.0  # Probability of zeroing out all tokens.
  zero_noise_std: float = 0.0  # Std of Gaussian noise to add to zero rewards.

  # Network parameters
  embed_dim: int = 64

  # Landmine settings
  landmine_reward: float = 0.0  # Penalty mean for landmine
  landmine_std: float = 0.0

  # Async Actor parameters
  actor_sigma: float = 0.0  # Stddev for actor network logit noise
  actor_fixed_noise: bool = True  # Fixed noise on actor network
  sampler_delay: int = 0  # Delay for sampler network updates
  uniform_delay: bool = True  # Uniform delay for sampler network updates
  actor_bug_prob: float = 0.0  # Probability of injecting landmine answer
  actor_correct_prob: float = 0.0  # Probability of injecting correct answer
  actor_random_prob: float = 0.0  # Probability of injecting random answer

  # Learning algorithm parameters
  loss: str = "dg"  # Loss config string

  # PPO parameters (only used if enlightened=False)
  loss_param_one: float = 1  # used for eta
  loss_param_two: float = 0  # used for beta KL
  lambda_learn: float = 0  # Delight threshold λ; -inf disables.
  pct_learn: float = 1  # Target fraction of tokens to learn.
  delight_noise_std: float = 0.0  # Std of noise added to gate input
  override_ucb: float | None = None  # Override UCB (advantage or surprisal)
  override_token_prob: float | None = 1.0  # Fixed prob for overridden tokens
  stochastic_gate: bool = False  # Use stochastic gate for Delightful

  # Standard training parameters
  learning_rate: float = 3e-4
  seed: int = 42
  num_steps: int = 1000
  prompts_per_batch: int = 10
  samples_per_prompt: int = 10
  log_details: bool = False


# Define flag structures using fancyflags
SweepFlags = ff.DEFINE_auto("sweep", SweepConfig, "Sweep configuration.")


def run_experiment(sweep_config: SweepConfig) -> pd.DataFrame:
  """Sets up and runs a single experiment based on the sweep config."""

  # --- 1. Environment Configuration ---
  inner_env_cfg = reverse_copy.EnvConfig(
      prompt_length=sweep_config.prompt_length,
      kappa=sweep_config.kappa,
      vocab_size=sweep_config.vocab_size,
      reward_to_first_error=sweep_config.reward_to_first_error,
      target_reverse=sweep_config.target_reverse,
      target_increment=sweep_config.target_increment,
  )

  env_cfg: base.MakeableConfig = inner_env_cfg
  if sweep_config.noise_std > 0:
    env_cfg = gaussian_noise_wrapper.EnvConfig(
        inner_env_config=env_cfg,
        noise_std=sweep_config.noise_std,
    )
  if sweep_config.p_noise > 0:
    env_cfg = bernoulli_noise_wrapper.EnvConfig(
        inner_env_config=env_cfg,
        p_noise=sweep_config.p_noise,
        min_val=0.0,
        max_val=1.0,
    )

  if sweep_config.prob_zero > 0:
    env_cfg = zeros_prompt_wrapper.EnvConfig(
        inner_env_config=env_cfg,
        prob_zero=sweep_config.prob_zero,
        zero_noise_std=sweep_config.zero_noise_std,
    )

  landmine_seq = jnp.zeros(sweep_config.prompt_length, dtype=jnp.int32)
  env_cfg = landmine_wrapper.EnvConfig(
      inner_env_config=env_cfg,
      landmine_sequence=landmine_seq,
      landmine_reward=sweep_config.landmine_reward,
      landmine_std=sweep_config.landmine_std,
  )

  # Materialize env to get vocab size
  sequence_length = sweep_config.prompt_length * 2

  # --- 2. Network Configuration ---
  base_net_cfg = transformers.NetworkConfig(
      vocab_size=sweep_config.vocab_size,
      sequence_length=sequence_length,
      embed_dim=sweep_config.embed_dim,
  )

  # Learner uses the base network
  learner_net_cfg = logit_noise.NetworkConfig(
      inner_network_config=base_net_cfg,
      vocab_size=sweep_config.vocab_size,
      sigma=0,
      fixed=sweep_config.actor_fixed_noise,
  )

  # Actor network has added logit noise only if sigma > 0
  sampler_net_cfg = logit_noise.NetworkConfig(
      inner_network_config=base_net_cfg,
      vocab_size=sweep_config.vocab_size,
      sigma=sweep_config.actor_sigma,
      fixed=sweep_config.actor_fixed_noise,
  )

  # --- 3. Actor Configuration ---
  actor_cfg = dreamer_bps.ActorConfig(
      env_config=env_cfg,
      sequence_length=sequence_length,
      prompts_per_batch=sweep_config.prompts_per_batch,
      samples_per_prompt=sweep_config.samples_per_prompt,
      sampler_network_config=sampler_net_cfg,  # Actor uses the biased network
      landmine_sequence=landmine_seq,
      bug_prob=sweep_config.actor_bug_prob,
      correct_prob=sweep_config.actor_correct_prob,
      random_prob=sweep_config.actor_random_prob,
      override_token_prob=sweep_config.override_token_prob,
  )

  # --- 4. Loss Configuration ---
  if sweep_config.loss == "dg":
    loss_cfg = dg.LossConfig(
        eta_learn=sweep_config.loss_param_one,
        beta_kl=sweep_config.loss_param_two,
        lambda_learn=sweep_config.lambda_learn,
        use_grouped_baseline=True,
        stochastic_gate=sweep_config.stochastic_gate,
        num_groups=sweep_config.prompts_per_batch,
        override_ucb=sweep_config.override_ucb,
        delight_noise_std=sweep_config.delight_noise_std,
    )
  elif sweep_config.loss == "ppo":
    loss_cfg = catalog.Loss.PPO.config(
        clip_epsilon=sweep_config.loss_param_one,
        beta_kl=sweep_config.loss_param_two,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  elif sweep_config.loss == "reinforce":
    print("Overriding PPO loss with REINFORCE")
    loss_cfg = catalog.Loss.PPO.config(
        clip_epsilon=1e9,
        beta_kl=0.0,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  elif sweep_config.loss == "pmpo":
    loss_cfg = catalog.Loss.PMPO.config(
        alpha=sweep_config.loss_param_one,
        beta=sweep_config.loss_param_two,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  else:
    raise ValueError(f"Unknown loss config: {sweep_config.loss}")

  learner_cfg = learners.LearnerConfig(
      model_config=learner_net_cfg,  # Learner uses the clean network
      loss_config=loss_cfg,
      prompt_length=sweep_config.prompt_length,
      learning_rate=sweep_config.learning_rate,
  )

  # --- 5. Trainer Configuration ---
  trainer_cfg = vanilla_async.TrainerConfig(
      steps=sweep_config.num_steps,
      seed=sweep_config.seed,
      sampler_delay=sweep_config.sampler_delay,
      uniform_delay=sweep_config.uniform_delay,
      log_details=sweep_config.log_details,
  )

  # --- 6. Run the training loop ---
  actor = actor_cfg.make()
  learner = learner_cfg.make()
  trainer = trainer_cfg.make()

  df = trainer(actor, learner)
  return logging.add_config_to_df(df, sweep_config)


def main(argv: tp.Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  sweep_config = SweepFlags.value()
  run_experiment(sweep_config)


if __name__ == "__main__":
  app.run(main)
