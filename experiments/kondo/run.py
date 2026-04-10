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

"""Run the costly-compute (Kondo gate) experiment on token reversal.

Reproduces the experiments from the costly paper (§5):
  Figure 7: PG vs DG vs DG-K learning curves at challenging (H, M).
  Figure 8: Compute speedup vs backward cost at increasing (H, M).

Losses:
  delightful: DG with sigmoid gate (no backward-pass savings).
  kondo: DG-K with binary Kondo gate (skips backward passes).
  reinforce: PG baseline.
  ppo, pmpo: standard RL baselines.
"""

import dataclasses
import enum
import typing as tp

from absl import app
from egg import base
from egg import learners
from egg.actors import dreamer_bps
from egg.environments import landmine_wrapper
from egg.environments import reverse_copy
from egg.environments import zeros_prompt_wrapper
from egg.lib import logging
from egg.losses import catalog
from egg.losses import dg
from egg.losses import kondo
from egg.networks import logit_noise
from egg.networks import transformers
from egg.trainers import vanilla_async
import fancyflags as ff
import jax.numpy as jnp
import pandas as pd


class LossType(enum.StrEnum):
  DG = 'dg'
  KONDO = 'kondo'
  PPO = 'ppo'
  REINFORCE = 'reinforce'
  PMPO = 'pmpo'


@dataclasses.dataclass
class SweepConfig:
  """Configuration options to be overridden by flags."""

  prompt_length: int = 10
  vocab_size: int = 2
  kappa: float = 1
  reward_to_first_error: bool = False
  target_reverse: bool = True

  prob_zero: float = 0.0
  zero_noise_std: float = 0.0

  embed_dim: int = 64

  loss: str = LossType.KONDO

  loss_param_one: float = 1.0  # main param
  loss_param_two: float = 0.0  # Used for Beta KL
  lambda_learn: float = 0.0
  stochastic_gate: bool = False

  # 'delight', 'advantage', 'abs_advantage', 'surprisal', 'uniform', 'additive'
  priority: str = 'delight'
  alpha_additive: float = 0.5  # Only used if priority = additive

  learning_rate: float = 3e-4
  seed: int = 42
  num_steps: int = 1000
  prompts_per_batch: int = 10
  samples_per_prompt: int = 10
  log_details: bool = False


SweepFlags = ff.DEFINE_from_instance(
    'sweep', SweepConfig(), 'Sweep configuration.'
)


def run_experiment(sweep_config: SweepConfig) -> pd.DataFrame:
  """Sets up and runs a single experiment based on the sweep config."""

  env_cfg: base.MakeableConfig = reverse_copy.EnvConfig(
      prompt_length=sweep_config.prompt_length,
      kappa=sweep_config.kappa,
      vocab_size=sweep_config.vocab_size,
      reward_to_first_error=sweep_config.reward_to_first_error,
      target_reverse=sweep_config.target_reverse,
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
  )

  sequence_length = sweep_config.prompt_length * 2

  base_net_cfg = transformers.NetworkConfig(
      vocab_size=sweep_config.vocab_size,
      sequence_length=sequence_length,
      embed_dim=sweep_config.embed_dim,
  )

  learner_net_cfg = logit_noise.NetworkConfig(
      inner_network_config=base_net_cfg,
      vocab_size=sweep_config.vocab_size,
      sigma=0,
      fixed=True,
  )

  sampler_net_cfg = logit_noise.NetworkConfig(
      inner_network_config=base_net_cfg,
      vocab_size=sweep_config.vocab_size,
      sigma=0,
      fixed=True,
  )

  actor_cfg = dreamer_bps.ActorConfig(
      env_config=env_cfg,
      sequence_length=sequence_length,
      prompts_per_batch=sweep_config.prompts_per_batch,
      samples_per_prompt=sweep_config.samples_per_prompt,
      sampler_network_config=sampler_net_cfg,
      landmine_sequence=landmine_seq,
  )

  if sweep_config.loss == 'dg':
    loss_cfg = dg.LossConfig(
        eta_learn=sweep_config.loss_param_one,
        beta_kl=sweep_config.loss_param_two,
        lambda_learn=sweep_config.lambda_learn,
        stochastic_gate=sweep_config.stochastic_gate,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  elif sweep_config.loss == 'kondo':
    loss_cfg = kondo.LossConfig(
        pct_learn=sweep_config.loss_param_one,
        priority=sweep_config.priority,
        alpha_additive=sweep_config.alpha_additive,
        beta_kl=sweep_config.loss_param_two,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  elif sweep_config.loss == 'ppo':
    loss_cfg = catalog.Loss.PPO.config(
        clip_epsilon=sweep_config.loss_param_one,
        beta_kl=sweep_config.loss_param_two,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  elif sweep_config.loss == 'reinforce':
    loss_cfg = catalog.Loss.PPO.config(
        clip_epsilon=1e9,
        beta_kl=0.0,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  elif sweep_config.loss == 'pmpo':
    loss_cfg = catalog.Loss.PMPO.config(
        alpha=sweep_config.loss_param_one,
        beta=sweep_config.loss_param_two,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
    )
  else:
    raise ValueError(f'Unknown loss config: {sweep_config.loss}')

  learner_cfg = learners.LearnerConfig(
      model_config=learner_net_cfg,
      loss_config=loss_cfg,
      prompt_length=sweep_config.prompt_length,
      learning_rate=sweep_config.learning_rate,
  )

  trainer_cfg = vanilla_async.TrainerConfig(
      steps=sweep_config.num_steps,
      seed=sweep_config.seed,
      log_details=sweep_config.log_details,
  )

  actor = actor_cfg.make()
  learner = learner_cfg.make()
  trainer = trainer_cfg.make()

  df = trainer(actor, learner)
  return logging.add_config_to_df(df, sweep_config)


def main(argv: tp.Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  sweep_config = SweepFlags.value()
  run_experiment(sweep_config)


if __name__ == '__main__':
  app.run(main)
