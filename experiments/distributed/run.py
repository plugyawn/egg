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
from egg import learners
from egg.actors import dreamer_bps
from egg.environments import catalog as env_catalog
from egg.lib import logging
from egg.losses import catalog
from egg.losses import dg
from egg.networks import logit_noise
from egg.networks import transformers
from egg.trainers import vanilla_async
import fancyflags as ff
import jax.numpy as jnp
import pandas as pd

VALID_ENVS = ["bit_parity", "key_value", "reverse_copy", "subsequence_match"]


@dataclasses.dataclass
class SweepConfig:
  """Configuration options to be overridden by flags."""

  # Environment parameters
  env: str = "reverse_copy"  # Environment config string

  # Learning algorithm parameters
  loss: str = "dg"  # Loss config string
  loss_param_one: float = 1  # used for eta
  loss_param_two: float = 1e-6  # used for lambda
  override_token_prob: float | None = 1.0  # Fixed prob for overridden tokens

  # Standard training parameters
  learning_rate: float = 3e-4
  seed: int = 42
  num_steps: int = 1000
  prompts_per_batch: int = 10
  samples_per_prompt: int = 10


# Define flag structures using fancyflags
SweepFlags = ff.DEFINE_from_instance(
    "sweep", SweepConfig(), "Sweep configuration."
)


def run_experiment(sweep_config: SweepConfig) -> pd.DataFrame:
  """Sets up and runs a single experiment based on the sweep config."""
  assert sweep_config.env in VALID_ENVS, f"Invalid env: {sweep_config.env}"

  env_cfg = env_catalog.EnvironmentConfigs[
      sweep_config.env.upper()
  ].get_config()
  spec = env_cfg.make().spec

  # Materialize env to get vocab size
  sequence_length = spec.prompt_length + spec.answer_length

  # --- 2. Network Configuration ---
  base_net_cfg = transformers.NetworkConfig(
      vocab_size=spec.vocab_size,
      sequence_length=sequence_length,
  )

  # Learner uses the base network
  net_cfg = logit_noise.NetworkConfig(
      inner_network_config=base_net_cfg,
      vocab_size=spec.vocab_size,
      sigma=0,
  )

  # --- 3. Actor Configuration ---
  actor_cfg = dreamer_bps.ActorConfig(
      env_config=env_cfg,
      sequence_length=sequence_length,
      prompts_per_batch=sweep_config.prompts_per_batch,
      samples_per_prompt=sweep_config.samples_per_prompt,
      sampler_network_config=net_cfg,  # Actor uses the biased network
      landmine_sequence=jnp.zeros(spec.prompt_length, dtype=jnp.int32),
      override_token_prob=sweep_config.override_token_prob,
  )

  # --- 4. Loss Configuration ---
  if sweep_config.loss == "dg":
    loss_cfg = dg.LossConfig(
        eta_learn=sweep_config.loss_param_one,
        lambda_learn=sweep_config.loss_param_two,
        use_grouped_baseline=True,
        num_groups=sweep_config.prompts_per_batch,
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
      model_config=net_cfg,  # Learner uses the clean network
      loss_config=loss_cfg,
      prompt_length=spec.prompt_length,
      learning_rate=sweep_config.learning_rate,
  )

  # --- 5. Trainer Configuration ---
  trainer_cfg = vanilla_async.TrainerConfig(
      steps=sweep_config.num_steps,
      seed=sweep_config.seed,
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
