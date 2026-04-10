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

"""Run the RL experiment."""

import dataclasses
import typing as tp

from absl import app
from egg import learners
from egg.actors import fixed_bps
from egg.environments import subsequence_match
from egg.lib import logging
from egg.losses import catalog
from egg.networks import transformers
from egg.trainers import vanilla_sync
import fancyflags as ff
import pandas as pd


@dataclasses.dataclass
class SweepConfig:
  """Configuration options to be overridden by flags."""

  prompt_length: int = 4  # Prompt sequence length.
  answer_length: int = 12  # Answer sequence length.
  vocab_size: int = 5  # Vocabulary size.
  use_baseline: bool = False  # Use grouped baseline.
  learning_rate: float = 1e-4  # Learning rate.
  seed: int = 42  # Random seed.
  num_steps: int = 1000  # Total training steps.


# Define flag structures using fancyflags
SweepFlags = ff.DEFINE_auto('sweep', SweepConfig, 'Sweep configuration.')


def run_experiment(sweep_config: SweepConfig) -> pd.DataFrame:
  """Creates a Config object from a SweepConfig object."""
  env_cfg = subsequence_match.EnvConfig(
      prompt_length=sweep_config.prompt_length,
      answer_length=sweep_config.answer_length,
      vocab_size=sweep_config.vocab_size,
  )

  net_cfg = transformers.NetworkConfig(
      vocab_size=env_cfg.vocab_size,
      sequence_length=env_cfg.prompt_length + env_cfg.answer_length,
  )
  loss_cfg = catalog.Loss.REINFORCE.config(
      use_grouped_baseline=sweep_config.use_baseline,
  )
  learner_cfg = learners.LearnerConfig(
      model_config=net_cfg,
      loss_config=loss_cfg,
      prompt_length=env_cfg.prompt_length,
      learning_rate=sweep_config.learning_rate,
  )

  actor_cfg = fixed_bps.ActorConfig(
      env_config=env_cfg,
      sequence_length=env_cfg.prompt_length + env_cfg.answer_length,
      prompts_per_batch=8,
      samples_per_prompt=8,
  )

  trainer_cfg = vanilla_sync.TrainerConfig(
      steps=sweep_config.num_steps,
      seed=sweep_config.seed,
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
