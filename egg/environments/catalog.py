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

"""Catalog of environments, exposable as MakeableConfigs."""

from __future__ import annotations

import enum

from egg import base
from egg.environments import bit_parity
from egg.environments import finite_prompt_wrapper
from egg.environments import gaussian_noise_wrapper
from egg.environments import key_value
from egg.environments import landmine_wrapper
from egg.environments import noisy_reward
from egg.environments import poison_token_wrapper
from egg.environments import reverse_copy
from egg.environments import subsequence_match
from egg.environments import zeros_prompt_wrapper


class EnvironmentConfigs(enum.Enum):
  """Enum for base environment configurations."""

  BIT_PARITY = bit_parity.EnvConfig
  KEY_VALUE = key_value.EnvConfig
  NOISY_REWARD = noisy_reward.EnvConfig
  REVERSE_COPY = reverse_copy.EnvConfig
  SUBSEQUENCE_MATCH = subsequence_match.EnvConfig

  def make(self, **kwargs) -> base.Environment:
    """Returns an environment instance with the given config overrides."""
    return self.value(**kwargs).make()

  def get_config(self, **kwargs) -> base.MakeableConfig[base.Environment]:
    """Returns an environment config instance with the given overrides."""
    return self.value(**kwargs)


class WrapperConfigs(enum.Enum):
  """Enum for wrapper environment configurations."""

  FINITE_PROMPT_WRAPPER = finite_prompt_wrapper.EnvConfig
  GAUSSIAN_NOISE_WRAPPER = gaussian_noise_wrapper.EnvConfig
  LANDMINE_WRAPPER = landmine_wrapper.EnvConfig
  POISON_TOKEN_WRAPPER = poison_token_wrapper.EnvConfig
  ZEROS_PROMPT_WRAPPER = zeros_prompt_wrapper.EnvConfig

  def make(
      self, inner_env_config: base.MakeableConfig[base.Environment], **kwargs
  ) -> base.Environment:
    """Returns an environment instance with the given config overrides."""
    return self.value(inner_env_config=inner_env_config, **kwargs).make()

  def get_config(
      self, inner_env_config: base.MakeableConfig[base.Environment], **kwargs
  ) -> base.MakeableConfig[base.Environment]:
    """Returns an environment config instance with the given overrides."""
    return self.value(inner_env_config=inner_env_config, **kwargs)
