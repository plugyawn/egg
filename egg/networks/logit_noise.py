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

"""A flax linen wrapper to add Gaussian noise to a network's output logits."""

from __future__ import annotations

import dataclasses

from egg import base
import flax.linen as nn
import jax


@dataclasses.dataclass
class NetworkConfig(base.MakeableConfig[nn.Module]):
  """Config for the logit-wrapping network."""

  inner_network_config: base.MakeableConfig[nn.Module]  # The network to wrap.
  vocab_size: int  # Vocabulary size, needed for noise shape.
  sigma: float  # Standard deviation of the Gaussian noise on logits.
  fixed: bool = True  # Whether to use fixed noise.

  def make(self) -> LogitNoiseNetwork:
    """Returns a LogitNoiseNetwork with the given configuration."""
    return LogitNoiseNetwork(
        inner_network=self.inner_network_config.make(),
        vocab_size=self.vocab_size,
        sigma=self.sigma,
        fixed=self.fixed,
    )


class LogitNoiseNetwork(nn.Module):
  """A flax linen wrapper to add Gaussian noise to a network's output logits."""

  inner_network: nn.Module
  vocab_size: int
  sigma: float
  fixed: bool

  def setup(self):
    if self.fixed:
      # Define the unit noise parameter. Created regardless of self.sigma.
      # It will be stored in the 'params' collection.
      self.unit_fixed_noise = self.param(
          "unit_fixed_logit_noise",
          jax.random.normal,  # Initializer function
          (self.vocab_size,),  # Shape of the parameter
      )

  @nn.compact
  def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
    """Calls the inner network and adds the logit noise."""
    base_logits = self.inner_network(x, *args, **kwargs)

    if self.sigma == 0.0:
      return base_logits

    if self.fixed:
      # Scale the unit noise by the instance's sigma.
      noise = self.unit_fixed_noise * self.sigma
    else:
      # Sample noise on every forward pass using the 'noise' RNG stream.
      noise = (
          jax.random.normal(self.make_rng("noise"), shape=(self.vocab_size,))
          * self.sigma
      )

    return base_logits + noise
