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

"""A flax linen wrapper to add a fixed logit bias to a network's output."""

from __future__ import annotations

import dataclasses

from egg import base
import flax.linen as nn
import jax


@dataclasses.dataclass
class NetworkConfig(base.MakeableConfig[nn.Module]):
  """Config for the logit-wrapping network."""

  inner_network_config: base.MakeableConfig[nn.Module]  # The network to wrap.
  logit_bias: jax.Array  # Logit bias to add to the network's output.

  def make(self) -> LogitWrapperNetwork:
    """Returns a LogitWrapperNetwork with the given configuration."""
    return LogitWrapperNetwork(
        inner_network=self.inner_network_config.make(),
        logit_bias=self.logit_bias,
    )


class LogitWrapperNetwork(nn.Module):
  """A flax linen wrapper to add a fixed logit bias to a network's output."""

  inner_network: nn.Module
  logit_bias: jax.Array

  @nn.compact
  def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
    """Calls the inner network and adds the logit bias."""
    # Pass input through the wrapped network
    base_logits = self.inner_network(x, *args, **kwargs)

    # Ensure the bias shape is compatible with the logits shape.

    # Add the bias to the logits
    return base_logits + self.logit_bias
