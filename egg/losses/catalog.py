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

"""Catalog of available loss functions."""

from __future__ import annotations

import enum

from egg import base
from egg.losses import dg
from egg.losses import kondo
from egg.losses import online_star
from egg.losses import pmpo
from egg.losses import ppo
from egg.losses import reinforce
from egg.losses import xent


_REGISTRY: dict[str, type[base.MakeableConfig[base.LossFn]]] = {
    "dg": dg.LossConfig,
    "kondo": kondo.LossConfig,
    "online_star": online_star.LossConfig,
    "pmpo": pmpo.LossConfig,
    "ppo": ppo.LossConfig,
    "reinforce": reinforce.LossConfig,
    "xent": xent.LossConfig,
}


class Loss(enum.StrEnum):
  """Enum for losses. Serializable as plain strings."""

  DG = "dg"
  KONDO = "kondo"
  ONLINE_STAR = "online_star"
  PMPO = "pmpo"
  PPO = "ppo"
  REINFORCE = "reinforce"
  XENT = "xent"

  def config(self, **kwargs) -> base.MakeableConfig[base.LossFn]:
    """Returns a serializable config for this loss."""
    return _REGISTRY[self.value](**kwargs)  # pytype: disable=not-callable

  def make(self, **kwargs) -> base.LossFn:
    """Returns an instantiated loss function."""
    return self.config(**kwargs).make()
