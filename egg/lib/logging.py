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

"""Logging utilities."""

import collections
import dataclasses
import hashlib
import json
import typing as tp

import jax
import numpy as np
import pandas as pd

# We imagine we will have a config that is a datclass.
_ConfigClass: tp.TypeAlias = tp.Any


def add_config_to_df(df: pd.DataFrame, config: _ConfigClass) -> pd.DataFrame:
  """Adds a config to a dataframe."""
  if not dataclasses.is_dataclass(config):
    raise ValueError(f"Config {type(config)} must be a dataclass.")
  config_dict = dataclasses.asdict(config)
  scalar_config_dict: dict[str, tp.Any] = {}
  for k, v in config_dict.items():
    if isinstance(v, (list, tuple, dict)):
      try:
        scalar_config_dict[k] = json.dumps(v, sort_keys=True)
      except TypeError:
        scalar_config_dict[k] = str(v)  # Fallback to string
    elif isinstance(v, (int, float, str, bool, type(None))):
      scalar_config_dict[k] = v
    else:
      # Handle functions or other non-serializable types
      scalar_config_dict[k] = str(v)
  return df.assign(**scalar_config_dict, hash=_hash_dict_config(config_dict))


def _default_json_serializer(obj):
  """Helper for JSON dumping, converts non-serializable to str."""
  try:
    return json.dumps(obj)
  except TypeError:
    return str(obj)


def _hash_dict_config(config_dict: dict[str, tp.Any]) -> str:
  # Use the default serializer to handle non-serializable types in hash
  config_json = json.dumps(
      config_dict, sort_keys=True, default=_default_json_serializer
  )
  return hashlib.sha1(config_json.encode()).hexdigest()[:8]


def logarithmic_logging(t: int) -> bool:
  """Returns `True` only at specific ratios of 10**exponent."""
  ratios = (1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)  # pylint: disable=line-too-long
  if t <= 0:
    return False
  exponent = int(np.log10(t))
  power_of_10 = 10**exponent
  return t in (int(ratio * power_of_10) for ratio in ratios)


class RunningMeanLogger:
  """Computes running means of scalar metrics in a numerically stable way."""

  def __init__(self):
    self._means = collections.defaultdict(float)
    self._counts = collections.defaultdict(int)

  def record(self, metrics: dict[str, float | int | jax.Array]) -> None:
    """Updates the running means with new metric values."""
    for key, value in metrics.items():
      val = float(value)
      self._counts[key] += 1
      count = self._counts[key]
      old_mean = self._means[key]
      # Welford's online algorithm for mean
      delta = val - old_mean
      new_mean = old_mean + delta / count
      self._means[key] = float(new_mean)

  def write(self) -> dict[str, float]:
    """Returns the current means and resets the accumulators."""
    means = dict(self._means)
    self._means.clear()
    self._counts.clear()
    return means
