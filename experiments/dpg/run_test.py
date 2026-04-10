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

"""Tests that this can run a few steps."""

from absl.testing import absltest
from absl.testing import parameterized
from egg.experiments.dpg import run


class RunTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="dg", loss="dg"),
      dict(testcase_name="reinforce", loss="reinforce"),
      dict(testcase_name="ppo", loss="ppo"),
      dict(testcase_name="pmpo", loss="pmpo"),
  )
  def test_run_smoke(self, loss: str):
    """Basic smoke test to ensure run completes a few steps."""
    sweep_config = run.SweepConfig(
        num_steps=3,
        prompt_length=2,
        loss=loss,
        # Use default values from SweepConfig for other parameters
    )

    try:
      df = run.run_experiment(sweep_config)
      self.assertIsNotNone(df)
      self.assertFalse(df.empty)
      self.assertGreaterEqual(len(df), 1)

    except Exception as e:  # pylint: disable=broad-except
      self.fail(
          f"run.run_experiment(config with loss_config={loss}) raised an"
          f" exception: {e}"
      )


if __name__ == "__main__":
  absltest.main()
