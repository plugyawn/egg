# egg 🥚

A minimal, CPU-friendly simulation environment for exploring RL on LLMs — before
they hatch into full-scale distributed training.

<a href="egg_logo.png" target="_blank">
  <img src="egg_logo.png" width="300"/>
</a>

## Goals

-   **Tiny + Transparent**: No big framework, no magic.
-   **Algorithm-first**: Code should resemble paper pseudocode.
-   **Rapid iteration**: Launch and test ideas in seconds.
-   **Insightful**: Concepts here should transfer to larger scale.

## Installation

`egg` can be installed using `pip`. We recommend using a virtual environment:

```bash
python3 -m venv egg_venv
source egg_venv/bin/activate
pip install --upgrade pip
pip install .
```

To run tests, you can install the test dependencies:

```bash
pip install .[test]
pytest
```

## Usage

You can run the baseline experiment with the following command:

```bash
python3 experiments/baseline/run.py --sweep.num_steps=10
```

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```
@misc{osband2026delightfulpolicygradient,
      title={Delightful Policy Gradient},
      author={Ian Osband},
      year={2026},
      eprint={2603.14608},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.14608},
}
@misc{osband2026delightfuldistributedpolicygradient,
      title={Delightful Distributed Policy Gradient},
      author={Ian Osband},
      year={2026},
      eprint={2603.20521},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.20521},
}
@misc{osband2026doesgradientsparkjoy,
      title={Does This Gradient Spark Joy?},
      author={Ian Osband},
      year={2026},
      eprint={2603.20526},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.20526},
}
```

## License and disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
