# VAE-Balance-Loss

Reproducibility package for the paper:

> **Balanced Synthetic Data Generation with Variational AutoEncoders: A Trade-Off Analysis of Balance, Fairness, Utility and Privacy**

This repository contains the full source code, datasets, and experiment scripts needed to reproduce all results reported in the paper.

---

## Overview

We propose and evaluate three VAE-based loss functions for tabular synthetic data generation, studied under the lens of the Simpson's paradox effect on sensitive features:

| Loss model | Description |
|---|---|
| `vanilla` | Standard VAE loss (baseline) |
| `weight` | Reconstruction loss re-weighted by a Simpson-aware exponential term per sensitive feature |
| `term` | An explicit balance penalty term added to the VAE loss, scaled per sensitive feature |

Experiments are run on **3 fairness benchmark datasets** with **30 random seeds** each, measuring the trade-off between balance, fairness (Statistical Parity, Disparate Impact), utility (TSTR accuracy), and privacy (DCR).

---

## Repository structure

```
├── clearbox_engine/        # Core VAE engine (JAX/Flax)
│   ├── VAE/                # VAE models: vanilla, weight, term
│   ├── engine/             # TabularEngine wrappers
│   ├── synthesizer/        # LabeledSynthesizer / UnlabeledSynthesizer
│   ├── metrics/            # Privacy (DCR) and distinguishability metrics
│   ├── preprocessor/       # Tabular data preprocessor
│   ├── transformers/       # Categorical, ordinal, datetime transformers
│   └── dataset/            # Dataset abstraction
├── experiment/             # Experiment pipeline
│   ├── experiment.py       # Single experiment runner
│   ├── analysis.py         # Results aggregation and analysis
│   ├── fairness.py         # Fairness metrics (AIF360)
│   ├── source_dataset.py   # Dataset definitions and metadata
│   └── utils.py
├── data_framework/         # Imbalance and data quality utilities
├── fairness_datasets/      # Raw input datasets (see Datasets section)
├── synthetic-datasets/     # Generated outputs (created by usage.py)
├── tests/                  # Unit tests
├── usage.py                # Main experiment entry point
├── environment.yml         # Conda environment definition
├── requirements.txt        # pip dependencies
└── install-cuda.sh         # Build and install script
```

---

## Requirements

- Linux (tested on Ubuntu 22.04)
- CUDA 12 compatible GPU (recommended; CPU-only execution is not tested)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

---

## Installation

**1. Create and activate the conda environment:**

```shell
conda env create -f environment.yml
conda activate <env-name>
```

**2. Build and install the engine (includes Cython extension compilation):**

```shell
bash install-cuda.sh
```

This script:
1. Cleans any previous build artifacts
2. Installs pip dependencies from `requirements.txt`
3. Compiles the Cython extension (`gower_matrix_c`)
4. Builds and installs the `clearbox_engine` wheel
5. Installs `jax[cuda12_pip]`

---

## Datasets

All datasets are publicly available. The `fairness_datasets/` folder already contains the raw files. Original sources:

| # | Dataset | Source |
|---|---|---|
| 01 | Adult | https://archive.ics.uci.edu/dataset/2/adult |
| 02 | South German Credit | https://archive.ics.uci.edu/dataset/522/south+german+credit |
| 03 | COMPAS Two Years | https://github.com/propublica/compas-analysis |

---

## Reproducing the experiments

Open `usage.py` and configure the top of the `__main__` block:

```python
run_experiment = True          # True to run, or set to a folder path to only analyse
datasets = [compas_score_two_years]  # list of dataset objects from source_dataset.py
split_seeds = [0, 42, ...]     # list of 30 seeds used in the paper (already set)
```

Then run:

```shell
python usage.py
```

Results are saved under `synthetic-datasets/<dataset-id>/<timestamp>-<repetitions>-<combinations>/`.

Each individual experiment produces a timestamped subfolder containing:

```
<exp-timestamp>/
├── datasets/          # Synthetic CSV and (preprocessed) original CSV
├── balance/           # Imbalance statistics
├── states/            # Saved VAE model parameters
├── viz/               # Loss curves and latent space plots
├── logs/              # debug.log and info.log
└── experiment.json    # Full experiment parameters and results
```

After all repetitions complete, `analysis.py` aggregates all `experiment.json` files into a summary DataFrame and generates comparison plots.

---

## Running on a specific dataset only

To reproduce results for a single dataset, edit the `datasets` list in `usage.py`:

```python
from experiment.source_dataset import adult
datasets = [adult]
```

Available dataset objects: `adult`, `south_german_credit`, `compas_score_two_years`.

---

## Analysing existing results without re-running

Set `run_experiment` to the path of an existing experiment group folder:

```python
run_experiment = 'synthetic-datasets/01_adult/250202-111013-30-22/'
```

`usage.py` will skip training and directly run `analyse_folder` on the existing outputs.
