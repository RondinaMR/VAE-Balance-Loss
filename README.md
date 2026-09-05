# VAE-Balance-Loss

Reproducibility package for the paper:

> **Balanced Synthetic Data Generation with Variational AutoEncoders: A Trade-Off Analysis of Balance, Fairness, Utility and Privacy**

This repository contains the full source code, datasets, and experiment scripts needed to reproduce all results reported in the paper.

---

## Overview

We propose and evaluate three VAE-based loss functions for tabular synthetic data generation:

| Loss model | Description |
|---|---|
| `vanilla` | Standard VAE loss (baseline) |
| `weight` | Reconstruction loss re-weighted by a Simpson-based exponential term per sensitive feature |
| `term` | An explicit balance penalty term added to the VAE loss, scaled per sensitive feature |

Experiments are run on **3 fairness benchmark datasets** with **30 random seeds** each, measuring the trade-off between balance, fairness (Statistical Parity, Disparate Impact), utility (TSTR accuracy), and privacy (DCR).

---

## Repository structure

```
├── cbx_engine/             # Core VAE engine (JAX/Flax)
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
├── synthetic-datasets/     # Per-run results of the published experiments
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

**2. Install pip dependencies and JAX CUDA backend:**

```shell
bash install-cuda.sh
```

This script:
1. Installs pip dependencies from `requirements.txt`
2. Installs `jax[cuda12_pip]`

> The Cython extension (`gower_matrix_c`) is compiled automatically at first run via `pyximport`.

---

## Datasets

All datasets are publicly available. The `fairness_datasets/` folder already contains the raw files. Original sources:

| # | Dataset | Source |
|---|---|---|
| 01 | Adult | https://archive.ics.uci.edu/dataset/2/adult |
| 02 | South German Credit | https://archive.ics.uci.edu/dataset/522/south+german+credit |
| 03 | COMPAS Two Years | https://github.com/propublica/compas-analysis |

---

## Experiment results

The results of every experiment reported in the paper are included in this
repository, under `synthetic-datasets/`: 17,400 files covering the three
datasets above, one folder per run.

```
synthetic-datasets/<dataset>/<group>/<run>/
├── experiment.json     # parameters and all computed metrics
├── epoch-losses.csv    # per-epoch training losses
├── epoch_losses.pdf    # the same losses, plotted
├── datasets/
│   ├── *_od_TRAIN.csv  # real training split fed to the VAE
│   ├── *_od_TEST.csv   # real holdout split
│   └── *_sd.csv        # synthetic dataset produced by that run
└── balance/            # per-run balance and frequency tables
```

The three experiment groups are the ones listed in `DATASET_RUNS` in
`experiment/statistical_validation_p.py`.

Model checkpoints (`states/*.npy`), the plots in `viz/` and the training logs
are **not** published: they weigh around 37 GB and are not needed to reproduce
any number in the paper. Every statistic reported in the paper is derived from
the `experiment.json` files; the CSVs under `datasets/` are additionally read
by `experiment/fairness.py` when the fairness metrics are recomputed from
scratch.

To regenerate the four LaTeX tables of the paper without re-running any
experiment:

```shell
python experiment/statistical_export_p.py --outdir experiment/statistical_validation
```

This takes a few minutes: the `spd` table re-reads every synthetic and
training CSV. Use `--metrics` and `--datasets` to build a subset, and
`--outdir` to change the destination.


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

After all repetitions complete, `analysis.py` aggregates all `experiment.json` files into a summary DataFrame and generates comparison plots. The copies published in this repository omit `states/`, `viz/` and `logs/` — see [Experiment results](#experiment-results).


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
