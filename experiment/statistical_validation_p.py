"""Numeric side of the four tables published in the paper.

This is the reduced counterpart of :mod:`experiment.statistical_validation_all`: it keeps only what the LaTeX tables of the paper are computed from, and nothing else.
It reads the ``experiment.json`` files, extracts the four metrics, and reduces every metric family to the two quantities the tables show: the raw baseline values and, per (epochs, configuration, dataset), the seed count and the paired differences it pools.
It computes and returns numbers and writes nothing: saving them is the job of :mod:`experiment.statistical_export_p`, which is also the command to run.

Everything the tables do not need lives in :mod:`experiment.statistical_validation_all`: the bootstrap intervals, the effect sizes, the medians, the minima and maxima, the per-seed difference frames, the counts against an absolute reference and the relative columns.

Four metric families are supported:

* ``balance`` -- the metric of ``boxplot_e{50,1250}_imbalance_loss_final.pdf``, one value per experiment.
* ``spd`` -- the statistical parity improvement of ``boxplot_e{50,1250}_statistical_parity_diff_improvement_<attribute>.pdf``, one value per experiment and per sensitive attribute.
* ``tstr`` -- the utility metric of ``boxplot_e{50,1250}_tstr_accuracy_diff.pdf``, one value per experiment.
* ``privacy`` -- the privacy metric of ``boxplot_e{50,1250}_synth_holdout_test.pdf``, one value per experiment.

The tables it builds are paired: for a given split seed, all configurations come from the same train/validation split.

Sign conventions differ between the families and are handled explicitly, see ``METRICS``.

All default paths are resolved against the repository root, so the module works from any working directory.
"""

import contextlib
import glob
import io
import json
import os
import re
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

try:  # Imported as part of the package: python -m experiment.statistical_validation_p.
    from experiment.utils import imbalance_statistics
    from experiment.fairness import binary_fmetrics_improvement
except ImportError:  # Run directly as a script: python statistical_validation_p.py.
    from utils import imbalance_statistics
    from fairness import binary_fmetrics_improvement

# Repository root, one level above this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Result folders used for the plots in the paper.
DATASET_RUNS: Dict[str, str] = {name: os.path.join(ROOT, path) for name, path in {
    'adult': 'synthetic-datasets/01_adult/250202-111013-30-22/',
    'south_german_credit': 'synthetic-datasets/03_south-german-credit/250202-111210-30-22/',
    'compas': 'synthetic-datasets/04_compas-two-years/260304-105454-30-14/',
}.items()}

# Default parent folder for every file produced, one subfolder per metric family.
DEFAULT_OUTDIR = os.path.join(ROOT, 'statistical_validation')

BASELINE = 'vanilla'

# Same configurations shown in the boxplots (experiment/plotting.py:146-147).
# The Adult and South German Credit runs also contain the weights 0.01 and 0.05, which do not appear in the paper plots and are therefore excluded.
CONFIGS: List[str] = [
    BASELINE,
    'weight_0.1', 'weight_0.5', 'weight_1.0',
    'term_0.1', 'term_0.5', 'term_1.0',
]

# The configurations compared against the baseline, that is every one except the baseline itself.
MODIFICATIONS: List[str] = [c for c in CONFIGS if c != BASELINE]

EPOCHS_SETTINGS: List[int] = [50, 1250]

# Metric families, computation parameters only. Their wording and layout live in statistical_export_p.PRESENTATION, keyed the same way.
# ``dir`` is the output subfolder of the family, and the name the table is labelled and captioned with.
# ``better`` is the sign of (modification - vanilla) that indicates the modification is better than the baseline.
# ``count_delta`` is the threshold of the seed count: None counts a strict improvement over vanilla, 0 counts every seed that is not worse.
# ``excess`` makes degradation one-sided around a reference value: only the part of the metric beyond it counts as a degradation, so moving further onto the safe side is neither counted nor measured as worse.
METRICS: Dict[str, dict] = {
    'balance': {'dir': 'balance', 'better': -1, 'count_delta': None, 'excess': None},
    'spd': {'dir': 'fairness', 'better': +1, 'count_delta': None, 'excess': None},
    'tstr': {'dir': 'utility', 'better': +1, 'count_delta': 0.0, 'excess': None},
    'privacy': {'dir': 'privacy', 'better': -1, 'count_delta': None, 'excess': 50.0},
}

# Names of the wide CSV files, for example spd_native_country_adult_e50. The layout is described here because it is the format of the data, and the fast path reads it back.
FILE_PATTERN = re.compile(r'^(?P<group>.+)_(?P<dataset>' + '|'.join(DATASET_RUNS) + r')_e(?P<epochs>\d+)$')


def balance(experiment: dict) -> float:
    """Balance metric of the synthetic dataset of a single experiment.

    Sum, over all sensitive features, of ``1 - Simpson index`` computed on the synthetic dataset.
    Lower values indicate a more balanced dataset.

    This is the same call used for the boxplots (experiment/plotting.py:171), so the exported values match exactly the ones shown in the plots.
    """
    return imbalance_statistics(
        experiment['results']['synthetic_imbalance'],
        stat='sensitiveloss',
        sensitive_features=experiment['parameters']['sensitive_features'],
    )


def statistical_parity(experiment: dict, attribute: str) -> float:
    """Statistical parity improvement of a single experiment for one sensitive attribute.

    Returns ``-(|SPD_synth| - |SPD_train|)``, so positive values mean the synthetic dataset is fairer than the training split it was generated from.
    Note that this is the opposite sign convention of :func:`balance`.

    This is the same call used for the boxplots (experiment/plotting.py:176), so the exported values match exactly the ones shown in the plots.
    The dataset paths stored in experiment.json are relative to the repository root and are resolved here, so the script does not depend on the working directory.
    """
    experiment = {**experiment, 'path': {k: os.path.join(ROOT, v) for k, v in experiment['path'].items()}}
    mapping = {attribute: experiment['parameters']['fair_column_mappings'][attribute]}
    return binary_fmetrics_improvement(experiment, fair_column_mapping=mapping)['statistical_parity_difference']


def tstr_accuracy(experiment: dict) -> float:
    """TSTR accuracy difference of a single experiment.

    Returns ``accuracy(synthetic) - accuracy(training)``, the accuracy of a classifier trained on the synthetic dataset minus the accuracy of the same classifier trained on the real training split, both evaluated on the real test set.
    Higher values mean less utility is lost, so this uses the opposite sign convention of :func:`balance`.

    This is the same computation used for the boxplots (experiment/plotting.py:169), so the exported values match exactly the ones shown in the plots.
    """
    accuracy = experiment['results']['tstr']['accuracy']
    return accuracy['synthetic'] - accuracy['training']


def privacy(experiment: dict) -> float:
    """Privacy metric of a single experiment.

    Returns the percentage of synthetic rows that are closer to the training set than to the holdout set (clearbox_engine/metrics/privacy/privacy.py:650).
    A value of 50 means the synthetic data is equally close to both, so nothing of the training set has been memorised; values above 50 indicate leakage.

    This is the same value used for the boxplots (experiment/plotting.py:168), so the exported values match exactly the ones shown in the plots.
    """
    return experiment['results']['synthetic_holdout_metrics']['synth_holdout_test']


def config_label(loss_model: str, simpson_weight) -> str:
    """Reproduce the label used in the boxplots (experiment/plotting.py:188-191)."""
    if loss_model == BASELINE:
        return BASELINE
    return f'{loss_model}_{simpson_weight}'


def read_records(folder: str, metrics: List[str]) -> pd.DataFrame:
    """Read the experiment.json files of a folder in long format.

    ``metrics`` selects which metric families to compute; ``spd`` reads the synthetic and training CSV files of every experiment and is therefore much slower than ``balance``.
    Warnings printed by the fairness code are collected and reported once instead of being repeated for every experiment.

    Returns
    -------
    DataFrame with columns: epochs, group, config, seed, value.
    ``group`` is ``balance`` for the balance metric and ``spd_<attribute>`` for the statistical parity improvement of each sensitive attribute.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'Result folder not found: {folder}')

    records = []
    noise = io.StringIO()
    entries = sorted(os.listdir(folder))
    for entry in tqdm(entries, desc=f'Reading {os.path.basename(folder.rstrip("/"))}', leave=False):
        json_path = os.path.join(folder, entry, 'experiment.json')
        if not os.path.isfile(json_path):
            continue
        with open(json_path) as file:
            experiment = json.load(file)
        parameters = experiment['parameters']
        label = config_label(parameters['loss_model'], parameters.get('simpson_weight'))
        if label not in CONFIGS:
            continue
        values = {}
        if 'balance' in metrics:
            values['balance'] = balance(experiment)
        if 'tstr' in metrics:
            values['tstr'] = tstr_accuracy(experiment)
        if 'privacy' in metrics:
            values['privacy'] = privacy(experiment)
        if 'spd' in metrics:
            with contextlib.redirect_stdout(noise):
                for attribute in parameters['fair_column_mappings']:
                    values[f'spd_{attribute}'] = statistical_parity(experiment, attribute)
        for group, value in values.items():
            records.append({'epochs': parameters['epochs'], 'group': group, 'config': label,
                            'seed': parameters['split_seed'], 'value': value})

    for warning in sorted(set(noise.getvalue().splitlines())):
        print(f'  [fairness] {warning}')
    if not records:
        raise ValueError(f'No usable experiment.json found in {folder}')
    return pd.DataFrame.from_records(records)


def paired_table(records: pd.DataFrame, group: str, epochs: int) -> pd.DataFrame:
    """Paired seed x configuration table for one metric group and one number of epochs.

    Returns
    -------
    DataFrame with one row per split seed and the columns ``seed, vanilla, weight_0.1, weight_0.5, weight_1.0, term_0.1, term_0.5, term_1.0``.
    """
    subset = records[(records['group'] == group) & (records['epochs'] == epochs)]
    if subset.empty:
        raise ValueError(f'No experiment with group={group} and epochs={epochs}')

    duplicated = subset.duplicated(subset=['config', 'seed'])
    if duplicated.any():
        raise ValueError(f'Unbalanced design: {int(duplicated.sum())} duplicated (config, seed) combinations for group={group} and epochs={epochs}')

    table = subset.pivot(index='seed', columns='config', values='value')
    missing = [c for c in CONFIGS if c not in table.columns]
    if missing:
        raise ValueError(f'Missing configurations for group={group} and epochs={epochs}: {missing}')

    table = table[CONFIGS].sort_index().reset_index()
    table.columns.name = None
    return table


def metric_family(group: str) -> str:
    """Return the metric family a group belongs to, e.g. ``spd`` for ``spd_sex``."""
    return group.split('_')[0]


def wide_files(outdir: str):
    """Yield (group, dataset, epochs, path) for every wide CSV under ``outdir``, skipping the difference files."""
    for family in METRICS.values():
        for path in sorted(glob.glob(os.path.join(outdir, family['dir'], '*.csv'))):
            name = os.path.basename(path)[:-len('.csv')]
            if '_diff_' in name:
                continue
            match = FILE_PATTERN.match(name)
            if match:
                yield match.group('group'), match.group('dataset'), int(match.group('epochs')), path


def records_from_csv(outdir: str, datasets: List[str] = None, metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Rebuild the long records of every dataset from the wide CSV files already on disk.

    This is the fast path. It returns the same frames :func:`read_records` would have returned, so :func:`compute_differences` runs on them unchanged, but it skips reading the experiments, which is what costs minutes: the fairness metric reopens the synthetic and training CSV of every experiment.
    It can therefore only see the metric families whose files are already there.
    """
    collected: Dict[str, list] = {}
    for group, dataset, epochs, path in wide_files(outdir):
        if datasets is not None and dataset not in datasets:
            continue
        if metrics is not None and metric_family(group) not in metrics:
            continue
        # ``round_trip`` selects the correctly rounded parser: the default one loses the last digit, and on tstr, whose
        # values are already rounded to 4 decimals, that is enough to move a rounding.
        long = pd.read_csv(path, float_precision='round_trip').melt(id_vars='seed', var_name='config', value_name='value')
        long['group'], long['epochs'] = group, epochs
        collected.setdefault(dataset, []).append(long)
    if not collected:
        raise FileNotFoundError(f'No wide CSV found under {outdir}: run without --from-csv first')
    return {dataset: pd.concat(collected[dataset], ignore_index=True)[['epochs', 'group', 'config', 'seed', 'value']]
            for dataset in sorted(collected)}


def compute_differences(all_records: Dict[str, pd.DataFrame], epochs_settings: List[int]) -> dict:
    """Reduce every metric family to the two quantities the published tables are built from.

    The quantity the paired differences are measured on is the same one the seed count uses, so that the two never describe different scales.
    For the families that declare an ``excess`` reference the quantity is the paired difference of the excess beyond it; for the others it is the plain paired difference.
    Nothing is written to disk.

    Returns
    -------
    dict with the keys:

    * ``cells`` -- family -> list of one entry per (group, dataset, epochs), each holding the wide table the reference row of the table is averaged from.
    * ``counted_seeds`` -- family -> (epochs, configuration) -> dataset -> {count, total, diffs}, pooled over the groups of the dataset. ``diffs`` drops the missing values, so its length can be shorter than ``total``, which keeps counting every seed the design declares.
    * ``count_names`` -- family -> ``improved`` where the research question asks for one, ``not_worse`` where it asks for the absence of a degradation. It selects the wording of the caption.
    """
    cells: Dict[str, List[dict]] = {}
    counted_seeds: Dict[str, Dict[tuple, Dict[str, dict]]] = {}
    count_names: Dict[str, str] = {}
    for dataset, records in all_records.items():
        for group in sorted(records['group'].unique()):
            family = metric_family(group)
            better, excess, count_delta = (METRICS[family][k] for k in ('better', 'excess', 'count_delta'))
            for epochs in epochs_settings:
                table = paired_table(records, group, epochs)
                d = table[MODIFICATIONS].sub(table[BASELINE], axis=0)
                if excess is not None:
                    # Degradation is one-sided around the reference: only the part of the metric beyond it is a degradation, so a seed that moves further onto the safe side is neither counted nor measured as worse.
                    risk = (better * (excess - table[CONFIGS])).clip(lower=0)
                    magnitude = risk[MODIFICATIONS].sub(risk[BASELINE], axis=0)
                    counted, count_name = (magnitude <= 0).sum(), 'not_worse'
                elif count_delta is not None:
                    magnitude = d
                    counted, count_name = (d * better >= -count_delta).sum(), 'not_worse'
                else:
                    magnitude = d
                    counted, count_name = (d * better > 0).sum(), 'improved'
                cells.setdefault(family, []).append(
                    {'group': group, 'dataset': dataset, 'epochs': epochs, 'table': table})
                count_names[family] = count_name
                for config in MODIFICATIONS:
                    cell = counted_seeds.setdefault(family, {}).setdefault((epochs, config), {}).setdefault(
                        dataset, {'count': 0, 'total': 0, 'diffs': []})
                    cell['count'] += int(counted[config])
                    cell['total'] += len(d)
                    cell['diffs'] += magnitude[config].dropna().tolist()
    return {'cells': cells, 'counted_seeds': counted_seeds, 'count_names': count_names}
