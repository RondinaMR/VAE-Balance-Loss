"""The four LaTeX tables published in the paper, and nothing else.

This is the reduced counterpart of :mod:`experiment.statistical_export_all`: it holds only the output code the paper tables need.
It computes nothing on its own: the numbers come from :mod:`experiment.statistical_validation_p`. Everything here is presentation: table layout, number formatting, prose and labels.

Everything the tables do not need lives in :mod:`experiment.statistical_export_all`: the CSV artefacts, the per-family markdown reports, the cross-metric summary and the Friedman report.
That module remains the one to run to produce the CSV files this one can read back with ``--from-csv``.

Files written under the output folder:

* ``statistical_latex.tex`` -- the four per-question tables.

Usage
-----
    python statistical_export_p.py
    python statistical_export_p.py --from-csv        # seconds instead of minutes, from the CSV files already on disk
    python statistical_export_p.py --metrics balance --outdir /tmp/out
"""

import argparse
import os
from typing import Dict, List

import numpy as np

try:  # Imported as part of the package: python -m experiment.statistical_export_p.
    from experiment import statistical_validation_p as sv
except ImportError:  # Run directly as a script: python statistical_export_p.py.
    import statistical_validation_p as sv

# Display names of the datasets, in the order the tables use. Datasets missing here are appended in the order they are read, under their raw name.
DATASET_LABELS: Dict[str, str] = {
    'adult': 'Adult',
    'south_german_credit': 'South German Credit',
    'compas': 'Compas',
}

# Presentation of each metric family, keyed exactly like ``statistical_validation_p.METRICS``.
# ``rq`` is the research question the family answers, which is what the tables are segmented by; balance and spd both answer RQ1.
# ``decimals`` is how many decimal places the mean and standard deviation are printed with, chosen per family because the four metrics live on very different scales.
# ``scale_note`` warns that the metric does not share one scale across datasets, so that the means are not read across columns.
# ``magnitude_note`` explains, where the seed count is not a plain difference against vanilla, what the mean and standard deviation are computed on.
# Both notes are plain sentences with no markup, because they are appended verbatim to the caption.
PRESENTATION: Dict[str, dict] = {
    'balance': {
        'rq': 'RQ1',
        'decimals': 4,
        'scale_note': 'Balance is a sum over the sensitive features of the dataset, so its scale is the number of those features: 3 for Adult and Compas, 6 for South German Credit. The means below are therefore comparable within a column but not across columns.',
        'magnitude_note': None,
    },
    'spd': {
        'rq': 'RQ1',
        'decimals': 4,
        'scale_note': None,
        'magnitude_note': None,
    },
    'tstr': {
        'rq': 'RQ2',
        'decimals': 4,
        'scale_note': None,
        'magnitude_note': None,
    },
    'privacy': {
        'rq': 'RQ3',
        'decimals': 2,
        'scale_note': None,
        'magnitude_note': 'The mean and standard deviation are the paired difference of the excess over 50, which is the same quantity the percentage counts, so a negative mean means the modification carries less leakage than vanilla. A seed where neither the modification nor vanilla goes above 50 contributes exactly zero, so a dataset that never leaks shows a column of zeros: that is the metric saying there was nothing to leak, not a missing value.',
    },
}

if set(PRESENTATION) != set(sv.METRICS):
    raise ImportError(f'PRESENTATION and statistical_validation_p.METRICS describe different families: '
                      f'{sorted(set(PRESENTATION) ^ set(sv.METRICS))}')


def dataset_order(datasets) -> List[str]:
    """Return the datasets in the order used by the tables, appending the unknown ones at the end."""
    known = [d for d in DATASET_LABELS if d in datasets]
    return known + [d for d in datasets if d not in DATASET_LABELS]


def label_of(dataset: str) -> str:
    """Display name of a dataset."""
    return DATASET_LABELS.get(dataset, dataset)


# Escaped because they carry a meaning in LaTeX. The notes reused in the captions contain none of them, but a future edit might.
LATEX_ESCAPES = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_'}


def latex_escape(text: str) -> str:
    """Escape the characters LaTeX would otherwise read as commands."""
    return ''.join(LATEX_ESCAPES.get(character, character) for character in text)


# One letter per loss model, so the first column stays narrow: the full names are in the caption.
LATEX_MODEL_SYMBOLS = {'weight': 'w', 'term': 't'}


def latex_row_label(config: str, epochs: int) -> str:
    """Render the row label of a configuration at a given number of epochs, as ``$w_{0.1}$/e50``."""
    model, _, magnitude = config.partition('_')
    symbol = LATEX_MODEL_SYMBOLS.get(model, model[:1])
    return f'${symbol}_{{{magnitude}}}$/e{epochs}'


def latex_magnitude(cell, decimals: int) -> str:
    """Render one cell as its percentage above the mean and standard deviation, with the numbers in math mode.

    The two halves are stacked with ``\\makecell`` so that the column stays narrow: the percentage is what the eye scans down a column, the effect size is the detail read once the row matters.
    """
    if cell is None:
        return '--'
    diffs = np.asarray(cell['diffs'], dtype=float)
    percentage = f'{100 * cell["count"] / cell["total"]:.1f}\\%'
    effect = f'(${diffs.mean():+.{decimals}f} \\pm {diffs.std(ddof=1):.{decimals}f}$)'
    return f'\\makecell{{{percentage} \\\\ {effect}}}'


def baseline_values(cells: List[dict], dataset: str, epochs: int) -> np.ndarray:
    """Pool the raw baseline values of one (dataset, epochs), over the groups the family splits into.

    For spd that means the three or two sensitive attributes of the dataset, the same pooling the modification rows use.
    """
    return np.concatenate([cell['table'][sv.BASELINE].dropna().to_numpy()
                           for cell in cells if cell['dataset'] == dataset and cell['epochs'] == epochs])


def latex_baseline(values: np.ndarray, decimals: int) -> str:
    """Render the absolute level of the baseline, with no percentage: it is the reference, not a comparison against one."""
    return f'${values.mean():+.{decimals}f} \\pm {values.std(ddof=1):.{decimals}f}$'


def latex_caption(family: str, count_name: str) -> str:
    """Build the caption of a per-question table, including the note the table cannot be read correctly without."""
    held = ('improves on vanilla' if count_name == 'improved'
            else 'does not do worse than vanilla')
    caption = (f'{sv.METRICS[family]["dir"].capitalize()} ({PRESENTATION[family]["rq"]}). '
               f'Percentage of the split seeds where the modification {held} on the same split, '
               f'with the mean and standard deviation of the paired differences in parentheses, in the units of the metric. '
               f'A configuration with no systematic effect splits the seeds evenly, so 50\\% means no difference. '
               f'The reference row is the unmodified baseline, $w$ is the reweighted reconstruction loss, $t$ is the added balance term, and the subscript is the magnitude.')
    for extra in ('scale_note', 'magnitude_note'):
        if PRESENTATION[family][extra]:
            caption += ' ' + latex_escape(PRESENTATION[family][extra])
    return caption


def write_latex_tables(result: dict, datasets: List[str], epochs_settings: List[int], outdir: str) -> str:
    """Write the four per-question tables into a single .tex file and return its path.

    One table per metric family, rows are the six loss modifications with the 50 epochs block first and a rule between the two blocks, columns are the datasets.
    The file declares no preamble: it expects ``\\usepackage{booktabs}``, ``\\usepackage{makecell}`` and ``\\usepackage{arydshln}`` in the document that includes it.
    """
    counted_seeds = result['counted_seeds']
    families = [f for f in sv.METRICS if f in counted_seeds]
    lines = ['% Percentage of seeds and size of the effect, by research question.',
             '% Generated by experiment/statistical_export.py, same numbers as the last section of stats_summary.md.',
             '% Requires \\usepackage{booktabs}, \\usepackage{makecell} and \\usepackage{arydshln} in the including document.',
             '% arydshln must be loaded after array; if it clashes with booktabs, drop the \\hdashline commands and keep the plain \\hline.', '']
    for family in families:
        decimals = PRESENTATION[family]['decimals']
        columns = 'p{1.5cm}' + 'c' * len(datasets)
        lines += [r'\begin{table}[t]', r'\centering', r'\small',
                  f'\\caption{{{latex_caption(family, result["count_names"][family])}}}',
                  f'\\label{{tab:stats-{sv.METRICS[family]["dir"]}}}',
                  f'\\begin{{tabular}}{{{columns}}}', r'\toprule',
                  'Configuration & ' + ' & '.join(latex_escape(label_of(d)) for d in datasets) + r' \\',
                  r'\midrule']
        for position, epochs in enumerate(epochs_settings):
            if position:
                lines.append(r'\midrule')
            block = [f'Ref. e{epochs} & ' + ' & '.join(
                latex_baseline(baseline_values(result['cells'][family], dataset, epochs), decimals)
                for dataset in datasets)]
            for config in sv.MODIFICATIONS:
                cells = [latex_magnitude(counted_seeds[family].get((epochs, config), {}).get(dataset), decimals)
                         for dataset in datasets]
                block.append(latex_row_label(config, epochs) + ' & ' + ' & '.join(cells))
            # A solid rule under the reference row, because it is an absolute value while the rows below it are
            # differences, and a dashed one between those, which are the same quantity for different configurations.
            # The last row of a block needs neither: a \midrule or a \bottomrule already sits under it.
            lines.append(block[0] + r' \\ \hline')
            lines += [row + r' \\ \hdashline' for row in block[1:-1]] + [block[-1] + r' \\']
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'statistical_latex.tex')
    with open(path, 'w') as file:
        file.write('\n'.join(lines))
    print(f'{path}: {len(families)} tables written')
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Build the four LaTeX tables published in the paper.')
    parser.add_argument('--datasets', nargs='+', choices=sorted(sv.DATASET_RUNS),
                        default=sorted(sv.DATASET_RUNS),
                        help='Datasets to process (default: all).')
    parser.add_argument('--metrics', nargs='+', choices=sorted(sv.METRICS),
                        default=sorted(sv.METRICS),
                        help='Metric families to process, one table each (default: all). Computing spd is much slower because it re-reads every synthetic and training CSV.')
    parser.add_argument('--epochs', nargs='+', type=int, default=sv.EPOCHS_SETTINGS,
                        help='Epoch settings to process (default: 50 1250).')
    parser.add_argument('--outdir', default=sv.DEFAULT_OUTDIR,
                        help='Output folder of statistical_latex.tex (default: statistical_validation/).')
    parser.add_argument('--run', nargs=2, action='append', metavar=('DATASET', 'FOLDER'),
                        default=[],
                        help='Override the result folder of a dataset.')
    parser.add_argument('--from-csv', action='store_true',
                        help='Read the wide CSV files already in the output folder instead of the experiments. Seconds instead of minutes; it covers only the metric families whose files are already there. Those files are written by statistical_export_all.py.')
    args = parser.parse_args(argv)

    runs = dict(sv.DATASET_RUNS)
    runs.update({dataset: folder for dataset, folder in args.run})

    if args.from_csv:
        all_records = sv.records_from_csv(args.outdir, args.datasets, args.metrics)
    else:
        all_records = {dataset: sv.read_records(runs[dataset], args.metrics) for dataset in args.datasets}
    result = sv.compute_differences(all_records, args.epochs)
    write_latex_tables(result, dataset_order(all_records), args.epochs, args.outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
