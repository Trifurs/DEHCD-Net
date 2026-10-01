"""Verify a complete campaign, compare paired seeds, export raw percentage tables."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_multiseed import aggregate
from utils.campaign_layout import summary_path
from utils.protocol import atomic_json
from utils.comparison_protocol import comparability, pair_policy
from utils.experiment_statistics import bootstrap_mean, paired_seed_comparison, holm_adjust


def make_report(campaign, references, metrics, comparators=None, repeats=20000, seed=20260929):
    campaign = Path(campaign)
    plan = json.loads((campaign / 'protocol.json').read_text())
    checked = aggregate(plan, campaign)
    if checked['status'] != 'complete':
        raise ValueError('Statistical reporting requires all planned seeds; do not discard failed runs')
    with summary_path(campaign, plan, 'per_seed.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        groups.setdefault(row['experiment'], {})[int(row['seed'])] = row
    if not set(references) <= groups.keys() or comparators and not set(comparators) <= groups.keys():
        raise ValueError('Unknown reference/comparator experiment')
    tables, comparisons, excluded = [], [], []
    for name, observations in groups.items():
        for metric in metrics:
            if not all(metric in row for row in observations.values()):
                raise ValueError(f'Metric {metric} unavailable for {name}')
            values = [float(row[metric]) for row in observations.values()]
            if not all(0 <= value <= 1 for value in values):
                raise ValueError('Only 0..1 accuracy metrics may be expressed as percentages')
            tables.append({'experiment': name, 'metric': metric, 'n': len(values),
                           'mean_percent': 100 * statistics.mean(values),
                           'sample_std_percentage_points': 100 * statistics.stdev(values) if len(values) > 1 else None,
                           'ci95_percent': [100*x for x in bootstrap_mean(values, repeats=repeats, seed=seed)] if len(values)>1 else None})
    for reference in references:
        ref_jobs = [j for j in plan['jobs'] if j['experiment'] == reference]
        dataset = ref_jobs[0]['dataset']
        names = comparators or sorted({j['experiment'] for j in plan['jobs'] if j['dataset'] == dataset})
        for name in names:
            if name == reference:
                continue
            jobs = [j for j in plan['jobs'] if j['experiment'] == name]
            if jobs[0]['dataset'] != dataset:
                excluded.append({'reference': reference, 'candidate': name, 'reason': 'different datasets'})
                continue
            if not comparators:
                ref_event = ref_jobs[0]['config'].get('experiment', {})
                event = jobs[0]['config'].get('experiment', {})
                if any(ref_event.get(k) != event.get(k) for k in ('test_event', 'validation_event')):
                    excluded.append({'reference': reference, 'candidate': name, 'reason': 'different event folds'})
                    continue
            # Explicit invalid pairs are errors. Automatic selection includes
            # main architectures/direct controls, and records other exclusions.
            try:
                pair_policy(ref_jobs[0]['config'], jobs[0]['config'])
            except ValueError as exc:
                if comparators:
                    raise
                excluded.append({'reference': reference, 'candidate': name, 'reason': str(exc)})
                continue
            contract = comparability(plan, ref_jobs + jobs)
            for metric in metrics:
                result = paired_seed_comparison(
                    {s: float(row[metric]) for s, row in groups[reference].items()},
                    {s: float(row[metric]) for s, row in groups[name].items()}, repeats=repeats, seed=seed)
                comparisons.append({'reference': reference, 'comparator': name, 'dataset': dataset,
                                    'metric': metric, 'comparison_contract': contract, **result})
    # One declared family across all exported comparisons/endpoints, not one test at a time.
    for row, p in zip(comparisons, holm_adjust([r['p_value'] for r in comparisons])):
        row['p_holm'] = p
        row['mean_difference_percentage_points'] = 100 * row['mean_difference']
    return {'unit': 'training seed, fixed train/val/test split', 'tables': tables, 'comparisons': comparisons,
            'excluded_comparisons': excluded,
            'bootstrap': {'method': 'percentile, resample whole seed blocks', 'repeats': repeats, 'seed': seed},
            'multiplicity': 'Holm family comprises every comparison and endpoint in this report; CIs are marginal, unadjusted',
            'interpretation': 'Seed variation does not estimate cross-event or geographic generalization. Small n yields coarse p values and unstable CIs; n=5 exact two-sided p cannot be smaller than 0.0625. No significance labels are generated.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--reference', nargs='+', required=True, help='Predeclared experiment ids, e.g. bright_dehcd_l')
    p.add_argument('--comparators', nargs='+')
    p.add_argument('--metrics', nargs='+', default=['mean_iou', 'foreground_miou', 'oa'])
    p.add_argument('--bootstrap-repeats', type=int, default=20000)
    p.add_argument('--bootstrap-seed', type=int, default=20260929)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report = make_report(a.campaign, a.reference, a.metrics, a.comparators, a.bootstrap_repeats, a.bootstrap_seed)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'statistics.json', report)
    with (out / 'raw_metrics_percent.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['tables'][0])); writer.writeheader(); writer.writerows(report['tables'])
    lines = ['# Raw test metrics across seeds', '', 'All values are percentages; no model-wise min–max normalization.', '',
             '| Experiment | Metric | n | Mean (%) | SD (percentage points) |', '|---|---|---:|---:|---:|']
    for row in report['tables']:
        sd = row['sample_std_percentage_points']
        lines.append(f"| {row['experiment']} | {row['metric']} | {row['n']} | {row['mean_percent']:.3f} | {sd:.3f} |" if sd is not None else
                     f"| {row['experiment']} | {row['metric']} | {row['n']} | {row['mean_percent']:.3f} | NA |")
    (out / 'raw_metrics.md').write_text('\n'.join(lines) + '\n\n' + report['interpretation'] + '\n')
    print(out / 'statistics.json')


if __name__ == '__main__':
    main()
