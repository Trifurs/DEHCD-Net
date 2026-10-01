"""Create all event-held-out folds from explicit metadata, without moving images."""
import argparse
import copy
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.make_event_split import assign_events
from utils.split_manifest import read_manifest
from utils.config import XMLConfigParser
from utils.protocol import atomic_json, file_digest


def event_folds(rows):
    events = sorted({r['event'] for r in rows})
    if len(events) < 3:
        raise ValueError('Need at least three explicitly identified events for train/val/test holdouts')
    groups = {}
    for row in rows:
        previous = groups.setdefault(row['group'], row['event'])
        if previous != row['event']:
            raise ValueError(f"Source scene {row['group']} belongs to multiple events")
    # Validation selection is deterministic and fixed before seeing predictions.
    return [(test, events[(i + 1) % len(events)],
             assign_events(rows, {test}, {events[(i + 1) % len(events)]})) for i, test in enumerate(events)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', required=True, help='CSV: id,source_split,split,group,event; no guessed geography')
    p.add_argument('--configs', nargs='+', required=True, help='Base models to compare on EVERY fold')
    p.add_argument('--dataset', required=True, help='Dataset key used by run_multiseed --datasets/--data-root')
    p.add_argument('--output', required=True, help='New output directory, refuses overwrite')
    a = p.parse_args()
    folds = event_folds(read_manifest(a.metadata))
    configs = [XMLConfigParser(path).parse().as_dict() for path in a.configs]
    names = [c.get('experiment', {}).get('id', Path(path).stem) for c,path in zip(configs, a.configs)]
    if len(names) != len(set(names)):
        raise ValueError('Base experiment names must be unique')
    for cfg in configs:
        reference = cfg.get('experiment', {}).get('comparison', {}).get('reference')
        if reference and reference not in names:
            raise ValueError(f'Include declared reference {reference} in --configs for each controlled fold')
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    catalog, evidence = [], []
    for index, (test, val, rows) in enumerate(folds):
        token = re.sub(r'[^a-zA-Z0-9_-]', '_', test)[:40] or 'event'
        token += '_' + hashlib.sha256(test.encode()).hexdigest()[:8]
        manifest = out / f'fold_{index:02d}_{token}.csv'
        with manifest.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id','source_split','split','group','event'])
            writer.writeheader();writer.writerows(rows)
        read_manifest(manifest)
        evidence.append({'fold': index, 'test_event': test, 'validation_event': val, 'manifest': str(manifest)})
        for name, config in zip(names, configs):
            cfg = copy.deepcopy(config); identifier = f'{name}__event_{index:02d}_{token}'
            cfg.setdefault('dataset', {})['manifest'] = str(manifest)
            cfg.setdefault('training', {}).update(early_stop_patience=0, deterministic=True)
            cfg.setdefault('inference', {})['test_time_augmentation'] = 'none'
            cfg.setdefault('experiment', {}).update(id=identifier, suite='cross_event', dataset=a.dataset,
                test_event=test, validation_event=val, split_protocol='leave_one_event_out_rotating_validation')
            comparison = cfg['experiment'].get('comparison', {})
            reference = comparison.get('reference')
            if reference:
                comparison['reference'] = f'{reference}__event_{index:02d}_{token}'
            cfg.setdefault('logging', {})['run_name'] = identifier
            path = out / 'configs' / f'{identifier}.json'; atomic_json(path, cfg)
            catalog.append({'id': identifier, 'config': str(path), 'dataset': a.dataset, 'suite': 'cross_event'})
    atomic_json(out / 'catalog.json', {'schema_version': 1, 'experiments': catalog})
    atomic_json(out / 'folds.json', {'metadata_sha256': file_digest(a.metadata), 'folds': evidence,
        'validation_rule': 'next event in sorted event-name order, fixed before any training',
        'unit': 'event; source scene groups remain intact; events are not inferred from tile names'})
    print(f'Generated {len(folds)} event holdouts x {len(configs)} models. No training started: {out / "catalog.json"}')


if __name__ == '__main__':
    main()
