#!/usr/bin/env python3
"""Repartition paired patches by event and label composition, preserving counts.

The default uses complete source scenes with shared events. The optional patch
mode allows adjacent patches of a scene in different splits. Neither mode is
an event-held-out benchmark. No patch is duplicated between splits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.protocol import atomic_json, digest

SPLITS = ('train', 'val', 'test')
SUFFIXES = {'pre-event': '_pre_disaster.tif', 'post-event': '_post_disaster.tif',
            'target': '_building_damage.tif'}
ID_PATTERN = re.compile(r'^(?P<event>.+)_(?P<scene>\d{8})_r\d+_c\d+$')


def inventory(root):
    rows = []
    seen = set()
    for split in SPLITS:
        maps = {}
        for modality, suffix in SUFFIXES.items():
            folder = root / split / modality
            maps[modality] = {p.name[:-len(suffix)]: p for p in folder.glob('*' + suffix)}
        ids = set(maps['target'])
        if any(set(files) != ids for files in maps.values()):
            raise ValueError(f'Incomplete paired sample sets: {split}')
        for sample_id in sorted(ids):
            if sample_id in seen:
                raise ValueError(f'Duplicate sample ID: {sample_id}')
            seen.add(sample_id)
            match = ID_PATTERN.fullmatch(sample_id)
            if not match:
                raise ValueError(f'Cannot identify event/scene: {sample_id}')
            files = {}
            for modality, mapping in maps.items():
                path = mapping[sample_id]
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f'Expected a regular source file: {path}')
                stat = path.stat()
                files[modality] = {'name': path.name, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            rows.append({'id': sample_id, 'source_split': split, 'event': match['event'],
                         'group': match['event'] + '_' + match['scene'], 'files': files})
    if any(not any(row['source_split'] == split for row in rows) for split in SPLITS):
        raise ValueError('All three source splits must be nonempty')
    return rows


def label_counts(root, rows, cache_root, workers):
    counts = np.zeros((len(rows), 4), dtype=np.int64)
    by_id = {row['id']: index for index, row in enumerate(rows)}
    filled = set()
    for path in sorted(cache_root.glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        samples = manifest.get('samples', [])
        records = manifest.get('identities', {}).get('stat', {}).get('records', [])
        if len(samples) != len(records):
            continue
        valid = []
        for i, (sample, record) in enumerate(zip(samples, records)):
            j = by_id.get(sample['id'])
            if j is None or j in filled:
                continue
            row = rows[j]
            expected = [root / row['source_split'] / modality / row['files'][modality]['name']
                        for modality in SUFFIXES]
            if any(str(p) != f['path'] or p.stat().st_size != f['bytes'] or p.stat().st_mtime_ns != f['mtime_ns']
                   for p, f in zip(expected, record['files'])):
                continue
            valid.append((i, j))
        if not valid:
            continue
        info = manifest['files']['label.bin']
        label_file = path.parent / 'label.bin'
        if label_file.stat().st_size != info['bytes'] or label_file.stat().st_mtime_ns != info['mtime_ns']:
            continue
        stat_path = path.parent / 'label_counts.npy'
        if stat_path.exists():
            values = np.load(stat_path, allow_pickle=False)
            for i, j in valid:
                counts[j] = values[i]
                filled.add(j)
        else:
            labels = np.memmap(label_file, mode='r', dtype=manifest['tensors']['label']['dtype'])
            for i, j in valid:
                start, shape = manifest['offsets'][i]['label']
                values = labels[start:start + int(np.prod(shape))]
                if not np.isin(values, [0, 1, 2, 3, 255]).all():
                    raise ValueError(f'Invalid cached labels: {rows[j]["id"]}')
                counts[j] = np.bincount(values[values < 4], minlength=4)[:4]
                filled.add(j)
        print(f'Label counts from verified file-matched cache: {len(valid)} ({path.parent.name})', flush=True)
    missing = [i for i in range(len(rows)) if i not in filled]
    if missing:
        import rasterio
        def read(index):
            row = rows[index]
            with rasterio.open(root / row['source_split'] / 'target' / row['files']['target']['name']) as dataset:
                values = dataset.read(1).reshape(-1)
            if not np.isin(values, [0, 1, 2, 3, 255]).all():
                raise ValueError(f'Invalid original labels: {row["id"]}')
            return index, np.bincount(values[values < 4].astype(np.int64), minlength=4)[:4]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, values in pool.map(read, missing):
                counts[index] = values
    if (counts.sum(axis=1) <= 0).any():
        raise ValueError('A source sample has no valid labelled pixels')
    return counts


def balanced_round(sizes, target):
    """Round a three-column proportional table with exact row/column totals."""
    sizes = np.asarray(sizes, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if sizes.sum() != target.sum():
        raise ValueError('Rounding totals differ')
    ideal = sizes[:, None] * (target / target.sum())[None, :]
    result = np.floor(ideal).astype(np.int64)
    residual = sizes - result.sum(axis=1)
    deficits = target - result.sum(axis=0)
    pending = [i for i in range(len(sizes)) if residual[i]]
    pending.sort(key=lambda i: (-residual[i], -sizes[i], i))
    for offset, i in enumerate(pending):
        remaining_rows = len(pending) - offset - 1
        feasible = []
        for columns in combinations(range(3), int(residual[i])):
            after = deficits.copy()
            after[list(columns)] -= 1
            if (after >= 0).all() and (after <= remaining_rows).all():
                cost = sum(1 - (ideal[i, j] - result[i, j]) for j in columns)
                feasible.append((cost, columns, after))
        if not feasible:
            raise ValueError('No feasible balanced rounding step')
        _, columns, deficits = min(feasible, key=lambda item: (item[0], item[1]))
        result[i, list(columns)] += 1
    if not np.array_equal(result.sum(axis=0), target) or not np.array_equal(result.sum(axis=1), sizes):
        raise ValueError('Balanced rounding changed a split/sample total')
    return result


def repartition(rows, counts, seed, iterations):
    rng = np.random.default_rng(seed)
    total = len(rows)
    sizes = np.array([sum(row['source_split'] == split for row in rows) for split in SPLITS])
    events = sorted({row['event'] for row in rows})
    event_ids = {event: np.array([i for i, row in enumerate(rows) if row['event'] == event]) for event in events}
    quotas = balanced_round([len(event_ids[event]) for event in events], sizes)
    assignment = np.full(total, -1, dtype=np.int8)
    strata = []
    features = counts[:, [1, 2, 3]].astype(np.float64)
    event_sums = np.zeros((len(events), 3, 3), dtype=np.float64)
    event_targets = np.zeros_like(event_sums)
    for e, event in enumerate(events):
        indices = event_ids[event]
        foreground = counts[indices, 1:].sum(axis=1) / counts[indices].sum(axis=1)
        presence = ((counts[indices, 1:] > 0) * [1, 2, 4]).sum(axis=1)
        bins = defaultdict(list)
        for mask in sorted(set(presence)):
            local = np.where(presence == mask)[0]
            cuts = np.unique(np.quantile(foreground[local], [0.2, 0.4, 0.6, 0.8]))
            rare_share = counts[indices[local], 2:].sum(axis=1) / np.maximum(counts[indices[local], 1:].sum(axis=1), 1)
            rare_cuts = np.unique(np.quantile(rare_share, [1 / 3, 2 / 3]))
            for j, rare in zip(local, rare_share):
                key = (int(mask), int(np.searchsorted(cuts, foreground[j], side='right')),
                       int(np.searchsorted(rare_cuts, rare, side='right')))
                bins[key].append(int(indices[j]))
        keys = sorted(bins)
        stratum_quotas = balanced_round([len(bins[key]) for key in keys], quotas[e])
        for key, q in zip(keys, stratum_quotas):
            shuffled = rng.permutation(bins[key])
            start = 0
            buckets = []
            for split, number in enumerate(q):
                selected = shuffled[start:start + number]
                assignment[selected] = split
                buckets.append(list(selected))
                start += int(number)
            strata.append((e, buckets))
        event_targets[e] = quotas[e, :, None] / len(indices) * features[indices].sum(axis=0)
        for split in range(3):
            event_sums[e, split] = features[indices[assignment[indices] == split]].sum(axis=0)
    target = sizes[:, None] / total * features.sum(axis=0)
    sums = np.array([features[assignment == split].sum(axis=0) for split in range(3)])
    global_scale = np.maximum(target, 1)
    event_scale = np.maximum(event_targets, 65536)
    usable = [(e, buckets) for e, buckets in strata if len([b for b in buckets if b]) > 1]
    accepted = 0
    for _ in range(iterations):
        e, buckets = usable[rng.integers(len(usable))]
        choices = [i for i, bucket in enumerate(buckets) if bucket]
        a, b = rng.choice(choices, size=2, replace=False)
        ia, ib = rng.integers(len(buckets[a])), rng.integers(len(buckets[b]))
        first, second = buckets[a][ia], buckets[b][ib]
        change = features[second] - features[first]
        old_g = (sums[[a, b]] - target[[a, b]]) / global_scale[[a, b]]
        new_g = old_g + np.array([change, -change]) / global_scale[[a, b]]
        old_e = (event_sums[e, [a, b]] - event_targets[e, [a, b]]) / event_scale[e, [a, b]]
        new_e = old_e + np.array([change, -change]) / event_scale[e, [a, b]]
        # Match global class ratios and each event's class ratios, without
        # changing event/stratum quotas or looking at model predictions.
        if np.square(new_g).sum() + 0.05 * np.square(new_e).sum() < np.square(old_g).sum() + 0.05 * np.square(old_e).sum():
            sums[a] += change
            sums[b] -= change
            event_sums[e, a] += change
            event_sums[e, b] -= change
            assignment[first], assignment[second] = b, a
            buckets[a][ia], buckets[b][ib] = second, first
            accepted += 1
    if (assignment < 0).any() or not np.array_equal(np.bincount(assignment, minlength=3), sizes):
        raise ValueError('Repartition did not preserve all samples and split sizes')
    return assignment, accepted


def exact_group_subset(groups, weights, target, rng):
    """Randomized integer subset sum; return complete scenes of exact size."""
    if target == 0:
        return []
    reachable = 1
    mask = (1 << (target + 1)) - 1
    parent = {}
    for group in rng.permutation(groups):
        weight = int(weights[group])
        fresh = ((reachable << weight) & mask) & ~reachable
        while fresh:
            bit = fresh & -fresh
            value = bit.bit_length() - 1
            parent[value] = (int(group), value - weight)
            fresh -= bit
        reachable |= (reachable << weight) & mask
        if (reachable >> target) & 1:
            selected = []
            value = target
            while value:
                group, value = parent[value]
                selected.append(group)
            return selected
    raise ValueError(f'Cannot assign exactly {target} patches using complete scenes')


def repartition_scenes(rows, counts, seed, iterations):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['event'], row['group'])].append(i)
    keys = sorted(groups)
    members = [np.array(groups[key]) for key in keys]
    weights = np.array([len(indices) for indices in members])
    sizes = np.array([sum(row['source_split'] == split for row in rows) for split in SPLITS])
    events = sorted({key[0] for key in keys})
    event_map = {event: np.array([i for i, key in enumerate(keys) if key[0] == event]) for event in events}
    quotas = balanced_round([sum(weights[event_map[event]]) for event in events], sizes)
    # Pixels, per-patch class presence, and foreground coverage are balanced.
    foreground = counts[:, 1:].sum(axis=1) / counts.sum(axis=1)
    sample_features = np.concatenate([counts[:, :].astype(float), (counts[:, 1:] > 0).astype(float),
        (foreground[:, None] > np.array([0.05, 0.2, 0.5, 0.8])).astype(float)], axis=1)
    features = np.array([sample_features[indices].sum(axis=0) for indices in members])
    assignment = np.zeros(len(keys), dtype=np.int8)
    event_targets = []
    buckets = []
    for e, event in enumerate(events):
        indices = event_map[event]
        for attempt in range(30):
            try:
                selected_test = exact_group_subset(indices, weights, int(quotas[e, 2]), rng)
                remaining = np.array(sorted(set(indices) - set(selected_test)))
                selected_val = exact_group_subset(remaining, weights, int(quotas[e, 1]), rng)
                break
            except ValueError:
                if attempt == 29:
                    raise
        assignment[selected_test] = 2
        assignment[selected_val] = 1
        for weight in sorted(set(weights[indices])):
            equal_size = indices[weights[indices] == weight]
            portions = [list(equal_size[assignment[equal_size] == split]) for split in range(3)]
            if len([portion for portion in portions if portion]) > 1:
                buckets.append((e, portions))
        event_targets.append(quotas[e, :, None] / weights[indices].sum() * features[indices].sum(axis=0))
    event_targets = np.array(event_targets)
    event_sums = np.array([[features[ids[assignment[ids] == split]].sum(axis=0) for split in range(3)]
                          for ids in event_map.values()])
    targets = sizes[:, None] / len(rows) * features.sum(axis=0)
    sums = np.array([features[assignment == split].sum(axis=0) for split in range(3)])
    scales = np.maximum(targets, 1)
    event_scales = np.maximum(event_targets, [65536] * 4 + [1] * 7)
    metric_weights = np.array([1, 1, 1, 1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
    accepted = 0
    for _ in range(iterations):
        e, portions = buckets[rng.integers(len(buckets))]
        choices = [i for i, portion in enumerate(portions) if portion]
        a, b = rng.choice(choices, size=2, replace=False)
        ia, ib = rng.integers(len(portions[a])), rng.integers(len(portions[b]))
        first, second = portions[a][ia], portions[b][ib]
        change = np.array([features[second] - features[first], features[first] - features[second]])
        old_g = (sums[[a, b]] - targets[[a, b]]) / scales[[a, b]]
        new_g = old_g + change / scales[[a, b]]
        old_e = (event_sums[e, [a, b]] - event_targets[e, [a, b]]) / event_scales[e, [a, b]]
        new_e = old_e + change / event_scales[e, [a, b]]
        if ((np.square(new_g) - np.square(old_g)) * metric_weights).sum() + 0.02 * ((np.square(new_e) - np.square(old_e)) * metric_weights).sum() < 0:
            sums[[a, b]] += change
            event_sums[e, [a, b]] += change
            assignment[first], assignment[second] = b, a
            portions[a][ia], portions[b][ib] = second, first
            accepted += 1
    patch_assignment = np.empty(len(rows), dtype=np.int8)
    for split, indices in zip(assignment, members):
        patch_assignment[indices] = split
    if not np.array_equal(np.bincount(patch_assignment, minlength=3), sizes):
        raise ValueError('Scene assignment did not preserve exact split counts')
    return patch_assignment, accepted


def js_divergence(probabilities):
    p, q = np.asarray(probabilities[0]), np.asarray(probabilities[1])
    mean = (p + q) / 2
    def kl(x):
        active = x > 0
        return float(np.sum(x[active] * np.log2(x[active] / mean[active])))
    return (kl(p) + kl(q)) / 2


def profile(rows, counts, assignment):
    events = sorted({row['event'] for row in rows})
    splits = {}
    scenes = defaultdict(set)
    for row, split in zip(rows, assignment):
        scenes[row['group']].add(SPLITS[int(split)])
    for index, name in enumerate(SPLITS):
        indices = np.where(assignment == index)[0]
        event_count = Counter(rows[i]['event'] for i in indices)
        pixels = counts[indices].sum(axis=0)
        splits[name] = {'samples': len(indices), 'events': dict(sorted(event_count.items())),
            'event_ratios': {event: event_count[event] / len(indices) for event in events},
            'class_pixels': pixels.tolist(), 'class_ratios': (pixels / pixels.sum()).tolist(),
            'class_present_samples': (counts[indices] > 0).sum(axis=0).tolist()}
    pairs = {}
    for a, b in combinations(SPLITS, 2):
        pairs[a + '_vs_' + b] = {
            'event_js_bits': js_divergence([[splits[s]['event_ratios'][event] for event in events] for s in (a, b)]),
            'class_js_bits': js_divergence([splits[s]['class_ratios'] for s in (a, b)])}
    test_indices = np.where(assignment == 2)[0]
    return {'splits': splits, 'distribution_divergence': pairs,
            'scenes': len(scenes), 'cross_split_scenes': sum(len(s) > 1 for s in scenes.values()),
            'test_samples_sharing_scene_with_train': sum('train' in scenes[rows[i]['group']] for i in test_indices),
            'exact_sample_id_overlap': 0}


def make_plan(args):
    root = args.root.expanduser().resolve()
    report = args.report_dir.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError('A regular dataset root directory is required')
    if (report / 'plan.json').exists():
        raise FileExistsError(f'Plan already exists: {report}; use a new report directory')
    rows = inventory(root)
    counts = label_counts(root, rows, args.cache_root.expanduser().resolve(), args.workers)
    before_assignment = np.array([SPLITS.index(row['source_split']) for row in rows], dtype=np.int8)
    method = repartition_scenes if args.unit == 'scene' else repartition
    assignment, accepted = method(rows, counts, args.seed, args.iterations)
    summary = {'schema_version': 1, 'source_root': str(root), 'seed': args.seed,
               'created_at': datetime.now(timezone.utc).isoformat(),
               'protocol': f'event_and_label_stratified_{args.unit}_level_v1',
               'policy': {'preserve_exact_split_counts': True, 'duplicate_patches_between_splits': False,
                          'source_scene_overlap_allowed': args.unit == 'patch', 'model_metrics_used_for_assignment': False},
               'before': profile(rows, counts, before_assignment), 'after': profile(rows, counts, assignment),
               'accepted_label_balance_swaps': accepted}
    report.mkdir(parents=True, exist_ok=True)
    for row, split, pixel_counts in zip(rows, assignment, counts):
        row['split'] = SPLITS[int(split)]
        row['class_pixels'] = pixel_counts.tolist()
    plan = {'source_root': str(root), 'inventory_sha256': digest(inventory(root)), 'rows': rows,
            'summary': summary}
    plan['plan_sha256'] = digest(plan)
    atomic_json(report / 'plan.json', plan)
    atomic_json(report / 'distribution_report.json', summary)
    with (report / 'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'source_split', 'split', 'group', 'event',
                                                  'background_pixels', 'intact_pixels', 'damaged_pixels', 'destroyed_pixels'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in ('id', 'source_split', 'split', 'group', 'event')},
                             **dict(zip(writer.fieldnames[-4:], row['class_pixels']))})
    print(json.dumps({'report_dir': str(report), 'before': summary['before'], 'after': summary['after'],
                      'accepted_swaps': accepted}, ensure_ascii=False, indent=2), flush=True)


def check_readers(root):
    """Do not replace input directories while a process has a source file open."""
    readers = set()
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            for descriptor in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(descriptor)
                    if target.startswith(str(root) + '/'):
                        readers.add(proc.name)
                except OSError:
                    pass
            tokens = [token.decode(errors='replace') for token in (proc / 'cmdline').read_bytes().split(b'\0') if token]
            entries = ('tools/train.py', 'tools/test.py', 'tools/evaluate.py', 'tools/infer.py', 'tools/smoke_test.py')
            if any(token.endswith(entries) for token in tokens):
                cwd = (proc / 'cwd').resolve()
                config_paths = []
                for i, token in enumerate(tokens[:-1]):
                    if token == '--config':
                        config_paths.append(Path(tokens[i + 1]).expanduser())
                    if token == '--runs':
                        for run_name in tokens[i + 1:]:
                            if run_name.startswith('--'):
                                break
                            config_paths.append(Path(run_name).expanduser() / 'config_snapshot.json')
                for config_path in config_paths:
                    if not config_path.is_absolute():
                        config_path = cwd / config_path
                    if config_path.is_file():
                        from utils.config import XMLConfigParser
                        config = XMLConfigParser(config_path).parse().as_dict()
                        data_path = Path(config.get('dataset', {}).get('root', '')).expanduser()
                        if not data_path.is_absolute():
                            data_path = cwd / data_path
                        if data_path.resolve() == root:
                            readers.add(proc.name)
        except (OSError, ValueError):
            continue
    if readers:
        raise RuntimeError(f'Dataset is used by active process(es): {sorted(readers)}; stop them before activation')


def write_active_manifest(metadata, rows):
    fields = ['id', 'source_split', 'split', 'group', 'event', 'origin_split',
              'background_pixels', 'intact_pixels', 'damaged_pixels', 'destroyed_pixels']
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=metadata,
                                     prefix='.manifest.', suffix='.csv', delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({'id': row['id'], 'source_split': row['split'], 'split': row['split'],
                'origin_split': row['source_split'], 'group': row['group'], 'event': row['event'],
                **dict(zip(fields[-4:], row['class_pixels']))})
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, metadata / 'manifest.csv')


def refresh_manifest(args):
    report = args.report_dir.expanduser().resolve()
    plan = json.loads((report / 'plan.json').read_text())
    plan_hash = plan.pop('plan_sha256')
    if digest(plan) != plan_hash:
        raise ValueError('Plan checksum mismatch')
    activation = json.loads((report / 'activation.json').read_text())
    root = args.root.expanduser().resolve()
    if str(root) != activation['root'] or activation['plan_sha256'] != plan_hash:
        raise ValueError('Active dataset does not match this plan')
    expected = {row['id']: row for row in plan['rows']}
    current = inventory(root)
    if len(current) != len(expected):
        raise ValueError('Active inventory sample count differs')
    for row in current:
        original = expected[row['id']]
        if row['source_split'] != original['split'] or row['files'] != original['files']:
            raise ValueError('Active inventory differs from this plan')
    shutil.copy2(report / 'manifest.csv', root / 'metadata/original_assignment_manifest.csv')
    write_active_manifest(root / 'metadata', plan['rows'])
    print('Active manifest paths verified and origin split provenance retained', flush=True)


def apply_plan(args):
    report = args.report_dir.expanduser().resolve()
    plan = json.loads((report / 'plan.json').read_text())
    expected_hash = plan.pop('plan_sha256')
    if digest(plan) != expected_hash:
        raise ValueError('Plan checksum mismatch')
    root = Path(plan['source_root'])
    if root != args.root.expanduser().resolve() or root.name != 'BRIGHT1':
        raise ValueError('Activation is restricted to the validated BRIGHT1 root')
    if (report / 'activation.json').exists():
        raise FileExistsError('This plan was already activated')
    if digest(inventory(root)) != plan['inventory_sha256']:
        raise ValueError('Source inventory changed after planning')
    check_readers(root)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stage = root.parent / f'.{root.name}_stratified_stage_{uuid.uuid4().hex[:12]}'
    backup = root.parent / f'{root.name}_before_stratified_{stamp}'
    if backup.exists():
        raise FileExistsError(backup)
    stage.mkdir()
    for split in SPLITS:
        for modality in SUFFIXES:
            (stage / split / modality).mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(plan['rows'], 1):
        for modality, info in row['files'].items():
            original = root / row['source_split'] / modality / info['name']
            destination = stage / row['split'] / modality / info['name']
            os.link(original, destination)
            if not os.path.samefile(original, destination):
                raise ValueError('Hardlink validation failed')
        if number % 5000 == 0:
            print(f'Staged {number}/{len(plan["rows"])} paired samples', flush=True)
    new_inventory = inventory(stage)
    if {r['id'] for r in new_inventory} != {r['id'] for r in plan['rows']}:
        raise ValueError('Staged sample inventory differs')
    expected = {row['id']: row for row in plan['rows']}
    for row in new_inventory:
        if row['source_split'] != expected[row['id']]['split'] or row['files'] != expected[row['id']]['files']:
            raise ValueError(f'Staged file/assignment mismatch: {row["id"]}')
    # Read the source labels independently of the previous tensor cache before
    # switching directories; filenames and all label pixels must agree.
    import rasterio
    def verify_label(row):
        with rasterio.open(stage / row['split'] / 'target' / row['files']['target']['name']) as dataset:
            values = dataset.read(1).reshape(-1)
        if not np.isin(values, [0, 1, 2, 3, 255]).all():
            raise ValueError(f'Invalid staged labels: {row["id"]}')
        actual_counts = np.bincount(values[values < 4].astype(np.int64), minlength=4)[:4]
        if actual_counts.tolist() != row['class_pixels']:
            raise ValueError(f'Cached/original label counts differ: {row["id"]}')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(verify_label, plan['rows']):
            pass
    print('All original raster labels independently verified', flush=True)
    metadata = stage / 'metadata'
    metadata.mkdir()
    for name in ('manifest.csv', 'distribution_report.json'):
        shutil.copy2(report / name, metadata / name)
    shutil.copy2(report / 'manifest.csv', metadata / 'original_assignment_manifest.csv')
    write_active_manifest(metadata, plan['rows'])
    old_stats = root / 'crop256_stats.json'
    if old_stats.exists():
        shutil.copy2(old_stats, metadata / 'original_crop256_stats.json')
        stats = json.loads(old_stats.read_text())
        stats['split_protocol'] = plan['summary']['protocol']
        stats['original_crop_statistics'] = 'metadata/original_crop256_stats.json'
        for split, details in plan['summary']['after']['splits'].items():
            stats['splits'][split] = {'kept_patches': details['samples'],
                'class_pixel_counts': {str(i): value for i, value in enumerate(details['class_pixels'])}}
        atomic_json(stage / 'crop256_stats.json', stats)
    record = {'status': 'activated', 'created_at': datetime.now(timezone.utc).isoformat(),
              'root': str(root), 'backup_root': str(backup), 'plan_sha256': expected_hash,
              'storage': 'Hardlinked immutable files; original bytes and mtime are unchanged',
              'protocol': plan['summary']['protocol'],
              'source_scene_overlap_allowed': plan['summary']['policy']['source_scene_overlap_allowed']}
    atomic_json(metadata / 'split_info.json', record)
    # Last check immediately before the reversible directory swap.
    if digest(inventory(root)) != plan['inventory_sha256']:
        raise ValueError('Source changed during staging')
    check_readers(root)
    os.rename(root, backup)
    try:
        os.rename(stage, root)
    except BaseException:
        os.rename(backup, root)
        raise
    atomic_json(report / 'activation.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=ROOT / 'runs/cache/bright')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'runs/data_preparation/bright')
    parser.add_argument('--unit', choices=['scene', 'patch'], default='scene')
    parser.add_argument('--seed', type=int, default=3402)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--refresh-manifest', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.iterations < 0:
        raise ValueError('Invalid workers/iterations')
    if args.apply and args.refresh_manifest:
        raise ValueError('Use either --apply or --refresh-manifest')
    if args.apply or args.refresh_manifest:
        lock_path = args.root.expanduser().resolve().parent / '.bright_repartition.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.apply:
                apply_plan(args)
            else:
                refresh_manifest(args)
    else:
        make_plan(args)


if __name__ == '__main__':
    main()
