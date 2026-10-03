"""Seed rounds, dataset boundaries, and stable experiment identities."""
from __future__ import annotations

import copy
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from tools.run_multiseed import make_plan
from utils.campaign_layout import run_path
from utils.campaign_reuse import dispatch_order

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 1051, 2060]
DATASETS = ['bright', 'haiti', 'xbd', 'cau_flood']


def plan_for(seeds=None, datasets=None, groups=None):
    return make_plan(SimpleNamespace(
        catalog=str(ROOT / 'configs/experiments/catalog.json'),
        suite=groups or ['all'], datasets=datasets or DATASETS,
        experiments=None, config=None, seeds=seeds or SEEDS,
        data_root=[], manifest=[], encoder_checkpoint=[], device='cpu',
        num_workers=0, epochs=None, batch_size=None,
        gradient_accumulation_steps=None, fingerprint='stat', runtime_profile='none'))


class SchedulingTests(unittest.TestCase):
    def test_full_plan_has_three_complete_seed_rounds(self):
        plan = plan_for()
        self.assertEqual(240, len(plan['jobs']))
        self.assertEqual(240, len({job['id'] for job in plan['jobs']}))
        for index, seed in enumerate(SEEDS):
            block = plan['jobs'][index * 80:(index + 1) * 80]
            self.assertEqual({seed}, {job['seed'] for job in block})
            self.assertEqual({'bright': 39, 'haiti': 23, 'xbd': 9, 'cau_flood': 9},
                             Counter(job['dataset'] for job in block))
            self.assertEqual(['bright'] * 39 + ['haiti'] * 23 + ['xbd'] * 9 + ['cau_flood'] * 9,
                             [job['dataset'] for job in block])

    def test_explicit_seed_order_preserves_task_identity_and_run_paths(self):
        before = plan_for()
        after = plan_for(seeds=[2060, 42, 1051], datasets=list(reversed(DATASETS)))
        self.assertEqual([2060] * 80 + [42] * 80 + [1051] * 80,
                         [job['seed'] for job in after['jobs']])
        def identities(plan):
            return {job['id']: (job['experiment'], job['seed'], job['config_sha256'],
                               str(run_path('/tmp/scheduling', job, plan)))
                    for job in plan['jobs']}
        self.assertEqual(identities(before), identities(after))

    def test_group_union_and_dataset_filter_still_have_complete_seed_rounds(self):
        plan = plan_for(datasets=['cau_flood', 'bright'],
                        groups=['main', 'scaling', 'main_bright'])
        self.assertEqual(63, len(plan['jobs']))
        self.assertEqual(63, len({job['id'] for job in plan['jobs']}))
        for index, seed in enumerate(SEEDS):
            block = plan['jobs'][index * 21:(index + 1) * 21]
            self.assertEqual([seed] * 21, [job['seed'] for job in block])
            self.assertEqual(['bright'] * 12 + ['cau_flood'] * 9,
                             [job['dataset'] for job in block])

    def test_action_priority_never_crosses_seed_or_dataset_boundaries(self):
        definitions = [
            ('new_bright_42', 42, 'bright', 'train_new'),
            ('complete_bright_1051', 1051, 'bright', 'reuse_complete'),
            ('resume_haiti_2060', 2060, 'haiti', 'resume_training'),
            ('eval_xbd_42', 42, 'xbd', 'reevaluate_only'),
            ('new_haiti_42', 42, 'haiti', 'train_new'),
            ('eval_bright_42', 42, 'bright', 'reevaluate_only'),
            ('resume_bright_42', 42, 'bright', 'resume_training'),
            ('new_cau_42', 42, 'cau_flood', 'train_new'),
            ('new_bright_1051', 1051, 'bright', 'train_new'),
            ('resume_bright_1051', 1051, 'bright', 'resume_training'),
        ]
        plan = {'seeds': SEEDS, 'jobs': [
            {'id': name, 'seed': seed, 'dataset': dataset}
            for name, seed, dataset, _ in definitions]}
        manifest = {'tasks': [{'id': name, 'state': state}
                              for name, _, _, state in definitions]}
        original_plan, original_manifest = copy.deepcopy(plan), copy.deepcopy(manifest)
        ordered = dispatch_order(plan, manifest)
        self.assertEqual([
            'eval_bright_42', 'resume_bright_42', 'new_bright_42',
            'new_haiti_42', 'eval_xbd_42', 'new_cau_42',
            'complete_bright_1051', 'resume_bright_1051', 'new_bright_1051',
            'resume_haiti_2060'], [job['id'] for job in ordered])
        self.assertEqual(original_plan, plan)
        self.assertEqual(original_manifest, manifest)


if __name__ == '__main__':
    unittest.main()
