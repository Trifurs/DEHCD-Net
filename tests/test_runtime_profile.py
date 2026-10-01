import copy
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tools.run_multiseed import make_plan
from tools.train import compute_label_distribution
from utils.comparison_protocol import audit_plan, validate_design
from utils.config import XMLConfigParser
from utils.protocol import config_digest
from utils.reproducibility import seed_epoch
from utils.runtime import apply_runtime_profile, configure_runtime
from utils.seeded_loader import DrawSeededDataset, DrawSeededSampler
from utils.training_health import gradient_statistics

ROOT = Path(__file__).resolve().parents[1]


class RandomSample(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), np.random.rand(), torch.rand(()).item()], dtype=torch.float64)


def loader(workers, persistent):
    sampler = WeightedRandomSampler([1., 3., 2.], 13, replacement=True, generator=torch.Generator())
    return DataLoader(DrawSeededDataset(RandomSample()), batch_size=4,
                      sampler=DrawSeededSampler(sampler, 42), num_workers=workers,
                      persistent_workers=persistent, generator=torch.Generator())


def rows(data, epoch):
    seed_epoch(data, 42, epoch, True)
    return torch.cat(list(data))


def close(data):
    if getattr(data, '_iterator', None) is not None:
        data._iterator._shutdown_workers()


class RuntimeProfileTests(unittest.TestCase):
    def test_persistent_workers_match_fresh_resume_and_worker_counts(self):
        persistent, fresh, parent = loader(2, True), loader(2, False), loader(0, False)
        try:
            first = rows(persistent, 1)
            resumed = rows(persistent, 7)
            self.assertTrue(torch.equal(resumed, rows(fresh, 7)))
            self.assertTrue(torch.equal(resumed, rows(parent, 7)))
            self.assertFalse(torch.equal(first, resumed))
            self.assertGreater(len(set(first[:, 1].tolist())), len(set(first[:, 0].tolist())))
        finally:
            for item in (persistent, fresh, parent): close(item)

    def test_run_seeds_do_not_share_shifted_augmentation_streams(self):
        dataset = DrawSeededDataset(RandomSample())
        streams = []
        for seed in (42, 43):
            sampler = DrawSeededSampler([1] * 16, seed)
            sampler.set_epoch(7)
            streams.append(torch.stack([dataset[item] for item in sampler]))
        self.assertFalse(torch.equal(streams[0][1:, 1:], streams[1][:-1, 1:]))

    def test_worker_zero_does_not_reset_model_rng(self):
        dataset = DrawSeededDataset(RandomSample())
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        dataset[(1, 999)]
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

    def test_profile_preserves_all_declared_controls_and_training_budget(self):
        catalog = json.loads((ROOT / 'configs/experiments/catalog.json').read_text())['experiments']
        configs = {}
        expected = {'bright': (4, 2), 'haiti': (24, 1), 'cau_flood': (4, 3), 'xbd': (4, 3)}
        for row in catalog:
            cfg = XMLConfigParser(ROOT / row['config']).parse().as_dict()
            original = copy.deepcopy(cfg)
            apply_runtime_profile(cfg, 'rtx5090')
            t, old = cfg['training'], original['training']
            self.assertEqual(expected[row['dataset']], (t['batch_size'], t['gradient_accumulation_steps']))
            self.assertEqual(old['batch_size'] * old['gradient_accumulation_steps'], t['batch_size'] * t['gradient_accumulation_steps'])
            for key in ('epochs', 'learning_rate', 'scheduler', 'warmup_epochs', 'loss', 'class_balanced_sampler', 'best_metric', 'val_interval'):
                self.assertEqual(old[key], t[key])
            for key in ('model', 'dataset', 'task', 'normalization', 'augmentation'):
                self.assertEqual(original[key], cfg[key])
            self.assertNotEqual(config_digest(original), config_digest(cfg))
            configs[row['id']] = cfg
        self.assertEqual(170, len(validate_design(configs)))

    def test_runtime_plan_has_522_jobs_and_profiles_references(self):
        args = SimpleNamespace(catalog=str(ROOT / 'configs/experiments/catalog.json'), suite=['all'],
            datasets=['bright', 'haiti', 'cau_flood', 'xbd'], experiments=None, config=None,
            seeds=[42, 1051, 2060], data_root=[], manifest=[], encoder_checkpoint=[],
            device='cuda:0', num_workers=None, epochs=None, batch_size=None,
            gradient_accumulation_steps=None, fingerprint='stat', runtime_profile='rtx5090')
        plan = make_plan(args)
        self.assertEqual(522, len(plan['jobs']))
        self.assertTrue(all(j['config']['training']['num_workers'] == 8 for j in plan['jobs']))
        args.experiments = ['haiti_no_bicsf']; args.num_workers = 2
        plan = make_plan(args)
        self.assertEqual('rtx5090_v1', plan['comparison_references']['haiti_dehcd_l']['training']['runtime_profile'])
        self.assertEqual(2, plan['comparison_references']['haiti_dehcd_l']['training']['num_workers'])
        audit_plan(plan)

    def test_foreach_norms_preserve_stability_detection(self):
        model = torch.nn.Sequential(torch.nn.Linear(7, 4), torch.nn.Linear(4, 2))
        model(torch.randn(3, 7)).square().sum().backward()
        a, detail_a = gradient_statistics(model, 0, foreach=False)
        b, detail_b = gradient_statistics(model, 0, foreach=True)
        torch.testing.assert_close(a, b)
        self.assertEqual(detail_a, detail_b)
        next(model.parameters()).grad[0, 0] = float('inf')
        bad, details = gradient_statistics(model, foreach=True)
        self.assertFalse(torch.isfinite(bad))
        self.assertIsNone(details[0]['norm'])

    def test_label_stats_cache_is_scoped_to_dataset_and_settings(self):
        class Labels:
            samples = ['a', 'b']
            reads = 0
            def load_label_for_stats(self, sample):
                self.reads += 1
                return torch.tensor([[0, 1], [255, 1]])
        dataset = Labels()
        a = compute_label_distribution(dataset, 2, 255)
        self.assertEqual(2, dataset.reads)
        b = compute_label_distribution(dataset, 2, 255)
        self.assertIs(a, b)
        self.assertEqual(2, dataset.reads)
        compute_label_distribution(dataset, 2, 255, max_samples=1)
        self.assertEqual(3, dataset.reads)

    def test_runtime_precision_is_explicit(self):
        a, b = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            info = configure_runtime({'allow_tf32': True}, torch.device('cpu'))
            self.assertTrue(info['matmul_allow_tf32']); self.assertTrue(info['cudnn_allow_tf32'])
            info = configure_runtime({'allow_tf32': False}, torch.device('cpu'))
            self.assertFalse(info['matmul_allow_tf32']); self.assertFalse(info['cudnn_allow_tf32'])
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = a, b


if __name__ == '__main__':
    unittest.main()
