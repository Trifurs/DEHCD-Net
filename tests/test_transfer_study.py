import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.protocol import atomic_json, file_digest
from utils.transfer_initialization import initialize_weights
from utils.transfer_study import (EVENTS, fewshot_partition, loeo_partition,
                                 source_class_weights, validate_partition, write_partition)


def rows_per_event():
    return [{"id": f"{event}_{scene:08d}_r00_c{tile:02d}", "source_split": "test",
             "group": f"{event}_{scene:08d}", "event": event,
             "class_pixels": [100, 30, 3, 2]}
            for event in EVENTS for scene in range(20) for tile in range(2)]


class TransferStudyTests(unittest.TestCase):
    def test_loeo_never_uses_held_out_events_in_train_or_validation(self):
        records = rows_per_event()
        for target in EVENTS:
            partition = loeo_partition(records, target, .1, 123)
            report = validate_partition(partition, target, True)
            self.assertEqual({target}, set(report['test']['events']))
            self.assertNotIn(target, report['train']['events'])
            self.assertNotIn(target, report['val']['events'])
            self.assertEqual(len(records), sum(len(rows) for rows in partition.values()))
            self.assertEqual(partition, loeo_partition(list(reversed(records)), target, .1, 123))

    def test_fewshot_scenes_are_whole_and_annotation_budget_includes_validation(self):
        partition = fewshot_partition(rows_per_event(), EVENTS[1], .1, .1, 88)
        report = validate_partition(partition)
        self.assertEqual((2, 2, 16), tuple(report[s]['scenes'] for s in ('train', 'val', 'test')))
        self.assertEqual((4, 4, 32), tuple(report[s]['samples'] for s in ('train', 'val', 'test')))
        self.assertTrue(all(all(c > 0 for c in r['class_pixels']) for r in report.values()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'split.csv'
            write_partition(path, partition)
            write_partition(path, partition)
            bad = copy.deepcopy(partition)
            bad['train'][0]['id'] += '_changed'
            with self.assertRaises(ValueError):
                write_partition(path, bad)

    def test_scene_overlap_and_missing_training_classes_fail(self):
        partition = fewshot_partition(rows_per_event(), EVENTS[1], .1, .1, 88)
        partition['test'].append(partition['train'][0])
        with self.assertRaises(ValueError):
            validate_partition(partition)
        with self.assertRaises(ValueError):
            source_class_weights([{'class_pixels': [100, 5, 1, 0]}])
        self.assertEqual([1.0, 1.0, 4.0, 8.0], source_class_weights([{'class_pixels': [500, 64, 4, 1]}]))

    def test_event_sampling_equalizes_probability_mass_not_sample_counts(self):
        from tools.train import build_class_balanced_sampler
        samples = [{'event': event} for event in ('large',) * 10 + ('small',) * 2]
        dataset = SimpleNamespace(samples=samples, __len__=lambda: len(samples))
        class SampleSet:
            def __init__(self): self.samples = samples
            def __len__(self): return len(samples)
        stats = {'sample_counts': [np.array([100, 20, 2, 1])] * 12,
                 'counts': np.array([1200, 240, 24, 12])}
        config = {'class_balanced_sampler': True, 'event_balanced_sampler': True,
                  'sampler_target_classes': [2, 3], 'sampler_min_pixels': 1, 'sampler_num_samples': 512}
        with patch('tools.train.compute_label_distribution', return_value=stats):
            sampler = build_class_balanced_sampler(SampleSet(), config, 4, 255)
        self.assertAlmostEqual(.5, float(sampler.weights[:10].sum() / sampler.weights.sum()))
        self.assertAlmostEqual(.5, float(sampler.weights[10:].sum() / sampler.weights.sum()))
        self.assertEqual(512, len(sampler))

    def test_validation_macro_does_not_let_a_larger_event_dominate(self):
        from tools.train import validate
        class Samples(Dataset):
            def __init__(self):
                self.samples = [{'id': str(i), 'event': 'large' if i < 9 else 'small'} for i in range(10)]
            def __len__(self): return len(self.samples)
            def __getitem__(self, index):
                values = [-1., 1.] if index < 9 else [1., -1.]
                image = torch.tensor(values).reshape(2, 1, 1)
                return {'id': str(index), 'optical': image, 'sar': image, 'label': torch.ones((1, 1), dtype=torch.long)}
        class Identity(torch.nn.Module):
            def forward(self, optical, sar): return optical
        config = {'training': {'amp': False, 'loss': 'ce_dice', 'validation_event_macro': True},
                  'task': {'num_classes': 2}, 'inference': {'prediction_rule': 'damage_argmax'}}
        _, metrics = validate(Identity(), DataLoader(Samples(), batch_size=3), torch.device('cpu'), config)
        self.assertAlmostEqual(.9, metrics['foreground_miou'])
        self.assertAlmostEqual(.5, metrics['event_macro_foreground_miou'])

    def test_discriminative_optimizer_covers_each_parameter_once(self):
        from tools.train import build_optimizer
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.optical_encoder = torch.nn.Linear(2, 2)
                self.sar_encoder = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 2)
        model = Tiny()
        optimizer = build_optimizer(model, {'learning_rate': .001, 'encoder_lr_multiplier': .25})
        self.assertEqual([.00025, .001], [group['lr'] for group in optimizer.param_groups])
        parameters = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(list(model.parameters())), len(parameters))
        self.assertEqual(len(parameters), len(set(parameters)))

    def test_loeo_preserves_donor_train_validation_boundary(self):
        records = rows_per_event()
        for row in records:
            if row['event'] == EVENTS[0]:
                row['source_split'] = 'val' if '_00000019_' in row['id'] else 'train'
        partition = loeo_partition(records, EVENTS[1], .1, 123)
        for split in ('train', 'val'):
            self.assertTrue(all(row['source_split'] == split for row in partition[split] if row['event'] == EVENTS[0]))

    def test_initialization_rejects_target_exposure_and_loads_only_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            path = run / 'checkpoints/best.pth'
            path.parent.mkdir()
            source_model, target_model = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
            config = {'model': {'name': 'tiny'}, 'task': {'num_classes': 2}, 'normalization': {}}
            torch.save({'model': source_model.state_dict(), 'config': config, 'epoch': 20}, path)
            atomic_json(run / 'protocol.json', {'datasets': {'train': {'records': [{'event': EVENTS[0]}]}}})
            target_config = {**copy.deepcopy(config), 'training': {'init_checkpoint': str(path),
                'init_checkpoint_sha256': file_digest(path)}, 'experiment': {'held_out_event': EVENTS[0]}}
            with self.assertRaisesRegex(ValueError, 'already saw'):
                initialize_weights(target_model, target_config)
            target_config['experiment']['held_out_event'] = EVENTS[1]
            record = initialize_weights(target_model, target_config)
            torch.testing.assert_close(target_model.weight, source_model.weight)
            self.assertFalse(record['optimizer_state_imported'])
            target_config['training']['init_checkpoint_sha256'] = 'invalid'
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                initialize_weights(target_model, target_config)

    def test_shared_tensor_view_matches_selected_source_rows_and_label_stats(self):
        from utils.prepared_data import (prepare_split, input_signature, preprocessing_identity,
                                         attach_prepared_split, _read_index)
        from utils.protocol import dataset_identity
        class Source:
            normalization_cfg = {}
            num_optical_channels = num_sar_channels = 1
            sar_channel_counts = [1]
            return_metadata = False
            num_classes, ignore_index = 4, 255
            def __init__(self, root, split):
                self.root, self.split = root, split
                self.dataset_cfg = {'root': str(root)}
                self.samples = [{'id': f'{split}_{i}', 'group': f'{split}_scene_{i}', 'event': split,
                    'optical': str(root/'raw'), 'sar': [str(root/'raw')], 'label': str(root/'raw')} for i in range(3)]
            def __len__(self): return len(self.samples)
            def _load_base_sample(self, i):
                return {'id': self.samples[i]['id'], 'optical': torch.full((1, 2, 2), float(i)),
                        'sar': torch.full((1, 2, 2), float(i + 10)), 'label': torch.full((2, 2), i, dtype=torch.long)}
            def load_label_for_stats(self, sample):
                return torch.full((2, 2), int(sample['id'].split('_')[-1]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'raw').write_text('fixture')
            index = {'preprocessing_sha256': preprocessing_identity(), 'splits': {}}
            sources = [Source(root, split) for split in ('train', 'test')]
            for source in sources:
                identity = dataset_identity(source)
                index['splits'][input_signature(source)] = prepare_split(source, {'stat': identity}, root/'cache', workers=0)
            index_path = root / 'index.json'
            atomic_json(index_path, index)
            manifest = root / 'manifest.csv'
            selected = [sources[1].samples[2], sources[0].samples[1]]
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'source_split', 'split', 'group', 'event'])
                writer.writeheader()
                writer.writerows({'id': r['id'], 'source_split': r['event'], 'split': 'train',
                                  'group': r['group'], 'event': r['event']} for r in selected)
            target = Source(root, 'train')
            target.dataset_cfg['manifest'] = str(manifest)
            with patch.dict('os.environ', {'DEHCD_PREPARED_INDEX': str(index_path)}):
                self.assertTrue(attach_prepared_split(target))
            self.assertEqual([r['id'] for r in selected], [r['id'] for r in target.samples])
            self.assertEqual(2., float(target._prepared_split.get(0)['optical'][0, 0, 0]))
            self.assertEqual(1., float(target._prepared_split.get(1)['optical'][0, 0, 0]))
            counts = target._prepared_split.label_distribution(4, 255)['counts']
            np.testing.assert_array_equal([0, 4, 4, 0], counts)
            cached_identity = dataset_identity(target)
            del target._prepared_identities
            self.assertEqual(cached_identity, dataset_identity(target))
            _read_index.cache_clear()


if __name__ == '__main__':
    unittest.main()
