"""CPU checks for evaluation provenance, display-only progress and complete resume."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from utils.checkpoint import CheckpointWriter, validate_training_checkpoint
from utils.damage_metrics import DamageHeadMeter, localization_is_supervised
from utils.metrics import ConfusionMatrixMeter
from utils.prediction import (predict_outputs, decode_predictions, evaluation_coverage,
                              validate_evaluation_labels, check_evaluation_tensors)
from utils.progress import RunProgress, training_health_text
from utils.reproducibility import BestTracker, capture_rng_state, restore_rng_state, set_seed
from utils.training_health import CollapseMonitor
from tools.validate_results import validate_result


class EvaluationProgressTests(unittest.TestCase):
    def test_unsupervised_localization_is_diagnostic_and_labels_are_ignored(self):
        target = torch.tensor([[[0, 1], [2, 255]]])
        pred = torch.tensor([[[0, 1], [2, 1]]])
        output = {'logits': torch.nn.functional.one_hot(pred, 3).permute(0, 3, 1, 2).float(),
                  'localization_logits': torch.nn.functional.one_hot((pred > 0).long(), 2).permute(0, 3, 1, 2).float()}
        meter = DamageHeadMeter(3)
        meter.update(output, target)
        result = meter.compute()
        self.assertFalse(result['localization_supervised'])
        self.assertIsNone(result['localization_f1'])
        self.assertEqual(1, result['diagnostic_localization_f1'])
        self.assertEqual(2, result['labeled_foreground_pixels'])
        self.assertEqual(3, sum(map(sum, result['localization_confusion_matrix'])))
        self.assertFalse(localization_is_supervised({'training': {'localization_loss_weight': 0}}))
        self.assertTrue(localization_is_supervised({'training': {'localization_loss_weight': .5}}))

    def test_argmax_ties_ignore_obsolete_threshold_and_reject_nonfinite_heads(self):
        class Network(torch.nn.Module):
            def forward(self, optical, sar):
                return {'logits': optical, 'localization_logits': sar}
        logits = torch.tensor([[[[0., 1.]], [[0., 2.]]]])
        model = Network()
        for tta in ('none', 'flips', 'd4'):
            output = predict_outputs(model, logits, logits, amp=False, tta_mode=tta)
            torch.testing.assert_close(decode_predictions(output, {'inference': {'threshold': .99}}),
                                       torch.tensor([[[0, 1]]]))
        for head in ('main', 'localization'):
            bad = logits.clone(); bad[0, 0, 0, 0] = float('nan')
            with self.assertRaises(FloatingPointError):
                predict_outputs(model, bad if head == 'main' else logits,
                                bad if head == 'localization' else logits, False, 'none')
        with self.assertRaises(FloatingPointError):
            check_evaluation_tensors({'loss': torch.tensor(float('inf'))})
        validate_evaluation_labels(torch.tensor([0, 1, 255]), 2)
        for invalid in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([.5]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError): validate_evaluation_labels(invalid, 2)

    def test_partial_and_unsupervised_formal_results_are_rejected(self):
        meter = ConfusionMatrixMeter(2).update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        old = {'schema_version': 1, 'split': 'test', 'confusion_matrix': meter.matrix.tolist(), 'metrics': meter.compute()}
        validate_result(old)  # Existing complete single-head schema remains supported.
        result = {**old, 'schema_version': 2, **evaluation_coverage(2, 2),
                  'dataset_identity': {'samples': 2}, 'runtime': {'max_batches': 0, 'prediction_rule': 'damage_argmax',
                                                                'test_time_augmentation': 'none'}}
        validate_result(result)
        partial = {**result, **evaluation_coverage(1, 2, limit=1)}
        with self.assertRaises(ValueError): validate_result(partial)
        self.assertFalse(evaluation_coverage(2, 2, limit=8)['complete'])
        for counts in ((0, 0), (0, 2), (3, 2)):
            with self.assertRaises(ValueError): evaluation_coverage(*counts)
        for heads in ({'localization_f1': .9}, {'localization_supervised': False, 'localization_f1': .9}):
            with self.assertRaises(ValueError): validate_result({**result, 'damage_head_metrics': heads})
        validate_result({**result, 'damage_head_metrics': {'localization_supervised': False,
                         'localization_f1': None, 'diagnostic_localization_f1': .9}})

    def test_progress_observes_tracker_and_guards_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = BestTracker(.4, .39, 3)
            monitor = CollapseMonitor({'collapse_patience': 4, 'collapse_warmup_epochs': 2,
                                       'foreground_stall_patience': 5, 'foreground_stall_warmup_epochs': 2})
            monitor.count, monitor.stall_count = 1, 2
            before = copy.deepcopy((tracker.__dict__, monitor.__dict__))
            reporter = RunProgress(Path(tmp) / 'progress.json', epochs=100, completed=9)
            reporter.training_health(9, 'foreground_miou', .4, 6, tracker, monitor, 0)
            self.assertEqual(before, (tracker.__dict__, monitor.__dict__))
            state = json.loads((Path(tmp) / 'progress.json').read_text())
            self.assertEqual(3, state['epochs_since_best'])
            text = training_health_text(state)
            for expected in ('foreground_miou=0.400000', '@epoch 6', 'disabled (fixed budget)', 'collapse=warning 1/4'):
                self.assertIn(expected, text)
            reporter.training_health(9, 'foreground_miou', .4, 6, tracker, monitor, 10)
            self.assertIn('early stop=3/10 checks', training_health_text(reporter.state))



    def test_campaign_eta_excludes_reused_training_and_completed_resume_epochs(self):
        from utils.campaign_progress import CampaignProgress
        from utils.campaign_layout import run_path
        from utils.protocol import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            jobs = [{'id': name, 'experiment': name, 'seed': 42, 'dataset': 'bright', 'data_key': 'd',
                     'config': {'model': {'name': 'same'}, 'dataset': {'patch_size': 1},
                                'training': {'epochs': 10}, 'experiment': {'suite': 'main'}}}
                    for name in ('complete', 'evaluate', 'resume')]
            plan = {'layout_version': 2, 'jobs': jobs, 'data': {'d': {'splits': {
                'train': {'samples': 100}, 'val': {'samples': 20}, 'test': {'samples': 20}}}}}
            atomic_json(output / 'reuse_manifest.json', {'tasks': [
                {'id': 'complete', 'state': 'reuse_complete'}, {'id': 'evaluate', 'state': 'reevaluate_only'},
                {'id': 'resume', 'state': 'resume_training', 'evidence': {'completed_epoch': 5}}]})
            atomic_json(run_path(output, jobs[0], plan) / 'training_summary.json', {'seconds': 1080})
            monitor = CampaignProgress(plan, output)
            self.assertEqual({'complete'}, monitor.completed)
            # 540 units for five remaining epochs + eight per test, twice.
            self.assertEqual(556, monitor.remaining(None, {}))
            self.assertEqual(551, monitor.remaining(jobs[1], {'stage': 'test', 'eta_seconds': 3}))
            monitor.close()

    def test_test_evaluate_and_infer_use_identical_predictions_on_cpu(self):
        from tools import test as test_tool, evaluate as evaluate_tool, infer as infer_tool
        from utils.protocol import atomic_json
        from PIL import Image
        class Dataset(torch.utils.data.Dataset):
            num_optical_channels = num_sar_channels = 1
            samples = [{'id': 'a'}, {'id': 'b'}]
            def __len__(self): return 2
            def __getitem__(self, index):
                # A logit tie must decode as background, regardless of threshold.
                image = torch.tensor([[[0., 1.], [-1., 1.]]])
                return {'id': self.samples[index]['id'], 'optical': image, 'sar': image,
                        'label': torch.tensor([[0, 1], [0, 255]])}
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(()))
            def forward(self, optical, sar):
                logits = torch.cat((-optical, optical), dim=1) * self.weight
                return {'logits': logits, 'localization_logits': logits.flip(1)}
        dataset = Dataset()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'train'; run.mkdir()
            config = {'model': {'name': 'test'}, 'task': {'num_classes': 2, 'ignore_index': 255},
                      'dataset': {'type': 'test'}, 'training': {'device': 'cpu', 'amp': False, 'num_workers': 0,
                       'batch_size': 1, 'loss': 'ce_dice', 'localization_loss_weight': 0, 'best_metric': 'foreground_miou'},
                      'inference': {'prediction_rule': 'damage_argmax', 'test_time_augmentation': 'none',
                                    'threshold': .99, 'save_visualization': False},
                      'logging': {'root_dir': str(root), 'run_name': 'tiny'}}
            atomic_json(run / 'config_snapshot.json', config)
            checkpoint = root / 'model.pth'
            torch.save({'model': Model().state_dict(), 'config': config, 'epoch': 1}, checkpoint)
            identity = {'samples': 2, 'sha256': 'test-identity', 'split': 'test'}
            with patch('datasets.build_dataset', return_value=dataset), patch('models.build_model', side_effect=lambda *a, **kw: Model()), patch.object(test_tool, 'dataset_identity', return_value=identity), patch('sys.argv', ['test.py', '--checkpoint', str(checkpoint), '--no-profile', '--no-confusion-plot', '--save-predictions']):
                args = test_tool.parse_args()
                result = test_tool.evaluate_run(run, args, root / 'tested')
            self.assertTrue(result['complete']); self.assertIsNone(result['damage_head_metrics']['localization_f1'])
            validate_result(result)
            evaluation_dir = root / 'evaluated'; evaluation_dir.mkdir()
            with patch('datasets.build_dataset', return_value=dataset), patch('models.build_model', side_effect=lambda *a, **kw: Model()), patch.object(evaluate_tool, 'dataset_identity', return_value=identity), patch.object(evaluate_tool, 'create_run_dir', return_value=evaluation_dir), patch.object(evaluate_tool, 'XMLConfigParser') as parser, patch('sys.argv', ['evaluate.py', '--checkpoint', str(checkpoint)]):
                parser.return_value.parse.return_value.as_dict.return_value = copy.deepcopy(config)
                evaluate_tool.main()
            evaluated = json.loads((evaluation_dir / 'metrics.json').read_text())
            self.assertEqual(result['metrics'], evaluated['metrics']); self.assertEqual(result['loss'], evaluated['loss'])
            validate_result(evaluated)
            inference_dir = root / 'inferred'; inference_dir.mkdir()
            with patch('datasets.build_dataset', return_value=dataset), patch('models.build_model', side_effect=lambda *a, **kw: Model()), patch.object(infer_tool, 'dataset_identity', return_value=identity), patch.object(infer_tool, 'create_run_dir', return_value=inference_dir), patch.object(infer_tool, 'XMLConfigParser') as parser, patch('sys.argv', ['infer.py', '--checkpoint', str(checkpoint)]):
                parser.return_value.parse.return_value.as_dict.return_value = copy.deepcopy(config)
                infer_tool.main()
            manifest = json.loads((inference_dir / 'prediction_manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            for sample in ('a', 'b'):
                actual = np.asarray(Image.open(inference_dir / 'predictions' / 'masks' / f'{sample}.png'))
                expected = np.asarray(Image.open(root / 'tested' / 'predictions' / 'masks' / f'{sample}.png'))
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(0, actual[0, 0]); self.assertEqual(255, actual[0, 1])

    def test_cpu_full_checkpoint_restores_optimizer_scheduler_rng_best_and_guards(self):
        set_seed(42, True)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        def step(network, opt, sch):
            x = torch.randn(4, 3) + float(np.random.random()) + random.random()
            opt.zero_grad(set_to_none=True)
            network(x).square().mean().backward(); opt.step(); sch.step()
        step(model, optimizer, scheduler)
        payload = {'epoch': 1, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                   'scaler': scaler.state_dict(), 'rng_state': capture_rng_state(), 'config': {'training': {'seed': 42}},
                   'best_metric': .3, 'best_metric_name': 'foreground_miou', 'best_epoch': 1,
                   'early_stop_reference': .29, 'epochs_without_improvement': 2, 'collapse_count': 1, 'foreground_stall_count': 2}
        with tempfile.TemporaryDirectory() as tmp:
            writer = CheckpointWriter(asynchronous=True)
            writer.submit(tmp, model, payload, improved=True); writer.close()
            step(model, optimizer, scheduler)
            expected_model = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_scheduler = scheduler.state_dict()
            expected_rng = capture_rng_state()
            checkpoint = torch.load(Path(tmp) / 'last.pth', map_location='cpu', weights_only=False)
            before_validation = capture_rng_state()
            validate_training_checkpoint(checkpoint)
            after_validation = capture_rng_state()
            self.assertEqual(before_validation['python'], after_validation['python'])
            self.assertEqual(before_validation['numpy'], after_validation['numpy'])
            torch.testing.assert_close(before_validation['torch'], after_validation['torch'], rtol=0, atol=0)
            for a, b in zip(before_validation['cuda'], after_validation['cuda']):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            broken_states = []
            broken = copy.deepcopy(checkpoint); broken['optimizer']['state'][0]['exp_avg'].fill_(float('nan')); broken_states.append(broken)
            broken = copy.deepcopy(checkpoint); broken['optimizer']['param_groups'][0]['lr'] = float('inf'); broken_states.append(broken)
            broken = copy.deepcopy(checkpoint); broken['scaler'] = {'scale': float('nan')}; broken_states.append(broken)
            broken = copy.deepcopy(checkpoint); broken['scheduler']['last_epoch'] += 1; broken_states.append(broken)
            broken = copy.deepcopy(checkpoint); broken['rng_state']['python'] = (99, (), None); broken_states.append(broken)
            broken = copy.deepcopy(checkpoint); broken['rng_state']['numpy'][1][0] = -1; broken_states.append(broken)
            broken = copy.deepcopy(checkpoint); broken['rng_state']['torch'] = torch.zeros(8, dtype=torch.uint8); broken_states.append(broken)
            broken = copy.deepcopy(checkpoint); broken['rng_state']['cuda'] = [torch.zeros(1, dtype=torch.uint8)]; broken_states.append(broken)
            for broken in broken_states:
                with self.assertRaises(ValueError): validate_training_checkpoint(broken)
            after_failures = capture_rng_state()
            self.assertEqual(before_validation['python'], after_failures['python'])
            self.assertEqual(before_validation['numpy'], after_failures['numpy'])
            torch.testing.assert_close(before_validation['torch'], after_failures['torch'], rtol=0, atol=0)
            for a, b in zip(before_validation['cuda'], after_failures['cuda']):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            resumed = torch.nn.Linear(3, 2); resumed.load_state_dict(checkpoint['model'])
            resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=.01)
            resumed_sch = torch.optim.lr_scheduler.StepLR(resumed_opt, step_size=1, gamma=.8)
            resumed_opt.load_state_dict(checkpoint['optimizer']); resumed_sch.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            tracker = BestTracker(checkpoint['best_metric'], checkpoint['early_stop_reference'], checkpoint['epochs_without_improvement'])
            monitor = CollapseMonitor({}); monitor.count = checkpoint['collapse_count']; monitor.stall_count = checkpoint['foreground_stall_count']
            self.assertEqual((.3, .29, 2, 1, 2), (tracker.best, tracker.reference, tracker.stale, monitor.count, monitor.stall_count))
            restore_rng_state(checkpoint['rng_state']); step(resumed, resumed_opt, resumed_sch)
            for key, tensor in expected_model.items(): torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(expected_scheduler, resumed_sch.state_dict())
            for parameter, state in expected_optimizer['state'].items():
                for key, tensor in state.items(): torch.testing.assert_close(tensor, resumed_opt.state_dict()['state'][parameter][key], rtol=0, atol=0)
            actual_rng = capture_rng_state()
            torch.testing.assert_close(expected_rng['torch'], actual_rng['torch'], rtol=0, atol=0)
            self.assertEqual(expected_rng['python'], actual_rng['python']); self.assertEqual(expected_rng['numpy'], actual_rng['numpy'])
            for removed in ('optimizer', 'scheduler', 'rng_state', 'best_epoch', 'collapse_count'):
                broken = dict(checkpoint); broken.pop(removed)
                with self.assertRaises(ValueError): validate_training_checkpoint(broken)
            with self.assertRaises(ValueError): validate_training_checkpoint({**checkpoint, 'checkpoint_kind': 'inference'})


if __name__ == '__main__':
    unittest.main()
