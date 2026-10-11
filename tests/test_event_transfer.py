import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools import run_event_transfer as runner
from utils.config import XMLConfigParser
from utils.metrics import ConfusionMatrixMeter
from utils.protocol import atomic_json, config_digest, file_digest


class EventTransferTests(unittest.TestCase):
    def test_sizes_train_then_test_each_event_without_combined_test(self):
        with tempfile.TemporaryDirectory() as temp:
            arguments = ['run_event_transfer.py', '--seeds', '3402', '--output', temp,
                         '--data-root', temp, '--cache', 'none']
            with patch('sys.argv', arguments), patch.object(runner.subprocess, 'run',
                    return_value=SimpleNamespace(returncode=0)) as children, \
                    patch.object(runner, 'event_summary') as summary, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
            calls = [call.args[0] for call in children.call_args_list]
            self.assertEqual(12, len(calls))
            for index, size in enumerate(('s', 'm', 'l')):
                campaign, *tests = calls[index * 4:(index + 1) * 4]
                self.assertIn('--skip-test', campaign)
                self.assertEqual(str(runner.CONFIGS[size]), campaign[campaign.index('--config') + 1])
                self.assertEqual(runner.DEFAULT_TEST_EVENTS,
                                 [command[command.index('--event') + 1] for command in tests])
                self.assertTrue(all('--skip-existing' in command for command in tests))
            summary.assert_called_once_with(Path(temp), ['s', 'm', 'l'], [3402], runner.DEFAULT_TEST_EVENTS)

    def test_preflight_never_starts_event_evaluation(self):
        with tempfile.TemporaryDirectory() as temp, patch('sys.argv', [
                'run_event_transfer.py', '--seeds', '3402', '--preflight-only', '--output', temp,
                '--data-root', temp]), patch.object(runner.subprocess, 'run',
                return_value=SimpleNamespace(returncode=0)) as children, \
                patch.object(runner, 'event_summary') as summary, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, runner.main())
            self.assertEqual(3, children.call_count)
            self.assertTrue(all('--preflight-only' in call.args[0] for call in children.call_args_list))
            summary.assert_not_called()

    def test_summary_keeps_model_sizes_separate_and_single_seed_std_null(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            event, seed = 'haiti-earthquake', 3402
            scores = {}
            for size, predictions in [('s', [0, 0, 0, 0]), ('m', [0, 1, 1, 1]), ('l', [0, 1, 2, 3])]:
                config = XMLConfigParser(runner.CONFIGS[size]).parse().as_dict()
                config['training']['seed'] = seed
                run = runner.run_directory(output, size, seed)
                atomic_json(run / 'config_snapshot.json', config)
                checkpoint = run / 'checkpoints/best.pth'
                checkpoint.parent.mkdir(parents=True)
                torch.save({'model': {}}, checkpoint)
                meter = ConfusionMatrixMeter(4).update(torch.tensor(predictions), torch.arange(4))
                metrics = meter.compute()
                result = {'schema_version': 2, 'split': 'test', 'status': 'complete', 'complete': True,
                    'seed': seed, 'evaluation_event': event, 'config_sha256': config_digest(config),
                    'checkpoint': str(checkpoint), 'checkpoint_sha256': file_digest(checkpoint),
                    'dataset_identity': {'samples': 4}, 'expected_samples': 4, 'evaluated_samples': 4,
                    'runtime': {'max_batches': 0, 'prediction_rule': 'damage_argmax',
                                'test_time_augmentation': 'none'},
                    'confusion_matrix': meter.matrix.tolist(), 'metrics': metrics}
                atomic_json(run / 'test' / event / 'result.json', result)
                scores[size] = metrics['foreground_miou']
            runner.event_summary(output, ['s', 'm', 'l'], [seed], [event])
            summary = json.loads((output / 'summary/event_metrics.json').read_text())
            for size in scores:
                metric = summary['models'][size]['events'][event]['metrics']['foreground_miou']
                self.assertEqual(scores[size], metric['mean'])
                self.assertIsNone(metric['sample_std'])
                self.assertEqual(1, metric['n'])

    def test_precision_is_explicit_and_custom_config_keeps_fast_runtime(self):
        from tools.run_multiseed import make_plan, parse_args
        from utils.runtime import autocast_dtype
        for size in ('s', 'm', 'l'):
            with patch('sys.argv', ['run_multiseed.py', '--config', str(runner.CONFIGS[size]),
                    '--output', '/tmp/event-plan', '--seeds', '3402']):
                config = make_plan(parse_args())['jobs'][0]['config']
            self.assertFalse(config['training']['deterministic'])
            self.assertEqual(torch.bfloat16, autocast_dtype(config['training']))
            self.assertEqual('dehcd_' + size, config['model']['backbone'])
        self.assertEqual(torch.float16, autocast_dtype({}))
        with self.assertRaises(ValueError):
            autocast_dtype({'amp_dtype': 'invalid'})


if __name__ == '__main__':
    unittest.main()
