import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models import build_model
from utils.comparison_protocol import audit_plan, comparability, pair_policy, validate_config, validate_design
from utils.config import XMLConfigParser
from utils.losses import segmentation_loss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ComparisonProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = json.loads((ROOT / 'configs/experiments/catalog.json').read_text())['experiments']
        cls.configs = {r['id']: XMLConfigParser(ROOT / r['config']).parse().as_dict() for r in rows}

    def plan(self, *names):
        jobs = []
        for name in names:
            for seed in (42, 1051):
                config = copy.deepcopy(self.configs[name]); config['training']['seed'] = seed
                jobs.append({'experiment': name, 'dataset': config['experiment']['dataset'], 'seed': seed,
                             'data_key': 'same', 'config': config})
        return {'jobs': jobs, 'data': {'same': {'splits': {
            split: {'sha256': split, 'mode': 'sha256'} for split in ('train', 'val', 'test')}}}}

    def test_every_declared_control_holds_other_settings_fixed(self):
        evidence = validate_design(self.configs)
        self.assertEqual(76, len(evidence))  # four dataset roots have no parent
        edges = {row['candidate']: row for row in evidence}
        self.assertEqual(['model.diffusion_steps'], list(edges['bright_m_irb_steps_0']['actual_changes']))
        self.assertEqual('bright_dehcd_m', edges['bright_m_irb_steps_0']['reference'])
        self.assertEqual(['model.hog_modulation_levels'], list(edges['bright_m_hog_levels_4']['actual_changes']))
        self.assertEqual({'model.use_hog', 'model.fusion_mode'}, set(edges['haiti_no_hog_no_dpm']['actual_changes']))
        self.assertEqual({'model.gcb_mode', 'model.bicsf_mode'}, set(edges['bright_bicsf_conv_matched']['actual_changes']))
        self.assertEqual(['training.class_balanced_sampler'], list(edges['bright_no_weighted_sampler']['actual_changes']))
        self.assertEqual(['training.class_weights'], list(edges['bright_no_class_weights']['actual_changes']))

    def test_complete_bicsf_ablation_and_capacity_control(self):
        for ds, channels in (('bright', (3, 1)), ('haiti', (3, 4))):
            for suffix in ('no_bicsf', 'no_hog_no_dpm_no_bicsf', 'all_off'):
                with self.subTest(dataset=ds, ablation=suffix):
                    model = build_model(self.configs[f'{ds}_{suffix}'], *channels)
                    self.assertIsInstance(model.global_context, torch.nn.Identity)
                    self.assertIsInstance(model.cross_scale_fusion, torch.nn.Identity)
                    out = model(torch.randn(2, channels[0], 32, 32), torch.randn(2, channels[1], 32, 32))
                    self.assertEqual((2, 4, 32, 32), tuple(out.shape))
        model = build_model(self.configs['bright_bicsf_conv_matched'], 3, 1)
        self.assertEqual('IndependentScaleConvControl', type(model.global_context).__name__)
        self.assertEqual('IndependentScaleConvControl', type(model.cross_scale_fusion).__name__)
        model(torch.randn(2, 3, 32, 32), torch.randn(2, 1, 32, 32)).square().mean().backward()
        for module in (model.global_context, model.cross_scale_fusion):
            self.assertLess(abs(module.relative_parameter_error), .05)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertTrue(all(p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_identical_primary_supervision_for_all_42_main_models(self):
        for ds in ('bright', 'haiti', 'cau_flood', 'xbd'):
            reference = self.configs[f'{ds}_dehcd_l']
            classes = reference['task']['num_classes']
            torch.manual_seed(17)
            initial = torch.randn(2, classes, 8, 8)
            labels = torch.randint(classes, (2, 8, 8)); labels[:, :1] = 255
            logits = initial.clone().requires_grad_()
            expected = segmentation_loss(logits, labels, reference['training'], classes)
            expected_grad, = torch.autograd.grad(expected, logits)
            for name, cfg in self.configs.items():
                if cfg['experiment']['suite'] != 'main' or cfg['experiment']['dataset'] != ds:
                    continue
                with self.subTest(experiment=name):
                    logits = initial.clone().requires_grad_()
                    loc = torch.full((2, 2, 8, 8), float('nan'), requires_grad=True)
                    loss = segmentation_loss({'logits': logits, 'localization_logits': loc}, labels, cfg['training'], classes)
                    grad, loc_grad = torch.autograd.grad(loss, (logits, loc), allow_unused=True)
                    torch.testing.assert_close(loss, expected)
                    torch.testing.assert_close(grad, expected_grad)
                    self.assertIsNone(loc_grad)
            loc = torch.full((2, 2, 8, 8), float('nan'), requires_grad=True)
            empty = segmentation_loss({'logits': logits, 'localization_logits': loc}, torch.full_like(labels, 255), reference['training'], classes)
            self.assertEqual(0, empty.item())

    def test_m_scans_cannot_be_relabelled_l_controls(self):
        for name in ('bright_m_irb_steps_0', 'bright_m_hog_levels_4'):
            cfg = self.configs[name]
            self.assertEqual('dehcd_m', cfg['model']['backbone'])
            self.assertEqual(24, cfg['model']['base_channels'])
            self.assertEqual('bright_dehcd_m', cfg['experiment']['comparison']['reference'])
            plan = self.plan('bright_dehcd_m', name)
            comparability(plan, plan['jobs'])
            wrong = self.plan('bright_dehcd_l', name)
            with self.assertRaisesRegex(ValueError, 'declared direct control'):
                comparability(wrong, wrong['jobs'])
        self.assertEqual('dehcd_l', self.configs['bright_no_irb']['model']['backbone'])

    def test_main_architectures_are_comparable_but_undeclared_recipe_drift_is_not(self):
        plan = self.plan('bright_dehcd_l', 'bright_changeos', 'bright_damageformer', 'bright_changemamba')
        self.assertEqual([42, 1051], comparability(plan, plan['jobs'])['paired_seeds'])
        for section, key, value in (('training', 'learning_rate', .003), ('training', 'class_weights', []),
                ('training', 'class_balanced_sampler', False), ('training', 'batch_size', 1),
                ('dataset', 'label_quality_ignore_policy', 'all_modalities'), ('dataset', 'patch_size', 128),
                ('dataset', 'eval_full_image', True), ('augmentation', 'random_flip', False),
                ('model', 'dropout', .22)):
            altered = copy.deepcopy(plan)
            for job in altered['jobs'][2:4]:
                job['config'][section][key] = value
            with self.subTest(field=f'{section}.{key}'), self.assertRaises(ValueError):
                comparability(altered, altered['jobs'])

    def test_train_validation_and_test_identity_and_seed_sets_are_mandatory(self):
        for split in ('train', 'val', 'test'):
            plan = self.plan('bright_dehcd_l', 'bright_changeos')
            plan['data']['other'] = copy.deepcopy(plan['data']['same'])
            plan['data']['other']['splits'][split]['sha256'] = 'different'
            for job in plan['jobs'][2:]: job['data_key'] = 'other'
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'identical train'):
                comparability(plan, plan['jobs'])
        plan = self.plan('bright_dehcd_l', 'bright_changeos')
        with self.assertRaisesRegex(ValueError, 'seed sets'):
            comparability(plan, plan['jobs'][:-1])

    def test_recipe_controls_require_their_declared_full_reference(self):
        for name in ('bright_ce_dice_full_sampling', 'bright_no_weighted_sampler', 'bright_no_class_weights'):
            plan = self.plan('bright_dehcd_l', name)
            comparability(plan, plan['jobs'])
            wrong = self.plan('bright_dehcd_m', name)
            with self.assertRaisesRegex(ValueError, 'declared direct control'):
                comparability(wrong, wrong['jobs'])
        plan = self.plan('bright_ce_dice_full_sampling', 'bright_no_weighted_sampler')
        with self.assertRaisesRegex(ValueError, 'declared direct control'):
            comparability(plan, plan['jobs'])

    def test_preflight_and_single_training_reject_unfair_main_settings(self):
        for section, key, value in (('model', 'encoder_checkpoint', '/tmp/weights.pth'),
                ('model', 'pretrained_backbone', True), ('training', 'amp', True),
                ('training', 'localization_loss_weight', 1.), ('training', 'early_stop_patience', 30),
                ('training', 'best_metric', 'oa'), ('inference', 'test_time_augmentation', 'flips')):
            cfg = copy.deepcopy(self.configs['bright_changeos']); cfg[section][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError): validate_config(cfg)
        plan = self.plan('bright_no_bicsf')
        plan['comparison_references'] = {'bright_dehcd_l': self.configs['bright_dehcd_l']}
        audit_plan(plan)
        for job in plan['jobs']: job['config']['training']['learning_rate'] = .003
        with self.assertRaisesRegex(ValueError, 'training.learning_rate'): audit_plan(plan)

    def test_missing_design_and_effective_adapter_resolution_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Missing declared'):
            pair_policy({'model': {}}, {'model': {}})
        from utils.model_metadata import model_metadata
        module = torch.nn.Conv2d(3, 4, 1)
        module.resize_to = 64
        info = model_metadata(module, 'cpu')
        self.assertEqual([64, 64], info['adapter_spatial_transform']['core_input_size'])

    def test_multiple_analysis_views_make_unique_seed_tasks(self):
        from tools.run_multiseed import make_plan
        args = SimpleNamespace(catalog=str(ROOT / 'configs/experiments/catalog.json'), config=None, experiments=None,
            seeds=[42, 1051, 2060], datasets=['bright', 'haiti', 'cau_flood', 'xbd'],
            suite=['main', 'ablation', 'sensitivity', 'scaling'], data_root=[], manifest=[],
            encoder_checkpoint=[], device='cpu', num_workers=0, epochs=None, batch_size=None,
            gradient_accumulation_steps=None, fingerprint='stat')
        plan = make_plan(args)
        self.assertEqual(213, len(plan['jobs']))
        keys = {(job['experiment'], job['seed']) for job in plan['jobs']}
        self.assertEqual(len(plan['jobs']), len(keys))
        self.assertEqual(3, sum(job['experiment'] == 'bright_dehcd_m' for job in plan['jobs']))
        self.assertEqual(3, sum(job['experiment'] == 'haiti_dehcd_l' for job in plan['jobs']))
        self.assertEqual(['bright', 'haiti', 'xbd', 'cau_flood'], list(dict.fromkeys(job['dataset'] for job in plan['jobs'])))


if __name__ == '__main__':
    unittest.main()
