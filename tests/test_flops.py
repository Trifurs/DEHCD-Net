from __future__ import annotations

import copy
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from utils.flops import FlopCounter, profile_forward, selective_scan_flops


class FlopTests(unittest.TestCase):
    def test_functional_grouped_and_transposed_convolution_include_bias(self):
        x = torch.randn(2, 4, 7, 9)
        w = torch.randn(6, 2, 3, 3)
        bias = torch.randn(6)
        with FlopCounter() as count:
            y = F.conv2d(x, w, bias, stride=2, groups=2)
        macs = y.numel() * 2 * 3 * 3
        self.assertEqual(2 * macs + y.numel(), count.flops)
        self.assertEqual(macs, count.matrix_macs)
        wt = torch.randn(4, 3, 3, 3)
        with FlopCounter() as count:
            z = F.conv_transpose2d(x, wt, torch.zeros(6), stride=2, groups=2)
        expected_macs = 2 * 7 * 9 * wt.numel()
        self.assertEqual(expected_macs, count.matrix_macs)
        self.assertEqual(2 * expected_macs + z.numel(), count.flops)

    def test_functional_linear_and_einsum_are_counted_once(self):
        x, w, b = torch.randn(2, 3), torch.randn(4, 3), torch.randn(4)
        with FlopCounter() as count:
            y = F.linear(x, w, b)
        self.assertEqual(2 * 2 * 3 * 4 + 2 * 4, count.flops)
        a, b = torch.randn(2, 3, 5), torch.randn(2, 5, 7)
        with FlopCounter() as count:
            torch.einsum('bik,bkj->bij', a, b)
        self.assertEqual(2 * 2 * 3 * 5 * 7, count.flops)
        self.assertEqual(2 * 3 * 5 * 7, count.matrix_macs)

    def test_fused_attention_counts_products_scaling_softmax_and_float_mask(self):
        q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
        scores, rows = 1 * 2 * 3 * 5, 1 * 2 * 3
        expected_macs = scores * (4 + 4)
        mask = torch.zeros(3, 5)
        with FlopCounter() as count:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertFalse(count.unsupported)
        self.assertEqual(expected_macs, count.matrix_macs)
        self.assertEqual(2 * expected_macs + 6 * scores - rows, count.flops)
        self.assertEqual((1, 2, 3, 4), y.shape)

    def test_norm_reduction_and_hog_scatter_scalar_counts(self):
        x = torch.randn(2, 4, 3, 3)
        with FlopCounter() as count:
            F.group_norm(x, 2, torch.ones(4), torch.zeros(4))
        self.assertEqual(7 * x.numel() + 2 * (2 * 2), count.flops)
        hist, index, source = torch.zeros(1, 6, 2, 2), torch.zeros(1, 1, 2, 2, dtype=torch.long), torch.ones(1, 1, 2, 2)
        with FlopCounter() as count:
            hist.scatter_add_(1, index, source)
        self.assertEqual(source.numel(), count.flops)
        values = torch.ones(3)
        with FlopCounter() as count:
            target = torch.zeros(5)
            target.index_put_((torch.tensor([0, 1, 1]),), values, accumulate=True)
        self.assertEqual(values.numel(), count.flops)
        with FlopCounter() as count:
            x.mean(dim=(2, 3))
        self.assertEqual(x.numel(), count.flops)

    def test_resampling_includes_blending_and_coordinates(self):
        x = torch.randn(1, 2, 4, 4)
        grid = torch.randn(1, 3, 5, 2)
        with FlopCounter() as count:
            y = F.grid_sample(x, grid, align_corners=True, padding_mode='border')
        self.assertEqual(7 * y.numel() + 14 * 3 * 5, count.flops)
        with FlopCounter() as count:
            y = F.interpolate(x, size=(3, 5), mode='bilinear', align_corners=False)
        self.assertEqual(7 * y.numel() + 4 * 3 * 5 + 5 * (3 + 5), count.flops)

    def test_scan_formula_equals_primitive_reference_and_wrapped_scan(self):
        from compare.official.changemamba.scan_backend import ReferenceScan, selective_scan_reference
        u = torch.randn(1, 4, 3)
        delta, A = torch.rand_like(u), -torch.rand(4, 2)
        b, c = torch.randn(1, 2, 2, 3), torch.randn(1, 2, 2, 3)
        d, bias = torch.randn(4), torch.randn(4)
        args = (u, delta, A, b, c, d, bias, True)
        with FlopCounter() as primitive:
            expected = selective_scan_reference(*args)
        self.assertFalse(primitive.unsupported)
        self.assertEqual(selective_scan_flops(*args), primitive.flops)
        self.assertEqual(12 * (8 * 2 - 1) + 12 * (2 + 1 + 3), primitive.flops)
        original = ReferenceScan.__dict__['apply']
        with FlopCounter() as count, count.opaque_kernels():
            actual = ReferenceScan.apply(*args)
        torch.testing.assert_close(actual, expected)
        self.assertIs(original, ReferenceScan.__dict__['apply'])
        self.assertEqual(primitive.flops, count.flops)
        self.assertEqual({'custom::selective_scan'}, set(count.operators))

    def test_unknown_arithmetic_never_publishes_a_complete_total(self):
        x = torch.ones(3)
        with FlopCounter() as count:
            torch.special.i0(x)
        report = count.report()
        self.assertFalse(report['complete'])
        self.assertIsNone(report['gflops'])
        self.assertTrue(report['unsupported_operators'])

    def test_profile_preserves_output_model_flags_parameters_rng_and_hooks(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.conv = nn.Conv2d(3, 4, 3, padding=1); self.drop = nn.Dropout()
            def forward(self, optical, sar):
                return {'logits': self.drop(self.conv(optical)) + sar}
        model = Model().train(); model.drop.eval()
        a, b = torch.randn(1, 3, 8, 8), torch.randn(1, 4, 8, 8)
        flags = [m.training for m in model.modules()]
        weights, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
        report = profile_forward(model, a, b)
        self.assertTrue(report['complexity']['complete'])
        self.assertEqual(flags, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertEqual(sum(r['flops'] for r in report['complexity']['operators'].values()), report['complexity']['flops'])
        self.assertEqual(sum(r['flops'] for r in report['complexity']['top_modules_by_flops']), report['complexity']['flops'])

    def test_profile_exception_restores_temporary_state(self):
        class Broken(nn.Module):
            def __init__(self):super().__init__(); self.child = nn.Identity()
            def forward(self, a, b):
                self.child(a)
                raise RuntimeError('intentional fixture failure')
        model = Broken().train()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            profile_forward(model, torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
        self.assertTrue(all(m.training and not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_testing_entry_point_and_declared_channel_contract(self):
        from tools.test import profile_model
        from tools.profile_models import input_channels
        from utils.config import XMLConfigParser
        root = Path(__file__).resolve().parents[1]
        channels = {'bright': (3, 1), 'haiti': (3, 4), 'xbd': (3, 3), 'cau_flood': (3, 1)}
        for name, expected in channels.items():
            cfg = XMLConfigParser(root / 'configs/datasets' / f'{name}.xml').parse().as_dict()
            self.assertEqual(expected, input_channels(cfg))
        with self.assertRaises(ValueError):input_channels({'model': {}, 'dataset': {'type': 'custom'}})
        self.assertEqual((7, 5), input_channels({}, optical=7, sar=5))
        class Pair(nn.Module):
            def forward(self, a, b):return a + b
        dataset = [{'optical': torch.zeros(1, 4, 4), 'sar': torch.zeros(1, 4, 4)}]
        report = profile_model(Pair(), dataset, 'cpu', SimpleNamespace(profile_size=[8, 8]))
        self.assertEqual(64, report['complexity']['flops'])
        self.assertEqual([1, 1, 8, 8], report['input_shape']['logits'])
        from tools.test import write_summary
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_summary([{'model_profile': report, 'metrics': {'f1': .8}}], output)
            row = next(csv.DictReader((output/'summary.csv').open()))
            self.assertAlmostEqual(64e-9, float(row['gflops']))
            self.assertEqual('True', row['flops_complete'])
            self.assertEqual(report['complexity']['convention'], row['flops_convention'])
            self.assertAlmostEqual(.8, float(row['f1']))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_scan_and_fused_attention_match_analytical_counts(self):
        from compare.official.changemamba import vmamba
        if not hasattr(vmamba, 'selective_scan_cuda_oflex'):
            self.skipTest('scan extension required')
        u = torch.randn(1, 4, 8, device='cuda')
        args = (u, torch.rand_like(u), -torch.rand(4, 2, device='cuda'),
                torch.randn(1, 2, 2, 8, device='cuda'), torch.randn(1, 2, 2, 8, device='cuda'),
                torch.randn(4, device='cuda'), torch.randn(4, device='cuda'), True)
        with FlopCounter() as count, count.opaque_kernels():
            vmamba.SelectiveScanOflex.apply(*args)
        self.assertEqual(selective_scan_flops(*args), count.flops)
        self.assertEqual(1, count.operators['custom::selective_scan']['calls'])
        q = torch.randn(1, 2, 3, 4, device='cuda')
        with FlopCounter() as attention:
            F.scaled_dot_product_attention(q, q, q)
        self.assertFalse(attention.unsupported)
        # FP32 may use the decomposed math backend, whose scaling is performed
        # on Q/K rather than on scores; both remain fully accounted.
        self.assertGreater(attention.flops, 2 * 1 * 2 * 3 * 3 * (4 + 4))


if __name__ == '__main__':
    unittest.main()
