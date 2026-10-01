import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from models import build_model
from tools.export_features import FeatureCapture, select_samples, write_features


class FeatureExportTests(unittest.TestCase):
    def test_hooks_preserve_output_and_capture_real_boundaries(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        cfg = {"model": {"backbone": "dehcd_s", "base_channels": 16,
               "norm_type": "group", "use_hog": True, "hog_modulation_levels": 2,
               "diffusion_steps": 3, "global_context": True, "cross_scale_fusion": True},
               "task": {"num_classes": 4}}
        model = build_model(cfg, 3, 1).eval()
        optical, sar = torch.randn(1, 3, 16, 16), torch.randn(1, 1, 16, 16)
        with torch.inference_mode():
            expected = model(optical, sar)
            with FeatureCapture(model) as capture:
                actual = model(optical, sar)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertIn("optical_hog_modulators.0__before", capture.features)
        self.assertIn("sar_hog_modulators.0__after", capture.features)
        self.assertIn("fusion_blocks.0__before__0", capture.features)
        self.assertIn("fusion_blocks.0__before__1", capture.features)
        self.assertIn("fusion_blocks.0__after", capture.features)
        self.assertIn("global_context__after__3", capture.features)
        self.assertIn("cross_scale_fusion__after__3", capture.features)
        self.assertIn("diffusion_refine__after", capture.features)
        self.assertTrue(all(np.isfinite(array).all() for array in capture.features.values()))
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in capture.modules.values()))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with FeatureCapture(model) as interrupted:
                raise RuntimeError("intentional")
        self.assertFalse(interrupted.handles)

    def test_fixed_samples_are_lexical_unique_and_restricted(self):
        samples = [{"id": f"tile_{i:02}"} for i in reversed(range(20))]
        pool, selected = select_samples(samples, count=2)
        self.assertEqual(["tile_00", "tile_01"], [name for name, _ in selected])
        self.assertEqual(16, len(pool))
        with self.assertRaises(ValueError):
            select_samples(samples, ["tile_19"])
        with self.assertRaises(ValueError):
            select_samples(samples, ["tile_01", "tile_01"])

    def test_raw_features_and_display_scale_are_separate(self):
        feature = np.arange(24, dtype=np.float32).reshape(3, 2, 4) - 10
        with tempfile.TemporaryDirectory() as temp:
            rows = write_features(temp, {"module__before": feature})
            np.testing.assert_array_equal(feature, np.load(Path(temp) / rows[0]["raw_tensor"], allow_pickle=False))
            self.assertEqual(float(np.abs(feature).mean(0).max()), rows[0]["response_max"])
            self.assertTrue((Path(temp) / rows[0]["response_png"]).is_file())


if __name__ == "__main__":
    unittest.main()
