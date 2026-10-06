from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from models import build_model
from models.fusion import PlainFusionBlock
from utils.config import XMLConfigParser
from utils.losses import segmentation_loss
from utils.metrics import ConfusionMatrixMeter
from utils.prediction import predict_logits
from utils.protocol import assert_disjoint, config_digest, verify_checkpoint_config
from utils.reproducibility import BestTracker, set_seed
from utils.split_manifest import read_manifest
from utils.training_health import CollapseMonitor
from tools.make_event_split import assign_events
from tools.validate_results import validate_result

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ExperimentTests(unittest.TestCase):
    def test_catalog_and_ablation_semantics(self):
        catalog = json.loads((ROOT / "configs/experiments/catalog.json").read_text())["experiments"]
        self.assertEqual(42, sum(r["suite"] == "main" for r in catalog))
        self.assertEqual(len(catalog), len({r["id"] for r in catalog}))
        for row in catalog:
            cfg = XMLConfigParser(ROOT / row["config"]).parse().as_dict()
            self.assertEqual(row["id"], cfg["experiment"]["id"])
            expected_patience = 100 if cfg["experiment"]["dataset"] == "haiti" else 30
            self.assertEqual(expected_patience, cfg["training"]["early_stop_patience"])
            self.assertFalse(cfg["training"]["amp"], "All experiment models must share FP32 precision")
        base = XMLConfigParser(ROOT / "configs/dehcd/haiti_s.xml").parse().as_dict()
        self.assertEqual("class_mean", base["training"]["hier_binary_reduction"])
        self.assertEqual([1., 1.], base["training"]["hier_binary_class_weights"])
        for switches, flow, gate in (({}, True, True), ({"align_fusion": False}, False, True),
                                    ({"difference_gate": False}, True, False)):
            cfg = copy.deepcopy(base); cfg["model"].update(switches)
            model = build_model(cfg, 3, 4)
            self.assertTrue(all(m.eps == .001 for m in model.modules() if isinstance(m, torch.nn.GroupNorm)))
            self.assertEqual(flow, model.fusion_blocks[0].flow_head is not None)
            self.assertEqual(gate, model.fusion_blocks[0].change_gate is not None)
            output = model(torch.randn(2, 3, 32, 32), torch.randn(2, 4, 32, 32))
            self.assertEqual((2, 4, 32, 32), tuple(output.shape))

    def test_capacity_control_uses_all_parameters(self):
        model = PlainFusionBlock(32, matched=True, adaptive_modality_weight=True)
        self.assertLess(model.relative_parameter_error, .01)
        model(torch.randn(2, 32, 8, 8), torch.randn(2, 32, 8, 8)).square().mean().backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_haiti_loss_large_half_precision_and_empty_masks(self):
        cfg = XMLConfigParser(ROOT / "configs/datasets/haiti.xml").parse()["training"]
        set_seed(42)
        logits = torch.randn(24, 4, 128, 128, dtype=torch.float16, requires_grad=True)
        labels = torch.randint(0, 4, (24, 128, 128))
        loss = segmentation_loss(logits, labels, cfg, 4)
        self.assertEqual(torch.float32, loss.dtype)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        for recipe in (cfg, {"loss": "compound"}):
            x = torch.randn(1, 4, 8, 8, requires_grad=True)
            loss = segmentation_loss(x, torch.full((1, 8, 8), 255), recipe, 4)
            self.assertEqual(0., loss.item()); loss.backward()
            self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_unknown_loss_rejected(self):
        with self.assertRaises(ValueError):
            segmentation_loss(torch.randn(1, 2, 4, 4), torch.zeros(1, 4, 4, dtype=torch.long), {"loss": "typo"}, 2)

    def test_tta_inverse_on_rectangular_images(self):
        class Pointwise(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat([x[:, :1] + y, x[:, 1:2] - y], 1)
        x, y = torch.randn(2, 3, 7, 11), torch.randn(2, 1, 7, 11)
        expected = Pointwise()(x, y)
        for mode in ("none", "flips", "d4"):
            torch.testing.assert_close(expected, predict_logits(Pointwise(), x, y, False, mode))

    def test_best_is_independent_of_early_stop_delta(self):
        tracker = BestTracker()
        self.assertTrue(tracker.update(.5, .01))
        self.assertTrue(tracker.update(.501, .01))
        self.assertEqual(.501, tracker.best)
        self.assertEqual(1, tracker.stale)
        self.assertTrue(tracker.update(.52, .01))
        self.assertEqual(0, tracker.stale)

    def test_foreground_collapse_guard(self):
        monitor = CollapseMonitor({"collapse_patience": 2, "collapse_warmup_epochs": 1})
        metrics = {"target_foreground_ratio": .1, "pred_foreground_ratio": 0.}
        monitor.update(20, .5, 0., metrics)
        with self.assertRaisesRegex(RuntimeError, "Foreground collapse"):
            monitor.update(21, .5, 0., metrics)

    def test_stalled_foreground_is_detected_without_a_good_previous_score(self):
        monitor = CollapseMonitor({"foreground_stall_patience": 2, "foreground_stall_warmup_epochs": 40})
        metrics = {"target_foreground_ratio": .05, "pred_foreground_ratio": 0.}
        monitor.update(39, .001, 0., metrics)
        self.assertEqual(0, monitor.stall_count)
        monitor.update(40, .001, 0., metrics)
        with self.assertRaisesRegex(RuntimeError, "learning stalled"):
            monitor.update(41, .001, 0., metrics)

    def test_balanced_hierarchical_ce_is_invariant_to_background_replication(self):
        # Identical class scores, one foreground pixel and repeated background.
        cfg = {"loss": "hierarchical_change", "hier_binary_reduction": "class_mean",
               "hier_binary_class_weights": [1., 1.], "hier_binary_dice_weight": 0.,
               "hier_subclass_ce_weight": 0., "hier_subclass_dice_weight": 0.}
        values = []
        for background_count in (1, 30):
            x = torch.tensor([.4, .1, -.3, .2]).reshape(1,4,1,1).repeat(1,1,1,background_count+1).requires_grad_()
            y = torch.zeros(1,1,background_count+1,dtype=torch.long);y[0,0,-1]=2
            loss = segmentation_loss(x,y,cfg,4);values.append(loss.detach());loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum(),0.)
        torch.testing.assert_close(values[0],values[1])

    def test_raw_confusion_verification_and_validation_rejection(self):
        meter = ConfusionMatrixMeter(3)
        meter.matrix = torch.tensor([[20, 0, 0], [0, 0, 0], [1, 0, 3]], dtype=torch.float64)
        result = {"schema_version": 1, "split": "test", "confusion_matrix": meter.matrix.tolist(), "metrics": meter.compute()}
        validate_result(result)
        self.assertEqual(24., result["metrics"]["valid_pixels"])
        result["metrics"]["mean_iou"] += .1
        with self.assertRaises(ValueError): validate_result(result)
        result["split"] = "val"
        with self.assertRaisesRegex(ValueError, "split=test"): validate_result(result)
        self.assertEqual(0., ConfusionMatrixMeter(4).compute()["valid_pixels"])
        with self.assertRaises(ValueError):
            ConfusionMatrixMeter(2).update(torch.tensor([2]), torch.tensor([1]))

    def test_manifest_leakage_and_event_assignment(self):
        rows = [{"id": str(i), "source_split": "train", "split": "train", "group": f"g{i}", "event": f"e{i}"} for i in range(3)]
        assigned = assign_events(rows, {"e2"}, {"e1"})
        self.assertEqual(["train", "val", "test"], [r["split"] for r in assigned])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            def write(values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(values)
            write(assigned); self.assertEqual(3, len(read_manifest(path)))
            assigned[1]["group"] = assigned[0]["group"]; write(assigned)
            with self.assertRaises(ValueError): read_manifest(path)
        record = {"id": "tile", "group": None, "files": [{"path": "/same/file"}]}
        with self.assertRaises(ValueError):
            assert_disjoint({"train": {"records": [record]}, "test": {"records": [record]}})

    def test_config_identity(self):
        cfg = {"model": {"hog_bins": 6}, "task": {}, "training": {"seed": 42}}
        moved = copy.deepcopy(cfg); moved["logging"] = {"run_dir": "/new/path"}
        self.assertEqual(config_digest(cfg), config_digest(moved))
        moved["model"]["hog_bins"] = 10
        self.assertNotEqual(config_digest(cfg), config_digest(moved))
        with self.assertRaises(ValueError): verify_checkpoint_config({"config": cfg}, moved)

    def test_binary_adapter_preserves_sigmoid_probability(self):
        from compare.official_adapters import OfficialCompareAdapter
        adapter = OfficialCompareAdapter(3, 1, num_classes=2, output_channels=1)
        z = torch.tensor([[[[-2., 0., 2.]]]])
        output = adapter._finish_logits(z, (1, 3))
        torch.testing.assert_close(torch.softmax(output, 1)[:, 1], torch.sigmoid(z[:, 0]))

    def test_background_crops_are_retained_by_default(self):
        import numpy as np
        from PIL import Image
        from tools.dataset_tools.xbd_split_crop_1024_to_256 import crop_partition, Sample
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "background.png"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
            sample = Sample("scene", "event", "tier3", path, path, path)
            common = dict(samples=[sample], dst_root=Path(tmp), image_cls=Image,
                          patch_size=2, source_size=8, ignore_values={5, 255}, min_foreground_pixels=1,
                          dry_run=True, tqdm=lambda values, **kwargs: values)
            kept = crop_partition(split="test", **common)
            self.assertEqual(16, kept["kept_patches"])
            filtered = crop_partition(split="train", drop_background=True, **common)
            self.assertEqual(0, filtered["kept_patches"])


class OptimizerSafetyTests(unittest.TestCase):
    def train(self, model, batches, optimizer, accumulation=1):
        from tools.train import train_one_epoch
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        cfg = {"task": {"num_classes": 2}, "training": {"amp": False, "loss": "compound",
               "ce_weight": 1., "dice_weight": 0., "grad_clip_norm": 0., "gradient_accumulation_steps": accumulation}}
        return train_one_epoch(model, batches, optimizer, scaler, torch.device("cpu"), cfg, 1)

    def make_model(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.conv = torch.nn.Conv2d(1, 2, 1)
            def forward(self, optical, sar): return self.conv(optical)
        return Toy()

    def test_partial_accumulation_window_has_correct_scale(self):
        set_seed(4); model = self.make_model(); reference = copy.deepcopy(model)
        batches = [{"optical": torch.randn(1, 1, 2, 2), "sar": torch.zeros(1, 1, 2, 2),
                    "label": torch.randint(0, 2, (1, 2, 2)), "id": [str(i)]} for i in range(3)]
        opt = torch.optim.SGD(model.parameters(), lr=.1); ref_opt = torch.optim.SGD(reference.parameters(), lr=.1)
        self.train(model, batches, opt, 2)
        for window in (batches[:2], batches[2:]):
            ref_opt.zero_grad()
            loss = sum(torch.nn.functional.cross_entropy(reference(b["optical"], b["sar"]), b["label"]) for b in window) / len(window)
            loss.backward(); ref_opt.step()
        for p, q in zip(model.parameters(), reference.parameters()): torch.testing.assert_close(p, q)

    def test_nonfinite_gradient_does_not_change_parameters(self):
        model = self.make_model(); before = copy.deepcopy(model.state_dict())
        model.conv.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        batches = [{"optical": torch.ones(1, 1, 2, 2), "sar": torch.ones(1, 1, 2, 2), "label": torch.zeros(1, 2, 2, dtype=torch.long)}]
        with self.assertRaises(FloatingPointError): self.train(model, batches, torch.optim.SGD(model.parameters(), lr=.1))
        for name, value in model.state_dict().items(): torch.testing.assert_close(value, before[name])


if __name__ == "__main__":
    unittest.main()
