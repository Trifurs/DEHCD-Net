import copy
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from datasets import build_dataset
from tools.train import compute_label_distribution, build_class_balanced_sampler
from utils.checkpoint import CheckpointWriter
from utils.metrics import ConfusionMatrixMeter
from utils.prepared_data import input_signature, preprocessing_identity, prepare_split
from utils.progress import RunProgress
from utils.protocol import atomic_json, dataset_identity
from utils.reproducibility import set_seed
from utils.campaign_layout import run_path, result_path, summary_path


class PreparedInputTests(unittest.TestCase):
    def make_data(self, root):
        import rasterio
        rng = np.random.default_rng(314)
        folders = {"optical": "Pre_event/S2_20210804", "sar0": "Post_event/S1_ASC_20210817",
                   "sar1": "Post_event/S1_DESC_20210815", "label": "Annotations"}
        for i in range(3):
            for name, folder in folders.items():
                h, w = (11 + i, 13 + i) if name != "label" else (9 + i, 10 + i)
                channels = 4 if name == "optical" else (1 if name == "label" else 3)
                array = rng.normal(20, 4, (channels, h, w)).astype("float32")
                if name == "label": array = rng.integers(0, 4, (1, h, w)).astype("float32")
                else: array[-1] = rng.integers(0, 2, (h, w))
                path = root / "train" / folder / f"{i}.tif"
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=channels, dtype="float32") as dst:
                    dst.write(array)
        return {"dataset": {"type": "haiti", "root": str(root), "patch_size": 8},
                "task": {"num_classes": 4, "ignore_index": 255},
                "augmentation": {"enabled": True, "optical_noise_std": .02, "sar_noise_std": .02,
                                 "random_flip": True, "random_rotate90": True},
                "training": {"class_balanced_sampler": True, "sampler_target_classes": [1, 2, 3]}}

    def test_cache_preserves_haiti_masks_rng_crops_stats_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEHCD_PREPARED_INDEX", None)
            root = Path(tmp); cfg = self.make_data(root / "source")
            raw = build_dataset(cfg, "train", True)
            identity = dataset_identity(raw)
            expected_stats = compute_label_distribution(raw, 4, 255)
            expected_sampler = build_class_balanced_sampler(raw, cfg["training"], 4, 255)
            path = prepare_split(raw, {"stat": identity}, root / "cache", workers=0)
            index = root / "index.json"
            atomic_json(index, {"preprocessing_sha256": preprocessing_identity(), "splits": {input_signature(raw): path}})
            os.environ["DEHCD_PREPARED_INDEX"] = str(index)
            with patch.object(type(raw), "_build_index", side_effect=AssertionError("rescanned source")):
                cached = build_dataset(cfg, "train", True)
            for seed in (42, 1051, 2060):
                for i in range(len(raw)):
                    set_seed(seed); a = raw[i]
                    expected_rng = torch.get_rng_state()
                    set_seed(seed); b = cached[i]
                    for field in ("optical", "sar", "label"):
                        self.assertTrue(torch.equal(a[field], b[field]), (seed, i, field))
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertEqual(identity, dataset_identity(cached))
            with patch.object(type(cached), "load_label_for_stats", side_effect=AssertionError("reread labels")):
                actual_stats = compute_label_distribution(cached, 4, 255)
                actual_sampler = build_class_balanced_sampler(cached, cfg["training"], 4, 255)
            np.testing.assert_array_equal(expected_stats["sample_counts"], actual_stats["sample_counts"])
            torch.testing.assert_close(expected_sampler.weights, actual_sampler.weights, rtol=0, atol=0)
            base = cached._prepared_split.get(0)
            expected = base["optical"].clone(); base["optical"].fill_(1000)
            self.assertTrue(torch.equal(expected, cached._prepared_split.get(0)["optical"]))
            changed = copy.deepcopy(cfg); changed["dataset"]["masked_input_policy"] = "zero"
            with self.assertRaisesRegex(ValueError, "No prepared input"):
                build_dataset(changed, "train", True)
            os.environ.pop("DEHCD_PREPARED_INDEX")
            source = Path(raw.samples[0]["label"])
            stat = source.stat(); os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
            self.assertNotEqual(identity["sha256"], dataset_identity(raw)["sha256"])

    def test_corrupted_generated_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEHCD_PREPARED_INDEX", None)
            root = Path(tmp); cfg = self.make_data(root / "source")
            raw = build_dataset(cfg, "train", False)
            identities = {"stat": dataset_identity(raw)}
            manifest = Path(prepare_split(raw, identities, root / "cache", workers=0))
            data = manifest.parent / "optical.bin"
            before = data.read_bytes()
            data.write_bytes(b"broken")
            prepare_split(raw, identities, root / "cache", workers=0)
            self.assertEqual(before, data.read_bytes())


class WorkflowTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_confusion_counts_and_metrics_exactly_equal_cpu(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        for classes in (2, 4, 5):
            for all_background in (False, True):
                target = torch.randint(-1, classes + 1, (4, 256, 256))
                target.view(-1)[::9] = 255
                pred = torch.zeros_like(target) if all_background else torch.randint(classes, target.shape)
                cpu = ConfusionMatrixMeter(classes).update(pred, target)
                gpu = ConfusionMatrixMeter(classes, device=torch.device("cuda")).update(pred.cuda(), target.cuda())
                self.assertTrue(torch.equal(cpu.matrix, gpu.matrix.cpu()))
                self.assertEqual(cpu.compute(), gpu.compute())
        with self.assertRaises(ValueError):
            ConfusionMatrixMeter(2, device=torch.device("cuda")).update(torch.tensor([2], device="cuda"), torch.tensor([1], device="cuda"))

    def test_async_checkpoint_is_immutable_and_resume_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            model(torch.ones(2, 3)).sum().backward(); optimizer.step()
            expected = copy.deepcopy(model.state_dict())
            payload = {"epoch": 1, "optimizer": optimizer.state_dict(), "rng_state": {"torch": torch.get_rng_state()},
                       "config": {"training": {"learning_rate": .001}}}
            writer = CheckpointWriter(asynchronous=True, compact_best=True)
            writer.submit(tmp, model, payload, improved=True)
            with torch.no_grad():
                for parameter in model.parameters(): parameter.fill_(99)
            payload["config"]["training"]["learning_rate"] = 99
            writer.close()
            last = torch.load(Path(tmp) / "last.pth", weights_only=True)
            best = torch.load(Path(tmp) / "best.pth", weights_only=True)
            for key, value in expected.items():
                self.assertTrue(torch.equal(value, last["model"][key]))
                self.assertTrue(torch.equal(value, best["model"][key]))
            self.assertIn("optimizer", last); self.assertIn("rng_state", last)
            self.assertNotIn("optimizer", best); self.assertEqual("inference", best["checkpoint_kind"])
            self.assertEqual(.001, last["config"]["training"]["learning_rate"])
            failed = CheckpointWriter(asynchronous=True)
            with patch("utils.checkpoint._write_payload", side_effect=OSError("disk full")):
                failed.submit(tmp, model, payload)
                with self.assertRaisesRegex(OSError, "disk full"): failed.close()
            self.assertEqual(1, torch.load(Path(tmp) / "last.pth", weights_only=True)["epoch"])

    def test_full_best_link_survives_later_last_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 1)
            expected = copy.deepcopy(model.state_dict())
            writer = CheckpointWriter(asynchronous=True)
            writer.submit(tmp, model, {"epoch": 1, "optimizer": {"state": {1: {"step": 7}}}}, improved=True)
            writer.flush()
            self.assertEqual((Path(tmp) / "best.pth").stat().st_ino, (Path(tmp) / "last.pth").stat().st_ino)
            with torch.no_grad():
                for parameter in model.parameters(): parameter.add_(1)
            writer.submit(tmp, model, {"epoch": 2, "optimizer": {"state": {1: {"step": 8}}}})
            writer.close()
            best = torch.load(Path(tmp) / "best.pth", weights_only=True)
            last = torch.load(Path(tmp) / "last.pth", weights_only=True)
            self.assertEqual(1, best["epoch"]); self.assertEqual(2, last["epoch"])
            self.assertEqual(7, best["optimizer"]["state"][1]["step"])
            for name, value in expected.items(): self.assertTrue(torch.equal(value, best["model"][name]))

    def test_failed_test_is_incomplete_even_if_result_was_written(self):
        from tools.run_multiseed import aggregate
        with tempfile.TemporaryDirectory() as tmp:
            job = {"id": "x__seed_42", "dataset": "bright", "experiment": "x", "seed": 42,
                   "config": {"experiment": {"suite": "main"}}}
            plan = {"layout_version": 2, "jobs": [job]}
            run = run_path(tmp, job, plan)
            atomic_json(run / "training_summary.json", {"status": "complete"})
            atomic_json(result_path(tmp, job, plan), {})
            atomic_json(run / "test/failure.json", {"error": "disk write interrupted"})
            report = aggregate(plan, Path(tmp), audit=False)
            self.assertEqual("incomplete", report["status"])
            self.assertEqual([job["id"]], report["missing"])

    def test_status_eta_resume_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            progress = RunProgress(path, epochs=10)
            progress.start_epoch(1, 8, 2)
            progress.phase("train", 8)
            progress.epoch_started -= 5
            progress.batch(4); progress.write(force=True)
            self.assertGreater(progress.state["eta_seconds"], 0)
            self.assertGreater(progress.state["fraction"], 0)
            progress.epoch_done(1)
            resumed = RunProgress(path, epochs=10, completed=1)
            self.assertEqual(1, resumed.state["completed_epoch"])
            resumed.finish("failed")
            self.assertEqual("failed", json.loads(path.read_text())["status"])
            job = {"id": "bright_x__seed_42", "dataset": "bright", "experiment": "bright_x", "seed": 42,
                   "config": {"experiment": {"suite": "main"}}}
            self.assertEqual(Path(tmp) / "runs/bright/main/bright_x/seed_42", run_path(tmp, job, {"layout_version": 2}))
            self.assertEqual(Path(tmp) / "test/bright_x__seed_42.json", result_path(tmp, job, {}))
            self.assertEqual(Path(tmp) / "summary/per_seed.csv", summary_path(tmp, {"layout_version": 2}, "per_seed.csv"))
            job["dataset"] = "../escape"
            with self.assertRaises(ValueError): run_path(tmp, job, {"layout_version": 2})


if __name__ == "__main__":
    unittest.main()
