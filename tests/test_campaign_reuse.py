"""CPU evidence fixtures for conservative reuse and canonical result views.

Model construction is replaced with a tiny linear module. Actual local files,
optimizer, scheduler, RNG, histories, test counts and hashes are still checked.
"""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.run_multiseed import aggregate, make_plan
from tools.train import build_optimizer, build_scheduler
from utils.campaign_layout import run_path
from utils.campaign_reuse import (STATES, audit_campaign, inspect_run, install_imports,
                                  source_changes, source_review_for, verify_source_import)
from utils.checkpoint import validate_training_checkpoint
from utils.config import XMLConfigParser
from utils.metrics import ConfusionMatrixMeter
from utils.protocol import atomic_json, config_digest, digest, file_digest
from utils.reproducibility import capture_rng_state

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1051, 2060)
ENV = {"python": "fixture", "torch": "fixture", "cuda_build": None, "cudnn": None, "gpu": []}


def source(label):
    files = {"models/toy.py": digest(label), "tools/train.py": digest("loop")}
    return {"files": files, "sha256": digest(files), "git_head": "fixture"}


def review(before, after):
    return {"from_sha256": before["sha256"], "to_sha256": after["sha256"],
            "status": "training_compatible", "changes": source_changes(before, after)}


def full_plan():
    args = SimpleNamespace(catalog=str(ROOT / "configs/experiments/catalog.json"), suite=["all"],
        datasets=["bright", "haiti", "xbd", "cau_flood"], experiments=None, config=None,
        seeds=list(SEEDS), data_root=[], manifest=[], encoder_checkpoint=[], device="cpu",
        num_workers=0, epochs=None, batch_size=None, gradient_accumulation_steps=None,
        fingerprint="stat", runtime_profile="none")
    return make_plan(args)


class RunFixture:
    def __init__(self, output, *, completed=True, evaluated=True):
        self.output = Path(output)
        self.cfg = {"task": {"num_classes": 2, "ignore_index": 255},
            "dataset": {"root": "/synthetic", "patch_size": 4}, "model": {"name": "fixture", "variant": "m"},
            "training": {"seed": 42, "epochs": 2, "batch_size": 1, "gradient_accumulation_steps": 1,
                "device": "cpu", "num_workers": 0, "persistent_workers": False, "amp": False,
                "allow_tf32": False, "deterministic": True, "optimizer": "adamw", "learning_rate": .01,
                "weight_decay": 0., "scheduler": "cosine", "warmup_epochs": 0, "loss": "compound",
                "best_metric": "foreground_miou", "early_stop_patience": 0},
            "inference": {"test_time_augmentation": "none", "prediction_rule": "damage_argmax"},
            "experiment": {"id": "bright_fixture", "canonical_id": "bright_fixture", "dataset": "bright",
                "suite": "main", "roles": ["main"], "uses": []}}
        self.job = {"id": "bright_fixture__seed_42", "experiment": "bright_fixture", "dataset": "bright",
                    "seed": 42, "config": self.cfg, "config_sha256": config_digest(self.cfg), "data_key": "data"}
        self.identities = {s: {"mode": "stat", "sha256": digest(s), "samples": 2, "records": []}
                           for s in ("train", "val", "test")}
        self.plan = {"schema_version": 3, "layout_version": 2, "jobs": [self.job], "seeds": list(SEEDS),
                     "source": source("original"), "data": {"data": {"splits": self.identities}}}
        self.run = run_path(self.output, self.job, self.plan)
        (self.run / "checkpoints").mkdir(parents=True)
        self.execution = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False,
                          "cudnn_deterministic": True, "deterministic_algorithms": True}
        atomic_json(self.run / "config_snapshot.json", self.cfg)
        atomic_json(self.run / "protocol.json", {"config_sha256": self.job["config_sha256"],
            "source": self.plan["source"], "environment": ENV, "execution": self.execution,
            "datasets": {s: self.identities[s] for s in ("train", "val")}})
        model = self.model()
        optimizer = build_optimizer(model, self.cfg["training"])
        scheduler = build_scheduler(optimizer, self.cfg["training"], 2)
        epoch_count = 2 if completed else 1
        rows = []
        for epoch in range(1, epoch_count + 1):
            optimizer.zero_grad()
            model(torch.ones(2, 2)).square().mean().backward()
            optimizer.step(); scheduler.step()
            value = .5 + epoch / 10
            self.payload = {"checkpoint_kind": "training", "model": model.state_dict(), "config": copy.deepcopy(self.cfg),
                "config_sha256": self.job["config_sha256"], "epoch": epoch, "best_epoch": epoch,
                "best_metric": value, "best_metric_name": "foreground_miou", "early_stop_reference": value,
                "epochs_without_improvement": 0, "collapse_count": 0, "foreground_stall_count": 0,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": {},
                "rng_state": capture_rng_state(), "optical_channels": 1, "sar_channels": 1}
            rows.append({"epoch": epoch, "val": {"foreground_miou": value}, "best_epoch": epoch,
                         "best_metric": value, "early_stop_stale": 0})
        torch.save(self.payload, self.run / "checkpoints/last.pth")
        torch.save(self.payload, self.run / "checkpoints/best.pth")
        (self.run / "history.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if completed:
            atomic_json(self.run / "training_summary.json", {"status": "complete", "completed_epoch": 2,
                "best_epoch": 2, "best_metric": .7, "best_metric_name": "foreground_miou", "seed": 42,
                "stop_reason": "epochs_completed", "config_sha256": self.job["config_sha256"]})
        if completed and evaluated:
            meter = ConfusionMatrixMeter(2)
            meter.matrix = torch.tensor([[6., 2.], [1., 7.]], dtype=torch.float64)
            best = self.run / "checkpoints/best.pth"
            self.result = {"schema_version": 2, "status": "complete", "complete": True, "split": "test",
                "config_sha256": self.job["config_sha256"], "seed": 42, "experiment": self.cfg["experiment"],
                "confusion_matrix": meter.matrix.tolist(), "metrics": meter.compute(),
                "dataset_identity": self.identities["test"], "evaluated_samples": 2, "expected_samples": 2,
                "checkpoint": str(best), "checkpoint_sha256": file_digest(best), "checkpoint_selector": "best", "checkpoint_epoch": 2,
                "runtime": {"test_time_augmentation": "none", "prediction_rule": "damage_argmax",
                            "amp": False, "batch_size": 1, "execution": self.execution}}
            atomic_json(self.run / "test/result.json", self.result)

    @staticmethod
    def model(*args, **kwargs):
        return torch.nn.Linear(2, 2)

    def inspect(self, *, reviews=()):
        with patch("models.build_model", side_effect=self.model):
            return inspect_run(self.run, self.job, self.plan, self.output, reviews, ENV)

    def audit(self, *, reviews=()):
        with patch("models.build_model", side_effect=self.model):
            return audit_campaign(self.plan, self.output, reviews=reviews, environment_info=ENV)


class SourceReviewTests(unittest.TestCase):
    def test_source_pair_requires_both_complete_digests_and_exact_changes(self):
        a, b = source("a"), source("b")
        accepted = review(a, b)
        self.assertEqual("identical", source_review_for(a, a, [])["status"])
        self.assertEqual(accepted, source_review_for(a, b, [accepted]))
        for field, value in (("from_sha256", digest("wrong")), ("to_sha256", digest("wrong")),
                             ("changes", {}), ("status", "unreviewed")):
            candidate = copy.deepcopy(accepted); candidate[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                source_review_for(a, b, [candidate])
        forged = copy.deepcopy(a); forged["files"]["models/toy.py"] = digest("tampered")
        with self.assertRaises(ValueError): source_review_for(a, forged, [])
        with self.assertRaises(ValueError): source_review_for({"sha256": "same"}, {"sha256": "same"}, [])

    def test_import_record_does_not_authorize_another_source_or_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp)
            original, current = fixture.plan["source"], source("reviewed")
            fixture.plan["source"] = current
            record = review(original, current)
            manifest = fixture.audit(reviews=[record])
            self.assertEqual("reuse_complete", manifest["tasks"][0]["state"])
            install_imports(fixture.plan, manifest, tmp)
            verify_source_import(fixture.run, original, current)
            with self.assertRaises(ValueError): verify_source_import(fixture.run, original, source("unreviewed"))
            config = copy.deepcopy(fixture.cfg); config["training"]["loss"] = "other"
            atomic_json(fixture.run / "config_snapshot.json", config)
            with self.assertRaisesRegex(ValueError, "different scientific"):
                verify_source_import(fixture.run, original, current)


class ReuseTests(unittest.TestCase):
    def test_complete_reevaluate_and_resume_are_distinct(self):
        for completed, evaluated, expected in ((True, True, "reuse_complete"),
                (True, False, "reevaluate_only"), (False, False, "resume_training")):
            with self.subTest(state=expected), tempfile.TemporaryDirectory() as tmp:
                fixture = RunFixture(tmp, completed=completed, evaluated=evaluated)
                self.assertEqual(expected, fixture.inspect()["state"])
                self.assertEqual(expected, fixture.audit()["tasks"][0]["state"])

    def test_metadata_changes_preserve_reuse_but_effective_changes_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp)
            fixture.cfg["experiment"].update(roles=["scaling", "main"], paper_refs=["another caption"])
            fixture.cfg["logging"] = {"run_name": "a display name"}
            self.assertEqual("reuse_complete", fixture.inspect()["state"])
            for section, field, value in (("model", "variant", "l"), ("training", "loss", "other"),
                    ("dataset", "patch_size", 8), ("training", "amp", True), ("training", "allow_tf32", True),
                    ("training", "batch_size", 2), ("training", "gradient_accumulation_steps", 2),
                    ("training", "num_workers", 4)):
                original = fixture.cfg[section][field]
                fixture.cfg[section][field] = value
                fixture.job["config_sha256"] = config_digest(fixture.cfg)
                with self.subTest(field=field):
                    self.assertEqual("retrain_required", fixture.inspect()["state"])
                fixture.cfg[section][field] = original
            fixture.job["config_sha256"] = config_digest(fixture.cfg)
            for split in ("train", "val"):
                original = fixture.identities[split]["sha256"]
                fixture.identities[split]["sha256"] = digest("changed files")
                with self.subTest(split=split):
                    self.assertEqual("retrain_required", fixture.inspect()["state"])
                fixture.identities[split]["sha256"] = original

    def test_unknown_source_and_incomplete_state_remain_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp, completed=False, evaluated=False)
            original = fixture.plan["source"]
            fixture.plan["source"] = source("unknown code change")
            self.assertEqual("blocked_review", fixture.audit()["tasks"][0]["state"])
            fixture.plan["source"] = original
            for mutation in ("missing_optimizer", "inference", "missing_rng", "nonfinite_model"):
                payload = copy.deepcopy(fixture.payload)
                if mutation == "missing_optimizer": payload.pop("optimizer")
                elif mutation == "inference": payload["checkpoint_kind"] = "inference"
                elif mutation == "missing_rng": payload["rng_state"].pop("torch")
                else: payload["model"]["weight"][0, 0] = float("nan")
                torch.save(payload, fixture.run / "checkpoints/last.pth")
                with self.subTest(mutation=mutation):
                    self.assertEqual("blocked_review", fixture.audit()["tasks"][0]["state"])

    def test_other_seed_and_other_variant_never_supply_a_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp)
            fixture.cfg["training"]["seed"] = 1051
            fixture.job.update(seed=1051, id="bright_fixture__seed_1051", config_sha256=config_digest(fixture.cfg))
            self.assertEqual("train_new", fixture.audit()["tasks"][0]["state"])
            fixture.cfg["training"]["seed"] = 42
            fixture.cfg["model"]["variant"] = "l"
            fixture.job.update(seed=42, id="bright_fixture__seed_42", config_sha256=config_digest(fixture.cfg))
            self.assertEqual("retrain_required", fixture.audit()["tasks"][0]["state"])

    def test_early_completion_and_truncated_or_wrong_checkpoint_test_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp)
            for field, value in (("checkpoint_selector", "last"), ("evaluated_samples", 1),
                                 ("complete", False), ("seed", 1051)):
                result = copy.deepcopy(fixture.result); result[field] = value
                atomic_json(fixture.run / "test/result.json", result)
                with self.subTest(field=field):
                    self.assertEqual("reevaluate_only", fixture.inspect()["state"])
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp, completed=False, evaluated=False)
            atomic_json(fixture.run / "training_summary.json", {"status": "complete", "stop_reason": "early_stopping"})
            self.assertEqual("retrain_required", fixture.inspect()["state"])

    def test_invalid_optimizer_scheduler_and_rng_are_not_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp, completed=False, evaluated=False)
            for mutation in ("nonfinite_optimizer", "scheduler_epoch", "rng_torch"):
                payload = copy.deepcopy(fixture.payload)
                if mutation == "nonfinite_optimizer":
                    state = next(iter(payload["optimizer"]["state"].values()))
                    state["exp_avg"].reshape(-1)[0] = float("nan")
                elif mutation == "scheduler_epoch":
                    payload["scheduler"]["last_epoch"] += 10
                else:
                    payload["rng_state"]["torch"] = torch.zeros(3, dtype=torch.uint8)
                torch.save(payload, fixture.run / "checkpoints/last.pth")
                with self.subTest(mutation=mutation):
                    self.assertEqual("blocked_review", fixture.audit()["tasks"][0]["state"])

    def test_independent_aggregate_checks_completion_selection_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp)
            baseline = aggregate(fixture.plan, fixture.output)
            self.assertEqual("incomplete", baseline["status"])
            self.assertEqual(1, baseline["experiments"][0]["n"])
            result_path = fixture.run / "test/result.json"
            for field, value in (("checkpoint_selector", "last"), ("checkpoint_epoch", 1)):
                candidate = copy.deepcopy(fixture.result); candidate[field] = value
                atomic_json(result_path, candidate)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    aggregate(fixture.plan, fixture.output)
            for field, value in (("amp", True), ("batch_size", 2)):
                candidate = copy.deepcopy(fixture.result); candidate["runtime"][field] = value
                atomic_json(result_path, candidate)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    aggregate(fixture.plan, fixture.output)
            candidate = copy.deepcopy(fixture.result)
            candidate["runtime"]["execution"]["matmul_allow_tf32"] = True
            atomic_json(result_path, candidate)
            with self.assertRaises(ValueError): aggregate(fixture.plan, fixture.output)
            atomic_json(result_path, fixture.result)
            history = fixture.run / "history.jsonl"
            content = history.read_text()
            rows = [json.loads(line) for line in content.splitlines()]
            rows[0]["train_loss"] = float("nan")
            history.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError): aggregate(fixture.plan, fixture.output)
            history.write_text(content)
            summary_path = fixture.run / "training_summary.json"
            summary = json.loads(summary_path.read_text()); summary["stop_reason"] = "early_stopping"
            atomic_json(summary_path, summary)
            with self.assertRaises(ValueError): aggregate(fixture.plan, fixture.output)

    def test_all_240_target_states_are_exhaustive_and_unique(self):
        plan = full_plan()
        with tempfile.TemporaryDirectory() as tmp:
            first = run_path(tmp, plan["jobs"][0], plan)
            first.mkdir(parents=True); (first / "unclassified.txt").write_text("fixture")
            manifest = audit_campaign(plan, tmp, environment_info=ENV)
            self.assertEqual(240, manifest["target_tasks"])
            self.assertEqual(set(STATES), set(manifest["counts"]))
            self.assertEqual(240, sum(manifest["counts"].values()))
            self.assertEqual(239, manifest["counts"]["train_new"])
            self.assertEqual(1, manifest["counts"]["blocked_review"])
            self.assertEqual(240, len({row["id"] for row in manifest["tasks"]}))

    def test_alias_migration_preserves_checkpoint_bytes_and_repeat_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp)
            canonical = fixture.run
            alias = fixture.output / "runs/bright/recipe/bright_full_recipe/seed_42"
            alias.parent.mkdir(parents=True)
            canonical.rename(alias)
            old_config = copy.deepcopy(fixture.cfg)
            old_config["experiment"].update(id="bright_full_recipe", canonical_id="bright_full_recipe", suite="recipe")
            old_config["logging"] = {"run_dir": str(alias)}
            old_config["training"]["checkpoint_dir"] = str(alias / "checkpoints")
            atomic_json(alias / "config_snapshot.json", old_config)
            for name in ("best", "last"):
                payload = copy.deepcopy(fixture.payload); payload["config"] = old_config
                torch.save(payload, alias / f"checkpoints/{name}.pth")
            result = copy.deepcopy(fixture.result)
            result["experiment"] = old_config["experiment"]
            result["checkpoint"] = str(alias / "checkpoints/best.pth")
            result["checkpoint_sha256"] = file_digest(result["checkpoint"])
            atomic_json(alias / "test/result.json", result)
            expected = {name: file_digest(alias / f"checkpoints/{name}.pth") for name in ("best", "last")}
            manifest = fixture.audit()
            self.assertEqual("reuse_complete", manifest["tasks"][0]["state"])
            self.assertEqual(str(alias), manifest["tasks"][0]["run_dir"])
            install_imports(fixture.plan, manifest, fixture.output)
            self.assertFalse(alias.exists())
            self.assertTrue(canonical.exists())
            for name, checksum in expected.items():
                path = canonical / f"checkpoints/{name}.pth"
                self.assertEqual(checksum, file_digest(path))
                payload = torch.load(path, map_location="cpu", weights_only=False)
                fixture.model().load_state_dict(payload["model"])
                self.assertEqual("bright_full_recipe", payload["config"]["experiment"]["id"])
            imported = json.loads((canonical / "source_import.json").read_text())
            self.assertEqual("bright_full_recipe", imported["original"]["config_snapshot"]["experiment"]["id"])
            self.assertEqual("moved_and_verified", imported["migration"]["status"])
            current = json.loads((canonical / "test/result.json").read_text())
            self.assertEqual(str(canonical / "checkpoints/best.pth"), current["checkpoint"])
            self.assertEqual("bright_fixture", current["experiment"]["id"])
            self.assertEqual("reuse_complete", fixture.audit()["tasks"][0]["state"])
            self.assertEqual(1, aggregate(fixture.plan, fixture.output)["experiments"][0]["n"])

    def test_alias_migration_rejects_symlink_destination_parent(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            fixture = RunFixture(tmp)
            original = fixture.run
            alias = fixture.output / "train/alias"
            alias.parent.mkdir()
            original.rename(alias)
            result = copy.deepcopy(fixture.result)
            result["checkpoint"] = str(alias / "checkpoints/best.pth")
            atomic_json(alias / "test/result.json", result)
            manifest = fixture.audit()
            self.assertEqual("reuse_complete", manifest["tasks"][0]["state"])
            original.parent.rmdir()
            original.parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                install_imports(fixture.plan, manifest, fixture.output)
            self.assertTrue(alias.exists())
            self.assertEqual([], list(Path(outside).iterdir()))

    def test_duplicate_asset_uses_fixed_order_without_extra_repetitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RunFixture(tmp)
            other = fixture.output / "runs/bright/main/duplicate/seed_42"
            other.mkdir(parents=True)
            atomic_json(other / "config_snapshot.json", fixture.cfg)
            manifest = fixture.audit()
            self.assertEqual(1, len(manifest["tasks"]))
            self.assertEqual("reuse_complete", manifest["tasks"][0]["state"])
            self.assertEqual([str(other)], manifest["tasks"][0]["duplicates"])


class DataPreflightTests(unittest.TestCase):
    def test_geometry_and_sampling_profiles_are_counted_once_each(self):
        from utils.data_preflight import audit_prepared_inputs
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            cache = output / "cache/fixture"; cache.mkdir(parents=True)
            manifest = {"identities": {"stat": {"sha256": digest("train")}},
                        "offsets": [{"label": [0, [4, 4]]}, {"label": [16, [8, 8]]}],
                        "tensors": {"label": {"dtype": "uint8"}}}
            atomic_json(cache / "manifest.json", manifest)
            (cache / "label.bin").write_bytes(bytes([1] * 80))
            atomic_json(output / "cache/index.json", {"preprocessing_sha256": "fixture", "splits": {}})
            class PreparedDataset:
                _prepared_split = SimpleNamespace(path=cache / "manifest.json", manifest=manifest)
                def __len__(self): return 2
            config = {"dataset": {"patch_size": 4, "rare_crop_classes": [1], "rare_crop_prob": .5,
                                   "positive_crop_prob": .5, "train_random_crop": True},
                      "task": {"ignore_index": 255, "num_classes": 2},
                      "training": {"class_balanced_sampler": True}}
            off = copy.deepcopy(config); off["training"]["class_balanced_sampler"] = False
            jobs = [{"dataset": "bright", "data_key": "shared", "experiment": name, "config": cfg}
                    for name, cfg in (("main", config), ("another_seed", config), ("uniform", off))]
            plan = {"jobs": jobs, "data": {"shared": {"splits": {split: {"samples": 2, "sha256": digest(split)}
                      for split in ("train", "val", "test")}}}}
            def sampler(ds, cfg, classes, ignored):
                return SimpleNamespace(weights=torch.tensor([1., 3.], dtype=torch.float64), info={}) if cfg["class_balanced_sampler"] else None
            with patch("utils.prepared_data.preprocessing_identity", return_value="fixture"), \
                 patch("datasets.build_dataset", return_value=PreparedDataset()), \
                 patch("tools.train.build_class_balanced_sampler", side_effect=sampler):
                report = audit_prepared_inputs(plan, output)
            self.assertEqual("verified", report["status"])
            self.assertEqual(2, len(report["datasets"]))
            weighted, uniform = report["datasets"]
            self.assertEqual(["another_seed", "main"], weighted["experiments"])
            self.assertEqual(["uniform"], uniform["experiments"])
            self.assertEqual(1, weighted["crop_freedom_count"])
            self.assertAlmostEqual(.5, weighted["targeted_crop_eligible_ratio"])
            self.assertTrue(weighted["sampler_nonuniform"])
            self.assertFalse(uniform["sampler_nonuniform"])
            self.assertAlmostEqual(.5625, weighted["expected_targeted_branch_rate_per_draw"])
            self.assertAlmostEqual(.375, uniform["expected_targeted_branch_rate_per_draw"])


class GroupReportTests(unittest.TestCase):
    def test_shared_reference_once_in_tables_and_correct_paired_direction(self):
        from tools.compare_results import make_report
        plan = full_plan()
        plan["data"] = {"common": {"splits": {s: {"mode": "stat", "sha256": digest(s)}
                            for s in ("train", "val", "test")}}}
        for job in plan["jobs"]: job["data_key"] = "common"
        values = {"bright_dehcd_m": [.3, .5, .7], "bright_m_irb_steps_0": [.4, .6, .8],
                  "bright_m_hog_levels_1": [.2, .4, .6], "haiti_dehcd_l": [.4, .5, .6],
                  "haiti_hog_bins_2": [.3, .4, .5]}
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            atomic_json(campaign / "protocol.json", plan)
            def evaluated_rows(*args, **kwargs):
                output = campaign / "summary/per_seed.csv"; output.parent.mkdir()
                with output.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["experiment", "seed", "foreground_miou", "checkpoint", "config_sha256"])
                    writer.writeheader()
                    for name, scores in values.items():
                        for seed, value in zip(SEEDS, scores):
                            writer.writerow({"experiment": name, "seed": seed, "foreground_miou": value,
                                             "checkpoint": f"fixture/{name}/{seed}/best.pth", "config_sha256": digest(name)})
                return {"status": "incomplete"}
            with patch("tools.compare_results.aggregate", side_effect=evaluated_rows):
                report = make_report(campaign, metrics=["foreground_miou"], repeats=1000,
                                     groups=["irb_steps", "hog_levels", "hog_bins"])
            self.assertEqual("incomplete", report["status"])
            self.assertEqual(17, len(report["tables"]))
            rows = {row["experiment"]: row for row in report["tables"]}
            self.assertEqual(17, len(rows))
            shared = rows["bright_dehcd_m"]
            self.assertEqual(3, shared["n"])
            self.assertEqual(3, shared["expected_n"])
            self.assertAlmostEqual(50., shared["mean_percent"])
            self.assertAlmostEqual(20., shared["sample_std_percentage_points"])
            self.assertEqual(0, rows["bright_m_irb_steps_8"]["n"])
            self.assertIsNone(rows["bright_m_irb_steps_8"]["mean_percent"])
            self.assertFalse(rows["bright_m_irb_steps_8"]["complete"])
            pairs = [(row["reference"], row["comparator"]) for row in report["comparisons"]]
            self.assertEqual(len(pairs), len(set(pairs)))
            delta = next(row for row in report["comparisons"] if row["comparator"] == "bright_m_irb_steps_0")
            self.assertEqual("bright_dehcd_m", delta["reference"])
            self.assertEqual(3, delta["n"])
            self.assertAlmostEqual(delta["mean_difference"] * 100, delta["mean_difference_percentage_points"])
            self.assertAlmostEqual(-10., delta["mean_difference_percentage_points"])
            self.assertGreaterEqual(delta["p_value"], .25)


if __name__ == "__main__":
    unittest.main()
