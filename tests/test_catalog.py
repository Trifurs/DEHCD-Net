"""Canonical experiment coverage, scientific identity, and shared views."""
from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from utils.comparison_protocol import differences, comparison_settings, validate_design
from utils.config import XMLConfigParser
from utils.experiment_catalog import analysis_groups, derive_catalog, load_catalog, select_experiments
from utils.protocol import config_digest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1051, 2060)
BASELINES = {"icif_net", "dminet", "hfa_panet", "wavehfg", "hrsicd", "haff"}


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_catalog(ROOT)
        cls.configs = {row["id"]: XMLConfigParser(ROOT / row["config"]).parse().as_dict() for row in cls.rows}
        cls.groups = analysis_groups(cls.rows)

    def test_unique_configurations_and_tasks(self):
        self.assertEqual(80, len(self.rows))
        counts = Counter(row["dataset"] for row in self.rows)
        self.assertEqual({"bright": 39, "haiti": 23, "cau_flood": 9, "xbd": 9}, counts)
        tasks = {(row["canonical_id"], seed) for row in self.rows for seed in SEEDS}
        self.assertEqual(240, len(tasks))
        self.assertEqual(80, len({config_digest(config) for config in self.configs.values()}))
        formal = {str(path.relative_to(ROOT)) for path in (ROOT / "configs/experiments").rglob("*.xml")}
        self.assertEqual({row["config"] for row in self.rows}, formal)

    def test_complete_main_families(self):
        for ds in ("bright", "haiti", "cau_flood", "xbd"):
            suffixes = BASELINES | {"dehcd_s", "dehcd_m", "dehcd_l"}
            if ds in ("bright", "haiti"):
                suffixes |= {"changeos", "damageformer", "changemamba"}
            self.assertEqual({f"{ds}_{suffix}" for suffix in suffixes},
                             {m["canonical_id"] for m in self.groups[f"main_{ds}"]})
        self.assertEqual(42, len(self.groups["main"]))
        self.assertEqual(12, len(self.groups["scaling"]))
        for row in self.rows:
            if row["suite"] == "main":
                self.assertEqual(.15, self.configs[row["id"]]["model"]["dropout"])

    def test_group_union_precedes_seed_expansion(self):
        rows = select_experiments(self.rows, ["main", "ablation", "sensitivity", "scaling"])
        self.assertEqual(71, len(rows))
        self.assertEqual(213, len({(r["id"], s) for r in rows for s in SEEDS}))
        full = select_experiments(self.rows, ["main", "ablation", "sensitivity", "scaling", "capacity", "dpm", "recipe"])
        self.assertEqual(80, len(full))
        self.assertEqual(39, len(select_experiments(self.rows, ["all"], ["bright"])))
        with self.assertRaisesRegex(ValueError, "Unknown analysis"):
            select_experiments(self.rows, ["missing"])

    def test_fig17_switches_and_unique_full_references(self):
        expected = {
            "dehcd_l": [1, 1, 1, 1], "no_dpm": [1, 0, 1, 1], "no_bicsf": [1, 1, 0, 1],
            "no_irb": [1, 1, 1, 0], "no_hog": [0, 1, 1, 1], "no_hog_no_dpm": [0, 0, 1, 1],
            "no_hog_no_dpm_no_bicsf": [0, 0, 0, 1], "all_off": [0, 0, 0, 0],
        }
        for ds in ("bright", "haiti"):
            self.assertEqual(8, len(self.groups[f"fig17_{ds}"]))
            for member in self.groups[f"fig17_{ds}"]:
                cfg = self.configs[member["canonical_id"]]
                model = cfg["model"]
                actual = [int(model["use_hog"]), int(model["fusion_mode"] == "dpm"),
                          int(model["global_context"] and model["cross_scale_fusion"]), int(model["diffusion_steps"] > 0)]
                self.assertEqual(expected[member["canonical_id"][len(ds) + 1:]], actual)
                self.assertEqual(actual, member["modules"])
                self.assertEqual("dehcd_l", model["backbone"])
                if not model["cross_scale_fusion"]:
                    self.assertFalse(model["global_context"])

    def test_prespecified_scan_coverage_and_defaults(self):
        scans = {"hog_bins": ([2, 4, 6, 8, 10], "haiti_dehcd_l", "hog_bins"),
                 "irb_steps": (list(range(9)), "bright_dehcd_m", "diffusion_steps"),
                 "hog_levels": ([1, 2, 3, 4], "bright_dehcd_m", "hog_modulation_levels")}
        for group, (points, reference, field) in scans.items():
            members = self.groups[group]
            self.assertEqual(points, [m["x"] for m in members])
            self.assertEqual([reference], [m["canonical_id"] for m in members if m["role"] == "default"])
            for member in members:
                cfg = self.configs[member["canonical_id"]]
                self.assertEqual(member["x"], cfg["model"][field])
                changed = differences(comparison_settings(self.configs[reference]), comparison_settings(cfg))
                self.assertEqual(set() if member["role"] == "default" else {f"model.{field}"}, set(changed))
                if group == "hog_levels":
                    self.assertEqual(list(range(member["x"])), member["levels"])
        self.assertNotEqual(config_digest(self.configs["bright_m_irb_steps_0"]), config_digest(self.configs["bright_no_irb"]))
        self.assertEqual("dehcd_m", self.configs["bright_m_irb_steps_0"]["model"]["backbone"])
        self.assertEqual("dehcd_l", self.configs["bright_no_irb"]["model"]["backbone"])

    def test_declared_scientific_design(self):
        evidence = validate_design(self.configs)
        self.assertEqual(76, len(evidence))
        for cfg in self.configs.values():
            self.assertEqual(0, cfg["training"]["early_stop_patience"])
            self.assertEqual(0, cfg["training"]["localization_loss_weight"])
            self.assertEqual("foreground_miou", cfg["training"]["best_metric"])

    def test_presentation_metadata_is_not_a_training_identity(self):
        before = self.configs["bright_dehcd_m"]
        after = copy.deepcopy(before)
        after["experiment"].update(roles=["another_view"], paper_refs=["different caption"], uses=[])
        after["logging"]["run_name"] = "different_display_name"
        self.assertEqual(config_digest(before), config_digest(after))
        for section, key, value in (("model", "diffusion_steps", 5), ("training", "loss", "different"),
                                    ("dataset", "patch_size", 128), ("training", "amp", True)):
            after = copy.deepcopy(before)
            after[section][key] = value
            self.assertNotEqual(config_digest(before), config_digest(after))

    def test_derived_indices_reproduce_formal_metadata(self):
        derived = derive_catalog(ROOT, [ROOT / row["config"] for row in self.rows])
        self.assertEqual(self.rows, derived)
        stored = json.loads((ROOT / "configs/experiments/groups.json").read_text())
        self.assertEqual(list(SEEDS), stored["seeds"])
        self.assertEqual(self.groups, stored["groups"])
        for group, members in self.groups.items():
            for member in members:
                self.assertNotIn("model", member)
                self.assertNotIn("training", member)
        for row in self.rows:
            self.assertTrue(row["roles"])
            self.assertTrue(row["uses"])
            self.assertTrue(row["paper_refs"])
            self.assertEqual(row["id"], row["canonical_id"])


if __name__ == "__main__":
    unittest.main()
