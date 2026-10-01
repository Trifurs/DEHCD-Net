"""Reject inconsistent metrics and validation scores presented as test scores."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.protocol import file_digest


def validate_result(result, expected_config=None, expected_dataset=None, verify_files=False):
    import torch
    from utils.metrics import ConfusionMatrixMeter
    if result.get("split") != "test":
        raise ValueError("Publication summary requires split=test; validation scores cannot be relabelled")
    if result.get("runtime", {}).get("max_batches", 0):
        raise ValueError("Truncated test evaluation is not a complete result")
    schema = result.get("schema_version")
    if schema not in {1, 2}:
        raise ValueError("Legacy result has no verifiable provenance; re-evaluate the saved checkpoint")
    if schema == 2:
        if result.get("complete") is not True or result.get("status") != "complete":
            raise ValueError("Partial or failed evaluation is not a complete result")
        expected = result.get("dataset_identity", {}).get("samples")
        if not isinstance(expected, int) or expected <= 0 or result.get("expected_samples") != expected:
            raise ValueError("Complete evaluation requires the expected split sample count")
        runtime = result.get("runtime", {})
        if runtime.get("prediction_rule") != "damage_argmax" or runtime.get("test_time_augmentation") != "none":
            raise ValueError("Formal results require common main-logits argmax and no TTA")
    heads = result.get("damage_head_metrics")
    if heads is not None:
        if not isinstance(heads.get("localization_supervised"), bool):
            raise ValueError("Localization supervision is not recorded; re-evaluate this dual-head checkpoint")
        if not heads["localization_supervised"] and heads.get("localization_f1") is not None:
            raise ValueError("Unsupervised localization_f1 must be null; use diagnostic_localization_f1")
        for key in ("localization_f1", "diagnostic_localization_f1", "conditional_damage_hmean_f1"):
            if heads.get(key) is not None and not math.isfinite(float(heads[key])):
                raise ValueError(f"Non-finite damage head metric: {key}")
    if result.get("loss") is not None and not math.isfinite(float(result["loss"])):
        raise ValueError("Non-finite evaluation loss")
    if "dataset_identity" in result and result.get("evaluated_samples") != result["dataset_identity"]["samples"]:
        raise ValueError("Evaluated sample count does not match the complete test split")
    cm = torch.as_tensor(result["confusion_matrix"], dtype=torch.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not torch.isfinite(cm).all() or (cm < 0).any() or not torch.equal(cm, cm.round()):
        raise ValueError("Confusion matrix must contain finite, nonnegative integer counts")
    if cm.sum() <= 0:
        raise ValueError("Test result has no valid labelled pixels")
    meter = ConfusionMatrixMeter(cm.shape[0]); meter.matrix = cm
    recomputed = meter.compute()
    for key, value in recomputed.items():
        if key not in result["metrics"] or not math.isclose(float(result["metrics"][key]), value, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"Metric does not match raw confusion matrix: {key}")
    if expected_config and result["config_sha256"] != expected_config:
        raise ValueError("Result/config hash mismatch")
    if expected_dataset and result["dataset_identity"]["sha256"] != expected_dataset:
        raise ValueError("Test split/data fingerprint mismatch")
    if verify_files and file_digest(result["checkpoint"]) != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after evaluation")
    return recomputed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    for item in args.results:
        result = json.loads(Path(item).read_text())
        validate_result(result, verify_files=args.verify_checkpoints)
        print(f"Verified test metrics and provenance: {item}")


if __name__ == "__main__":
    main()
