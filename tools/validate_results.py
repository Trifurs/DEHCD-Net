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
    if result.get("schema_version") != 1:
        raise ValueError("Legacy result has no verifiable provenance; re-evaluate the saved checkpoint")
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
