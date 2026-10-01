"""Data geometry and sampler checks reused across every model and seed."""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
from utils.protocol import file_digest, digest


def audit_prepared_inputs(plan, output):
    from datasets import build_dataset
    from tools.train import build_class_balanced_sampler
    from utils.prepared_data import attach_prepared_split, PreparedSplit, preprocessing_identity
    index_path = Path(output) / "cache/index.json"
    if not index_path.exists():
        return {"status": "pending_preparation", "reason": "Run --prepare-only for full geometry and sampling checks"}
    index = json.loads(index_path.read_text())
    if index["preprocessing_sha256"] != preprocessing_identity():
        raise ValueError("Cached preprocessing identity changed; rebuild explicitly before reuse")
    rows = []
    seen = set()
    original = os.environ.get("DEHCD_PREPARED_INDEX")
    os.environ["DEHCD_PREPARED_INDEX"] = str(index_path)
    try:
        def audit_signature(job):
            cfg = job["config"]
            return digest({"data_key": job["data_key"], "sampler": {k: v for k, v in cfg["training"].items()
                if k.startswith("sampler_") or k == "class_balanced_sampler"}})
        for job in plan["jobs"]:
            signature = audit_signature(job)
            if signature in seen: continue
            seen.add(signature)
            cfg = job["config"]
            ds = build_dataset(cfg, "train", True)
            prepared = ds._prepared_split
            m = prepared.manifest
            expected = plan["data"][job["data_key"]]["splits"]["train"]
            if m["identities"]["stat"]["sha256"] != expected.get("stat_sha256", expected["sha256"]):
                raise ValueError("Prepared geometry does not match current source data")
            patch = int(cfg["dataset"]["patch_size"])
            rare_classes = cfg["dataset"].get("rare_crop_classes", [])
            ignored = int(cfg["task"].get("ignore_index", 255))
            geometry, rare, positive, probabilities, shapes = [], [], [], [], {}
            label = np.memmap(prepared.path.parent / "label.bin", mode="r", dtype=m["tensors"]["label"]["dtype"])
            rare_prob = float(cfg["dataset"].get("rare_crop_prob", 0))
            positive_prob = float(cfg["dataset"].get("positive_crop_prob", 0))
            random_crop = bool(cfg["dataset"].get("train_random_crop", True))
            for sample in m["offsets"]:
                offset, shape = sample["label"]
                height, width = shape[-2:]
                key = f"{height}x{width}"; shapes[key] = shapes.get(key, 0) + 1
                free = max(height - patch, 0) > 0 or max(width - patch, 0) > 0
                geometry.append(free)
                values = label[offset:offset + int(np.prod(shape))]
                r = bool(np.isin(values, rare_classes).any()) if rare_classes and free else False
                f = bool(((values > 0) & (values != ignored)).any()) if free else False
                rare.append(r and free and random_crop)
                positive.append(f and free and random_crop)
                pr = rare_prob if rare[-1] else 0.0
                pp = positive_prob if positive[-1] else 0.0
                probabilities.append(pr + (1 - pr) * pp)
            classes = int(cfg["task"]["num_classes"])
            sampler = build_class_balanced_sampler(ds, cfg["training"], classes, ignored)
            weights = sampler.weights.numpy() if sampler is not None else np.ones(len(ds))
            weights = weights / weights.sum()
            applicable = np.asarray(probabilities) > 0
            count = len(ds)
            rows.append({"dataset": job["dataset"], "experiments": sorted({j["experiment"] for j in plan["jobs"] if audit_signature(j) == signature}), "samples": count, "patch_size": patch, "shapes": shapes,
                "crop_freedom_count": int(sum(geometry)), "crop_freedom_ratio": float(np.mean(geometry)),
                "targeted_crop_eligible_count": int(applicable.sum()), "targeted_crop_eligible_ratio": float(applicable.mean()),
                "sampler_weighted_targeted_eligible_ratio": float(weights @ applicable),
                "expected_targeted_branch_rate_per_draw": float(weights @ np.asarray(probabilities)),
                "targeted_crop_note": "Eligibility and analytical branch probability, not observed training draws or a guaranteed change of crop origin.",
                "sampler_enabled": sampler is not None, "sampler_nonuniform": bool(sampler is not None and sampler.weights.max() > sampler.weights.min()),
                "sampler": sampler.info if sampler is not None else None,
                "manifest_sha256": file_digest(prepared.path),
                "splits": {s: d["samples"] for s, d in plan["data"][job["data_key"]]["splits"].items()}})
    finally:
        if original is None: os.environ.pop("DEHCD_PREPARED_INDEX", None)
        else: os.environ["DEHCD_PREPARED_INDEX"] = original
    return {"status": "verified", "datasets": rows}
