"""Serializable experiment evidence; no training or model imports required."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def scientific_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    for section in ("logging", "evaluation", "experiment"):
        cfg.pop(section, None)
    for key in ("resume", "checkpoint_dir", "output_dir", "best_metric_resolved", "device",
                "num_workers", "persistent_workers"):
        cfg.get("training", {}).pop(key, None)
    for key in ("checkpoint", "save_dir", "save_visualization", "split"):
        cfg.get("inference", {}).pop(key, None)
    return cfg


def config_digest(config: dict) -> str:
    return digest(scientific_config(config))


def source_identity(root) -> dict:
    root = Path(root)
    files = {str(p.relative_to(root)): file_digest(p)
             for folder in ("models", "utils", "datasets", "compare", "tools")
             for p in sorted((root / folder).rglob("*"))
             if p.is_file() and p.suffix in {".py", ".cpp", ".cu", ".cuh", ".h"}}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"sha256": digest(files), "files": files, "git_head": commit}


def environment() -> dict:
    import torch
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "determinism_note": "Seeded epoch/worker/sampler streams. CUDA grid_sample backward may be nondeterministic."}


def dataset_identity(dataset, mode: str = "stat") -> dict:
    if mode not in {"stat", "sha256"}:
        raise ValueError("fingerprint mode must be stat or sha256")
    prepared = getattr(dataset, "_prepared_identities", {})
    if mode in prepared:
        return prepared[mode]
    records = []
    for sample in dataset.samples:
        paths = [sample["optical"], *sample["sar"], sample["label"]]
        files = []
        for item in paths:
            p = Path(item).resolve()
            s = p.stat()
            files.append({"path": str(p), "bytes": s.st_size,
                          **({"sha256": file_digest(p)} if mode == "sha256" else {"mtime_ns": s.st_mtime_ns})})
        records.append({"id": sample["id"], "group": sample.get("group"),
                        "event": sample.get("event"), "files": files})
    result = {"split": dataset.split, "samples": len(records), "mode": mode, "records": records}
    result["sha256"] = digest(result)
    return result


def assert_disjoint(identities: dict) -> dict:
    """Check ids, physical paths, and provided source-scene groups across splits."""
    seen_ids, seen_paths, seen_groups = {}, {}, {}
    verified_groups = True
    for split, identity in identities.items():
        for row in identity["records"]:
            for value, seen, kind in ((row["id"], seen_ids, "sample id"),
                                       (row.get("group"), seen_groups, "source group")):
                if value:
                    if value in seen and seen[value] != split:
                        raise ValueError(f"Cross-split {kind} overlap: {value} ({seen[value]}, {split})")
                    seen[value] = split
            verified_groups &= bool(row.get("group"))
            for file in row["files"]:
                value = file["path"]
                if value in seen_paths and seen_paths[value] != split:
                    raise ValueError(f"Cross-split file overlap: {value}")
                seen_paths[value] = split
    return {"sample_ids_disjoint": True, "physical_paths_disjoint": True,
            "source_groups_verified": verified_groups,
            "note": "Without explicit group metadata, geographic/scene overlap is NOT verified."}


def verify_checkpoint_config(checkpoint: dict, config: dict) -> None:
    recorded = checkpoint.get("config", {})
    # Runtime batch size/AMP/TTA may change; architecture, labels and input transforms may not.
    for key in ("model", "task", "normalization"):
        if recorded and recorded.get(key, {}) != config.get(key, {}):
            raise ValueError(f"Checkpoint/config mismatch in {key}; use its saved configuration")
