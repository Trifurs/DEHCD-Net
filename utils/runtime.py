"""Explicit execution profiles, shared by every model in a comparison."""
from __future__ import annotations

import json
from pathlib import Path


def apply_runtime_profile(config, name, root=None):
    if not name or name == "none":
        return
    if name != "rtx5090":
        raise ValueError(f"Unknown runtime profile: {name}")
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    profile = json.loads((root / "configs/runtime" / f"{name}.json").read_text())
    dataset = config.get("experiment", {}).get("dataset")
    if dataset not in profile["datasets"]:
        raise ValueError(f"Runtime profile has no batch setting for dataset {dataset!r}")
    config.setdefault("training", {}).update(profile["training"])
    config["training"].update(profile["datasets"][dataset])
    config["training"]["runtime_profile"] = profile["id"]
    config.setdefault("logging", {}).update(profile.get("logging", {}))
    config.setdefault("inference", {}).update(profile.get("inference", {}))


def configure_runtime(training, device):
    """No precision fallback: explicitly requested TF32 is recorded in snapshots."""
    import torch
    threads = training.get("cpu_threads")
    if threads is not None:
        if int(threads) < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(int(threads))
    if "allow_tf32" in training:
        enabled = bool(training["allow_tf32"])
        if enabled and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("The selected GPU does not support the requested TF32 profile")
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
    if training.get("runtime_profile"):
        deterministic = bool(training.get("deterministic", True))
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    return runtime_metadata()


def runtime_metadata():
    import torch
    return {"cpu_threads": torch.get_num_threads(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
