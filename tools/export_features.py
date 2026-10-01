"""Export fixed-sample module responses from an existing seed-42 DEHCD checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def feature_modules(model):
    """Public module boundaries; post-fusion features form one shared stream."""
    names = dict(model.named_modules())
    selected = [name for name in names if re.fullmatch(
        r"(?:optical|sar)_hog_modulators\.\d+|fusion_blocks\.\d+|global_context|cross_scale_fusion|diffusion_refine", name)]
    if not selected or "fusion_blocks.0" not in names:
        raise ValueError("Feature export requires a DEHCD model with declared module boundaries")
    return {name: names[name] for name in selected}


def _tensors(value, path=""):
    import torch
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{path}.{index}" if path else str(index))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensors(item, f"{path}.{key}" if path else str(key))


class FeatureCapture:
    """Copy feature tensors without changing outputs; hooks always release."""
    def __init__(self, model):
        self.modules = feature_modules(model)
        self.features = {}
        self.handles = []

    def __enter__(self):
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_pre_hook(self._before(name)))
            self.handles.append(module.register_forward_hook(self._after(name)))
        return self

    def _copy(self, name, when, values):
        import torch
        for suffix, value in _tensors(values):
            if value.ndim != 4:
                continue
            if value.shape[0] != 1:
                raise ValueError("Feature export requires exactly one sample per forward")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Non-finite feature at {name}/{when}/{suffix}")
            key = name + "__" + when + ("__" + suffix if suffix else "")
            if key in self.features:
                raise ValueError(f"Module boundary was evaluated twice in one capture: {key}")
            self.features[key] = value.detach()[0].to(device="cpu", dtype=torch.float32, copy=True).numpy()

    def _before(self, name):
        def hook(module, inputs):
            # HOG modulation receives both a feature map and a histogram; only
            # the first tensor is the before-feature. Fusion input has two streams.
            values = inputs[0] if "hog_modulators" in name or len(inputs) == 1 else inputs
            self._copy(name, "before", values)
        return hook

    def _after(self, name):
        def hook(module, inputs, output):
            self._copy(name, "after", output)
        return hook

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def select_samples(samples, requested=None, count=16):
    """Fixed lexical pool, never score-based selection."""
    if not 1 <= int(count) <= 16:
        raise ValueError("--count must be between 1 and 16")
    by_id = {sample["id"]: index for index, sample in enumerate(samples)}
    if len(by_id) != len(samples):
        raise ValueError("Duplicate sample IDs in feature dataset")
    fixed_pool = sorted(by_id)[:16]
    if requested:
        if len(requested) != len(set(requested)) or not set(requested) <= set(fixed_pool):
            raise ValueError("--sample-id must be a unique subset of the predeclared first 16 test IDs")
        selected = [name for name in fixed_pool if name in requested]
    else:
        selected = fixed_pool[:count]
    if not selected:
        raise ValueError("Feature export requires nonempty test samples")
    return fixed_pool, [(name, by_id[name]) for name in selected]


def write_features(directory, features):
    import numpy as np
    from PIL import Image
    from matplotlib import colormaps
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, feature in features.items():
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        np.save(directory / (safe + ".npy"), feature, allow_pickle=False)
        response = np.abs(feature).mean(axis=0)
        low, high = float(response.min()), float(response.max())
        scaled = (response - low) / (high - low) if high > low else np.zeros_like(response)
        image = (colormaps["magma"](scaled)[..., :3] * 255).round().astype(np.uint8)
        Image.fromarray(image).save(directory / (safe + ".png"))
        rows.append({"module_boundary": name, "shape_CHW": list(feature.shape),
                     "raw_tensor": safe + ".npy", "response_png": safe + ".png",
                     "response_min": low, "response_max": high})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing seed-42 run with saved configuration")
    parser.add_argument("--checkpoint", default="best", choices=["best"], help="Fixed validation-selected checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, help="Postprocessing folder, separate from formal test metrics")
    parser.add_argument("--sample-id", nargs="+", help="Subset of the predeclared first 16 lexical test IDs")
    parser.add_argument("--count", type=int, default=16, help="Prefix of the fixed 16-ID pool, for bounded exports")
    args = parser.parse_args()
    import torch
    import numpy as np
    from PIL import Image
    from datasets import build_dataset
    from models import build_model
    from tools.test import load_config_snapshot, resolve_checkpoint
    from utils.checkpoint import load_model_state
    from utils.model_metadata import model_metadata
    from utils.prediction import (predict_outputs, decode_predictions, prediction_rule, resolve_tta,
                                  validate_evaluation_labels, check_evaluation_tensors)
    from utils.protocol import config_digest, verify_checkpoint_config, file_digest, atomic_json, dataset_identity
    from utils.runtime import configure_runtime

    run = Path(args.run_dir).expanduser().resolve()
    config = load_config_snapshot(run)
    if int(config.get("training", {}).get("seed", -1)) != 42:
        raise ValueError("Qualitative export uses the predeclared display seed 42")
    if resolve_tta(config) != "none" or prediction_rule(config) != "damage_argmax":
        raise ValueError("Feature export requires the formal no-TTA primary-argmax protocol")
    checkpoint_path = resolve_checkpoint(run, args.checkpoint)
    # This command accepts only the explicitly selected local run checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Feature export requires a trusted local task checkpoint with config/model")
    verify_checkpoint_config(checkpoint, config)
    if config_digest(checkpoint["config"]) != config_digest(config):
        raise ValueError("Checkpoint and run snapshot scientific settings differ")
    device = torch.device(args.device)
    execution = configure_runtime(config.get("training", {}), device)
    dataset = build_dataset(config, split="test", training=False)
    fixed_pool, selected = select_samples(dataset.samples, args.sample_id, args.count)
    prior_test = run / "test/result.json"
    identity_mode = "stat"
    if prior_test.exists():
        previous = json.loads(prior_test.read_text()).get("dataset_identity")
        if previous:
            identity_mode = previous["mode"]
    identity = dataset_identity(dataset, identity_mode)
    if prior_test.exists() and previous and previous != identity:
        raise ValueError("Test data no longer match the saved result evidence")
    model = build_model(config, dataset.num_optical_channels, dataset.num_sar_channels,
                        initialize_encoder=False).to(device).eval()
    load_model_state(model, checkpoint["model"])
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Feature output must be empty; existing artifacts will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "scope": "qualitative module-response export; no accuracy result",
        "run": str(run), "seed": 42, "split": "test", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_digest(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
        "config_sha256": config_digest(config), "dataset_identity": identity,
        "prediction_rule": "damage_argmax", "tta": "none", "implementation": model_metadata(model, device),
        "execution": execution, "selection": {"policy": "first 16 lexical test IDs, fixed before inference",
        "fixed_pool": fixed_pool, "selected": [name for name, _ in selected]},
        "response_definition": "mean absolute activation over channels; PNG min-max scaled independently for each image and module boundary; raw CHW FP32 arrays retained",
        "interpretation": "PNG colors cannot compare absolute activation across modules/images; heatmaps do not establish causal synergy or physical separation. Fusion input 0/1 is optical/SAR; post-DPM and subsequent tensors form one shared stream.",
        "samples": []}
    del checkpoint
    with torch.inference_mode():
        for sample_id, index in selected:
            sample = dataset[index]
            optical, sar = sample["optical"].unsqueeze(0).to(device), sample["sar"].unsqueeze(0).to(device)
            label = sample["label"]
            validate_evaluation_labels(label, int(config["task"]["num_classes"]), int(config["task"].get("ignore_index", 255)))
            check_evaluation_tensors({"optical": optical, "sar": sar})
            with FeatureCapture(model) as capture:
                heads = predict_outputs(model, optical, sar, False, "none")
                prediction = decode_predictions(heads, config)[0].cpu().numpy().astype(np.uint8)
            name = re.sub(r"[^a-zA-Z0-9_.-]", "_", sample_id)[:100] + "_" + hashlib.sha256(sample_id.encode()).hexdigest()[:8]
            directory = output / name
            rows = write_features(directory, capture.features)
            Image.fromarray(prediction).save(directory / "prediction.png")
            np.save(directory / "label.npy", label.cpu().numpy(), allow_pickle=False)
            manifest["samples"].append({"id": sample_id, "directory": name, "features": rows})
            atomic_json(output / "manifest.json", manifest)
            print(f"Feature export {len(manifest['samples'])}/{len(selected)}: {sample_id}")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
