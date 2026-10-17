"""Measure structural forward FLOPs without training or reading image data."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import XMLConfigParser
from utils.experiment_catalog import load_catalog, select_experiments
from utils.local_settings import launch_settings
from utils.protocol import atomic_json, config_digest


def input_channels(config, optical=None, sar=None):
    """Resolve the declared input contract; never inspect held-out labels."""
    ds, model = config.get("dataset", {}), config.get("model", {})
    def explicit(name):
        value = model.get(name, "auto")
        return None if value in (None, "auto", "") else int(value)
    optical = optical if optical is not None else explicit("optical_channels")
    sar = sar if sar is not None else explicit("sar_channels")
    if optical is None and ds.get("optical_band_indices"):
        optical = len(ds["optical_band_indices"])
    if sar is None and ds.get("sar_band_indices"):
        bands = ds["sar_band_indices"]
        sar = sum(map(len, bands)) if isinstance(bands[0], (list, tuple)) else len(bands) * len(ds.get("sar_dirs", [None]))
    kind = str(ds.get("type", "")).lower()
    known = {"bright", "bright1", "haiti", "cau", "cau1", "cau_flood", "xbd", "xbd1", "event_transfer"}
    if kind in known:
        optical = 3 if optical is None else optical
        if sar is None:
            sar = (3 if kind in {"xbd", "xbd1"} or ds.get("second_modality_rgb") else 1) * len(ds.get("sar_dirs", [None]))
    if optical is None or sar is None or min(optical, sar) <= 0:
        raise ValueError("Input channels are not fully declared; supply --optical-channels and --sar-channels")
    return optical, sar


def main():
    settings = launch_settings(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, help="Explicit XML/JSON configuration; repeat to profile several")
    parser.add_argument("--groups", nargs="+", default=["all"], help="Configuration groups, used when --config is omitted")
    parser.add_argument("--datasets", nargs="+", default=["bright", "haiti", "xbd", "cau_flood"])
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="Otherwise use each configuration's patch size")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--optical-channels", type=int)
    parser.add_argument("--sar-channels", type=int)
    parser.add_argument("--device", default=settings["device"])
    parser.add_argument("--output", type=Path, default=Path(settings["output"]) / "model_profiles")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.size and min(args.size) <= 0):
        parser.error("Batch and spatial dimensions must be positive")

    import torch
    from models import build_model
    from utils.flops import profile_forward
    from utils.model_metadata import model_metadata
    from utils.runtime import configure_runtime

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if args.config:
        paths = [p.expanduser().resolve(strict=True) for p in args.config]
    else:
        paths = [ROOT / row["config"] for row in select_experiments(load_catalog(ROOT), args.groups, args.datasets)]
    if not paths:
        parser.error("No configurations selected")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for index, path in enumerate(paths, 1):
        cfg = XMLConfigParser(path).parse().as_dict()
        name = cfg.get("experiment", {}).get("id", path.stem)
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Configuration identifier must be a filename component")
        model = None
        try:
            channels = input_channels(cfg, args.optical_channels, args.sar_channels)
            patch = cfg.get("dataset", {}).get("patch_size", 256)
            size = args.size or ([patch, patch] if isinstance(patch, int) else list(patch))
            execution = configure_runtime(cfg.get("training", {}), device)
            # Forking protects callers that import this tool; no training RNG is consumed.
            devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(0)
                model = build_model(cfg, *channels, initialize_encoder=False).to(device).float().eval()
                optical = torch.randn(args.batch_size, channels[0], *size, device=device)
                sar = torch.randn(args.batch_size, channels[1], *size, device=device)
                report = profile_forward(model, optical, sar)
            report.update(config_sha256=config_digest(cfg), configuration=name, device=str(device),
                          torch_version=str(torch.__version__), dtype="float32", execution=execution,
                          weights_source="random_initialization_for_structural_profiling",
                          implementation=model_metadata(model, device))
            complexity = report["complexity"]
            row = {"configuration": name, "dataset": cfg.get("experiment", {}).get("dataset", cfg.get("dataset", {}).get("type")),
                   "optical_shape": "x".join(map(str, report["input_shape"]["optical"])),
                   "sar_shape": "x".join(map(str, report["input_shape"]["sar"])),
                   "parameters": report["parameters"]["total"], "parameters_millions": report["parameters"]["total_millions"],
                   "gflops": complexity["gflops"], "matrix_gmacs": complexity["matrix_gmacs"],
                   "complete": complexity["complete"], "convention": complexity["convention"]}
            if not complexity["complete"]:
                failed.append(name)
                state = "UNSUPPORTED: " + ", ".join(complexity["unsupported_operators"])
            else:
                state = f"{row['parameters_millions']:.3f} M parameters | {row['gflops']:.4f} GFLOPs"
            rows.append(row)
            atomic_json(output / "profiles" / f"{name}.json", report)
            print(f"[{index}/{len(paths)}] {name}: {state}", flush=True)
        except Exception as exc:
            failed.append(name)
            atomic_json(output / "profiles" / f"{name}.json", {"configuration": name,
                "complexity": {"complete": False, "gflops": None}, "profile_error": str(exc)})
            print(f"[{index}/{len(paths)}] {name}: ERROR: {exc}", file=sys.stderr, flush=True)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    atomic_json(output / "index.json", {"complete": not failed, "profiles": [str(p) for p in paths], "failed": failed})
    print(f"Profiles: {output}", flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
