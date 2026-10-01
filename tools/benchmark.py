"""Measure model-only batch latency on a declared device, shape and precision."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import XMLConfigParser
from utils.model_metadata import model_metadata
from utils.protocol import atomic_json, environment, config_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--optical-channels", type=int, required=True)
    p.add_argument("--sar-channels", type=int, required=True)
    p.add_argument("--size", nargs=2, type=int, default=[256, 256])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if min(a.size + [a.batch_size, a.iterations]) <= 0 or a.warmup < 0:
        raise ValueError("Positive input sizes/iterations and nonnegative warmup are required")
    import torch
    from models import build_model
    from utils.prediction import predict_logits
    config = XMLConfigParser(a.config).parse().as_dict()
    device = torch.device(a.device)
    from utils.runtime import configure_runtime
    execution = configure_runtime(config.get("training", {}), device)
    model = build_model(config, a.optical_channels, a.sar_channels, initialize_encoder=False).to(device).eval()
    metadata = model_metadata(model, device)
    if a.amp and device.type == "cuda" and metadata["core_precision_policy"] == "fp32":
        raise ValueError("This baseline enforces an FP32 core. Omit --amp and benchmark all compared models in FP32.")
    optical = torch.randn(a.batch_size, a.optical_channels, *a.size, device=device)
    sar = torch.randn(a.batch_size, a.sar_channels, *a.size, device=device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(a.warmup):
            predict_logits(model, optical, sar, a.amp and device.type == "cuda", "none")
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(a.iterations):
            sync(); start = time.perf_counter()
            predict_logits(model, optical, sar, a.amp and device.type == "cuda", "none")
            sync(); timings.append((time.perf_counter() - start) * 1000)
    atomic_json(a.output, {"config_sha256": config_digest(config), "environment": environment(),
        "implementation": metadata, "execution": execution, "device": str(device), "amp": a.amp and device.type == "cuda", "input_shape": [a.batch_size, *a.size],
        "channels": [a.optical_channels, a.sar_channels], "warmup": a.warmup, "iterations": a.iterations,
        "parameters": sum(p.numel() for p in model.parameters()), "latency_ms_median": statistics.median(timings),
        "latency_ms_mean": statistics.mean(timings), "latency_ms_samples": timings,
        "throughput_images_per_second": a.batch_size * 1000 / statistics.mean(timings),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "scope": "forward only, synthetic device-resident inputs, no loading/writing/TTA; no FLOPs estimate"})
    print(a.output)


if __name__ == "__main__":
    main()
