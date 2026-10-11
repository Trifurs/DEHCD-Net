#!/usr/bin/env python3
"""Run the DEHCD-Net event-level earthquake transfer experiment."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.protocol import atomic_json

CONFIGS = {size: ROOT / "configs/event_transfer" / f"dehcd_{size}.xml" for size in ("s", "m", "l")}
DEFAULT_DATA_ROOT = Path("/media/whu/0d7bb559-7b14-4875-843f-08befb3ca56b/myData/event_transfer_earthquake")
DEFAULT_OUTPUT = Path("/home/whu/桌面/myResult/DEHCD-Net/event_transfer_earthquake")
DEFAULT_TEST_EVENTS = ["haiti-earthquake", "morocco-earthquake", "noto-earthquake"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1051, 2060])
    parser.add_argument("--sizes", nargs="+", choices=list(CONFIGS), default=["s", "m", "l"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--test-events", nargs="+", choices=DEFAULT_TEST_EVENTS, default=DEFAULT_TEST_EVENTS,
                        help="Held-out events to test sequentially, in the supplied order.")
    parser.add_argument("--cache", choices=["normalized", "none"], default="normalized")
    parser.add_argument("--prepare-workers", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def run_directory(output: Path, size: str, seed: int) -> Path:
    return output / "sizes" / size / "runs/event_transfer/custom" / f"dehcd_{size}" / f"seed_{seed}"


def event_summary(output: Path, sizes, seeds, events) -> None:
    from tools.validate_results import validate_result
    from tools.test import load_config_snapshot, resolve_checkpoint
    from utils.protocol import file_digest

    summary = {"schema_version": 2, "status": "complete", "seeds": list(seeds),
               "expected_n": len(seeds), "test_order": list(events), "models": {}}
    rows = []
    for size in sizes:
        model_summary = {}
        for event in events:
            by_seed = {}
            sample_counts = set()
            for seed in seeds:
                run = run_directory(output, size, seed)
                result_path = run / "test" / event / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                config = load_config_snapshot(run)
                metrics = validate_result(result, expected_config=config)
                if (result.get("seed") != seed or result.get("evaluation_event") != event
                        or result.get("checkpoint_sha256") != file_digest(resolve_checkpoint(run, "best"))):
                    raise ValueError(f"Event result identity/checkpoint mismatch: {result_path}")
                by_seed[seed] = metrics
                sample_counts.add(result["evaluated_samples"])
                rows.append({"size": size, "event": event, "seed": seed,
                             "samples": result["evaluated_samples"], "result": str(result_path), **metrics})
            if len(sample_counts) != 1:
                raise ValueError(f"Event sample counts differ across seeds: {size}/{event}")
            model_summary[event] = {"samples": sample_counts.pop(), "metrics": {
                name: {"mean": statistics.mean(row[name] for row in by_seed.values()),
                       "sample_std": statistics.stdev(row[name] for row in by_seed.values()) if len(seeds) > 1 else None,
                       "n": len(seeds), "by_seed": {str(seed): by_seed[seed][name] for seed in seeds}}
                for name in next(iter(by_seed.values()))}}
        summary["models"][size] = {"events": model_summary}
    destination = output / "summary/event_metrics.json"
    atomic_json(destination, summary)
    with (destination.parent / "event_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved event-level summary: {destination}")


def main() -> int:
    args = parse_args()
    control_only = any(getattr(args, name) for name in ("audit_only", "dry_run", "preflight_only", "prepare_only"))
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must contain at least one distinct integer")
    if not args.test_events or len(args.test_events) != len(set(args.test_events)):
        raise ValueError("--test-events must contain at least one distinct event")
    if len(args.sizes) != len(set(args.sizes)):
        raise ValueError("--sizes must be distinct")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Seeds must be in [0, 2**32)")
    if not args.data_root.expanduser().is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")
    if args.prepare_workers < 0:
        raise ValueError("--prepare-workers must be non-negative")
    for name in ("batch_size", "epochs"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output = args.output.expanduser().resolve()
    if args.audit_only:
        return run_experiments(args, output, control_only)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".event_transfer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another event-transfer command is writing this output directory") from exc
        return run_experiments(args, output, control_only)


def run_experiments(args, output, control_only):
    for size in args.sizes:
        print(f"\nDEHCD-Net-{size.upper()}: train seeds {args.seeds}, then test {args.test_events}", flush=True)
        size_output = output / "sizes" / size
        command = campaign_command(args, size, size_output, output / "cache")
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
        if control_only:
            continue
        test_env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
        test_env.pop("DEHCD_PREPARED_INDEX", None)
        index = size_output / "cache/index.json"
        if args.cache == "normalized":
            test_env["DEHCD_PREPARED_INDEX"] = str(index)
        for seed in args.seeds:
            for event in args.test_events:
                run = run_directory(output, size, seed)
                event_output = run / "test" / event
                test_command = [
                    sys.executable, str(ROOT / "tools/test.py"),
                    "--train-root", str(size_output / "runs"), "--runs", str(run),
                    "--output-root", str(event_output), "--result-file", str(event_output / "result.json"),
                    "--split", "test", "--event", event, "--checkpoint", "best",
                    "--save-sample-metrics", "--no-confusion-plot", "--no-profile", "--skip-existing",
                ]
                print(f"Testing size={size}, seed={seed}, event={event}", flush=True)
                tested = subprocess.run(test_command, cwd=ROOT, env=test_env, check=False)
                if tested.returncode != 0:
                    return tested.returncode
    if not control_only:
        event_summary(output, args.sizes, args.seeds, args.test_events)
    return 0


def campaign_command(args, size, output, cache_root):
    command = [
        sys.executable,
        str(ROOT / "tools/run_multiseed.py"),
        "--config", str(CONFIGS[size]),
        "--seeds", *map(str, args.seeds),
        "--data-root", f"event_transfer={args.data_root.expanduser().resolve()}",
        "--device", args.device,
        "--output", str(output),
        "--cache-root", str(cache_root),
        "--cache", args.cache,
        "--prepare-workers", str(args.prepare_workers),
        "--quiet-warnings",
        "--resume",
        "--skip-test",
    ]
    if args.num_workers is not None:
        command += ["--num-workers", str(args.num_workers)]
    if args.batch_size is not None:
        command += ["--batch-size", str(args.batch_size)]
    if args.epochs is not None:
        command += ["--epochs", str(args.epochs)]
    for name in ("audit_only", "dry_run", "preflight_only", "prepare_only"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))

    return command


if __name__ == "__main__":
    raise SystemExit(main())
