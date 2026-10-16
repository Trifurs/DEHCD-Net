#!/usr/bin/env python3
"""Run scene-disjoint leave-one-event-out and supervised few-shot transfer studies."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
import signal
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_event_transfer import CONFIGS, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT, run_directory
from utils.config import XMLConfigParser, merge_dicts
from utils.local_settings import launch_settings
from utils.defaults import DEFAULT_SEEDS
from utils.protocol import atomic_json, config_digest, dataset_identity, digest, file_digest, source_identity
from utils.transfer_study import (EVENTS, fewshot_partition, loeo_partition, scan_records,
    seal_json, source_class_weights, validate_partition, write_partition)


def seal_study_plan(path, plan):
    """Keep sealed scientific settings while accepting only checked source changes."""
    path = Path(path)
    if path.exists():
        from utils.campaign_reuse import source_review_for
        from utils.local_settings import local_source_reviews
        previous = json.loads(path.read_text(encoding="utf-8"))
        source_review_for(previous["source"], plan["source"], local_source_reviews(ROOT))
        compatible = dict(plan, source=previous["source"])
        seal_json(path, compatible)
    else:
        seal_json(path, plan)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", nargs="+", choices=["loeo", "fewshot"], default=["loeo"])
    parser.add_argument("--sizes", nargs="+", choices=["s", "m", "l"], default=["s", "m", "l"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--target-exposure-note", default=launch_settings(ROOT)["target_exposure_note"],
                        help="Optional declaration of prior target-data inspection")
    parser.add_argument("--events", nargs="+", choices=EVENTS)
    parser.add_argument("--arms", nargs="+", choices=["zero_shot", "finetune", "scratch"],
                        default=["zero_shot", "finetune", "scratch"], help="Few-shot comparisons on the same fixed held-out scenes")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT.parent / "event_transfer_study")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-seed", type=int, default=3402, help="Existing Turkey-only weights shared across adaptation seeds")
    parser.add_argument("--loeo-init", choices=["auto", "scratch"], default="auto",
                        help="Reuse Turkey-only weights only when Turkey is a source event; its held-out fold always starts fresh")
    parser.add_argument("--split-seed", type=int, default=20261015)
    parser.add_argument("--train-fraction", type=float, default=0.1, help="Few-shot fraction of original scenes, not fraction of patches")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, help="Override training budget for both protocols")
    parser.add_argument("--fewshot-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, help="Uniform head learning rate override for trainable arms")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prepare-workers", type=int, default=8)
    parser.add_argument("--device", default=launch_settings(ROOT)["device"])
    parser.add_argument("--cache", choices=["normalized", "none"], default="normalized")
    parser.add_argument("--dry-run", action="store_true", help="Write and verify manifests/configs without training or tensor caching")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def make_config(args, protocol, event, size, seed, manifest, partition, arm):
    size_config = XMLConfigParser(CONFIGS[size]).parse().as_dict()
    study_config = XMLConfigParser(ROOT / "configs/event_transfer/study.xml").parse().as_dict()
    config = merge_dicts(size_config, study_config)
    config["model"] = copy.deepcopy(size_config["model"])
    config["dataset"].update(root=str(args.data_root.resolve()), candidate_roots=[str(args.data_root.resolve())],
                              manifest=str(manifest))
    training = config["training"]
    weights = source_class_weights(partition["train"])
    training.update(seed=seed, device=args.device, num_workers=args.num_workers,
                    class_weights=weights, hier_subclass_weights=weights[1:],
                    batch_size=args.batch_size or size_config["training"]["batch_size"],
                    checkpoint_dir="auto", resume=None)
    if protocol == "fewshot":
        training.update(epochs=args.fewshot_epochs, learning_rate=0.0003, warmup_epochs=3,
                        encoder_lr_multiplier=0.25, early_stop_patience=15,
                        sampler_num_samples=max(512, len(partition["train"])),
                        validation_event_macro=False, best_metric="foreground_miou")
    else:
        training.update(validation_event_macro=True, best_metric="event_macro_foreground_miou",
                        sampler_num_samples=min(6000, len(partition["train"])))
        if arm == "finetune":
            training.update(epochs=80, learning_rate=0.0003, encoder_lr_multiplier=0.5)
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if arm == "finetune":
        source_run = run_directory(args.source_root, size, args.source_seed)
        checkpoint = source_run / "checkpoints/best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No source checkpoint for {size}: {checkpoint}")
        training.update(init_checkpoint=str(checkpoint), init_checkpoint_sha256=file_digest(checkpoint))
    config["experiment"].update(id=f"{protocol}_{event}_{size}_{arm}", suite="transfer_study",
        held_out_event=event, model_variant=size, transfer_protocol=protocol, arm=arm)
    config["logging"].update(log_dir="auto", run_name=config["experiment"]["id"],
                              async_checkpoint=True, compact_checkpoints=False, validation_images=0)
    return config


def make_plan(args, records):
    jobs, cases = [], {}
    for protocol in args.protocols:
        events = list(args.events or (EVENTS if protocol == "loeo" else EVENTS[1:]))
        if protocol == "fewshot" and "turkey-earthquake" in events:
            raise ValueError("Turkey-only source weights cannot be used for a Turkey-target few-shot study; select the other events")
        for event in events:
            partition = (loeo_partition(records, event, args.val_fraction, args.split_seed) if protocol == "loeo" else
                         fewshot_partition(records, event, args.train_fraction, args.val_fraction, args.split_seed))
            case_id = f"{protocol}/{event}"
            case_dir = args.output / case_id
            manifest = case_dir / "manifest.csv"
            write_partition(manifest, partition)
            report = validate_partition(partition, event, require_event_disjoint=protocol == "loeo")
            target_total = sum(1 for row in records if row["event"] == event)
            case = {"protocol": protocol, "held_out_event": event, "manifest": str(manifest),
                "split_algorithm": "scene_class_presence_and_pixel_balance_v2",
                "manifest_sha256": file_digest(manifest), "splits": report,
                "target_training_labels": report["train"]["samples"] if protocol == "fewshot" else 0,
                "target_validation_labels": report["val"]["samples"] if protocol == "fewshot" else 0,
                "target_total_samples": target_total,
                "label_budget_note": "Reference labels stratify benchmark scenes; target train and validation annotations are reported separately",
                "grouping_note": "Original scene IDs are disjoint; absolute geographic separation cannot be established without footprint metadata",
                "historical_test_exposure": getattr(args, "target_exposure_note", "Not declared")}
            seal_json(case_dir / "split_audit.json", case)
            cases[case_id] = case
            arms = (["finetune"] if args.loeo_init == "auto" and event != "turkey-earthquake" else ["scratch"]) if protocol == "loeo" else args.arms
            for size in args.sizes:
                for seed in args.seeds:
                    for arm in arms:
                        if arm == "zero_shot" and seed != args.seeds[0]:
                            continue
                        job_seed = args.source_seed if arm == "zero_shot" else seed
                        run = case_dir / f"dehcd_{size}" / f"seed_{job_seed}" / arm
                        if arm == "zero_shot":
                            source = run_directory(args.source_root, size, args.source_seed)
                            if not (source / "checkpoints/best.pth").is_file():
                                raise FileNotFoundError(f"Missing source checkpoint: {source}")
                            config = json.loads((source / "config_snapshot.json").read_text())
                            if Path(config["dataset"]["root"]).resolve() != args.data_root.resolve():
                                raise ValueError("Zero-shot source checkpoint uses a different dataset root")
                            source_protocol = json.loads((source / "protocol.json").read_text())
                            seen_events = {r.get("event") for value in source_protocol["datasets"].values()
                                           for r in value["records"]}
                            if event in seen_events:
                                raise ValueError(f"Zero-shot checkpoint has already seen target event {event}")
                            source_checkpoint_sha256 = file_digest(source / "checkpoints/best.pth")
                        else:
                            source = run
                            config = make_config(args, protocol, event, size, seed, manifest, partition, arm)
                            source_checkpoint_sha256 = config["training"].get("init_checkpoint_sha256")
                        config_path = case_dir / "configs" / f"{size}_seed_{job_seed}_{arm}.json"
                        seal_json(config_path, config)
                        jobs.append({"case": case_id, "protocol": protocol, "event": event, "size": size,
                            "seed": job_seed, "arm": arm, "run": str(run), "evaluation_run": str(source),
                            "config": str(config_path), "config_sha256": config_digest(config),
                            "source_checkpoint_sha256": source_checkpoint_sha256,
                            "result": str(run / "test/result.json")})
    return {"schema_version": 1, "source": source_identity(ROOT), "dataset_sha256": digest(records),
            "split_seed": args.split_seed, "seeds": args.seeds, "source_seed": args.source_seed,
            "cache": args.cache, "cases": cases, "jobs": jobs}


def prepare_cache(args, plan):
    if args.cache == "none":
        return None
    from datasets import build_dataset
    from utils.prepared_data import input_signature, preprocessing_identity, prepare_split

    config = XMLConfigParser(ROOT / "configs/event_transfer_earthquake.xml").parse().as_dict()
    config["dataset"].update(root=str(args.data_root.resolve()), candidate_roots=[str(args.data_root.resolve())])
    config["dataset"].pop("manifest", None)
    required = {"train", "val", "test"} if "loeo" in args.protocols else {"test"}
    index = {"preprocessing_sha256": preprocessing_identity(), "splits": {}}
    for split in sorted(required):
        print(f"Prepare shared source tensors: {split}", flush=True)
        dataset = build_dataset(config, split, training=False)
        index["splits"][input_signature(dataset)] = prepare_split(dataset,
            {"stat": dataset_identity(dataset)}, args.output / "cache/source", workers=args.prepare_workers)
    path = args.output / "cache/index.json"
    atomic_json(path, index)
    return path


def run_child(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(map(str, command)), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Child process failed ({code}); log: {log}")


def train_job(job, env, plan):
    if job["arm"] == "zero_shot":
        return
    run = Path(job["run"])
    config = json.loads(Path(job["config"]).read_text())
    snapshot = run / "config_snapshot.json"
    if snapshot.exists() and config_digest(json.loads(snapshot.read_text())) != job["config_sha256"]:
        raise ValueError(f"Existing run uses different hyperparameters: {run}")
    protocol_path = run / "protocol.json"
    if protocol_path.exists():
        from utils.campaign_reuse import verify_source_import
        verify_source_import(run, json.loads(protocol_path.read_text())["source"], plan["source"])
    summary_path = run / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "complete" and summary.get("config_sha256") == job["config_sha256"]:
            if summary["stop_reason"] not in {"epochs_completed", "early_stopping"} or not (run / "checkpoints/best.pth").exists():
                raise ValueError(f"Incomplete training evidence: {run}")
            print(f"Reuse completed training: {run}", flush=True)
            return
    command = [sys.executable, str(ROOT / "tools/train.py"), "--config", job["config"], "--run-dir", str(run)]
    last = run / "checkpoints/last.pth"
    if last.exists():
        command += ["--resume", str(last)]
    else:
        command += ["--restart-incomplete"]
    run_child(command, run / "logs/console.log", env)


def summarize(plan, output):
    from tools.validate_results import validate_result

    rows, groups = [], {}
    for job in plan["jobs"]:
        result = json.loads(Path(job["result"]).read_text())
        metrics = validate_result(result, expected_config=job["config_sha256"], verify_files=True)
        if result["evaluation_event"] != job["event"]:
            raise ValueError("Evaluation event mismatch")
        cm = np_array(result["confusion_matrix"])
        tp = cm[2:, 2:].sum()
        union = cm[2:, :].sum() + cm[:, 2:].sum() - tp
        metrics["damaged_union_iou"] = float(tp / union) if union else 0.0
        case = plan["cases"][job["case"]]
        row = {key: job[key] for key in ("protocol", "event", "size", "seed", "arm")}
        row.update(target_training_labels=case["target_training_labels"] if job["arm"] != "zero_shot" else 0,
                   target_validation_labels=case["target_validation_labels"] if job["arm"] != "zero_shot" else 0,
                   test_samples=result["evaluated_samples"], result=job["result"], **metrics)
        rows.append(row)
        key = "/".join(job[k] for k in ("protocol", "event", "size", "arm"))
        groups.setdefault(key, []).append(row)
    aggregates = {}
    for key, values in groups.items():
        expected_n = 1 if values[0]["arm"] == "zero_shot" else len(plan["seeds"])
        aggregates[key] = {"n": len(values), "expected_n": expected_n,
            "metrics": {metric: {"mean": statistics.mean(row[metric] for row in values),
                "sample_std": statistics.stdev(row[metric] for row in values) if len(values) > 1 else None}
                for metric in ("foreground_miou", "class_1_iou", "class_2_iou", "class_3_iou", "damaged_union_iou")}}
    loeo_macro = {}
    for size in sorted({row["size"] for row in rows if row["protocol"] == "loeo"}):
        by_seed = {}
        for seed in plan["seeds"]:
            selected = [row for row in rows if row["protocol"] == "loeo" and row["size"] == size and row["seed"] == seed]
            by_seed[str(seed)] = statistics.mean(row["foreground_miou"] for row in selected)
        loeo_macro[size] = {"metric": "unweighted mean of held-out-event foreground_miou",
            "events": sorted({row["event"] for row in rows if row["protocol"] == "loeo" and row["size"] == size}),
            "by_seed": by_seed, "mean": statistics.mean(by_seed.values()),
            "sample_std": statistics.stdev(by_seed.values()) if len(by_seed) > 1 else None}
    atomic_json(output / "summary/aggregate.json", {"status": "complete", "groups": aggregates,
        "loeo_event_macro": loeo_macro,
        "interpretation": "LOEO is event-held-out generalization; few-shot uses explicitly reported target training and validation labels",
        "historical_test_exposure": {key: case.get("historical_test_exposure", "Not declared") for key, case in plan["cases"].items()}})
    with (output / "summary/per_seed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved study summary: {output / 'summary/aggregate.json'}", flush=True)


def np_array(value):
    import numpy as np
    return np.asarray(value, dtype=np.float64)


def main():
    args = parse_args()
    args.output, args.data_root, args.source_root = (p.expanduser().resolve() for p in (args.output, args.data_root, args.source_root))
    for field in ("protocols", "sizes", "seeds", "arms"):
        value = getattr(args, field)
        if not value or len(value) != len(set(value)):
            raise ValueError(f"--{field} must have distinct entries")
    if any(seed < 0 or seed >= 2**32 for seed in [*args.seeds, args.split_seed, args.source_seed]):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    if not 0 < args.val_fraction < 0.5 or not 0 < args.train_fraction < 1 - args.val_fraction:
        raise ValueError("Invalid scene split fractions")
    if args.num_workers < 0 or args.prepare_workers < 0 or args.fewshot_epochs < 1:
        raise ValueError("Invalid worker count or epoch budget")
    if args.epochs is not None and args.epochs < 1 or args.batch_size is not None and args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.pop("DEHCD_PREPARED_INDEX", None)
        events = None if "loeo" in args.protocols else (args.events or EVENTS[1:])
        records = scan_records(args.data_root, workers=args.prepare_workers, events=events)
        plan = make_plan(args, records)
        seal_study_plan(args.output / "study_plan.json", plan)
        print(f"Verified {len(plan['cases'])} scene-disjoint cases; {len(plan['jobs'])} model/evaluation jobs", flush=True)
        for name, case in plan["cases"].items():
            print(name, {split: (row["samples"], row["scenes"]) for split, row in case["splits"].items()}, flush=True)
        if args.dry_run:
            return
        index = prepare_cache(args, plan)
        if args.prepare_only:
            return
        env = dict(os.environ, PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONWARNINGS="ignore")
        if index:
            env["DEHCD_PREPARED_INDEX"] = str(index)
        # All training/checkpoint selection precedes final test evaluation.
        for number, job in enumerate(plan["jobs"], 1):
            atomic_json(args.output / "status.json", {"stage": "train", "job": number, "total_jobs": len(plan["jobs"]),
                        "case": job["case"], "size": job["size"], "arm": job["arm"]})
            train_job(job, env, plan)
        for number, job in enumerate(plan["jobs"], 1):
            atomic_json(args.output / "status.json", {"stage": "test", "job": number, "total_jobs": len(plan["jobs"]),
                        "case": job["case"], "size": job["size"], "arm": job["arm"]})
            run = Path(job["run"])
            command = [sys.executable, str(ROOT / "tools/test.py"), "--train-root", str(args.output),
                "--runs", job["evaluation_run"], "--output-root", str(run / "test"), "--result-file", job["result"],
                "--manifest", plan["cases"][job["case"]]["manifest"], "--event", job["event"],
                "--checkpoint", "best", "--save-sample-metrics", "--no-confusion-plot", "--no-profile", "--skip-existing",
                "--device", args.device, "--num-workers", str(args.num_workers)]
            run_child(command, run / "logs/evaluation.log", env)
        summarize(plan, args.output)
        atomic_json(args.output / "status.json", {"stage": "complete", "total_jobs": len(plan["jobs"])})


if __name__ == "__main__":
    main()
