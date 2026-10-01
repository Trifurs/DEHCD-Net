"""Isolated, resumable repeat experiments with immutable code/config/data contracts."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import XMLConfigParser
from utils.protocol import atomic_json, config_digest, source_identity, dataset_identity, assert_disjoint, digest
from tools.validate_results import validate_result
from utils.comparison_protocol import audit_plan, declaration
from utils.campaign_layout import run_path, result_path, summary_path
from utils.progress import read_status, write_status, duration
from utils.campaign_progress import CampaignProgress


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default=str(ROOT / "configs/experiments/catalog.json"))
    p.add_argument("--suite", nargs="+", default=["main"], choices=["main", "ablation", "recipe", "sensitivity", "adapter", "auxiliary", "head_recipe", "optimizer_recipe", "cross_event", "all"])
    p.add_argument("--datasets", nargs="+", default=["bright", "haiti", "cau_flood", "xbd"])
    p.add_argument("--experiments", nargs="+", help="Exact experiment ids; selects across suites")
    p.add_argument("--config", help="One custom XML/JSON config, e.g. an event-holdout configuration")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1051, 2060, 3069, 4078])
    p.add_argument("--output", required=True, help="New campaign directory; never use a historical run directory")
    p.add_argument("--data-root", action="append", default=[], metavar="DATASET=PATH")
    p.add_argument("--encoder-checkpoint", action="append", default=[], metavar="MODEL=PATH",
                   help="Explicit local encoder-only weights for a new damage baseline; all seeds share its SHA256")
    p.add_argument("--manifest", action="append", default=[], metavar="DATASET=CSV")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=None, help="Override config/profile worker count")
    p.add_argument("--runtime-profile", choices=["none", "rtx5090"], default="none",
                   help="Shared measured execution settings; does not shorten training or tune individual models")
    p.add_argument("--epochs", type=int, help="Uniform override; recorded in every configuration")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--gradient-accumulation-steps", type=int, help="Uniform accumulation override, recorded for all models")
    p.add_argument("--fingerprint", choices=["stat", "sha256"], default="stat")
    p.add_argument("--cache", choices=["normalized", "none"], default="normalized",
                   help="Prepare lossless pre-augmentation tensors and label counts once for the campaign")
    p.add_argument("--prepare-workers", type=int, default=8)
    p.add_argument("--prepare-only", action="store_true", help="Seal data and build shared caches, without training")
    p.add_argument("--quiet-warnings", action="store_true", help="Suppress Python/logging warnings in parent and child consoles; errors remain visible")
    p.add_argument("--dry-run", action="store_true", help="Write the plan without loading datasets or models")
    p.add_argument("--preflight-only", action="store_true", help="Validate data/splits and seal the protocol, without training")
    p.add_argument("--resume", action="store_true", help="Continue the same sealed protocol from last.pth")
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def assignments(values):
    out = {}
    for value in values:
        key, sep, path = value.partition("=")
        if not sep or not path or key in out:
            raise ValueError(f"Expected unique DATASET=PATH: {value}")
        out[key] = str(Path(path).expanduser().resolve())
    return out


def make_plan(args):
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    catalog = json.loads(Path(args.catalog).read_text())["experiments"]
    if args.config:
        config_path = Path(args.config).resolve()
        config = XMLConfigParser(config_path).parse().as_dict()
        selected = [{"id": config_path.stem, "config": str(config_path), "suite": "custom",
                     "dataset": config.get("experiment", {}).get("dataset", config.get("dataset", {}).get("type", "custom"))}]
    else:
        selected = [e for e in catalog if e["id"] in args.experiments] if args.experiments else [
            e for e in catalog if e["dataset"] in args.datasets and ("all" in args.suite or e["suite"] in args.suite)]
        if args.experiments and set(args.experiments) != {e["id"] for e in selected}:
            raise ValueError("Unknown experiment id in --experiments")
    if not selected:
        raise ValueError("No experiments selected")
    roots, manifests = assignments(args.data_root), assignments(args.manifest)
    if (set(roots) | set(manifests)) - {e["dataset"] for e in selected}:
        raise ValueError("Data root/manifest key does not match a selected dataset")
    weights = assignments(getattr(args, "encoder_checkpoint", []))
    from utils.protocol import file_digest
    weight_hashes = {name: file_digest(path) for name, path in weights.items()}
    used_weights = set()
    def materialize(exp, seed):
        config = XMLConfigParser(ROOT / exp["config"]).parse().as_dict()
        ds = exp["dataset"]
        model_cfg = config.setdefault("model", {})
        model_name = str(model_cfg.get("compare_model") or model_cfg.get("name", "")).lower()
        if model_name in weights:
            model_cfg.update(encoder_checkpoint=weights[model_name], encoder_checkpoint_sha256=weight_hashes[model_name])
            used_weights.add(model_name)
        if model_cfg.get("encoder_checkpoint"):
            weight_path = str(Path(model_cfg["encoder_checkpoint"]).expanduser().resolve(strict=True))
            model_cfg.update(encoder_checkpoint=weight_path, encoder_checkpoint_sha256=file_digest(weight_path))
        if ds in roots:
            config["dataset"].update(root=roots[ds], candidate_roots=[roots[ds]])
        if ds in manifests:
            config["dataset"]["manifest"] = manifests[ds]
        from utils.runtime import apply_runtime_profile
        apply_runtime_profile(config, getattr(args, "runtime_profile", "none"), ROOT)
        train = config.setdefault("training", {})
        if train.get("max_train_batches", 0) or train.get("max_val_batches", 0):
            raise ValueError("Campaigns require full train/validation splits; use train.py for smoke tests")
        train.update(seed=seed, device=args.device, deterministic=True, resume=None, checkpoint_dir="auto")
        if args.num_workers is not None:
            train["num_workers"] = args.num_workers
        if args.epochs is not None:
            train["epochs"] = args.epochs
        if args.batch_size is not None:
            train["batch_size"] = args.batch_size
        if getattr(args, "gradient_accumulation_steps", None) is not None:
            if args.gradient_accumulation_steps < 1:
                raise ValueError("gradient accumulation must be positive")
            train["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        config.setdefault("logging", {}).update(log_dir="auto", wandb=False,
            async_checkpoint=True, compact_checkpoints=False)
        config.setdefault("experiment", {}).update(id=exp["id"], suite=exp["suite"], dataset=ds)
        name = f"{exp['id']}__seed_{seed}"
        return {"id": name, "experiment": exp["id"], "dataset": ds, "seed": seed,
                "config": config, "config_sha256": config_digest(config)}
    jobs = [materialize(exp, seed) for exp in selected for seed in args.seeds]
    dataset_order = list(dict.fromkeys(job["dataset"] for job in jobs))
    jobs.sort(key=lambda job: dataset_order.index(job["dataset"]))
    if set(weights) - used_weights:
        raise ValueError("--encoder-checkpoint contains a model absent from the selected experiments")
    # References are checked even when scheduled in a different invocation;
    # they do not become extra training jobs. Uniform CLI overrides also apply.
    by_id = {e["id"]: e for e in catalog + selected}
    registry = {j["experiment"]: j["config"] for j in jobs}
    references = {}
    pending = list(registry)
    while pending:
        name = pending.pop()
        ref = declaration(registry[name]).get("reference")
        if ref and ref not in registry:
            if ref not in by_id:
                raise ValueError(f"Declared reference {ref} is missing from the experiment catalog")
            registry[ref] = materialize(by_id[ref], args.seeds[0])["config"]
            references[ref] = registry[ref]
            pending.append(ref)
    plan = {"schema_version": 2, "layout_version": 2,
            "checkpoint_retention": "best_and_last",
            "seeds": args.seeds, "source": source_identity(ROOT),
            "fingerprint_mode": args.fingerprint, "jobs": jobs, "comparison_references": references}
    plan["comparison_design"] = audit_plan(plan)
    return plan


def seal_datasets(plan):
    from datasets import build_dataset
    identities, prepared_sources = {}, {}
    for job in plan["jobs"]:
        cfg = job["config"]
        key = digest({k: cfg.get(k) for k in ("dataset", "task", "normalization")})
        if key not in identities:
            splits = {}
            for split in ("train", "val", "test"):
                dataset = build_dataset(cfg, split=split, training=False)
                stat = dataset_identity(dataset)
                modes = {"stat": stat}
                if plan["fingerprint_mode"] == "sha256":
                    modes["sha256"] = dataset_identity(dataset, "sha256")
                    splits[split] = {**modes["sha256"], "stat_sha256": stat["sha256"]}
                else:
                    splits[split] = stat
                prepared_sources[(key, split)] = (dataset, modes)
            identities[key] = {"splits": splits, "checks": assert_disjoint(splits)}
            print(f"Data checked: {job['dataset']} " + "/".join(str(splits[s]["samples"]) for s in ("train", "val", "test")), flush=True)
        job["data_key"] = key
    plan["data"] = identities
    plan["comparison_design"] = audit_plan(plan)
    return prepared_sources


def prepare_inputs(plan, sources, output, workers):
    from tqdm import tqdm
    from utils.prepared_data import input_signature, preprocessing_identity, prepare_split
    index = {"preprocessing_sha256": preprocessing_identity(), "splits": {}}
    unique = {}
    for (key, split), (dataset, identities) in sources.items():
        signature = input_signature(dataset)
        unique.setdefault(signature, (key, split, dataset, identities))
    total_samples = sum(len(entry[2]) for entry in unique.values())
    completed_samples, started, last_write = 0, time.monotonic(), 0.0
    for number, (signature, (key, split, dataset, identities)) in enumerate(unique.items(), 1):
        name = next(j["dataset"] for j in plan["jobs"] if j["data_key"] == key)
        print(f"Prepare {number}/{len(unique)}: {name}/{split} ({len(dataset)} samples)", flush=True)
        with tqdm(total=len(dataset), desc=f"Cache {name}/{split}", dynamic_ncols=True,
                  mininterval=1.0, disable=not sys.stderr.isatty()) as bar:
            def progress(done, total, reused):
                nonlocal last_write
                bar.update(done - bar.n)
                now = time.monotonic()
                if now - last_write < 2.0 and done < total and done != 0: return
                last_write = now
                finished = completed_samples + done
                eta = (now - started) * (total_samples - finished) / finished if finished else None
                state = {"status": "preparing", "phase": f"{name}/{split}", "prepared_split": number,
                         "total_splits": len(unique), "prepared_samples": finished, "total_samples": total_samples,
                         "estimated_remaining_seconds": eta, "completed_jobs": 0, "total_jobs": len(plan["jobs"]),
                         "cache_reused": reused, "updated_at": time.time()}
                write_status(output / "progress.json", state)
                (output / "progress.txt").write_text(f"Preparing {number}/{len(unique)}: {name}/{split}\n"
                    f"Samples {finished}/{total_samples} | preparation ETA ~{duration(eta)} (estimate)\n")
            progress(0, len(dataset), False)
            index["splits"][signature] = prepare_split(dataset, identities, output / "cache" / name,
                                                       workers=workers, progress=progress)
        completed_samples += len(dataset)
    path = output / "cache/index.json"
    atomic_json(path, index)
    write_status(output / "progress.json", {"status": "ready", "completed_jobs": 0,
        "total_jobs": len(plan["jobs"]), "prepared_splits": len(unique), "updated_at": time.time()})
    (output / "progress.txt").write_text(f"Preparation complete: {len(unique)} shared splits.\n"
        f"Ready for {len(plan['jobs'])} training runs.\n")
    return path


def aggregate(plan, output, verify_checkpoints=True, *, audit=True, cache=None):
    if audit: audit_plan(plan)
    summary_path(output, plan, "aggregate.json").parent.mkdir(parents=True, exist_ok=True)
    rows, groups, missing = [], {}, []
    for job in plan["jobs"]:
        path = result_path(output, job, plan)
        summary = run_path(output, job, plan) / "training_summary.json"
        if not path.exists() or not summary.exists():
            missing.append(job["id"])
            continue
        if (summary.parent / "failure.json").exists() or (summary.parent / "test/failure.json").exists():
            missing.append(job["id"])
            continue
        stamp = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (path, summary))
        if cache is not None and not verify_checkpoints and job["id"] in cache and cache[job["id"]][0] == stamp:
            row = cache[job["id"]][1]
            rows.append(row)
            groups.setdefault(job["experiment"], []).append(row)
            continue
        completion = json.loads(summary.read_text())
        if completion.get("status") != "complete" or completion["config_sha256"] != job["config_sha256"]:
            raise ValueError(f"Invalid training completion: {job['id']}")
        result = json.loads(path.read_text())
        test_identity = plan["data"][job["data_key"]]["splits"]["test"]
        # Evaluator uses stat fingerprint; SHA256 campaign also retains a stat fingerprint below.
        metrics = validate_result(result, job["config_sha256"], test_identity.get("stat_sha256", test_identity["sha256"]), verify_checkpoints)
        from utils.prediction import resolve_tta, prediction_rule
        if result["runtime"]["test_time_augmentation"] != resolve_tta(job["config"]):
            raise ValueError(f"TTA protocol mismatch: {path}")
        if result["runtime"].get("prediction_rule", "damage_argmax") != prediction_rule(job["config"]):
            raise ValueError(f"Prediction rule mismatch: {path}")
        if result["seed"] != job["seed"] or result["experiment"]["id"] != job["experiment"]:
            raise ValueError(f"Seed/experiment identity mismatch: {path}")
        row = {"experiment": job["experiment"], "dataset": job["dataset"], "seed": job["seed"],
               "split": "test", "config_sha256": job["config_sha256"], "checkpoint": result["checkpoint"], **metrics}
        if cache is not None: cache[job["id"]] = (stamp, row)
        rows.append(row)
        groups.setdefault(job["experiment"], []).append(row)
    if rows:
        with summary_path(output, plan, "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))); writer.writeheader(); writer.writerows(rows)
    aggregates = []
    for name, items in groups.items():
        expected = sum(j["experiment"] == name for j in plan["jobs"])
        entry = {"experiment": name, "n": len(items), "expected_n": expected, "complete": len(items) == expected,
                 "seeds": [r["seed"] for r in items], "metrics": {}}
        for key in items[0]:
            if key in {"experiment", "dataset", "seed", "split", "config_sha256", "checkpoint"}:
                continue
            values = [r[key] for r in items]
            entry["metrics"][key] = {"mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None}
        aggregates.append(entry)
    report = {"status": "complete" if not missing else "incomplete", "missing": missing,
              "unit": "independent training seed on one fixed split; SD uses ddof=1", "experiments": aggregates}
    atomic_json(summary_path(output, plan, "aggregate.json"), report)
    return report


def run_child(command, env, run, job, monitor, stage):
    import signal
    log = run / "logs/console.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(env, DEHCD_PROGRESS_FILE=str(run / "progress.json"))
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{stage}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                state = read_status(run / "progress.json")
                if state.get("stage") != stage:
                    state = {"stage": stage, "phase": "initializing", "epochs": job["config"]["training"]["epochs"]}
                monitor.refresh(job, state)
                time.sleep(0.5)
        except BaseException:
            os.killpg(process.pid, signal.SIGINT)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            monitor.refresh(job, read_status(run / "progress.json"), status="interrupted", force=True)
            raise
    if process.returncode == 0 and stage == "test":
        (run / "test/failure.json").unlink(missing_ok=True)
    if process.returncode:
        failure = read_status(run / "failure.json")
        error = failure.get("error") if stage == "train" else None
        if not error:
            tail = log.read_text(errors="replace")[-12000:]
            marker = tail.rfind("Traceback (most recent call last):")
            error = tail[marker:] if marker >= 0 else f"process exit code {process.returncode}"
        if stage == "test": atomic_json(run / "test/failure.json", {"status": "failed", "error": error})
        monitor.refresh(job, read_status(run / "progress.json"), status="failed", force=True)
        raise RuntimeError(f"{job['id']} {stage} failed: {error}\nFull log: {log}")


def main():
    args = parse_args()
    if args.quiet_warnings:
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        import logging
        logging.getLogger().setLevel(logging.ERROR)
    if args.prepare_workers < 0:
        raise ValueError("--prepare-workers must be non-negative")
    # Verify source data afresh once per invocation, never a inherited cache view.
    os.environ.pop("DEHCD_PREPARED_INDEX", None)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / ".campaign.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another process is running this campaign") from exc
    plan = make_plan(args)
    if args.dry_run:
        atomic_json(output / "plan.json", plan)
        print(f"Planned {len(plan['jobs'])} jobs; no training started: {output / 'plan.json'}")
        return
    sources = seal_datasets(plan)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != plan:
            raise ValueError("Campaign source/config/seeds/data changed; choose a NEW output directory")
        if not (args.resume or args.preflight_only or args.prepare_only or args.aggregate_only):
            raise FileExistsError("Campaign exists; use --resume to continue the same protocol")
    else:
        if args.aggregate_only: raise FileNotFoundError("No sealed campaign protocol")
        atomic_json(protocol_path, plan)
    if args.preflight_only:
        print(f"Preflight passed for {len(plan['jobs'])} jobs; no training started")
        return
    result_cache = {}
    if not args.aggregate_only:
        index = prepare_inputs(plan, sources, output, args.prepare_workers) if args.cache == "normalized" else None
        if args.prepare_only:
            print(f"Prepared {len(plan['jobs'])} jobs; no training started. Output: {output}")
            return
        del sources
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
        if index: env["DEHCD_PREPARED_INDEX"] = str(index)
        # Validate existing results once. Subsequent aggregation caches validated
        # metric rows, not large repeated per-image identities or model weights.
        aggregate(plan, output, cache=result_cache, audit=False)
        monitor = CampaignProgress(plan, output)
        try:
            for job in plan["jobs"]:
                if job["id"] in monitor.completed:
                    continue
                if source_identity(ROOT)["sha256"] != plan["source"]["sha256"]:
                    raise ValueError("Source code changed during the campaign")
                run = run_path(output, job, plan)
                cfg_path = output / "campaign/configs" / f"{job['id']}.json"
                atomic_json(cfg_path, job["config"])
                if not (run / "training_summary.json").exists() or (run / "failure.json").exists():
                    command = [sys.executable, str(ROOT / "tools/train.py"), "--config", str(cfg_path), "--run-dir", str(run)]
                    if args.resume and (run / "checkpoints/last.pth").exists():
                        command += ["--resume", str(run / "checkpoints/last.pth")]
                    elif args.resume and (run / "config_snapshot.json").exists():
                        command += ["--restart-incomplete"]
                    run_child(command, env, run, job, monitor, "train")
                result = result_path(output, job, plan)
                if not result.exists() or (run / "test/failure.json").exists():
                    command = [sys.executable, str(ROOT / "tools/test.py"), "--train-root", str(output / "runs"),
                        "--runs", str(run), "--output-root", str(result.parent), "--result-file", str(result),
                        "--split", "test", "--checkpoint", "best", "--save-sample-metrics", "--no-confusion-plot"]
                    if not job["config"].get("inference", {}).get("profile_model", True): command += ["--no-profile"]
                    run_child(command, env, run, job, monitor, "test")
                aggregate(plan, output, verify_checkpoints=False, audit=False, cache=result_cache)
                monitor.finish_job(job)
            report = aggregate(plan, output)
            monitor.refresh(status=report["status"], force=True)
        finally:
            monitor.write_index()
            monitor.close()
    else:
        report = aggregate(plan, output)
    print(f"Campaign {report['status']}: {summary_path(output, plan, 'aggregate.json')}")
    if report["missing"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
