"""Evidence-based import of local runs into a canonical campaign.

A source transition is accepted only after a review records the exact pair of
complete file digests. Unknown code changes, data changes and missing evidence
remain blocked; this module never infers compatibility from model names.
"""
from __future__ import annotations
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path

from utils.protocol import atomic_json, config_digest, digest, file_digest, scientific_config
from utils.campaign_layout import run_path, result_path
from utils.comparison_protocol import differences

STATES = ("reuse_complete", "reevaluate_only", "resume_training", "train_new", "retrain_required", "blocked_review")


def _json(path):
    return json.loads(Path(path).read_text())


def source_changes(before, after):
    a, b = before.get("files", {}), after.get("files", {})
    return {name: {"before": a.get(name), "after": b.get(name)}
            for name in sorted(a.keys() | b.keys()) if a.get(name) != b.get(name)}


def source_review_for(before, after, reviews):
    for value in (before, after):
        if not value.get("files") or value.get("sha256") != digest(value["files"]):
            raise ValueError("Invalid source file-map digest")
    if before.get("sha256") == after.get("sha256"):
        return {"status": "identical", "changes": {}}
    for review in reviews:
        if (review.get("from_sha256") == before.get("sha256") and
            review.get("to_sha256") == after.get("sha256") and
            review.get("status") == "training_compatible" and
            review.get("changes") == source_changes(before, after)):
            return review
    raise ValueError("Unreviewed source transition; exact training/selection changes require inspection")


def verify_source_import(run_dir, before, after):
    if before.get("sha256") == after.get("sha256"):
        source_review_for(before, after, [])
        return
    path = Path(run_dir) / "source_import.json"
    if not path.exists():
        raise ValueError("Source changed without a verified import record")
    record = _json(path)
    source_review_for(before, after, record.get("source_reviews", []))
    snapshot = _json(Path(run_dir) / "config_snapshot.json")
    if config_digest(snapshot) != record["config_sha256"]:
        raise ValueError("Source import belongs to a different scientific configuration")


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def execution_signature(config):
    # config_digest retains physical batch, accumulation, precision and RNG.
    # These execution settings are also checked even where the historical
    # config digest omitted worker/device bookkeeping.
    t = config.get("training", {})
    keys = ("device", "num_workers", "persistent_workers", "prefetch_factor", "cpu_threads",
            "amp", "allow_tf32", "optimizer_fused", "foreach_gradient_clip",
            "deterministic", "augmentation_seed_mode")
    return {k: t.get(k) for k in keys}


def _safe_run(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Run escapes the result directory: {path}")
    for parent in [path, *path.parents]:
        if parent == root: break
        if parent.is_symlink(): raise ValueError(f"Symlink run path is not imported: {path}")
    return path


def _checkpoint(path, root):
    import torch
    path = _safe_run(path, root)
    if not path.is_file(): raise ValueError(f"Missing checkpoint: {path}")
    checksum = file_digest(path)
    # Explicitly scoped to user-owned, audited local training assets. Never
    # monkeypatch torch.load or apply this policy to downloaded encoder weights.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Not a full local training checkpoint")
    if not all(torch.isfinite(v).all().item() for v in payload["model"].values() if torch.is_tensor(v)):
        raise ValueError("Checkpoint contains non-finite model state")
    return payload, checksum


def _validate_execution(protocol, cfg, env):
    recorded = protocol.get("environment", {})
    for key in ("python", "torch", "cuda_build", "cudnn", "gpu"):
        if recorded.get(key) != env.get(key):
            raise ValueError(f"Runtime environment differs: {key}")
    t, actual = cfg["training"], protocol.get("execution", {})
    checks = {"matmul_allow_tf32": t.get("allow_tf32", False),
              "cudnn_allow_tf32": t.get("allow_tf32", False),
              "cudnn_benchmark": not t.get("deterministic", False),
              "cudnn_deterministic": t.get("deterministic", False),
              "deterministic_algorithms": t.get("deterministic", False)}
    if "cpu_threads" in t: checks["cpu_threads"] = t["cpu_threads"]
    if "optimizer_fused" in t: checks["optimizer_fused"] = t["optimizer_fused"]
    for key, value in checks.items():
        if actual.get(key) != value: raise ValueError(f"Actual execution differs: {key}")


def inspect_run(run, job, plan, output, reviews, env):
    cfg = job["config"]
    snapshot = _json(run / "config_snapshot.json")
    delta = differences(scientific_config(snapshot), scientific_config(cfg))
    extra = differences(execution_signature(snapshot), execution_signature(cfg))
    if delta or extra:
        return {"state": "retrain_required", "reason": "Effective configuration differs",
                "differences": {**delta, **{f"execution.{k}": v for k, v in extra.items()}}}
    protocol = _json(run / "protocol.json")
    if protocol.get("config_sha256") != config_digest(snapshot):
        raise ValueError("Snapshot/protocol configuration digest mismatch")
    review = source_review_for(protocol["source"], plan["source"], reviews)
    expected = plan["data"][job["data_key"]]["splits"]
    for split in ("train", "val"):
        a, b = protocol["datasets"][split], expected[split]
        wanted = b.get("stat_sha256", b["sha256"]) if a.get("mode") == "stat" else b["sha256"]
        if a["sha256"] != wanted:
            return {"state": "retrain_required", "reason": f"{split} file identities differ"}
    _validate_execution(protocol, cfg, env)
    last, last_sha = _checkpoint(run / "checkpoints/last.pth", output)
    best, best_sha = _checkpoint(run / "checkpoints/best.pth", output)
    for checkpoint in (last, best):
        if config_digest(checkpoint.get("config", {})) != job["config_sha256"] or checkpoint.get("config_sha256") != job["config_sha256"]:
            raise ValueError("Checkpoint configuration is inconsistent with the task")
    from utils.checkpoint import validate_training_checkpoint
    validate_training_checkpoint(last)
    if last.get("checkpoint_kind") == "inference":
        raise ValueError("Last checkpoint is inference-only")
    required = ("optimizer", "scheduler", "scaler", "rng_state", "epoch", "best_epoch", "best_metric",
                "best_metric_name", "early_stop_reference", "epochs_without_improvement", "collapse_count", "foreground_stall_count")
    if any(key not in last for key in required): raise ValueError("Incomplete continuation state in last checkpoint")
    if not last["optimizer"].get("state") or not last["rng_state"] or not last["scheduler"]:
        raise ValueError("Optimizer/scheduler/RNG state is empty")
    if last["best_metric_name"] != "foreground_miou": raise ValueError("Different best-checkpoint selection rule")
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    epoch = int(last["epoch"])
    durable = [r for r in rows if r["epoch"] <= epoch]
    if [r["epoch"] for r in durable] != list(range(1, epoch + 1)) or not _finite(rows):
        raise ValueError("History is non-finite, incomplete or not sequential")
    selected = max((r for r in durable if r.get("val")), key=lambda r: r["val"]["foreground_miou"])
    if (int(best["epoch"]) != selected["epoch"] or last["best_epoch"] != selected["epoch"] or
        not math.isclose(last["best_metric"], selected["val"]["foreground_miou"], abs_tol=1e-12)):
        raise ValueError("Best checkpoint does not match validation history")
    if durable[-1]["best_epoch"] != last["best_epoch"] or durable[-1]["early_stop_stale"] != last["epochs_without_improvement"]:
        raise ValueError("Tracker state disagrees with history")
    from models import build_model
    from utils.checkpoint import load_model_state
    from tools.train import build_optimizer, build_scheduler
    model = build_model(cfg, optical_channels=last["optical_channels"], sar_channels=last["sar_channels"], initialize_encoder=False)
    load_model_state(model, best["model"])
    load_model_state(model, last["model"])
    optimizer = build_optimizer(model, cfg["training"])
    optimizer.load_state_dict(last["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["training"], int(cfg["training"]["epochs"]))
    scheduler.load_state_dict(last["scheduler"])
    del model, optimizer, scheduler
    evidence = {"config_sha256": job["config_sha256"], "last_sha256": last_sha, "best_sha256": best_sha,
        "completed_epoch": epoch, "best_epoch": best["epoch"], "history_epochs": len(durable),
        "source_sha256": protocol["source"]["sha256"], "source_review": review.get("status"),
        "data_verification": "stat_verified" if protocol["datasets"]["train"].get("mode") == "stat" else "sha256_verified",
        "data_limitation": "Present SHA256 cannot prove content at an earlier stat-only observation.",
        "evidence_sha256": {name: file_digest(run / name) for name in ("config_snapshot.json", "protocol.json", "history.jsonl")}}
    del last, best
    budget = int(cfg["training"]["epochs"])
    if epoch > budget: return {"state": "retrain_required", "reason": "Training exceeded target budget", "evidence": evidence}
    summary_path = run / "training_summary.json"
    summary = _json(summary_path) if summary_path.exists() else {}
    failure = _json(run / "failure.json") if (run / "failure.json").exists() else {}
    if failure and "KeyboardInterrupt" not in failure.get("traceback", ""):
        raise ValueError("Numerical or unclassified training failure requires inspection")
    if epoch < budget:
        if summary.get("status") == "complete":
            return {"state": "retrain_required", "reason": "Early completion is not a fixed-budget experiment", "evidence": evidence}
        return {"state": "resume_training", "reason": "Complete compatible last state; continue at next epoch", "evidence": evidence}
    if not summary:
        return {"state": "resume_training", "reason": "Final checkpoint exists; finalize completion record without new epochs", "evidence": evidence}
    if (summary.get("status") != "complete" or summary.get("stop_reason") != "epochs_completed" or
        summary.get("completed_epoch") != budget or summary.get("config_sha256") != job["config_sha256"] or
        summary.get("seed") != job["seed"] or summary.get("best_epoch") != evidence["best_epoch"]):
        raise ValueError("Training completion summary disagrees with durable evidence")
    result_file = run / "test/result.json"
    if not result_file.exists() or (run / "test/failure.json").exists():
        return {"state": "reevaluate_only", "reason": "Compatible completed training; test is absent or failed", "evidence": evidence}
    try:
        from tools.validate_results import validate_result
        from utils.prediction import prediction_rule, resolve_tta
        result = _json(result_file)
        test = expected["test"]
        validate_result(result, job["config_sha256"], test.get("stat_sha256", test["sha256"]), True)
        if result.get("seed") != job["seed"] or result.get("experiment", {}).get("id") not in {job["experiment"], snapshot.get("experiment", {}).get("id")}:
            raise ValueError("Test result identity is unrelated to the verified run")
        if result.get("checkpoint_sha256") != evidence["best_sha256"] or result.get("checkpoint_selector") != "best":
            raise ValueError("Test does not use the selected best checkpoint")
        runtime = result["runtime"]
        if runtime.get("test_time_augmentation") != resolve_tta(cfg) or runtime.get("prediction_rule") != prediction_rule(cfg):
            raise ValueError("Test prediction policy differs")
        if runtime.get("amp") != cfg["training"].get("amp", True) or runtime.get("batch_size") != cfg["training"]["batch_size"]:
            raise ValueError("Test batch/precision differs")
        for k in ("matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms"):
            if runtime.get("execution", {}).get(k) != protocol["execution"].get(k): raise ValueError("Test backend differs")
        if result.get("damage_head_metrics") and "localization_supervised" not in result["damage_head_metrics"]:
            raise ValueError("Unsupervised localization reporting requires reevaluation")
        evidence["result_sha256"] = file_digest(result_file)
        return {"state": "reuse_complete", "reason": "Training, selection, full test and checkpoint verified", "evidence": evidence}
    except (ValueError, KeyError, OSError) as exc:
        return {"state": "reevaluate_only", "reason": str(exc), "evidence": evidence}


def audit_campaign(plan, output, *, reviews=(), environment_info=None):
    """Read-only classification; every planned seed receives exactly one state."""
    from utils.protocol import environment
    output = Path(output).resolve()
    env = environment_info or environment()
    inventories = []
    for parent in (output / "runs", output / "train"):
        if not parent.exists(): continue
        for snapshot in sorted(parent.rglob("config_snapshot.json")):
            run = _safe_run(snapshot.parent, output)
            try:
                config = _json(snapshot)
                inventories.append((run, config, config_digest(config)))
            except (ValueError, OSError):
                inventories.append((run, {}, None))
    rows = []
    for job in plan["jobs"]:
        matches = [(p, c) for p, c, sig in inventories if sig == job["config_sha256"] and c.get("training", {}).get("seed") == job["seed"]]
        # Known-name but scientifically different assets must not disappear as
        # a misleading train_new entry, including historical M/L aliases.
        related = [(p, c) for p, c, sig in inventories if c.get("experiment", {}).get("id") == job["experiment"] and c.get("training", {}).get("seed") == job["seed"]]
        candidates = matches or related
        row = {"id": job["id"], "canonical_id": job["experiment"], "dataset": job["dataset"], "seed": job["seed"]}
        if not candidates:
            target = _safe_run(run_path(output, job, plan), output)
            if target.exists() and any(target.iterdir()):
                row.update(state="blocked_review", reason="Unclassified files exist at the target run path")
            else:
                row.update(state="train_new", reason="No corresponding training asset exists")
        else:
            # Prefer a previously registered location; otherwise earliest
            # matching snapshot, then lexical path. Never use test metrics.
            target = _safe_run(run_path(output, job, plan), output)
            candidates.sort(key=lambda pair: (pair[0] != target, (pair[0] / "config_snapshot.json").stat().st_mtime_ns, str(pair[0])))
            run, cfg = candidates[0]
            row.update(run_dir=str(run), duplicates=[str(p) for p, _ in candidates[1:]])
            try:
                row.update(inspect_run(run, job, plan, output, reviews, env))
            except (ValueError, KeyError, OSError, RuntimeError, EOFError) as exc:
                row.update(state="blocked_review", reason=str(exc))
        rows.append(row)
    counts = {state: sum(r["state"] == state for r in rows) for state in STATES}
    assert sum(counts.values()) == len(plan["jobs"])
    return {"schema_version": 1, "target_tasks": len(rows), "counts": counts, "tasks": rows,
        "selection_rule": "Registered canonical location, then earliest snapshot, then lexical path; never test score",
        "not_needing_training": counts["reuse_complete"] + counts["reevaluate_only"],
        "full_new_or_retraining": counts["train_new"] + counts["retrain_required"],
        "source_reviews": list(reviews), "environment": env}


def install_imports(plan, manifest, output):
    """Keep original weights and snapshots; add a verifiable source connection."""
    output = Path(output).resolve()
    jobs = {j["id"]: j for j in plan["jobs"]}
    for row in manifest["tasks"]:
        if row["state"] not in {"reuse_complete", "reevaluate_only", "resume_training"}: continue
        job = jobs[row["id"]]
        run = _safe_run(row["run_dir"], output)
        target = _safe_run(run_path(output, job, plan), output)
        protocol = _json(run / "protocol.json")
        existing = _json(run / "source_import.json") if (run / "source_import.json").exists() else {}
        original_config = _json(run / "config_snapshot.json")
        original_protocol_sha = file_digest(run / "protocol.json")
        migration = None
        if run != target:
            if target.exists(): raise ValueError("Canonical target already exists; cannot overwrite an asset")
            if any(p.is_symlink() for p in run.rglob("*")):
                raise ValueError("Alias migration cannot depend on symlinks")
            origin = str(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(run, target)
            run = target
            # Keep weights and their embedded original configuration untouched.
            # Only output-path/display metadata in external evidence is updated.
            def relocate(value):
                if isinstance(value, str) and (value == origin or value.startswith(origin + "/")):
                    return str(run) + value[len(origin):]
                if isinstance(value, list): return [relocate(v) for v in value]
                if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
                return value
            snapshot = relocate(original_config)
            snapshot["experiment"] = job["config"]["experiment"]
            atomic_json(run / "config_snapshot.json", snapshot)
            for name in ("training_summary.json", "test/result.json"):
                path = run / name
                if path.exists():
                    value = relocate(_json(path))
                    if name == "test/result.json": value["experiment"] = job["config"]["experiment"]
                    atomic_json(path, value)
            for name in ("last", "best"):
                if file_digest(run / f"checkpoints/{name}.pth") != row["evidence"][f"{name}_sha256"]:
                    raise ValueError("Checkpoint changed during alias migration")
            migration = {"from": origin, "to": str(run), "status": "moved_and_verified", "checkpoint_bytes_unchanged": True}
            row["run_dir"] = str(run)
            row["migration"] = migration
        record = {"schema_version": 1, "canonical_id": job["experiment"], "seed": job["seed"],
                  "config_sha256": job["config_sha256"], "source_reviews": manifest["source_reviews"],
                  "original": existing.get("original", {"source": protocol["source"], "config_snapshot": original_config,
                      "protocol_sha256": original_protocol_sha, "environment": protocol.get("environment"), "execution": protocol.get("execution"),
                      "datasets": {s: {k: v for k, v in data.items() if k != "records"} for s, data in protocol["datasets"].items()}}),
                  "verification": row["evidence"], "migration": migration or existing.get("migration")}
        atomic_json(run / "source_import.json", record)
    atomic_json(output / "reuse_manifest.json", manifest)


def dispatch_order(plan, manifest):
    states = {r["id"]: r["state"] for r in manifest["tasks"]}
    priority = {"reuse_complete": 0, "reevaluate_only": 1, "resume_training": 2, "train_new": 3, "retrain_required": 4, "blocked_review": 5}
    datasets = {name: i for i, name in enumerate(("bright", "haiti", "xbd", "cau_flood"))}
    # Keep the requested dataset boundary, with reevaluation and continuation
    # first inside each dataset. Completed assets never enter the train queue.
    return sorted(plan["jobs"], key=lambda j: (datasets.get(j["dataset"], 99), priority[states[j["id"]]], plan["jobs"].index(j)))
