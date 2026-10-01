"""Check declared experimental factors before training and before paired reports.

A common recipe controls optimization/data, not model capacity or each model's
best possible hyperparameters. Native-recipe results are explicit controls.
"""
from __future__ import annotations

import copy
from collections import defaultdict

from utils.prediction import prediction_rule, resolve_tta

PROTOCOL = "controlled_comparison_v1"
KINDS = {"architecture", "control", "recipe_bundle"}
MODEL_INVARIANTS = ("norm_type", "norm_groups", "group_norm_eps", "dropout",
                    "optical_channels", "sar_channels", "input_adaptation")


def comparison_settings(config):
    """All input/training/evaluation semantics; omit only run bookkeeping/seed."""
    result = {key: copy.deepcopy(config.get(key, {})) for key in
              ("task", "dataset", "normalization", "augmentation", "model", "training", "inference")}
    for key in ("seed", "resume", "checkpoint_dir", "output_dir", "best_metric_resolved"):
        result["training"].pop(key, None)
    for key in ("checkpoint", "save_dir", "save_visualization", "split"):
        result["inference"].pop(key, None)
    result["inference"]["test_time_augmentation"] = resolve_tta(config)
    result["inference"]["prediction_rule"] = prediction_rule(config)
    result["inference"].pop("tta", None)
    # Explicitly absent localization supervision must not depend on return type.
    result["training"].setdefault("localization_loss_weight", 1.0)
    return result


def differences(reference, candidate, prefix=""):
    changes = {}
    for key in sorted(reference.keys() | candidate.keys()):
        path = f"{prefix}.{key}" if prefix else key
        if key not in reference or key not in candidate:
            changes[path] = {"reference": reference.get(key, "<unset>"),
                             "candidate": candidate.get(key, "<unset>")}
        elif isinstance(reference[key], dict) and isinstance(candidate[key], dict):
            changes.update(differences(reference[key], candidate[key], path))
        elif reference[key] != candidate[key]:
            changes[path] = {"reference": reference[key], "candidate": candidate[key]}
    return changes


def declaration(config):
    return config.get("experiment", {}).get("comparison", {})


def validate_config(config):
    """The controlled invariants apply equally to every model and control."""
    if config.get("experiment", {}).get("protocol") != PROTOCOL:
        return
    spec = declaration(config)
    if spec.get("kind") not in KINDS or not isinstance(spec.get("allowed_changes"), list):
        raise ValueError("Controlled experiments require a kind and explicit allowed_changes")
    m, t = config.get("model", {}), config.get("training", {})
    if m.get("pretrained_backbone") or m.get("encoder_checkpoint") not in (None, "", "none"):
        raise ValueError("controlled_comparison_v1 is a scratch comparison; pretrained weights require a separate declared protocol")
    if t.get("amp", True) or t.get("early_stop_patience", 0) or not t.get("deterministic", False):
        raise ValueError("Controlled comparisons require FP32, deterministic seeding and no ordinary early stopping")
    if float(t.get("aux_loss_weight", 0)) or float(t.get("feature_pair_loss_weight", 0)):
        raise ValueError("Undeclared deep/alignment auxiliary supervision in controlled protocol")
    if m.get("deep_supervision", False) or (m.get("compare_model") and m.get("compare_deep_supervision", True)):
        raise ValueError("Controlled protocol disables deep supervision for all models")
    if resolve_tta(config) != "none" or prediction_rule(config) != "damage_argmax":
        raise ValueError("Controlled protocol requires common no-TTA damage_argmax evaluation")
    if t.get("best_metric") != "foreground_miou":
        raise ValueError("Controlled protocol selects all checkpoints using validation foreground_miou")
    if spec["kind"] == "architecture":
        if spec["allowed_changes"] != ["model"]:
            raise ValueError("Architecture comparisons may change model settings only")
        if float(t.get("localization_loss_weight", 1)) or t.get("loss") in {"changeos_native", "damage_ce_lovasz"}:
            raise ValueError("Main architecture comparison uses the same primary task loss with localization supervision disabled")
        if m.get("compare_model") and not m.get("compare_adapt_batchnorm", True):
            raise ValueError("Original BatchNorm belongs in the declared adapter control, not the main table")
    else:
        for path in spec["allowed_changes"]:
            if not isinstance(path, str) or "." not in path or path.split(".")[0] not in {"model", "training", "dataset"}:
                raise ValueError(f"Controls must declare exact model/training/dataset fields, got {path!r}")
        if not spec.get("reference"):
            raise ValueError("A control requires its named reference experiment")


def check_changes(reference, candidate, allowed):
    a, b = comparison_settings(reference), comparison_settings(candidate)
    delta = differences(a, b)
    unexpected = [path for path in delta if path not in allowed and not ("model" in allowed and path.startswith("model."))]
    if "model" in allowed:
        # Model families may differ; shared external normalization and DEHCD
        # size regularization cannot drift silently with the architecture label.
        unexpected += [f"model.{key}" for key in MODEL_INVARIANTS if a["model"].get(key) != b["model"].get(key)]
    if unexpected:
        raise ValueError("Undeclared comparison differences: " + ", ".join(sorted(set(unexpected))))
    return delta


def validate_design(configs):
    """Validate a registry containing selected experiments and their references."""
    evidence, visited, active = [], set(), set()

    def visit(name):
        if name in visited:
            return
        if name in active:
            raise ValueError(f"Cyclic comparison references: {name}")
        active.add(name)
        config = configs[name]
        validate_config(config)
        spec = declaration(config)
        ref = spec.get("reference")
        if ref:
            if ref not in configs:
                raise ValueError(f"Missing declared reference {ref} for {name}")
            visit(ref)
            if config.get("experiment", {}).get("dataset") != configs[ref].get("experiment", {}).get("dataset"):
                raise ValueError(f"Reference crosses datasets: {ref} -> {name}")
            if config.get("experiment", {}).get("protocol") != configs[ref].get("experiment", {}).get("protocol"):
                raise ValueError(f"Reference crosses protocol versions: {ref} -> {name}")
            delta = check_changes(configs[ref], config, spec["allowed_changes"])
            evidence.append({"reference": ref, "candidate": name, "kind": spec["kind"],
                             "factor": spec.get("factor"), "actual_changes": delta})
        active.remove(name)
        visited.add(name)

    for name in configs:
        visit(name)
    return evidence


def pair_policy(reference, candidate):
    """No unannounced mixture of architecture and recipe comparisons."""
    a, b = declaration(reference), declaration(candidate)
    ref_id = reference.get("experiment", {}).get("id")
    cand_id = candidate.get("experiment", {}).get("id")
    if not a or not b:
        raise ValueError("Missing declared comparison design; legacy results may be described but not silently relabeled as controlled architecture comparisons")
    if a.get("kind") == b.get("kind") == "architecture":
        return {"kind": "architecture", "allowed_changes": ["model"]}
    if b.get("reference") == ref_id:
        return {**b, "direction": "candidate_vs_declared_reference"}
    if a.get("reference") == cand_id:
        return {**a, "direction": "declared_reference_vs_control"}
    raise ValueError(f"{ref_id} vs {cand_id} is not a declared direct control; compare each control to its named reference")


def comparability(plan, jobs):
    if not jobs:
        raise ValueError("No jobs to compare")
    groups = defaultdict(list)
    for job in jobs:
        validate_config(job["config"])
        groups[job["experiment"]].append(job)
    representatives, seed_sets = [], []
    data_reference = None
    for name, rows in groups.items():
        first = rows[0]["config"]
        seeds = [row["seed"] for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed in {name}")
        seed_sets.append(set(seeds))
        for row in rows:
            if declaration(first) != declaration(row["config"]):
                raise ValueError(f"Comparison declaration changes across seeds: {name}")
            if row["config"].get("training", {}).get("seed") != row["seed"]:
                raise ValueError(f"Configuration seed does not match job seed: {name}")
            check_changes(first, row["config"], [])
            splits = plan["data"][row["data_key"]]["splits"]
            identity = {split: {"mode": splits[split].get("mode", "stat"), "sha256": splits[split]["sha256"]}
                        for split in ("train", "val", "test")}
            if data_reference is not None and identity != data_reference:
                raise ValueError("Paired comparison requires identical train, validation AND test file identities")
            data_reference = identity
        representatives.append(first)
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("Paired comparison requires identical planned seed sets")
    base = representatives[0]
    checks = []
    for config in representatives[1:]:
        if base.get("experiment", {}).get("protocol") != config.get("experiment", {}).get("protocol"):
            raise ValueError("Paired comparison mixes experiment protocol versions")
        policy = pair_policy(base, config)
        changes = check_changes(base, config, policy["allowed_changes"])
        checks.append({"candidate": config.get("experiment", {}).get("id"),
                       "policy": policy, "actual_changes": changes})
    common = comparison_settings(base)
    common.pop("model")
    return {"data": data_reference, "paired_seeds": sorted(seed_sets[0]),
            "reference_settings": common, "comparisons": checks,
            "scope": "Declared common recipe or direct factor control; not a claim of equal capacity or optimal per-model tuning"}


def audit_plan(plan):
    """Preflight selected jobs against declared references; group by event fold."""
    registry = dict(plan.get("comparison_references", {}))
    for job in plan["jobs"]:
        validate_config(job["config"])
        name = job["experiment"]
        if name in registry:
            check_changes(registry[name], job["config"], [])
        registry[name] = job["config"]
    evidence = validate_design(registry)
    if "data" in plan:
        # Each experiment's own seeds must have an unchanged data protocol.
        names = {j["experiment"] for j in plan["jobs"]}
        for name in names:
            comparability(plan, [j for j in plan["jobs"] if j["experiment"] == name])
        # All main architectures on the same dataset/fold share every condition.
        cohorts = defaultdict(list)
        for job in plan["jobs"]:
            if declaration(job["config"]).get("kind") == "architecture":
                e = job["config"].get("experiment", {})
                cohorts[(job["dataset"], e.get("test_event"), e.get("validation_event"))].append(job)
        for rows in cohorts.values():
            comparability(plan, rows)
        # Check control/reference data too when both were selected for this run.
        for edge in evidence:
            if {edge["reference"], edge["candidate"]} <= names:
                comparability(plan, [j for j in plan["jobs"] if j["experiment"] in {edge["reference"], edge["candidate"]}])
    return evidence
