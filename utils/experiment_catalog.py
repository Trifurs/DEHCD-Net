"""Canonical definitions and ID-only views of shared experiment results."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from utils.config import XMLConfigParser
from utils.protocol import atomic_json, config_digest


def load_catalog(root):
    path = Path(root)
    if path.is_dir():
        path = path / "configs/experiments/catalog.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["experiments"]
    ids = [row.get("canonical_id", row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical_id in experiment catalog")
    return rows


def derive_catalog(root, paths):
    """Derive indexing metadata from executable files, never vice versa."""
    root = Path(root)
    rows, signatures = [], {}
    for path in paths:
        cfg = XMLConfigParser(path).parse().as_dict()
        exp = cfg["experiment"]
        if exp["id"] != exp["canonical_id"] or path.stem != exp["canonical_id"]:
            raise ValueError(f"Canonical ID/file mismatch: {path}")
        signature = config_digest(cfg)
        if signature in signatures:
            raise ValueError(f"Duplicate scientific definitions: {signatures[signature]} and {exp['id']}")
        signatures[signature] = exp["id"]
        rows.append({key: exp[key] for key in
                     ("id", "canonical_id", "dataset", "suite", "model_variant", "roles", "uses", "paper_refs", "comparison")})
        rows[-1]["config"] = str(path.relative_to(root))
    if len(rows) != len({row["canonical_id"] for row in rows}):
        raise ValueError("Duplicate canonical_id")
    return rows


def analysis_groups(rows):
    groups = {}
    for row in rows:
        for use in row.get("uses", []):
            name = use["group"]
            members = groups.setdefault(name, [])
            if any(member["canonical_id"] == row["canonical_id"] for member in members):
                raise ValueError(f"Repeated canonical_id in group {name}")
            members.append({"canonical_id": row["canonical_id"],
                            **{key: value for key, value in use.items() if key != "group"}})
    for members in groups.values():
        if all(isinstance(member.get("x"), (int, float)) for member in members):
            members.sort(key=lambda member: member["x"])
    return groups


def select_experiments(catalog, groups, datasets=None):
    """Union analysis groups before seeds are expanded; each ID appears once."""
    rows = catalog["experiments"] if isinstance(catalog, dict) else catalog
    indexed = analysis_groups(rows)
    requested = set(groups)
    unknown = requested - set(indexed) - {"all"}
    if unknown:
        raise ValueError("Unknown analysis groups: " + ", ".join(sorted(unknown)))
    ids = {row["canonical_id"] for row in rows} if "all" in requested else {
        member["canonical_id"] for group in requested for member in indexed[group]}
    return [row for row in rows if row["canonical_id"] in ids
            and (datasets is None or row["dataset"] in datasets)]


def write_catalog_views(root, rows, seeds=(42, 1051, 2060)):
    root = Path(root)
    counts = dict(Counter(row["dataset"] for row in rows))
    atomic_json(root / "configs/experiments/catalog.json", {
        "schema_version": 3, "seeds": list(seeds), "unique_configurations": len(rows),
        "target_tasks": len(rows) * len(seeds), "counts_by_dataset": counts, "experiments": rows})
    groups = analysis_groups(rows)
    atomic_json(root / "configs/experiments/groups.json", {
        "schema_version": 1, "seeds": list(seeds), "groups": groups,
        "note": "Views reference canonical IDs only. Union IDs before expanding seeds."})
    lines = ["# Experiment catalog", "", "Generated from the metadata in the formal XML definitions by `python tools/build_experiment_configs.py`.", "",
        f"There are **{len(rows)} unique configurations** and **{len(rows) * len(seeds)} target tasks** with seeds `{list(seeds)}`. Compatible completed tasks count toward this total; it is not the remaining training count.", "",
        "Shared `base`, `datasets`, `dehcd`, and runtime files are templates, not additional scheduled tasks. Each formal configuration lives in exactly one location. Group views reuse the same canonical ID and seed results.", "",
        "| Dataset | Unique configurations | Target tasks |", "|---|---:|---:|"]
    lines += [f"| {ds} | {count} | {count * len(seeds)} |" for ds, count in counts.items()]
    lines += ["", "## Coverage and analysis views", "",
        "Main metrics use all three complete test seeds, sample SD (`ddof=1`), `n`, and `expected_n=3`. A shared reference still has only three independent runs. Qualitative displays use seed 42; parameter counting and fixed-device efficiency profiling reuse checkpoints and do not train extra models.", "",
        "| Analysis group | Paper use | Canonical IDs (reference/default marked *) |", "|---|---|---|"]
    for name, members in groups.items():
        refs = list(dict.fromkeys(ref for member in members for ref in member["paper_refs"]))
        labels = [f"`{member['canonical_id']}`" + (" *" if member["role"] in ("reference", "default") else "")
                  + (f" (x={member['x']})" if "x" in member else "") for member in members]
        lines.append(f"| {name} | {', '.join(refs)} | {'<br>'.join(labels)} |")
    lines += ["", "## Fixed factors", "",
        "The Fig.17 order is HOG, DPM, BiCSF, IRB. Plain fusion replaces a disabled DPM; disabling BiCSF turns off both GCBM and WASM. Each dataset uses its existing L main model as full reference.", "",
        "Haiti HOG bins use L at K=[2,4,6,8,10], with K=6 supplied by `haiti_dehcd_l`. BRIGHT IRB uses M at T=0–8, with T=3 supplied by `bright_dehcd_m`. BRIGHT HOG guidance uses M at G=[1,2,3,4], with G=2 supplied by the same M run. Guidance covers the first G levels of both branches. All scans retain the other default factors (K=6, G=2, T=3).",
        "", "`bright_m_irb_steps_0` is an M model and is distinct from the L component control `bright_no_irb`. Former L scan IDs cannot be relabeled as M results. The M guidance assignment is supported by the historical definitions `Heterogeneous_LCD/configs/5090_x1/discussion_hog_levels_bright_m/bright_m_hog_levels_{1..4}.xml`, which inherit `bright_multiclass_hacf_m.xml`. Those definitions establish the variant only: their dropout=0.12, seed=1234 and other earlier settings do not make historical runs compatible with the current protocol.", "",
        "The three BRIGHT training controls separately compare the loss recipe, weighted sampler, and class weights. CE+Dice replaces several loss terms and smoothing and is a recipe comparison. Capacity and DPM controls are limited to BRIGHT. Matched parameter counts do not imply identical receptive fields or operation counts.", "",
        "Main architecture results share one from-scratch task objective and disable localization, feature-pair, and deep-supervision auxiliary losses. They measure adapted architectures under the common protocol, not each upstream implementation's best benchmark. Ordinary early stopping stays disabled; fixed budgets, validation foreground_miou selection, no TTA, and main-logits argmax remain in force.", "",
        "Table 1 is computed from actual dataset evidence. Confusion plots retain raw counts and explicit denominators; row normalization is not overall accuracy. Efficiency axes use validated full counts or measured latency/throughput under common settings. Postprocessing does not create additional training samples or tasks.", "",
        "## Unique definitions", "", "| Canonical ID | Variant | Roles | Declared reference | Allowed scientific changes |", "|---|---|---|---|---|"]
    for row in rows:
        comp = row["comparison"]
        lines.append(f"| `{row['canonical_id']}` | {row['model_variant']} | {', '.join(row['roles'])} | {comp['reference'] or 'dataset root'} | {', '.join(comp['allowed_changes'])} |")
    path = root / "docs/EXPERIMENT_CATALOG.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
