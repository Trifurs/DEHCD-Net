"""Source-checked three-seed summaries and shared canonical analysis views."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_multiseed import aggregate
from utils.campaign_layout import summary_path
from utils.protocol import atomic_json
from utils.comparison_protocol import comparability, pair_policy
from utils.experiment_catalog import analysis_groups
from utils.experiment_statistics import paired_seed_comparison, holm_adjust

EXPECTED_SEEDS = [42, 1051, 2060]


def make_report(campaign, references=None, metrics=None, comparators=None, repeats=20000, seed=20260929, groups=None):
    campaign = Path(campaign)
    metrics = metrics or ["mean_iou", "foreground_miou", "oa"]
    plan = json.loads((campaign / "protocol.json").read_text())
    checked = aggregate(plan, campaign)
    with summary_path(campaign, plan, "per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    observations = {}
    for row in rows:
        name, run_seed = row["experiment"], int(row["seed"])
        if run_seed not in EXPECTED_SEEDS: raise ValueError("Unexpected seed in formal summary")
        if run_seed in observations.setdefault(name, {}): raise ValueError("Duplicate training repetition")
        observations[name][run_seed] = row
    registry = {j["experiment"]: j["config"]["experiment"] for j in plan["jobs"]}
    # Custom smoke campaigns do not define manuscript groups.
    view_rows = [dict(v, canonical_id=v.get("canonical_id", name)) for name, v in registry.items()]
    views = analysis_groups(view_rows)
    selected_groups = groups or []
    unknown = set(selected_groups) - views.keys() - {"all"}
    if unknown: raise ValueError("Unknown analysis groups: " + ", ".join(sorted(unknown)))
    chosen_views = views if "all" in selected_groups else {g: views[g] for g in selected_groups}
    selected = {m["canonical_id"] for members in chosen_views.values() for m in members} if selected_groups else set(registry)
    if references and not set(references) <= registry.keys(): raise ValueError("Unknown reference")
    if comparators and not set(comparators) <= registry.keys(): raise ValueError("Unknown comparator")
    tables = []
    for name in registry:
        if name not in selected: continue
        values_by_seed = observations.get(name, {})
        for metric in metrics:
            values = [float(values_by_seed[s][metric]) for s in EXPECTED_SEEDS if s in values_by_seed]
            if not all(0 <= value <= 1 for value in values): raise ValueError("Only 0..1 accuracy metrics may be expressed as percentages")
            tables.append({"experiment": name, "metric": metric, "n": len(values), "expected_n": 3,
                "complete": sorted(values_by_seed) == sorted(EXPECTED_SEEDS),
                "mean_percent": 100 * statistics.mean(values) if values else None,
                "sample_std_percentage_points": 100 * statistics.stdev(values) if len(values) > 1 else None,
                "per_seed": {str(s): float(values_by_seed[s][metric]) for s in EXPECTED_SEEDS if s in values_by_seed},
                "sources": [{"seed": s, "checkpoint": values_by_seed[s]["checkpoint"], "config_sha256": values_by_seed[s]["config_sha256"]} for s in EXPECTED_SEEDS if s in values_by_seed]})
    pairs = set()
    if references:
        for ref in references:
            for name in comparators or sorted(selected):
                if ref != name: pairs.add((ref, name))
    else:
        for name in selected:
            ref = registry[name].get("comparison", {}).get("reference")
            if ref and ref in selected: pairs.add((ref, name))
    comparisons, excluded = [], []
    jobs_by_id = {name: [j for j in plan["jobs"] if j["experiment"] == name] for name in registry}
    for reference, name in sorted(pairs):
        ref_jobs, jobs = jobs_by_id[reference], jobs_by_id[name]
        if ref_jobs[0]["dataset"] != jobs[0]["dataset"]:
            excluded.append({"reference": reference, "candidate": name, "reason": "different datasets"}); continue
        try:
            policy = pair_policy(ref_jobs[0]["config"], jobs[0]["config"])
            contract = comparability(plan, ref_jobs + jobs)
        except ValueError as exc:
            if comparators: raise
            excluded.append({"reference": reference, "candidate": name, "reason": str(exc)}); continue
        a, b = observations.get(reference, {}), observations.get(name, {})
        paired = [s for s in EXPECTED_SEEDS if s in a and s in b]
        for metric in metrics:
            delta = {str(s): float(a[s][metric]) - float(b[s][metric]) for s in paired}
            entry = {"reference": reference, "comparator": name, "dataset": jobs[0]["dataset"],
                "metric": metric, "comparison_contract": contract, "paired_seeds": paired,
                "expected_n": 3, "complete": paired == EXPECTED_SEEDS, "n": len(paired),
                "per_seed_differences": delta, "direction": "reference_minus_comparator", "reference_point": "declared default, not necessarily curve starting point",
                "mean_difference_percentage_points": 100 * statistics.mean(delta.values()) if delta else None,
                "p_value": None, "p_holm": None}
            if len(paired) == 3:
                result = paired_seed_comparison({s: float(a[s][metric]) for s in paired}, {s: float(b[s][metric]) for s in paired}, repeats=repeats, seed=seed)
                entry.update(result)
            comparisons.append(entry)
    tested = [r for r in comparisons if r.get("p_value") is not None]
    for row, adjusted in zip(tested, holm_adjust([r["p_value"] for r in tested])): row["p_holm"] = adjusted
    return {"status": "complete" if all(r["complete"] for r in tables) else "incomplete",
        "unit": "independent training seed on a fixed split", "tables": tables, "comparisons": comparisons,
        "analysis_groups": chosen_views, "excluded_comparisons": excluded,
        "qualitative_policy": plan.get("analysis_policy", {"display_seed": 42}),
        "multiplicity": "Holm over all complete exported paired comparisons and endpoints",
        "interpretation": "Mean and sample SD use ddof=1 across three seeds. Missing seeds remain incomplete, never zero-filled. Reused references do not increase n. Three seeds give a minimum two-sided exact sign-flip p of 0.25; no significance labels are generated. Seed variation does not demonstrate unseen-event generalization."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--reference", nargs="+")
    p.add_argument("--comparators", nargs="+")
    p.add_argument("--groups", nargs="+", help="ID-only analysis views; shared references never increase n")
    p.add_argument("--metrics", nargs="+", default=["mean_iou", "foreground_miou", "oa"])
    p.add_argument("--bootstrap-repeats", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260929)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = make_report(args.campaign, args.reference, args.metrics, args.comparators, args.bootstrap_repeats, args.bootstrap_seed, args.groups)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "statistics.json", report)
    with (out / "raw_metrics_percent.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["experiment", "metric", "n", "expected_n", "complete", "mean_percent", "sample_std_percentage_points"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(report["tables"])
    lines = ["# Test metrics across seeds", "", "| Experiment | Metric | n/3 | Mean (%) | Sample SD (pp) | Status |", "|---|---|---:|---:|---:|---|"]
    for row in report["tables"]:
        mean, sd = row["mean_percent"], row["sample_std_percentage_points"]
        mean = "NA" if mean is None else f"{mean:.3f}"
        sd = "NA" if sd is None else f"{sd:.3f}"
        state = "complete" if row["complete"] else "incomplete"
        lines.append(f"| {row['experiment']} | {row['metric']} | {row['n']}/3 | {mean} | {sd} | {state} |")
    (out / "raw_metrics.md").write_text("\n".join(lines) + "\n\n" + report["interpretation"] + "\n")
    print(f"{report['status']}: {out / 'statistics.json'}")


if __name__ == "__main__":
    main()
