from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.test import build_visual_style, colorize_mask_array, save_rgb_image, sanitize_filename, tensor_preview_array


DEHCD_SIZE_PRIORITY = {"l": 0, "m": 1, "s": 2}
COMPARE_DISPLAY_NAMES = {
    "dminet": "DMINet",
    "haff": "HAFF",
    "hfa_panet": "HFA-PANet",
    "hrsicd": "HRSICD",
    "icif_net": "ICIFNet",
    "wavehfg": "WaveHFG",
    "lightweight": "Lightweight",
}


@dataclass
class RunInfo:
    result_path: Path
    artifact_dir: Path
    train_run: str
    dataset: str
    model_key: str
    display_name: str
    role: str
    dehcd_size: str | None
    overall_score: float
    metric_name: str
    result: Dict[str, Any]
    config: Dict[str, Any]
    split: str
    sample_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class Candidate:
    dataset: str
    sample_id: str
    winner: RunInfo
    winner_score: float
    best_baseline: RunInfo
    best_baseline_score: float
    margin: float
    dehcd_scores: Dict[str, float]
    baseline_scores: Dict[str, float]
    foreground_ratio: float
    target_class_ratios: Dict[int, float]
    present_classes: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select paper-friendly qualitative samples where DEHCD beats comparison models."
    )
    parser.add_argument("--test-root", default="runs/test_paper_all", help="Output folder produced by tools/test.py.")
    parser.add_argument("--output-root", default="runs/paper_sample_selection", help="Where selected samples are written.")
    parser.add_argument(
        "--refresh-contact-sheets",
        action="store_true",
        help="Only rebuild contact_sheet.png files under --output-root from already exported sample images.",
    )
    parser.add_argument(
        "--split",
        default="auto",
        choices=["auto", "train", "val", "test"],
        help="Dataset split used by test.py. auto reads the split recorded in each result JSON.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Number of selected samples per dataset.")
    parser.add_argument("--min-margin", type=float, default=0.03, help="Minimum DEHCD minus best-baseline sample metric margin.")
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=1,
        help="For multiclass datasets, try to include at least this many selected samples for each foreground class.",
    )
    parser.add_argument(
        "--min-class-ratio",
        type=float,
        default=0.001,
        help="Minimum target_class_*_ratio for a class to count as present in multiclass sample balancing.",
    )
    parser.add_argument(
        "--metric",
        default="auto",
        help="Sample metric used for ranking. auto uses iou for binary and foreground_miou for multiclass.",
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=0.005,
        help="Skip nearly empty GT samples below this foreground ratio.",
    )
    parser.add_argument(
        "--max-foreground-ratio",
        type=float,
        default=1.0,
        help="Skip samples above this GT foreground ratio.",
    )
    parser.add_argument(
        "--include-extra-runs",
        action="store_true",
        help="Include discussion/ablation runs. By default they are ignored.",
    )
    parser.add_argument(
        "--require-all-predictions",
        action="store_true",
        help="Only select samples that have prediction masks for every retained model.",
    )
    parser.add_argument(
        "--contact-cols",
        type=int,
        default=6,
        help="Maximum columns in generated contact sheets.",
    )
    parser.add_argument(
        "--contact-tile-size",
        type=int,
        default=640,
        help="Contact-sheet subfigure width/height in pixels. Inputs are upscaled for publication clarity.",
    )
    parser.add_argument(
        "--contact-dpi",
        type=int,
        default=600,
        help="DPI metadata for contact_sheet.png.",
    )
    parser.add_argument(
        "--contact-font-size",
        type=int,
        default=80,
        help="Label font size in generated contact sheets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write summaries; do not export images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_root = Path(args.test_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if args.refresh_contact_sheets:
        refreshed = refresh_existing_contact_sheets(
            output_root,
            contact_cols=max(int(args.contact_cols), 2),
            tile_size=max(int(args.contact_tile_size), 128),
            dpi=max(int(args.contact_dpi), 72),
            font_size=max(int(args.contact_font_size), 10),
        )
        print(f"Refreshed {refreshed} contact sheet(s) under {output_root}")
        return
    if not test_root.exists():
        raise FileNotFoundError(f"test root not found: {test_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    runs = load_runs(test_root, include_extra_runs=bool(args.include_extra_runs))
    selected_runs = select_best_runs(runs)
    write_run_inventory(selected_runs, output_root / "selected_run_inventory.csv")

    by_dataset: Dict[str, list[RunInfo]] = {}
    for run in selected_runs:
        by_dataset.setdefault(run.dataset, []).append(run)

    summary: Dict[str, Any] = {"test_root": str(test_root), "datasets": {}}
    all_rows: list[Dict[str, Any]] = []
    for dataset, dataset_runs in sorted(by_dataset.items()):
        metric_name = resolve_metric_name(args.metric, dataset_runs)
        candidates = find_candidates(
            dataset=dataset,
            runs=dataset_runs,
            metric_name=metric_name,
            min_margin=float(args.min_margin),
            min_foreground_ratio=float(args.min_foreground_ratio),
            max_foreground_ratio=float(args.max_foreground_ratio),
            require_all_predictions=bool(args.require_all_predictions),
        )
        selected = select_top_candidates(
            candidates,
            runs=dataset_runs,
            top_k=max(int(args.top_k), 0),
            min_class_samples=max(int(args.min_class_samples), 0),
            min_class_ratio=float(args.min_class_ratio),
        )
        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        write_candidates_csv(candidates, dataset_dir / "candidate_ranking.csv", metric_name=metric_name)
        write_candidates_csv(selected, dataset_dir / "selected_samples.csv", metric_name=metric_name)
        class_coverage = selected_class_coverage(
            selected,
            dataset_runs,
            min_class_ratio=float(args.min_class_ratio),
        )
        warn_if_class_coverage_is_low(
            dataset=dataset,
            runs=dataset_runs,
            coverage=class_coverage,
            min_class_samples=max(int(args.min_class_samples), 0),
            top_k=max(int(args.top_k), 0),
        )
        if not args.dry_run:
            export_dataset_samples(
                dataset=dataset,
                runs=dataset_runs,
                candidates=selected,
                output_dir=dataset_dir,
                split=str(args.split),
                contact_cols=int(args.contact_cols),
                contact_tile_size=int(args.contact_tile_size),
                contact_dpi=int(args.contact_dpi),
                contact_font_size=int(args.contact_font_size),
                metric_name=metric_name,
            )
        summary["datasets"][dataset] = {
            "metric": metric_name,
            "retained_models": [run.display_name for run in dataset_runs],
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_samples": [candidate.sample_id for candidate in selected],
            "selected_class_coverage": class_coverage,
        }
        all_rows.extend(candidate_row(candidate, metric_name) for candidate in selected)

    with (output_root / "selection_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    write_dict_rows(all_rows, output_root / "selected_samples_all.csv")
    print(f"Loaded {len(runs)} test runs; retained {len(selected_runs)} best model runs.")
    print(f"Saved selected samples to {output_root}")


def load_runs(test_root: Path, include_extra_runs: bool) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for result_path in sorted(test_root.glob("*.json")):
        if result_path.name == "selection_summary.json":
            continue
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        train_run = str(result.get("train_run") or result_path.stem)
        if is_extra_run(train_run) and not include_extra_runs:
            continue
        artifact_dir = Path(result.get("artifact_dir") or (test_root / train_run))
        config = load_config_for_result(result)
        dataset = dataset_key(result, config)
        model_key, display_name, role, dehcd_size = identify_model(result, config)
        if is_extra_run(train_run) and include_extra_runs:
            model_key = f"{model_key}_{sanitize_filename(train_run)}"
            display_name = f"{display_name} ({short_run_name(train_run)})"
        metric_name = str(result.get("best_metric") or result.get("checkpoint_best_metric_name") or "primary_score")
        overall_score = result_metric(result, metric_name)
        sample_metrics = load_sample_metrics(artifact_dir / "sample_metrics.csv")
        if not sample_metrics:
            print(f"Warning: skip {train_run}, missing sample_metrics.csv", file=sys.stderr)
            continue
        runs.append(
            RunInfo(
                result_path=result_path,
                artifact_dir=artifact_dir,
                train_run=train_run,
                dataset=dataset,
                model_key=model_key,
                display_name=display_name,
                role=role,
                dehcd_size=dehcd_size,
                overall_score=overall_score,
                metric_name=metric_name,
                result=result,
                config=config,
                split=str(result.get("split") or "test"),
                sample_metrics=sample_metrics,
            )
        )
    return runs


def load_config_for_result(result: Dict[str, Any]) -> Dict[str, Any]:
    train_run_dir = Path(str(result.get("train_run_dir") or ""))
    snapshot = train_run_dir / "config_snapshot.json"
    if snapshot.exists():
        with snapshot.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def load_sample_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample_id = str(row.get("id") or "")
            if not sample_id:
                continue
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if key == "id":
                    continue
                parsed[key] = to_float(value)
            out[sample_id] = parsed
    return out


def select_best_runs(runs: list[RunInfo]) -> list[RunInfo]:
    best: Dict[tuple[str, str], RunInfo] = {}
    for run in runs:
        key = (run.dataset, run.model_key)
        current = best.get(key)
        if current is None or run.overall_score > current.overall_score:
            best[key] = run
    ordered = sorted(best.values(), key=run_sort_key)
    return ordered


def run_sort_key(run: RunInfo) -> tuple[str, int, str]:
    if run.role == "dehcd":
        rank = DEHCD_SIZE_PRIORITY.get(run.dehcd_size or "", 9)
    elif run.role == "compare":
        rank = 10
    else:
        rank = 20
    return run.dataset, rank, run.display_name.lower()


def resolve_metric_name(metric: str, runs: Iterable[RunInfo]) -> str:
    if str(metric).lower() not in {"", "auto", "default"}:
        return str(metric)
    num_classes = 2
    for run in runs:
        task_cfg = run.config.get("task", {})
        model_cfg = run.config.get("model", {})
        num_classes = int(task_cfg.get("num_classes", model_cfg.get("num_classes", run.result.get("dataset", {}).get("num_classes", 2))))
        break
    return "iou" if num_classes == 2 else "foreground_miou"


def find_candidates(
    dataset: str,
    runs: list[RunInfo],
    metric_name: str,
    min_margin: float,
    min_foreground_ratio: float,
    max_foreground_ratio: float,
    require_all_predictions: bool,
) -> list[Candidate]:
    dehcd_runs = [run for run in runs if run.role == "dehcd"]
    baseline_runs = [run for run in runs if run.role != "dehcd"]
    if not dehcd_runs or not baseline_runs:
        print(f"Warning: skip {dataset}, need both DEHCD and baseline runs.", file=sys.stderr)
        return []

    common_ids = set.intersection(*(set(run.sample_metrics) for run in runs))
    if not common_ids:
        return []

    candidates: list[Candidate] = []
    for sample_id in sorted(common_ids):
        if require_all_predictions and any(not prediction_mask_path(run, sample_id).exists() for run in runs):
            continue
        foreground_ratio = sample_foreground_ratio(next(iter(runs)).sample_metrics[sample_id])
        if foreground_ratio < min_foreground_ratio or foreground_ratio > max_foreground_ratio:
            continue
        dehcd_scores = {run.display_name: sample_score(run, sample_id, metric_name) for run in dehcd_runs}
        baseline_scores = {run.display_name: sample_score(run, sample_id, metric_name) for run in baseline_runs}
        target_class_ratios = sample_target_class_ratios(next(iter(runs)).sample_metrics[sample_id])
        present_classes = tuple(class_idx for class_idx, ratio in sorted(target_class_ratios.items()) if ratio > 0.0)
        winner = max(dehcd_runs, key=lambda run: sample_score(run, sample_id, metric_name))
        best_baseline = max(baseline_runs, key=lambda run: sample_score(run, sample_id, metric_name))
        winner_score = sample_score(winner, sample_id, metric_name)
        best_baseline_score = sample_score(best_baseline, sample_id, metric_name)
        margin = winner_score - best_baseline_score
        if margin < min_margin:
            continue
        candidates.append(
            Candidate(
                dataset=dataset,
                sample_id=sample_id,
                winner=winner,
                winner_score=winner_score,
                best_baseline=best_baseline,
                best_baseline_score=best_baseline_score,
                margin=margin,
                dehcd_scores=dehcd_scores,
                baseline_scores=baseline_scores,
                foreground_ratio=foreground_ratio,
                target_class_ratios=target_class_ratios,
                present_classes=present_classes,
            )
        )
    return sorted(candidates, key=candidate_sort_key)


def candidate_sort_key(candidate: Candidate) -> tuple[int, float, float, float]:
    priority = DEHCD_SIZE_PRIORITY.get(candidate.winner.dehcd_size or "", 9)
    foreground_class_count = sum(1 for class_idx in candidate.present_classes if class_idx > 0)
    return priority, -foreground_class_count, -candidate.margin, -candidate.winner_score


def select_top_candidates(
    candidates: list[Candidate],
    runs: list[RunInfo],
    top_k: int,
    min_class_samples: int,
    min_class_ratio: float,
) -> list[Candidate]:
    if top_k <= 0:
        return []
    num_classes = num_classes_from_runs(runs)
    if num_classes <= 2 or min_class_samples <= 0:
        return candidates[:top_k]

    required_classes = list(range(1, num_classes))
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    class_counts = {class_idx: 0 for class_idx in required_classes}

    def candidate_classes(candidate: Candidate) -> set[int]:
        return {
            class_idx
            for class_idx in required_classes
            if candidate.target_class_ratios.get(class_idx, 0.0) >= min_class_ratio
        }

    remaining = list(candidates)
    while len(selected) < top_k:
        unmet = {class_idx for class_idx, count in class_counts.items() if count < min_class_samples}
        if not unmet:
            break
        useful = [candidate for candidate in remaining if candidate.sample_id not in selected_ids and candidate_classes(candidate) & unmet]
        if not useful:
            break
        best = min(
            useful,
            key=lambda candidate: balanced_candidate_sort_key(candidate, candidate_classes(candidate), unmet),
        )
        selected.append(best)
        selected_ids.add(best.sample_id)
        for class_idx in candidate_classes(best):
            class_counts[class_idx] += 1

    for candidate in candidates:
        if len(selected) >= top_k:
            break
        if candidate.sample_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.sample_id)
    return selected


def balanced_candidate_sort_key(candidate: Candidate, present: set[int], unmet: set[int]) -> tuple[int, int, int, float, float]:
    priority = DEHCD_SIZE_PRIORITY.get(candidate.winner.dehcd_size or "", 9)
    newly_covered = len(present & unmet)
    foreground_class_count = len(present)
    return priority, -newly_covered, -foreground_class_count, -candidate.margin, -candidate.winner_score


def selected_class_coverage(candidates: list[Candidate], runs: list[RunInfo], min_class_ratio: float = 0.0) -> Dict[str, int]:
    num_classes = num_classes_from_runs(runs)
    counts = {str(class_idx): 0 for class_idx in range(1, max(num_classes, 1))}
    for candidate in candidates:
        for class_idx in range(1, num_classes):
            if candidate.target_class_ratios.get(class_idx, 0.0) >= min_class_ratio:
                counts[str(class_idx)] += 1
    return counts


def warn_if_class_coverage_is_low(
    dataset: str,
    runs: list[RunInfo],
    coverage: Dict[str, int],
    min_class_samples: int,
    top_k: int,
) -> None:
    num_classes = num_classes_from_runs(runs)
    if num_classes <= 2 or min_class_samples <= 0 or top_k <= 0:
        return
    low = {class_idx: count for class_idx, count in coverage.items() if count < min_class_samples}
    if low:
        print(
            f"Warning: {dataset} selected samples could not satisfy requested foreground-class coverage "
            f"(min_class_samples={min_class_samples}, coverage={coverage}).",
            file=sys.stderr,
        )


def num_classes_from_runs(runs: Iterable[RunInfo]) -> int:
    for run in runs:
        task_cfg = run.config.get("task", {})
        model_cfg = run.config.get("model", {})
        result_dataset = run.result.get("dataset", {}) if isinstance(run.result.get("dataset"), dict) else {}
        return int(task_cfg.get("num_classes", model_cfg.get("num_classes", result_dataset.get("num_classes", 2))))
    return 2


def sample_score(run: RunInfo, sample_id: str, metric_name: str) -> float:
    row = run.sample_metrics[sample_id]
    if metric_name in row:
        return float(row[metric_name])
    if run.result.get("best_metric") in row:
        return float(row[str(run.result["best_metric"])])
    for fallback in ("foreground_miou", "iou", "binary_foreground_iou", "mean_iou", "oa"):
        if fallback in row:
            return float(row[fallback])
    return float("nan")


def sample_foreground_ratio(row: Dict[str, float]) -> float:
    if "target_foreground_ratio" in row:
        return float(row["target_foreground_ratio"])
    total = 0.0
    foreground = 0.0
    for key, value in row.items():
        if key.startswith("target_class_") and key.endswith("_ratio"):
            total += value
            try:
                class_idx = int(key[len("target_class_") : -len("_ratio")])
            except ValueError:
                class_idx = 0
            if class_idx > 0:
                foreground += value
    return foreground if total <= 0 else foreground / max(total, 1e-12)


def sample_target_class_ratios(row: Dict[str, float]) -> Dict[int, float]:
    ratios: Dict[int, float] = {}
    for key, value in row.items():
        if key.startswith("target_class_") and key.endswith("_ratio"):
            try:
                class_idx = int(key[len("target_class_") : -len("_ratio")])
            except ValueError:
                continue
            ratios[class_idx] = float(value)
    return ratios


def export_dataset_samples(
    dataset: str,
    runs: list[RunInfo],
    candidates: list[Candidate],
    output_dir: Path,
    split: str,
    contact_cols: int,
    contact_tile_size: int,
    contact_dpi: int,
    contact_font_size: int,
    metric_name: str,
) -> None:
    if not candidates:
        return
    ref_run = choose_reference_run(runs)
    style = build_visual_style(ref_run.config, num_classes=int(ref_run.config.get("task", {}).get("num_classes", 2)))
    with (output_dir / "visual_style.json").open("w", encoding="utf-8") as file:
        json.dump(style, file, ensure_ascii=False, indent=2)
    resolver = DatasetResolver(runs, split_override=split)

    for rank, candidate in enumerate(candidates, start=1):
        sample_dir = output_dir / f"top-{rank}-{sanitize_filename(candidate.sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        located = resolver.find(candidate.sample_id, preferred=[candidate.winner, ref_run])
        if located is None:
            print(
                f"Warning: sample id not found in {dataset} dataset for split={resolver.preferred_split_name()}: "
                f"{candidate.sample_id}",
                file=sys.stderr,
            )
            continue
        dataset_obj, sample_index, source_run = located
        item = dataset_obj[sample_index]
        sample_style = build_visual_style(
            source_run.config,
            num_classes=int(source_run.config.get("task", {}).get("num_classes", style["num_classes"])),
        )
        export_inputs_and_gt(item, sample_dir, sample_style)
        prediction_paths: list[tuple[str, Path]] = []
        for model_idx, run in enumerate(runs):
            pred_path = export_prediction(run, candidate.sample_id, sample_dir, sample_style, prefix=model_idx)
            if pred_path is not None:
                prediction_paths.append((run.display_name, pred_path))
        write_sample_metadata(candidate, runs, sample_dir / "sample_selection.json", rank=rank, metric_name=metric_name)
        make_contact_sheet(
            sample_dir=sample_dir,
            prediction_paths=prediction_paths,
            output_path=sample_dir / "contact_sheet.png",
            max_cols=max(int(contact_cols), 2),
            tile_size=max(int(contact_tile_size), 128),
            dpi=max(int(contact_dpi), 72),
            font_size=max(int(contact_font_size), 10),
        )


def choose_reference_run(runs: list[RunInfo]) -> RunInfo:
    dehcd_lm = [run for run in runs if run.role == "dehcd" and run.dehcd_size in {"l", "m"}]
    if dehcd_lm:
        return max(dehcd_lm, key=lambda run: run.overall_score)
    dehcd = [run for run in runs if run.role == "dehcd"]
    return max(dehcd or runs, key=lambda run: run.overall_score)


class DatasetResolver:
    def __init__(self, runs: list[RunInfo], split_override: str = "auto"):
        self.runs = runs
        self.split_override = split_override
        self.cache: Dict[tuple[str, str], tuple[Any, Dict[str, int]]] = {}
        self.failed: set[tuple[str, str]] = set()

    def preferred_split_name(self) -> str:
        if self.split_override and self.split_override != "auto":
            return self.split_override
        split_names = sorted({self.run_split(run) for run in self.runs})
        return ",".join(split_names) if split_names else "auto"

    def run_split(self, run: RunInfo) -> str:
        if self.split_override and self.split_override != "auto":
            return self.split_override
        return run.split or "test"

    def find(self, sample_id: str, preferred: list[RunInfo] | None = None):
        ordered_runs = ordered_unique_runs((preferred or []) + self.runs)
        wanted = normalized_id_variants(sample_id)
        for run in ordered_runs:
            loaded = self.load(run)
            if loaded is None:
                continue
            dataset_obj, id_to_index = loaded
            for variant in wanted:
                if variant in id_to_index:
                    return dataset_obj, id_to_index[variant], run

        fuzzy_matches: list[tuple[Any, int, RunInfo]] = []
        for run in ordered_runs:
            loaded = self.load(run)
            if loaded is None:
                continue
            dataset_obj, id_to_index = loaded
            matches = {
                index
                for key, index in id_to_index.items()
                if any(key.endswith(variant) or variant.endswith(key) for variant in wanted if variant)
            }
            if len(matches) == 1:
                fuzzy_matches.append((dataset_obj, next(iter(matches)), run))
        return fuzzy_matches[0] if len(fuzzy_matches) == 1 else None

    def load(self, run: RunInfo):
        from datasets import build_dataset

        split = self.run_split(run)
        cache_key = (run.train_run, split)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if cache_key in self.failed:
            return None
        try:
            dataset_obj = build_dataset(run.config, split=split, training=False)
        except Exception as exc:
            self.failed.add(cache_key)
            print(f"Warning: could not build {run.train_run} split={split}: {exc}", file=sys.stderr)
            return None
        id_to_index: Dict[str, int] = {}
        for index, sample in enumerate(getattr(dataset_obj, "samples", [])):
            for alias in sample_aliases(sample):
                for variant in normalized_id_variants(alias):
                    id_to_index.setdefault(variant, index)
        self.cache[cache_key] = (dataset_obj, id_to_index)
        return self.cache[cache_key]


def ordered_unique_runs(runs: Iterable[RunInfo]) -> list[RunInfo]:
    out: list[RunInfo] = []
    seen: set[str] = set()
    for run in runs:
        if run.train_run in seen:
            continue
        seen.add(run.train_run)
        out.append(run)
    return out


def sample_aliases(sample: Dict[str, Any]) -> set[str]:
    aliases = {str(sample.get("id", ""))}
    for key in ("optical", "label"):
        value = sample.get(key)
        if value:
            aliases.update(path_aliases(value))
    for value in sample.get("sar", []) or []:
        aliases.update(path_aliases(value))
    return {alias for alias in aliases if alias}


def path_aliases(value: Any) -> set[str]:
    path = Path(str(value))
    return {path.name, path.stem}


def normalized_id_variants(value: Any) -> set[str]:
    text = normalize_sample_id(value)
    variants = {text}
    current = text
    for suffix in SAMPLE_ID_SUFFIXES:
        if current.endswith(suffix):
            variants.add(current[: -len(suffix)])
    if text.isdigit():
        variants.add(str(int(text)))
    return {variant for variant in variants if variant}


SAMPLE_ID_SUFFIXES = (
    "_pre_disaster",
    "_post_disaster",
    "_building_damage",
    "_building_localization",
    "_change",
    "_label",
    "_labels",
    "_mask",
    "_target",
    "_gt",
    "_vv",
    "_vh",
)


def normalize_sample_id(value: Any) -> str:
    text = Path(str(value)).stem.lower().strip()
    changed = True
    while changed:
        changed = False
        for suffix in SAMPLE_ID_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text


def export_inputs_and_gt(item: Dict[str, Any], sample_dir: Path, style: Dict[str, Any]) -> None:
    import torch

    save_rgb_image(tensor_preview_array(item["optical"], sar=False), sample_dir / "pre_optical.png")
    save_gray_or_rgb(tensor_preview_array(item["sar"], sar=True), sample_dir / "post_sar.png")
    save_sar_channel_grid(item["sar"], sample_dir / "post_sar_channels.png")
    label = item["label"].detach().cpu().long() if isinstance(item["label"], torch.Tensor) else torch.as_tensor(item["label"]).long()
    gt_color = colorize_mask_array(
        label,
        num_classes=int(style["num_classes"]),
        ignore_index=255,
        class_colors=style["class_colors"],
    )
    save_rgb_image(gt_color, sample_dir / "gt.png")
    save_label_png(label, sample_dir / "gt_raw_labels.png", num_classes=int(style["num_classes"]))


def export_prediction(run: RunInfo, sample_id: str, sample_dir: Path, style: Dict[str, Any], prefix: int) -> Path | None:
    import torch

    src = prediction_mask_path(run, sample_id)
    if not src.exists():
        print(f"Warning: missing prediction mask: {src}", file=sys.stderr)
        return None
    labels = read_prediction_labels(src, num_classes=int(style["num_classes"]))
    pred_dir = sample_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_filename(run.display_name)
    dst = pred_dir / f"{prefix:02d}_{safe_model}.png"
    pred_color = colorize_mask_array(
        torch.as_tensor(labels).long(),
        num_classes=int(style["num_classes"]),
        ignore_index=255,
        class_colors=style["class_colors"],
    )
    save_rgb_image(pred_color, dst)
    raw_dst = pred_dir / "raw_labels" / f"{prefix:02d}_{safe_model}.png"
    save_label_png(torch.as_tensor(labels).long(), raw_dst, num_classes=int(style["num_classes"]))
    return dst


def prediction_mask_path(run: RunInfo, sample_id: str) -> Path:
    safe_id = sanitize_filename(sample_id)
    return run.artifact_dir / "predictions" / "masks" / f"{safe_id}.png"


def read_prediction_labels(path: Path, num_classes: int):
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if int(num_classes) == 2 and arr.max(initial=0) > 1:
        arr = (arr > 0).astype("uint8")
    return arr.astype("uint8")


def save_label_png(label, path: Path, num_classes: int) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = label.detach().cpu().numpy().astype("uint8")
    if int(num_classes) == 2:
        arr = arr * 255
    Image.fromarray(arr).save(path)


def save_gray_or_rgb(array, path: Path) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def save_sar_channel_grid(sar, path: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    arr = sar.detach().cpu().float().numpy()
    if arr.ndim != 3 or arr.shape[0] <= 1:
        return
    tiles = []
    for idx in range(arr.shape[0]):
        img = robust_minmax_np(arr[idx])
        img = (img * 255).round().astype("uint8")
        tile = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 40, 16), fill=(0, 0, 0))
        draw.text((3, 2), f"ch{idx}", fill=(255, 255, 255))
        tiles.append(tile)
    cols = min(len(tiles), 4)
    rows = math.ceil(len(tiles) / cols)
    width, height = tiles[0].size
    sheet = Image.new("RGB", (cols * width, rows * height), color=(255, 255, 255))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * width, (idx // cols) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def make_contact_sheet(
    sample_dir: Path,
    prediction_paths: list[tuple[str, Path]],
    output_path: Path,
    max_cols: int,
    tile_size: int = 640,
    dpi: int = 600,
    font_size: int = 30,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    entries = [
        ("Pre-event Optical", sample_dir / "pre_optical.png"),
        ("Post-event SAR", sample_dir / "post_sar.png"),
        ("GT", sample_dir / "gt.png"),
    ]
    entries.extend(prediction_paths)
    entries = [(label, path) for label, path in entries if path.exists()]
    if not entries:
        return

    font = load_contact_font(font_size=font_size, bold=True)
    gap = max(tile_size // 28, 18)
    margin = max(tile_size // 24, 22)
    label_h = max(font_size + 24, 54)
    tile_w = tile_h = int(tile_size)
    cols = min(max_cols, len(entries))
    rows = math.ceil(len(entries) / cols)
    sheet_w = cols * tile_w + (cols - 1) * gap + 2 * margin
    sheet_h = rows * (tile_h + label_h) + (rows - 1) * gap + 2 * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, path) in enumerate(entries):
        image = Image.open(path).convert("RGB")
        is_mask = is_categorical_panel(path)
        image = resize_panel_image(image, size=(tile_w, tile_h), categorical=is_mask)
        col = idx % cols
        row = idx // cols
        x = margin + col * (tile_w + gap)
        y = margin + row * (tile_h + label_h + gap)
        title = compact_panel_label(label)
        draw_centered_text(draw, (x, y, x + tile_w, y + label_h), title, font=font)
        image_y = y + label_h
        sheet.paste(image, (x, image_y))
        draw.rectangle((x, image_y, x + tile_w - 1, image_y + tile_h - 1), outline=(185, 185, 185), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, dpi=(dpi, dpi))


def refresh_existing_contact_sheets(
    output_root: Path,
    contact_cols: int,
    tile_size: int,
    dpi: int,
    font_size: int,
) -> int:
    if not output_root.exists():
        raise FileNotFoundError(f"output root not found: {output_root}")
    refreshed = 0
    for sample_dir in sorted(output_root.glob("*/*")):
        if not sample_dir.is_dir() or not (sample_dir / "pre_optical.png").exists():
            continue
        prediction_paths = existing_prediction_paths(sample_dir)
        make_contact_sheet(
            sample_dir=sample_dir,
            prediction_paths=prediction_paths,
            output_path=sample_dir / "contact_sheet.png",
            max_cols=contact_cols,
            tile_size=tile_size,
            dpi=dpi,
            font_size=font_size,
        )
        refreshed += 1
    return refreshed


def existing_prediction_paths(sample_dir: Path) -> list[tuple[str, Path]]:
    pred_dir = sample_dir / "predictions"
    if not pred_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for path in sorted(pred_dir.glob("*.png")):
        out.append((label_from_prediction_filename(path), path))
    return out


def label_from_prediction_filename(path: Path) -> str:
    stem = path.stem
    if len(stem) >= 4 and stem[:2].isdigit() and stem[2] == "_":
        stem = stem[3:]
    return stem.replace("_", " ")


def load_contact_font(font_size: int, bold: bool = False):
    from PIL import ImageFont

    names = [
        "Times New Roman Bold.ttf" if bold else "Times New Roman.ttf",
        "timesbd.ttf" if bold else "times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "LiberationSerif-Bold.ttf" if bold else "LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def compact_panel_label(label: str) -> str:
    text = str(label).replace("_", " ").strip()
    replacements = {
        "Pre optical": "Pre-event Optical",
        "Post SAR": "Post-event SAR",
        "DEHCD-L": "Ours (l)",
        "DEHCD-M": "Ours (m)",
        "DEHCD-S": "Ours (s)",
        "HFA-PANet": "HFA-PANet",
    }
    return replacements.get(text, text)


def is_categorical_panel(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "predictions" in parts or name.startswith("gt") or "label" in name or "mask" in name


def resize_panel_image(image, size: tuple[int, int], categorical: bool):
    from PIL import Image, ImageOps

    resampling = Image.Resampling.NEAREST if categorical else Image.Resampling.LANCZOS
    image = ImageOps.contain(image, size, method=resampling)
    canvas = Image.new("RGB", size, color=(255, 255, 255))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_centered_text(draw, box: tuple[int, int, int, int], text: str, font) -> None:
    x0, y0, x1, y1 = box
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x = x0 + max((x1 - x0 - text_w) // 2, 0)
    y = y0 + max((y1 - y0 - text_h) // 2, 0)
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def write_sample_metadata(candidate: Candidate, runs: list[RunInfo], path: Path, rank: int, metric_name: str | None = None) -> None:
    metric_name = metric_name or ("iou" if candidate.dataset == "CAU" else "foreground_miou")
    payload = {
        "rank": rank,
        "dataset": candidate.dataset,
        "sample_id": candidate.sample_id,
        "winner": candidate.winner.display_name,
        "winner_run": candidate.winner.train_run,
        "winner_score": candidate.winner_score,
        "best_baseline": candidate.best_baseline.display_name,
        "best_baseline_run": candidate.best_baseline.train_run,
        "best_baseline_score": candidate.best_baseline_score,
        "margin": candidate.margin,
        "foreground_ratio": candidate.foreground_ratio,
        "target_class_ratios": candidate.target_class_ratios,
        "present_classes": list(candidate.present_classes),
        "dehcd_scores": candidate.dehcd_scores,
        "baseline_scores": candidate.baseline_scores,
        "models": [
            {
                "display_name": run.display_name,
                "train_run": run.train_run,
                "overall_score": run.overall_score,
                "role": run.role,
                "sample_score": sample_score(run, candidate.sample_id, metric_name),
            }
            for run in runs
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_candidates_csv(candidates: list[Candidate], path: Path, metric_name: str) -> None:
    rows = [candidate_row(candidate, metric_name) for candidate in candidates]
    write_dict_rows(rows, path)


def candidate_row(candidate: Candidate, metric_name: str) -> Dict[str, Any]:
    return {
        "dataset": candidate.dataset,
        "sample_id": candidate.sample_id,
        "metric": metric_name,
        "winner": candidate.winner.display_name,
        "winner_run": candidate.winner.train_run,
        "winner_score": candidate.winner_score,
        "best_baseline": candidate.best_baseline.display_name,
        "best_baseline_run": candidate.best_baseline.train_run,
        "best_baseline_score": candidate.best_baseline_score,
        "margin": candidate.margin,
        "foreground_ratio": candidate.foreground_ratio,
        "target_class_ratios": json.dumps(candidate.target_class_ratios, ensure_ascii=False),
        "present_classes": json.dumps(list(candidate.present_classes), ensure_ascii=False),
        "dehcd_scores": json.dumps(candidate.dehcd_scores, ensure_ascii=False),
        "baseline_scores": json.dumps(candidate.baseline_scores, ensure_ascii=False),
    }


def write_run_inventory(runs: list[RunInfo], path: Path) -> None:
    rows = [
        {
            "dataset": run.dataset,
            "model_key": run.model_key,
            "display_name": run.display_name,
            "role": run.role,
            "train_run": run.train_run,
            "overall_metric": run.metric_name,
            "overall_score": run.overall_score,
            "artifact_dir": str(run.artifact_dir),
        }
        for run in runs
    ]
    write_dict_rows(rows, path)


def write_dict_rows(rows: list[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as file:
            file.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def identify_model(result: Dict[str, Any], config: Dict[str, Any]) -> tuple[str, str, str, str | None]:
    model_cfg = config.get("model", {})
    train_run = str(result.get("train_run") or "")
    run_lower = train_run.lower()
    compare_model = str(model_cfg.get("compare_model") or "").strip().lower()
    if compare_model:
        return compare_model, COMPARE_DISPLAY_NAMES.get(compare_model, compare_model), "compare", None
    backbone = str(model_cfg.get("backbone") or "").strip().lower()
    name = str(model_cfg.get("name") or "").strip()
    size = None
    for candidate in ("l", "m", "s"):
        if backbone == f"dehcd_{candidate}" or f"_dehcd_{candidate}" in run_lower:
            size = candidate
            break
    if size is not None or "dehcd" in name.lower() or "dehcd" in run_lower:
        size = size or "unknown"
        display = f"DEHCD-{size.upper()}" if size in {"s", "m", "l"} else "DEHCD"
        return f"dehcd_{size}", display, "dehcd", size
    if "lightweight" in backbone or "lightweight" in run_lower:
        return "lightweight", "Lightweight", "baseline", None
    key = sanitize_filename(name or backbone or train_run).lower()
    display = name or backbone or short_run_name(train_run)
    return key, display, "baseline", None


def dataset_key(result: Dict[str, Any], config: Dict[str, Any]) -> str:
    dataset_cfg = result.get("dataset") or config.get("dataset", {})
    text = " ".join(
        str(value).lower()
        for value in [
            dataset_cfg.get("type", ""),
            dataset_cfg.get("name", ""),
            dataset_cfg.get("root", ""),
            config.get("task", {}).get("name", ""),
            config.get("logging", {}).get("run_name", ""),
            result.get("train_run", ""),
        ]
    )
    if "cau" in text or "flood" in text:
        return "CAU"
    if "haiti" in text:
        return "Haiti"
    if "bright" in text:
        return "BRIGHT"
    return sanitize_filename(dataset_cfg.get("name") or dataset_cfg.get("type") or "dataset")


def result_metric(result: Dict[str, Any], metric_name: str) -> float:
    if "best_metric_value" in result:
        return to_float(result["best_metric_value"])
    metrics = result.get("metrics") or {}
    if metric_name in metrics:
        return to_float(metrics[metric_name])
    if "primary_score" in metrics:
        return to_float(metrics["primary_score"])
    return 0.0


def is_extra_run(name: str) -> bool:
    lowered = str(name).lower()
    return any(token in lowered for token in ("discussion", "ablation", "ab_", "no_"))


def short_run_name(name: str) -> str:
    text = str(name)
    parts = text.split("_", 2)
    return parts[-1] if len(parts) >= 3 and parts[0].isdigit() else text


def robust_minmax_np(array):
    import numpy as np

    finite = np.isfinite(array)
    if not bool(finite.any()):
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(array[finite], [2, 98])
    return np.clip((array - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


if __name__ == "__main__":
    main()
