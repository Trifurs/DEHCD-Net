from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.select_paper_samples import (  # noqa: E402
    COMPARE_DISPLAY_NAMES,
    DEHCD_SIZE_PRIORITY,
    dataset_key,
    identify_model,
    is_extra_run,
    result_metric,
    to_float,
)
from tools.test import sanitize_filename  # noqa: E402


DATASET_ORDER = {"BRIGHT": 0, "CAU": 1, "Haiti": 2}
COMPARE_MODEL_ORDER = {
    "dminet": 0,
    "haff": 1,
    "hfa_panet": 2,
    "hrsicd": 3,
    "icif_net": 4,
    "wavehfg": 5,
    "lightweight": 6,
}
FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
STATISTIC_METRIC_COLUMNS = ("f1", "foreground_miou", "iou", "miou", "oa", "precision", "recall")
XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass
class MatrixRun:
    result_path: Path
    artifact_dir: Path
    train_run: str
    dataset: str
    model_key: str
    display_name: str
    role: str
    dehcd_size: str | None
    metric_name: str
    score: float
    matrix: list[list[float]]
    class_names: list[str]
    result: dict[str, Any]
    config: dict[str, Any]


@dataclass
class StatisticsTarget:
    dataset: str
    model_key: str
    display_name: str
    metrics: dict[str, float]
    sheet_name: str
    row_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw paper-ready multi-model confusion-matrix grids from tools/test.py outputs."
    )
    parser.add_argument(
        "--test-root",
        default="runs/test_paper_all_2",
        help="Folder containing per-run result JSON files and confusion_matrix.csv artifacts.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output folder. Defaults to <test-root>/confusion_matrix_grids.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to plot, e.g. BRIGHT CAU Haiti.",
    )
    parser.add_argument(
        "--metric",
        default="auto",
        help="Metric used to select the best duplicate run. auto uses each result's best_metric.",
    )
    parser.add_argument(
        "--statistics-xlsx",
        default="runs/statistics.xlsx",
        help="Workbook whose 2nd/3rd/4th sheets define the paper-table metrics used to select runs.",
    )
    parser.add_argument(
        "--ignore-statistics",
        action="store_true",
        help="Ignore --statistics-xlsx and select runs by the result JSON metric instead.",
    )
    parser.add_argument("--columns", type=int, default=3, help="Number of matrix panels per row.")
    parser.add_argument("--dpi", type=int, default=900, help="DPI for saved PNG figures.")
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats supported by matplotlib, e.g. png or png,pdf.",
    )
    parser.add_argument(
        "--annotate-threshold",
        type=float,
        default=0.005,
        help="Hide row-normalized cell annotations below this value. 0.005 means 0.5%%.",
    )
    parser.add_argument(
        "--include-extra-runs",
        action="store_true",
        help="Include ablation/discussion runs. By default only main DEHCD and comparison runs are used.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Do not print the selected metric under each model title.",
    )
    parser.add_argument(
        "--no-class-key",
        action="store_true",
        help="Do not print the abbreviation key below each figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_root = resolve_input_root(args.test_root)
    if not test_root.exists():
        raise FileNotFoundError(f"test root not found: {test_root}")
    output_root = Path(args.output_root).expanduser() if args.output_root else test_root / "confusion_matrix_grids"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    statistics_targets: list[StatisticsTarget] = []
    statistics_path = resolve_project_path(args.statistics_xlsx)
    if not args.ignore_statistics:
        if statistics_path.exists():
            statistics_targets = load_statistics_targets(statistics_path)
        else:
            print(f"Warning: statistics workbook not found: {statistics_path}", file=sys.stderr)

    runs = load_matrix_runs(
        test_root=test_root,
        metric=str(args.metric),
        include_extra_runs=bool(args.include_extra_runs or statistics_targets),
    )
    if statistics_targets:
        selected_runs = select_runs_from_statistics(runs, statistics_targets)
    else:
        selected_runs = select_best_runs(runs)
    if args.datasets:
        wanted = {normalize_dataset_name(name) for name in args.datasets}
        selected_runs = [run for run in selected_runs if normalize_dataset_name(run.dataset) in wanted]
    if not selected_runs:
        raise RuntimeError(f"no usable confusion matrices found under {test_root}")

    write_run_inventory(selected_runs, output_root / "selected_run_inventory.csv")
    grouped = group_by_dataset(selected_runs)
    outputs: dict[str, list[str]] = {}
    for dataset in sorted(grouped, key=dataset_sort_key):
        dataset_runs = sorted(grouped[dataset], key=model_sort_key)
        paths = plot_dataset_grid(
            dataset=dataset,
            runs=dataset_runs,
            output_root=output_root,
            columns=max(int(args.columns), 1),
            dpi=max(int(args.dpi), 72),
            formats=parse_formats(args.formats),
            annotate_threshold=max(float(args.annotate_threshold), 0.0),
            show_score=not bool(args.no_score),
            show_class_key=not bool(args.no_class_key),
        )
        outputs[dataset] = [str(path) for path in paths]

    with (output_root / "confusion_matrix_grid_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "test_root": str(test_root),
                "output_root": str(output_root),
                "statistics_xlsx": str(statistics_path) if statistics_targets else None,
                "selected_runs": [run_inventory_row(run) for run in selected_runs],
                "outputs": outputs,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Loaded {len(runs)} usable run(s); retained {len(selected_runs)} best model run(s).")
    for dataset, paths in outputs.items():
        print(f"{dataset}: " + ", ".join(paths))


def resolve_input_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_project_path(value: Any, fallback: Path | None = None) -> Path:
    if value in (None, ""):
        if fallback is None:
            return PROJECT_ROOT
        return fallback
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_matrix_runs(test_root: Path, metric: str, include_extra_runs: bool) -> list[MatrixRun]:
    runs: list[MatrixRun] = []
    for result_path in sorted(test_root.glob("*.json")):
        if result_path.name in {"selection_summary.json", "confusion_matrix_grid_summary.json"}:
            continue
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        train_run = str(result.get("train_run") or result_path.stem)
        if is_extra_run(train_run) and not include_extra_runs:
            continue
        artifact_dir = resolve_project_path(result.get("artifact_dir"), fallback=test_root / train_run)
        config = load_config_for_result(result)
        dataset = dataset_key(result, config)
        model_key, display_name, role, dehcd_size = identify_model_with_fallback(result, config)
        metric_name = resolve_score_metric(result, metric)
        score = result_metric(result, metric_name)
        matrix, csv_class_names = load_confusion_matrix(result, artifact_dir)
        if not matrix:
            print(f"Warning: skip {train_run}, missing confusion matrix.", file=sys.stderr)
            continue
        class_names = load_class_names(
            artifact_dir=artifact_dir,
            result=result,
            config=config,
            csv_class_names=csv_class_names,
            num_classes=len(matrix),
        )
        runs.append(
            MatrixRun(
                result_path=result_path,
                artifact_dir=artifact_dir,
                train_run=train_run,
                dataset=dataset,
                model_key=model_key,
                display_name=display_name,
                role=role,
                dehcd_size=dehcd_size,
                metric_name=metric_name,
                score=score,
                matrix=matrix,
                class_names=class_names,
                result=result,
                config=config,
            )
        )
    return runs


def load_config_for_result(result: dict[str, Any]) -> dict[str, Any]:
    train_run_dir = resolve_project_path(result.get("train_run_dir"), fallback=PROJECT_ROOT)
    snapshot = train_run_dir / "config_snapshot.json"
    if snapshot.exists():
        with snapshot.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def identify_model_with_fallback(result: dict[str, Any], config: dict[str, Any]) -> tuple[str, str, str, str | None]:
    model_key, display_name, role, dehcd_size = identify_model(result, config)
    train_run = str(result.get("train_run") or "").lower()
    if role == "compare":
        return model_key, display_name, role, dehcd_size

    for key in sorted(COMPARE_DISPLAY_NAMES, key=len, reverse=True):
        if f"compare_" in train_run and key in train_run:
            return key, COMPARE_DISPLAY_NAMES[key], "compare", None

    for size in ("l", "m", "s"):
        if f"_dehcd_{size}" in train_run:
            return f"dehcd_{size}", f"DEHCD-{size.upper()}", "dehcd", size
    return model_key, display_name, role, dehcd_size


def resolve_score_metric(result: dict[str, Any], metric: str) -> str:
    if str(metric).strip().lower() not in {"", "auto", "default"}:
        return str(metric)
    return str(result.get("best_metric") or result.get("checkpoint_best_metric_name") or "primary_score")


def load_confusion_matrix(result: dict[str, Any], artifact_dir: Path) -> tuple[list[list[float]], list[str]]:
    raw = result.get("confusion_matrix")
    if isinstance(raw, list) and raw:
        return [[to_float(value) for value in row] for row in raw], []
    return read_confusion_csv(artifact_dir / "confusion_matrix.csv")


def read_confusion_csv(path: Path) -> tuple[list[list[float]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)
    if not rows:
        return [], []
    class_names = [item.strip() for item in rows[0][1:]]
    matrix: list[list[float]] = []
    for row in rows[1:]:
        if len(row) <= 1:
            continue
        matrix.append([to_float(value) for value in row[1:]])
    return matrix, class_names


def load_class_names(
    artifact_dir: Path,
    result: dict[str, Any],
    config: dict[str, Any],
    csv_class_names: list[str],
    num_classes: int,
) -> list[str]:
    for candidate in (
        load_visual_style_class_names(artifact_dir / "visual_style.json"),
        nested_class_names(result.get("artifacts"), "visual_style"),
        nested_class_names(config, "task"),
        nested_class_names(config, "dataset"),
        csv_class_names,
    ):
        if candidate and len(candidate) >= num_classes:
            return [str(item) for item in candidate[:num_classes]]
    return ["background", *[f"class {idx}" for idx in range(1, num_classes)]]


def load_visual_style_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError:
        return []
    names = payload.get("class_names")
    return [str(item) for item in names] if isinstance(names, list) else []


def nested_class_names(payload: Any, key: str) -> list[str]:
    if not isinstance(payload, dict):
        return []
    nested = payload.get(key)
    if not isinstance(nested, dict):
        return []
    names = nested.get("class_names")
    return [str(item) for item in names] if isinstance(names, list) else []


def load_statistics_targets(path: Path) -> list[StatisticsTarget]:
    sheets = read_xlsx_sheets(path)
    targets: list[StatisticsTarget] = []
    for sheet_order, (sheet_name, rows) in enumerate(sheets[1:4], start=2):
        dataset = statistics_dataset_name(sheet_name)
        if not rows:
            continue
        header = [normalize_header(cell) for cell in rows[0]]
        for row_index, row in enumerate(rows[1:], start=2):
            record = {header[idx]: row[idx] for idx in range(min(len(header), len(row))) if header[idx]}
            model_name = str(record.get("model") or "").strip()
            if not model_name:
                continue
            metrics = {
                key: to_float(record[key])
                for key in STATISTIC_METRIC_COLUMNS
                if key in record and str(record.get(key, "")).strip() != ""
            }
            model_key, display_name = statistics_model_identity(model_name)
            targets.append(
                StatisticsTarget(
                    dataset=dataset,
                    model_key=model_key,
                    display_name=display_name,
                    metrics=metrics,
                    sheet_name=sheet_name,
                    row_index=row_index,
                )
            )
    return targets


def read_xlsx_sheets(path: Path) -> list[tuple[str, list[list[str]]]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheets: list[tuple[str, list[list[str]]]] = []
        for sheet in workbook.find("a:sheets", XLSX_NS) or []:
            sheet_name = str(sheet.attrib.get("name") or "")
            rel_id = sheet.attrib.get(f"{{{XLSX_NS['r']}}}id")
            target = rel_map.get(str(rel_id), "")
            sheet_path = "xl/" + target.lstrip("/")
            sheets.append((sheet_name, read_xlsx_rows(archive, sheet_path, shared_strings)))
        return sheets


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for item in root.findall("a:si", XLSX_NS):
        out.append("".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS)))
    return out


def read_xlsx_rows(archive: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.findall(".//a:sheetData/a:row", XLSX_NS):
        values: list[str] = []
        for cell in row.findall("a:c", XLSX_NS):
            index = xlsx_column_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append("")
            values[index] = xlsx_cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "s":
        value = cell.find("a:v", XLSX_NS)
        if value is None or value.text is None:
            return ""
        index = int(value.text)
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", XLSX_NS))
    value = cell.find("a:v", XLSX_NS)
    return value.text if value is not None and value.text is not None else ""


def xlsx_column_index(ref: str) -> int:
    column = "".join(char for char in str(ref) if char.isalpha())
    index = 0
    for char in column.upper():
        index = index * 26 + ord(char) - ord("A") + 1
    return max(index - 1, 0)


def normalize_header(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def statistics_dataset_name(sheet_name: str) -> str:
    lowered = str(sheet_name).strip().lower()
    if "bright" in lowered:
        return "BRIGHT"
    if "cau" in lowered:
        return "CAU"
    if "haiti" in lowered:
        return "Haiti"
    return str(sheet_name).strip()


def statistics_model_identity(model_name: str) -> tuple[str, str]:
    normalized = " ".join(model_name.lower().replace("-", "_").split())
    if "dehcd" in normalized:
        if "(l)" in normalized or " l" in normalized:
            return "dehcd_l", "Ours-L"
        if "(m)" in normalized or " m" in normalized:
            return "dehcd_m", "Ours-M"
        if "(s)" in normalized or " s" in normalized:
            return "dehcd_s", "Ours-S"
    aliases = {
        "dminet": ("dminet", "DMINet"),
        "haff": ("haff", "HAFF"),
        "hfa_panet": ("hfa_panet", "HFA-PANet"),
        "hfa panet": ("hfa_panet", "HFA-PANet"),
        "hrsicd": ("hrsicd", "HRSICD"),
        "icif_net": ("icif_net", "ICIFNet"),
        "icif net": ("icif_net", "ICIFNet"),
        "wavehfg": ("wavehfg", "WaveHFG"),
    }
    for token, identity in aliases.items():
        if token in normalized:
            return identity
    fallback = sanitize_filename(model_name).lower()
    return fallback, model_name


def select_runs_from_statistics(runs: Iterable[MatrixRun], targets: Iterable[StatisticsTarget]) -> list[MatrixRun]:
    run_list = list(runs)
    selected: list[MatrixRun] = []
    used_result_paths: set[Path] = set()
    for target in targets:
        candidates = [
            run
            for run in run_list
            if run.dataset == target.dataset and run.model_key == target.model_key and run.result_path not in used_result_paths
        ]
        if not candidates:
            print(
                f"Warning: no candidate run for {target.sheet_name} row {target.row_index}: "
                f"{target.display_name} ({target.model_key})",
                file=sys.stderr,
            )
            continue
        best = min(candidates, key=lambda run: statistics_match_key(run, target))
        error, count = statistics_match_error(best, target)
        if count > 0 and error / count > 5e-5:
            print(
                f"Warning: loose statistics match for {target.dataset} {target.display_name}: "
                f"mean_abs_error={error / count:.6g}, run={best.train_run}",
                file=sys.stderr,
            )
        selected.append(best)
        used_result_paths.add(best.result_path)
    return sorted(selected, key=lambda run: (dataset_sort_key(run.dataset), model_sort_key(run)))


def statistics_match_key(run: MatrixRun, target: StatisticsTarget) -> tuple[float, float, str]:
    error, count = statistics_match_error(run, target)
    score = error / max(count, 1)
    return score, -run.score, run.train_run


def statistics_match_error(run: MatrixRun, target: StatisticsTarget) -> tuple[float, int]:
    metrics = run.result.get("metrics") if isinstance(run.result.get("metrics"), dict) else {}
    error = 0.0
    count = 0
    for key, target_value in target.metrics.items():
        if key not in metrics:
            continue
        error += abs(to_float(metrics[key]) - float(target_value))
        count += 1
    return error, count


def select_best_runs(runs: Iterable[MatrixRun]) -> list[MatrixRun]:
    best: dict[tuple[str, str], MatrixRun] = {}
    for run in runs:
        key = (run.dataset, run.model_key)
        current = best.get(key)
        if current is None or run.score > current.score:
            best[key] = run
    return sorted(best.values(), key=lambda run: (dataset_sort_key(run.dataset), model_sort_key(run)))


def group_by_dataset(runs: Iterable[MatrixRun]) -> dict[str, list[MatrixRun]]:
    grouped: dict[str, list[MatrixRun]] = {}
    for run in runs:
        grouped.setdefault(run.dataset, []).append(run)
    return grouped


def normalize_dataset_name(value: Any) -> str:
    return str(value).strip().lower()


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    return DATASET_ORDER.get(dataset, 99), dataset.lower()


def model_sort_key(run: MatrixRun) -> tuple[int, str]:
    if run.role == "dehcd":
        rank = DEHCD_SIZE_PRIORITY.get(run.dehcd_size or "", 9)
        return rank, run.display_name.lower()
    if run.role == "compare":
        return 10 + COMPARE_MODEL_ORDER.get(run.model_key, 99), run.display_name.lower()
    return 30, run.display_name.lower()


def parse_formats(value: str) -> list[str]:
    formats = []
    for item in str(value or "png").split(","):
        cleaned = item.strip().lower().lstrip(".")
        if cleaned and cleaned not in formats:
            formats.append(cleaned)
    return formats or ["png"]


def setup_matplotlib():
    mpl_cache = Path(os.environ.get("MPLCONFIGDIR", "/tmp/dehcd_matplotlib"))
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import PercentFormatter

    cmap = LinearSegmentedColormap.from_list(
        "paper_blues",
        ["#fbfdff", "#e8f2f8", "#bfddeb", "#73b6d4", "#2479b7", "#083d77"],
        N=256,
    )
    return plt, Rectangle, PercentFormatter, cmap


def plot_dataset_grid(
    dataset: str,
    runs: list[MatrixRun],
    output_root: Path,
    columns: int,
    dpi: int,
    formats: list[str],
    annotate_threshold: float,
    show_score: bool,
    show_class_key: bool,
) -> list[Path]:
    import numpy as np

    plt, Rectangle, PercentFormatter, cmap = setup_matplotlib()
    class_names = harmonized_class_names(runs)
    short_names = short_class_names(class_names)
    num_classes = len(short_names)
    columns = min(max(columns, 1), max(len(runs), 1))
    rows = int(math.ceil(len(runs) / columns))
    panel = 2.48 if num_classes <= 2 else 2.82 if num_classes <= 4 else 3.08
    fig_width = columns * panel + 0.92
    fig_height = rows * (panel + 0.38) + 0.72 + (0.62 if show_class_key else 0.0)
    style = {
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#1f2933",
        "axes.linewidth": 1.0,
        "xtick.color": "#1f2933",
        "ytick.color": "#1f2933",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height), squeeze=False)
        fig.patch.set_facecolor("white")
        image = None
        for index, axis in enumerate(axes.flat):
            if index >= len(runs):
                axis.axis("off")
                continue
            run = runs[index]
            matrix = np.asarray(run.matrix, dtype=np.float64)
            normalized = normalize_rows(matrix)
            row_idx = index // columns
            col_idx = index % columns
            image = axis.imshow(normalized, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            decorate_axis(
                axis=axis,
                normalized=normalized,
                matrix=matrix,
                short_names=short_names,
                row_idx=row_idx,
                col_idx=col_idx,
                rows=rows,
                title=panel_title(index, run, show_score=show_score),
                annotate_threshold=annotate_threshold,
                rectangle_cls=Rectangle,
            )

        top = 0.965
        bottom = 0.118 if show_class_key else 0.075
        fig.subplots_adjust(left=0.078, right=0.888, top=top, bottom=bottom, wspace=0.20, hspace=0.40)
        cax = fig.add_axes([0.912, bottom + 0.015, 0.018, top - bottom - 0.03])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        colorbar.ax.tick_params(length=0, labelsize=12)
        colorbar.outline.set_visible(False)
        colorbar.set_label("Row-normalized (%)", fontsize=13, fontweight="bold", labelpad=8)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")

        if show_class_key:
            fig.text(
                0.5,
                0.034,
                class_key_text(dataset, short_names, class_names),
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="#1f2933",
            )

        outputs: list[Path] = []
        base = output_root / f"confusion_matrix_grid_{sanitize_filename(dataset).lower()}"
        for fmt in formats:
            path = base.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
            outputs.append(path)
        plt.close(fig)
    return outputs


def normalize_rows(matrix: Any):
    import numpy as np

    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=np.float64), where=row_sum > 0)


def decorate_axis(
    axis: Any,
    normalized: Any,
    matrix: Any,
    short_names: list[str],
    row_idx: int,
    col_idx: int,
    rows: int,
    title: str,
    annotate_threshold: float,
    rectangle_cls: Any,
) -> None:
    import numpy as np

    num_classes = len(short_names)
    axis.set_title(title, fontsize=14, pad=8)
    axis.set_xticks(np.arange(num_classes))
    axis.set_yticks(np.arange(num_classes))
    axis.set_xticklabels(short_names if row_idx == rows - 1 else [])
    axis.set_yticklabels(short_names if col_idx == 0 else [])
    axis.tick_params(axis="both", which="major", length=0, pad=4, labelsize=12)
    axis.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.25)
    axis.tick_params(which="minor", bottom=False, left=False)
    if row_idx == rows - 1:
        axis.set_xlabel("Pred.", fontsize=12, labelpad=5)
    if col_idx == 0:
        axis.set_ylabel("Target", fontsize=12, labelpad=5)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set_fontweight("bold")

    font_size = 12 if num_classes <= 2 else 10.5 if num_classes <= 4 else 8.5
    for y in range(num_classes):
        for x in range(num_classes):
            value = float(normalized[y, x])
            if matrix[y, x] <= 0 or value < annotate_threshold:
                continue
            axis.text(
                x,
                y,
                cell_label(value),
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color="white" if value >= 0.54 else "#17202a",
            )
    for idx in range(num_classes):
        axis.add_patch(rectangle_cls((idx - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="#0b253a", linewidth=1.0))


def panel_title(index: int, run: MatrixRun, show_score: bool) -> str:
    prefix = f"({chr(ord('a') + index)})"
    name = paper_model_label(run)
    if not show_score:
        return f"{prefix} {name}"
    return f"{prefix} {name}\n{metric_label(run.metric_name)} {run.score * 100.0:.2f}"


def paper_model_label(run: MatrixRun) -> str:
    if run.role == "dehcd" and run.dehcd_size in {"l", "m", "s"}:
        return f"Ours-{run.dehcd_size.upper()}"
    return run.display_name


def metric_label(metric_name: str) -> str:
    lowered = str(metric_name).lower()
    if lowered in {"iou", "change_iou", "binary_foreground_iou"}:
        return "IoU"
    if lowered == "foreground_miou":
        return "FmIoU"
    if lowered in {"miou", "mean_iou"}:
        return "mIoU"
    if lowered in {"f1", "foreground_f1", "change_f1"}:
        return "F1"
    if lowered == "oa":
        return "OA"
    return metric_name


def cell_label(value: float) -> str:
    percent = value * 100.0
    return f"{percent:.1f}"


def harmonized_class_names(runs: list[MatrixRun]) -> list[str]:
    if not runs:
        return []
    best = max(runs, key=lambda run: len(run.class_names))
    return best.class_names


def short_class_names(class_names: list[str]) -> list[str]:
    return ["B" if idx == 0 else f"C{idx}" for idx, _ in enumerate(class_names)]


def class_key_text(dataset: str, short_names: list[str], class_names: list[str]) -> str:
    parts = [
        f"{short}: {compact_class_name(name, idx, dataset)}"
        for idx, (short, name) in enumerate(zip(short_names, class_names))
    ]
    return "\n".join(textwrap.wrap("   |   ".join(parts), width=112))


def compact_class_name(name: str, idx: int, dataset: str) -> str:
    if dataset == "Haiti":
        haiti_names = {
            0: "Background",
            1: "Change class 1",
            2: "Change class 2",
            3: "Change class 3",
        }
        return haiti_names.get(idx, f"Change class {idx}")
    text = " ".join(str(name).replace("_", " ").split())
    if idx == 0 and text.lower() in {"background", "bg"}:
        return "Background"
    if text.lower().startswith("change class "):
        return f"Change class {idx}"
    return text[:1].upper() + text[1:]


def write_run_inventory(runs: list[MatrixRun], path: Path) -> None:
    rows = [run_inventory_row(run) for run in sorted(runs, key=lambda item: (dataset_sort_key(item.dataset), model_sort_key(item)))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_inventory_row(run: MatrixRun) -> dict[str, Any]:
    return {
        "dataset": run.dataset,
        "model_key": run.model_key,
        "display_name": paper_model_label(run),
        "role": run.role,
        "train_run": run.train_run,
        "metric": run.metric_name,
        "score": run.score,
        "artifact_dir": str(run.artifact_dir),
        "result_path": str(run.result_path),
    }


if __name__ == "__main__":
    main()
