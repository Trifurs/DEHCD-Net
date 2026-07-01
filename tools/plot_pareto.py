from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_CACHE_DIR = Path("/tmp/matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))


SPREADSHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
REL_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
OURS_PATTERN = re.compile(r"(dehcd|ours)", re.IGNORECASE)
OURS_SIZE_PATTERN = re.compile(r"\(([sml])\)", re.IGNORECASE)
OURS_SIZE_ORDER = {"s": 0, "m": 1, "l": 2}
METHOD_COLORS = {
    "DMINet": "#4E79A7",
    "HAFF": "#59A14F",
    "HFA-PANet": "#F28E2B",
    "HRSICD": "#9C755F",
    "ICIF-Net": "#76B7B2",
    "WaveHFG": "#8E6CBE",
    "Ours": "#D43F3A",
    "Ours (s)": "#D43F3A",
    "Ours (m)": "#D43F3A",
    "Ours (l)": "#D43F3A",
}
OURS_COLOR = "#D43F3A"
STYLE = {
    "figure": {
        "width_per_dataset": 7.2,
        "height": 7.0,
        "left": 0.055,
        "right": 0.995,
        "top": 0.86,
        "bottom": 0.34,
        "wspace": 0.27,
    },
    "font": {
        "title": 30,
        "axis_label": 26,
        "tick": 22,
        "legend": 28,
        "suptitle": 28,
        "break_label": 26,
        "annotation": 19,
        "annotation_ours": 20,
    },
    "marker": {
        "area_min": 230.0,
        "area_max": 1450.0,
        "ours_scale": 1.18,
        "edge_width": 2.4,
        "edge_width_ours": 2.7,
        "alpha": 0.9,
        "alpha_ours": 0.95,
        "legend_size": 18.0,
        "size_legend_scale": 1.45,
    },
    "line": {
        "ours_width": 5.2,
        "frontier_width": 4.0,
        "leader_width": 2.0,
        "grid_width": 1.25,
        "spine_width": 1.6,
    },
    "legend": {
        "rows": 2,
        "row_gap": 0.085,
        "handletextpad": 1.0,
        "columnspacing": 1.55,
        "anchor_y": 0.02,
    },
    "mean": {
        "figsize": 8.8,
        "left": 0.145,
        "right": 0.965,
        "top": 0.96,
        "bottom": 0.135,
        "title": 34,
        "axis_label": 31,
        "tick": 24,
        "label": 19,
        "label_ours": 21,
        "marker_scale": 1.08,
        "size_legend_title": 20,
        "size_legend": 18,
        "size_legend_scale": 0.72,
        "size_legend_anchor": (0.985, 0.985),
    },
    "break": {
        "shade_color": "#F6F8FB",
        "label_color": "#667085",
    },
}


@dataclass(frozen=True)
class Point:
    dataset: str
    model: str
    params: float
    gflops: float
    metric: float
    is_ours: bool
    ours_size: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a publication-ready accuracy-efficiency Pareto figure from statistics.xlsx."
    )
    parser.add_argument("--xlsx", default="runs/statistics.xlsx", help="Input workbook.")
    parser.add_argument(
        "--sheets",
        nargs="*",
        default=None,
        help="Dataset sheet names. Defaults to workbook sheets 2/3/4.",
    )
    parser.add_argument(
        "--sheet-indices",
        nargs="*",
        type=int,
        default=[2, 3, 4],
        help="1-based sheet indices used when --sheets is omitted.",
    )
    parser.add_argument("--metric", default="foreground_miou", help="Accuracy column for the y-axis.")
    parser.add_argument("--efficiency", default="gflops", help="Efficiency-cost column for the x-axis.")
    parser.add_argument("--params", default="params", help="Parameter-count column used for marker area.")
    parser.add_argument("--output-dir", default="runs/figures", help="Directory for exported figures and CSV.")
    parser.add_argument("--name", default="pareto_accuracy_efficiency", help="Output filename stem.")
    parser.add_argument(
        "--mean-name",
        default=None,
        help="Output filename stem for the mean-metric figure. Defaults to '<name>_mean'.",
    )
    parser.add_argument("--no-mean-plot", action="store_true", help="Do not export the mean-metric Pareto figure.")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf", "svg"], help="Figure formats to save.")
    parser.add_argument("--dpi", type=int, default=1600, help="Raster output DPI.")
    parser.add_argument("--linear-x", action="store_true", help="Deprecated; linear x-axis is now the default.")
    parser.add_argument("--log-x", action="store_true", help="Use a logarithmic GFLOPS axis.")
    parser.add_argument("--no-x-break", action="store_true", help="Do not compress the empty GFLOPS interval.")
    parser.add_argument("--x-break-start", type=float, default=60.0, help="Start of compressed GFLOPS interval.")
    parser.add_argument("--x-break-end", type=float, default=160.0, help="End of compressed GFLOPS interval.")
    parser.add_argument("--x-break-width", type=float, default=14.0, help="Displayed width of the compressed GFLOPS interval.")
    parser.add_argument("--x-max", type=float, default=180.0, help="Maximum GFLOPS tick shown on the x-axis.")
    parser.add_argument("--labels", action="store_true", help="Annotate individual methods with labels and leader lines.")
    parser.add_argument("--no-labels", action="store_true", help="Deprecated; labels are disabled by default.")
    parser.add_argument("--frontier", action="store_true", help="Draw the Pareto frontier line.")
    parser.add_argument("--no-frontier", action="store_true", help="Deprecated; frontier is disabled by default.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xlsx_path = Path(args.xlsx).expanduser()
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    workbook = read_xlsx_tables(xlsx_path)
    selected_sheet_names = resolve_sheet_names(workbook, args.sheets, args.sheet_indices)
    points_by_dataset: dict[str, list[Point]] = {}
    for sheet_name in selected_sheet_names:
        rows = workbook[sheet_name]
        points_by_dataset[sheet_name] = parse_points(
            sheet_name,
            rows,
            metric_col=str(args.metric),
            efficiency_col=str(args.efficiency),
            params_col=str(args.params),
        )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_plot_data(points_by_dataset, output_dir / f"{args.name}_data.csv")

    log_x = bool(args.log_x) and not bool(args.linear_x)
    x_break = None
    if not log_x and not bool(args.no_x_break) and float(args.x_break_end) > float(args.x_break_start):
        x_break = (float(args.x_break_start), float(args.x_break_end), max(float(args.x_break_width), 1.0))

    plot_pareto(
        points_by_dataset=points_by_dataset,
        output_dir=output_dir,
        stem=str(args.name),
        metric_col=str(args.metric),
        efficiency_col=str(args.efficiency),
        formats=[str(item).lstrip(".") for item in args.formats],
        dpi=max(int(args.dpi), 72),
        log_x=log_x,
        x_break=x_break,
        x_max=float(args.x_max) if float(args.x_max) > 0 else None,
        annotate=bool(args.labels) and not bool(args.no_labels),
        draw_frontier=bool(args.frontier) and not bool(args.no_frontier),
        title=str(args.title).strip(),
    )
    if not bool(args.no_mean_plot):
        mean_stem = str(args.mean_name or f"{args.name}_mean")
        mean_points_by_dataset = mean_points(points_by_dataset)
        write_plot_data(mean_points_by_dataset, output_dir / f"{mean_stem}_data.csv")
        plot_mean_pareto(
            points_by_dataset=mean_points_by_dataset,
            output_dir=output_dir,
            stem=mean_stem,
            metric_col=str(args.metric),
            efficiency_col=str(args.efficiency),
            formats=[str(item).lstrip(".") for item in args.formats],
            dpi=max(int(args.dpi), 72),
            log_x=log_x,
            x_break=x_break,
            x_max=float(args.x_max) if float(args.x_max) > 0 else None,
            draw_frontier=bool(args.frontier) and not bool(args.no_frontier),
            title=str(args.title).strip(),
        )
    print(f"Saved Pareto figure(s) to {output_dir}")


def read_xlsx_tables(path: Path) -> dict[str, list[dict[str, object]]]:
    rows_by_sheet: dict[str, list[list[object]]] = {}
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_targets = read_sheet_targets(archive)
        for sheet_name, target in sheet_targets:
            rows_by_sheet[sheet_name] = read_sheet_rows(archive, target, shared_strings)

    tables: dict[str, list[dict[str, object]]] = {}
    for sheet_name, rows in rows_by_sheet.items():
        if not rows:
            tables[sheet_name] = []
            continue
        header = [normalize_header(value) for value in rows[0]]
        table_rows: list[dict[str, object]] = []
        for row in rows[1:]:
            if not any(value not in (None, "") for value in row):
                continue
            record: dict[str, object] = {}
            for idx, key in enumerate(header):
                if not key:
                    continue
                record[key] = row[idx] if idx < len(row) else None
            table_rows.append(record)
        tables[sheet_name] = table_rows
    return tables


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in root.findall("a:si", SPREADSHEET_NS):
        out.append("".join(si.itertext()))
    return out


def read_sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", REL_NS)}
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.find("a:sheets", SPREADSHEET_NS).findall("a:sheet", SPREADSHEET_NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib[REL_ATTR]
        target = rel_map[rel_id]
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((name, target))
    return sheets


def read_sheet_rows(archive: zipfile.ZipFile, target: str, shared_strings: list[str]) -> list[list[object]]:
    root = ET.fromstring(archive.read(target))
    rows: list[list[object]] = []
    for row in root.findall(".//a:sheetData/a:row", SPREADSHEET_NS):
        values: list[object] = []
        for cell in row.findall("a:c", SPREADSHEET_NS):
            idx = cell_col_index(cell.attrib.get("r", ""))
            while len(values) <= idx:
                values.append(None)
            values[idx] = read_cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def cell_col_index(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference or "")
    letters = match.group(1) if match else ""
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - ord("A") + 1
    return max(value - 1, 0)


def read_cell_value(cell: ET.Element, shared_strings: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", SPREADSHEET_NS)
    if cell_type == "inlineStr":
        inline = cell.find("a:is", SPREADSHEET_NS)
        return "".join(inline.itertext()) if inline is not None else ""
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)] if raw else ""
    return parse_number(raw)


def parse_number(value: str) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def resolve_sheet_names(
    workbook: dict[str, list[dict[str, object]]],
    sheets: Iterable[str] | None,
    indices: Iterable[int],
) -> list[str]:
    names = list(workbook.keys())
    if sheets:
        lookup = {name.lower(): name for name in names}
        selected = []
        for item in sheets:
            key = str(item).lower()
            if key not in lookup:
                raise KeyError(f"Sheet not found: {item}. Available sheets: {', '.join(names)}")
            selected.append(lookup[key])
        return selected
    selected = []
    for idx in indices:
        if idx < 1 or idx > len(names):
            raise IndexError(f"Sheet index out of range: {idx}. Workbook has {len(names)} sheets.")
        selected.append(names[idx - 1])
    return selected


def parse_points(
    dataset: str,
    rows: list[dict[str, object]],
    metric_col: str,
    efficiency_col: str,
    params_col: str,
) -> list[Point]:
    metric_key = normalize_header(metric_col)
    efficiency_key = normalize_header(efficiency_col)
    params_key = normalize_header(params_col)
    model_key = "model"
    points: list[Point] = []
    for row in rows:
        model = str(row.get(model_key) or row.get("train_run") or "").strip()
        if not model:
            continue
        metric = to_float(row.get(metric_key))
        cost = to_float(row.get(efficiency_key))
        params = to_float(row.get(params_key))
        if metric is None or cost is None or params is None:
            continue
        ours_size = ours_size_from_name(model)
        is_ours = bool(OURS_PATTERN.search(model)) or ours_size is not None
        points.append(
            Point(
                dataset=dataset,
                model=clean_model_name(model),
                params=params,
                gflops=cost,
                metric=metric,
                is_ours=is_ours,
                ours_size=ours_size,
            )
        )
    return points


def normalize_header(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "_", str(value).strip().lower())


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ours_size_from_name(model: str) -> str | None:
    match = OURS_SIZE_PATTERN.search(model)
    return match.group(1).lower() if match else None


def clean_model_name(model: str) -> str:
    text = re.sub(r"\s+", " ", model).strip()
    if OURS_PATTERN.search(text):
        size = ours_size_from_name(text)
        return f"Ours ({size})" if size else "Ours"
    return text


def write_plot_data(points_by_dataset: dict[str, list[Point]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "model", "params_m", "gflops", "accuracy", "is_ours", "ours_size", "pareto"],
        )
        writer.writeheader()
        for dataset, points in points_by_dataset.items():
            frontier = set(pareto_frontier(points))
            for point in points:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "model": point.model,
                        "params_m": f"{point.params:.6g}",
                        "gflops": f"{point.gflops:.6g}",
                        "accuracy": f"{point.metric:.8g}",
                        "is_ours": int(point.is_ours),
                        "ours_size": point.ours_size or "",
                        "pareto": int(point in frontier),
                    }
                )


def mean_points(points_by_dataset: dict[str, list[Point]], dataset_name: str = "Average") -> dict[str, list[Point]]:
    required_count = len(points_by_dataset)
    grouped: dict[str, list[Point]] = {}
    for points in points_by_dataset.values():
        best_by_model: dict[str, Point] = {}
        for point in points:
            previous = best_by_model.get(point.model)
            if previous is None or point.metric > previous.metric:
                best_by_model[point.model] = point
        for model, point in best_by_model.items():
            grouped.setdefault(model, []).append(point)

    averaged: list[Point] = []
    for model, points in grouped.items():
        if len(points) != required_count:
            continue
        averaged.append(
            Point(
                dataset=dataset_name,
                model=model,
                params=sum(point.params for point in points) / len(points),
                gflops=sum(point.gflops for point in points) / len(points),
                metric=sum(point.metric for point in points) / len(points),
                is_ours=any(point.is_ours for point in points),
                ours_size=next((point.ours_size for point in points if point.ours_size), None),
            )
        )
    averaged.sort(key=lambda point: (not point.is_ours, OURS_SIZE_ORDER.get(point.ours_size or "", 99), point.gflops, point.model))
    return {dataset_name: averaged}


def pareto_frontier(points: list[Point]) -> list[Point]:
    ordered = sorted(points, key=lambda item: (item.gflops, -item.metric))
    frontier: list[Point] = []
    best = -float("inf")
    for point in ordered:
        if point.metric > best + 1e-12:
            frontier.append(point)
            best = point.metric
    return frontier


def plot_pareto(
    points_by_dataset: dict[str, list[Point]],
    output_dir: Path,
    stem: str,
    metric_col: str,
    efficiency_col: str,
    formats: list[str],
    dpi: int,
    log_x: bool,
    x_break: tuple[float, float, float] | None,
    x_max: float | None,
    annotate: bool,
    draw_frontier: bool,
    title: str,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to draw the Pareto figure.") from exc

    configure_matplotlib(mpl)
    datasets = list(points_by_dataset.keys())
    fig_cfg = STYLE["figure"]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(fig_cfg["width_per_dataset"] * len(datasets), fig_cfg["height"]),
        constrained_layout=False,
    )
    if len(datasets) == 1:
        axes = [axes]
    fig.subplots_adjust(
        left=fig_cfg["left"],
        right=fig_cfg["right"],
        top=fig_cfg["top"],
        bottom=fig_cfg["bottom"],
        wspace=fig_cfg["wspace"],
    )

    frontier_color = "#B8BEC8"

    all_params = [point.params for points in points_by_dataset.values() for point in points]
    min_params = min(all_params) if all_params else 1.0
    max_params = max(all_params) if all_params else 1.0
    all_gflops = [point.gflops for points in points_by_dataset.values() for point in points if point.gflops > 0]
    shared_x_limits = global_x_limits(all_gflops, log_x=log_x, x_break=x_break, x_max=x_max)

    for ax, dataset in zip(axes, datasets):
        points = points_by_dataset[dataset]
        baselines = [point for point in points if not point.is_ours]
        ours = sorted(
            [point for point in points if point.is_ours],
            key=lambda item: OURS_SIZE_ORDER.get(item.ours_size or "", 99),
        )

        if draw_frontier:
            frontier = pareto_frontier(points)
            if frontier:
                ax.plot(
                    [plot_x(point.gflops, x_break) for point in frontier],
                    [to_percent(point.metric) for point in frontier],
                    color=frontier_color,
                    linewidth=STYLE["line"]["frontier_width"],
                    linestyle="--",
                    zorder=1,
                )

        if ours:
            ax.plot(
                [plot_x(point.gflops, x_break) for point in ours],
                [to_percent(point.metric) for point in ours],
                color=OURS_COLOR,
                linewidth=STYLE["line"]["ours_width"],
                alpha=0.95,
                zorder=3,
            )

        drawable_points = sorted(
            points,
            key=lambda point: point_marker_area(point, min_params, max_params),
            reverse=True,
        )
        for order, point in enumerate(drawable_points):
            ax.scatter(
                plot_x(point.gflops, x_break),
                to_percent(point.metric),
                s=point_marker_area(point, min_params, max_params),
                marker="o",
                color=point_color(point),
                edgecolor="white",
                linewidth=STYLE["marker"]["edge_width_ours"] if point.is_ours else STYLE["marker"]["edge_width"],
                alpha=STYLE["marker"]["alpha_ours"] if point.is_ours else STYLE["marker"]["alpha"],
                zorder=4 + order * 0.05,
            )

        if annotate:
            annotate_points(ax, dataset, baselines, ours, x_break=x_break)

        ax.set_title(dataset, fontweight="bold", fontsize=STYLE["font"]["title"], pad=14)
        ax.set_xlabel(x_axis_label(efficiency_col), fontsize=STYLE["font"]["axis_label"], labelpad=8)
        ax.set_ylabel(y_axis_label(metric_col) if ax is axes[0] else "", fontsize=STYLE["font"]["axis_label"], labelpad=8)
        if log_x:
            ax.set_xscale("log")
        set_axis_limits(ax, points, log_x=log_x, shared_x_limits=shared_x_limits)
        if x_break is not None:
            real_upper = x_max or nice_upper_limit((max(all_gflops) if all_gflops else 180.0) * 1.08)
            apply_broken_x_axis(ax, x_break=x_break, real_max=real_upper)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        elif not log_x:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax.minorticks_off()
        ax.grid(True, which="major", axis="both", color="#E7EAF0", linewidth=STYLE["line"]["grid_width"])
        ax.grid(False, which="minor")
        ax.tick_params(axis="both", labelsize=STYLE["font"]["tick"], width=STYLE["line"]["spine_width"])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#98A2B3")
        ax.spines["bottom"].set_color("#98A2B3")
        ax.spines["left"].set_linewidth(STYLE["line"]["spine_width"])
        ax.spines["bottom"].set_linewidth(STYLE["line"]["spine_width"])

    method_names = ordered_method_names(points_by_dataset)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=OURS_COLOR if name.startswith("Ours") else "none",
            markerfacecolor=METHOD_COLORS.get(name, "#667085"),
            markeredgecolor="white",
            markeredgewidth=STYLE["marker"]["edge_width"],
            linewidth=STYLE["line"]["ours_width"] if name.startswith("Ours") else 0.0,
            markersize=STYLE["marker"]["legend_size"],
            label="Ours (s/m/l)" if name.startswith("Ours") else name,
        )
        for name in method_names
    ]
    if draw_frontier:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color=frontier_color,
                linestyle="--",
                linewidth=STYLE["line"]["frontier_width"],
                label="Pareto frontier",
            )
        )
    size_handles = size_legend_handles(min_params=min_params, max_params=max_params)
    param_handles = [Line2D([0], [0], color="none", label="Params (M):"), *size_handles] if size_handles else []
    legend_rows = max(1, int(STYLE["legend"]["rows"]))
    legend_cfg = STYLE["legend"]
    legend_common = {
        "loc": "lower center",
        "frameon": False,
        "prop": {"size": STYLE["font"]["legend"], "weight": "bold"},
        "handletextpad": legend_cfg["handletextpad"],
        "columnspacing": legend_cfg["columnspacing"],
    }
    if legend_rows >= 2 and param_handles:
        top_legend = fig.legend(
            handles=method_handles,
            bbox_to_anchor=(0.5, legend_cfg["anchor_y"] + legend_cfg["row_gap"]),
            ncol=len(method_handles),
            **legend_common,
        )
        fig.add_artist(top_legend)
        fig.legend(
            handles=param_handles,
            bbox_to_anchor=(0.5, legend_cfg["anchor_y"]),
            ncol=len(param_handles),
            **legend_common,
        )
    else:
        legend_handles = [*method_handles, *param_handles]
        fig.legend(
            handles=legend_handles,
            bbox_to_anchor=(0.5, legend_cfg["anchor_y"]),
            ncol=max(1, math.ceil(len(legend_handles) / legend_rows)),
            **legend_common,
        )
    if title:
        fig.suptitle(title, fontsize=STYLE["font"]["suptitle"], fontweight="bold", y=1.12)

    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def plot_mean_pareto(
    points_by_dataset: dict[str, list[Point]],
    output_dir: Path,
    stem: str,
    metric_col: str,
    efficiency_col: str,
    formats: list[str],
    dpi: int,
    log_x: bool,
    x_break: tuple[float, float, float] | None,
    x_max: float | None,
    draw_frontier: bool,
    title: str,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to draw the Pareto figure.") from exc

    configure_matplotlib(mpl)
    mean_cfg = STYLE["mean"]
    dataset, points = next(iter(points_by_dataset.items()))
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(mean_cfg["figsize"], mean_cfg["figsize"]),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=mean_cfg["left"],
        right=mean_cfg["right"],
        top=mean_cfg["top"],
        bottom=mean_cfg["bottom"],
    )

    frontier_color = "#B8BEC8"
    all_params = [point.params for point in points]
    min_params = min(all_params) if all_params else 1.0
    max_params = max(all_params) if all_params else 1.0
    all_gflops = [point.gflops for point in points if point.gflops > 0]
    shared_x_limits = global_x_limits(all_gflops, log_x=log_x, x_break=x_break, x_max=x_max)
    ours = sorted(
        [point for point in points if point.is_ours],
        key=lambda item: OURS_SIZE_ORDER.get(item.ours_size or "", 99),
    )

    if draw_frontier:
        frontier = pareto_frontier(points)
        if frontier:
            ax.plot(
                [plot_x(point.gflops, x_break) for point in frontier],
                [to_percent(point.metric) for point in frontier],
                color=frontier_color,
                linewidth=STYLE["line"]["frontier_width"],
                linestyle="--",
                zorder=1,
            )

    if ours:
        ax.plot(
            [plot_x(point.gflops, x_break) for point in ours],
            [to_percent(point.metric) for point in ours],
            color=OURS_COLOR,
            linewidth=STYLE["line"]["ours_width"],
            alpha=0.95,
            zorder=3,
        )

    drawable_points = sorted(
        points,
        key=lambda point: point_marker_area(point, min_params, max_params),
        reverse=True,
    )
    for order, point in enumerate(drawable_points):
        ax.scatter(
            plot_x(point.gflops, x_break),
            to_percent(point.metric),
            s=point_marker_area(point, min_params, max_params) * mean_cfg["marker_scale"],
            marker="o",
            color=point_color(point),
            edgecolor="white",
            linewidth=STYLE["marker"]["edge_width_ours"] if point.is_ours else STYLE["marker"]["edge_width"],
            alpha=STYLE["marker"]["alpha_ours"] if point.is_ours else STYLE["marker"]["alpha"],
            zorder=4 + order * 0.05,
        )

    annotate_mean_points(ax, points, x_break=x_break)

    ax.set_xlabel(x_axis_label(efficiency_col), fontsize=mean_cfg["axis_label"], labelpad=10)
    ax.set_ylabel(y_axis_label(metric_col), fontsize=mean_cfg["axis_label"], labelpad=10)
    if log_x:
        ax.set_xscale("log")
    set_axis_limits(ax, points, log_x=log_x, shared_x_limits=shared_x_limits)
    if x_break is not None:
        real_upper = x_max or nice_upper_limit((max(all_gflops) if all_gflops else 180.0) * 1.08)
        apply_broken_x_axis(ax, x_break=x_break, real_max=real_upper)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    elif not log_x:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        ax.minorticks_off()
    ax.set_box_aspect(1)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color="#E5EAF1", linewidth=STYLE["line"]["grid_width"])
    ax.grid(False, which="minor")
    ax.tick_params(axis="both", labelsize=mean_cfg["tick"], width=STYLE["line"]["spine_width"])
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#98A2B3")
    ax.spines["bottom"].set_color("#98A2B3")
    ax.spines["left"].set_linewidth(STYLE["line"]["spine_width"])
    ax.spines["bottom"].set_linewidth(STYLE["line"]["spine_width"])

    size_handles = size_legend_handles(
        min_params=min_params,
        max_params=max_params,
        scale=mean_cfg["size_legend_scale"],
    )
    if size_handles:
        legend = ax.legend(
            handles=size_handles,
            title="Params (M)",
            loc="upper right",
            bbox_to_anchor=mean_cfg["size_legend_anchor"],
            frameon=True,
            facecolor="white",
            edgecolor="#D0D5DD",
            framealpha=0.88,
            prop={"size": mean_cfg["size_legend"], "weight": "bold"},
            borderpad=0.8,
            labelspacing=1.0,
            handletextpad=1.0,
        )
        legend.get_title().set_fontsize(mean_cfg["size_legend_title"])
        legend.get_title().set_fontweight("bold")
        legend.get_frame().set_linewidth(1.1)

    if title:
        fig.suptitle(title, fontsize=STYLE["font"]["suptitle"], fontweight="bold", y=1.02)

    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def configure_matplotlib(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def to_percent(value: float) -> float:
    return value * 100.0 if abs(value) <= 1.5 else value


def marker_area(params: float, min_params: float, max_params: float) -> float:
    area_min = STYLE["marker"]["area_min"]
    area_max = STYLE["marker"]["area_max"]
    if max_params <= min_params:
        return area_min
    scaled = (params - min_params) / max(max_params - min_params, 1e-9)
    scaled = max(0.0, min(1.0, scaled)) ** 0.82
    return area_min + (area_max - area_min) * scaled


def point_marker_area(point: Point, min_params: float, max_params: float) -> float:
    area = marker_area(point.params, min_params, max_params)
    return area * STYLE["marker"]["ours_scale"] if point.is_ours else area


def point_color(point: Point) -> str:
    if point.is_ours:
        return OURS_COLOR
    return METHOD_COLORS.get(point.model, "#667085")


def ordered_method_names(points_by_dataset: dict[str, list[Point]]) -> list[str]:
    preferred = ["DMINet", "HAFF", "HFA-PANet", "HRSICD", "ICIF-Net", "WaveHFG"]
    names = {point.model for points in points_by_dataset.values() for point in points if not point.is_ours}
    ordered = [name for name in preferred if name in names]
    ordered.extend(sorted(names - set(ordered)))
    if any(point.is_ours for points in points_by_dataset.values() for point in points):
        ordered.append("Ours")
    return ordered


def size_legend_handles(min_params: float, max_params: float, scale: float | None = None):
    try:
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        return []
    candidates = [2.0, 10.0, 50.0]
    values = [value for value in candidates if min_params * 0.8 <= value <= max_params * 1.2]
    if not values:
        values = [min_params, (min_params + max_params) / 2.0, max_params]
    deduped: list[float] = []
    marker_scale = STYLE["marker"]["size_legend_scale"] if scale is None else scale
    for value in values:
        rounded = round(float(value), 1)
        if rounded not in deduped:
            deduped.append(rounded)
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#D0D5DD",
            markeredgecolor="#667085",
            markeredgewidth=STYLE["marker"]["edge_width"] * 0.55,
            markersize=math.sqrt(marker_area(value, min_params, max_params)) * marker_scale,
            label=f"{value:g}",
        )
        for value in deduped
    ]


def global_x_limits(
    xs: list[float],
    log_x: bool,
    x_break: tuple[float, float, float] | None,
    x_max: float | None,
) -> tuple[float, float] | None:
    if not xs:
        return None
    xmin, xmax = min(xs), max(xs)
    if log_x:
        return xmin / 1.35, xmax * 1.35
    upper = x_max if x_max is not None else nice_upper_limit(xmax * 1.08)
    upper = max(upper, xmax)
    return 0.0, plot_x(upper, x_break)


def nice_upper_limit(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10**exponent
    fraction = value / base
    if fraction <= 1.5:
        nice = 1.5
    elif fraction <= 2:
        nice = 2
    elif fraction <= 3:
        nice = 3
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return nice * base


def plot_x(value: float, x_break: tuple[float, float, float] | None) -> float:
    if x_break is None:
        return value
    start, end, width = x_break
    if value <= start:
        return value
    if value >= end:
        return start + width + (value - end)
    return start + (value - start) / max(end - start, 1e-9) * width


def apply_broken_x_axis(ax, x_break: tuple[float, float, float], real_max: float) -> None:
    start, end, width = x_break
    break_left = plot_x(start, x_break)
    break_right = plot_x(end, x_break)
    ax.axvspan(break_left, break_right, color=STYLE["break"]["shade_color"], zorder=0)
    ax.text(
        (break_left + break_right) / 2.0,
        0.015,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=STYLE["font"]["break_label"],
        fontweight="bold",
        color=STYLE["break"]["label_color"],
    )
    tick_values = [0, 20, 40, start, end, real_max]
    deduped = []
    for value in tick_values:
        if value <= max(real_max, end) and all(abs(value - old) > 1e-6 for old in deduped):
            deduped.append(value)
    tick_values = deduped
    ax.set_xticks([plot_x(value, x_break) for value in tick_values])
    ax.set_xticklabels([f"{int(value)}" if float(value).is_integer() else f"{value:g}" for value in tick_values])
    ax.minorticks_off()


def annotate_points(ax, dataset: str, baselines: list[Point], ours: list[Point], x_break: tuple[float, float, float] | None) -> None:
    offsets = label_offsets(dataset)
    for point in baselines + ours:
        dx, dy = offsets.get(point.model, default_offset(point))
        far_label = abs(dx) + abs(dy) >= 34
        arrowprops = None
        if far_label:
            color = point_color(point)
            arrowprops = {
                "arrowstyle": "-",
                "color": color,
                "alpha": 0.7,
                "linewidth": STYLE["line"]["leader_width"],
                "shrinkA": 3,
                "shrinkB": 5,
                "connectionstyle": "arc3,rad=0.12",
            }
        ax.annotate(
            point.model,
            xy=(plot_x(point.gflops, x_break), to_percent(point.metric)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=STYLE["font"]["annotation"] if not point.is_ours else STYLE["font"]["annotation_ours"],
            fontweight="bold" if point.is_ours else "normal",
            color="#1D2939" if point.is_ours else "#475467",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops=arrowprops,
            annotation_clip=False,
            zorder=8,
        )


def annotate_mean_points(ax, points: list[Point], x_break: tuple[float, float, float] | None) -> None:
    offsets = mean_label_offsets()
    mean_cfg = STYLE["mean"]
    for point in points:
        dx, dy = offsets.get(point.model, default_offset(point))
        ax.annotate(
            point.model,
            xy=(plot_x(point.gflops, x_break), to_percent(point.metric)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=mean_cfg["label_ours"] if point.is_ours else mean_cfg["label"],
            fontweight="bold",
            color=point_color(point),
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.74,
            },
            annotation_clip=False,
            zorder=9,
        )


def mean_label_offsets() -> dict[str, tuple[int, int]]:
    return {
        "DMINet": (-20, -14),
        "HAFF": (10, -12),
        "HFA-PANet": (10, -24),
        "HRSICD": (-22, 12),
        "ICIF-Net": (10, -4),
        "WaveHFG": (10, -14),
        "Ours (s)": (8, -20),
        "Ours (m)": (8, 20),
        "Ours (l)": (9, 10),
    }


def label_offsets(dataset: str) -> dict[str, tuple[int, int]]:
    common = {
        "DMINet": (5, 7),
        "HAFF": (5, -12),
        "HFA-PANet": (5, 8),
        "HRSICD": (-6, 7),
        "ICIF-Net": (5, -12),
        "WaveHFG": (5, 8),
        "Ours (s)": (5, 8),
        "Ours (m)": (5, 8),
        "Ours (l)": (5, 8),
    }
    dataset_key = dataset.lower()
    if dataset_key == "bright":
        common.update(
            {
                "DMINet": (-36, 19),
                "HFA-PANet": (-44, -29),
                "ICIF-Net": (16, -18),
                "HAFF": (24, -30),
                "Ours (s)": (12, 19),
                "Ours (m)": (14, 18),
                "Ours (l)": (-13, 18),
                "WaveHFG": (14, 16),
            }
        )
    elif dataset_key == "cau":
        common.update(
            {
                "DMINet": (-38, 16),
                "HFA-PANet": (18, -34),
                "WaveHFG": (18, 18),
                "ICIF-Net": (26, -26),
                "HAFF": (26, -30),
                "Ours (s)": (15, 22),
                "Ours (m)": (-34, -28),
                "Ours (l)": (12, 15),
            }
        )
    elif dataset_key == "haiti":
        common.update(
            {
                "DMINet": (-38, -18),
                "HFA-PANet": (18, 28),
                "ICIF-Net": (28, -10),
                "WaveHFG": (18, 18),
                "HAFF": (16, -24),
                "Ours (s)": (10, -32),
                "Ours (m)": (-42, 16),
                "Ours (l)": (12, 16),
            }
        )
    return common


def default_offset(point: Point) -> tuple[int, int]:
    return (10, 12) if point.is_ours else (10, 10)


def set_axis_limits(ax, points: list[Point], log_x: bool, shared_x_limits: tuple[float, float] | None) -> None:
    xs = [point.gflops for point in points if point.gflops > 0]
    ys = [to_percent(point.metric) for point in points]
    if shared_x_limits is not None:
        ax.set_xlim(*shared_x_limits)
    elif xs:
        xmin, xmax = min(xs), max(xs)
        if log_x:
            ax.set_xlim(xmin / 1.35, xmax * 1.35)
        else:
            pad = (xmax - xmin) * 0.12 or 1.0
            ax.set_xlim(max(0.0, xmin - pad), xmax + pad)
    if ys:
        ymin, ymax = min(ys), max(ys)
        pad = max((ymax - ymin) * 0.24, 1.6)
        ax.set_ylim(ymin - pad, ymax + pad)


def x_axis_label(column: str) -> str:
    key = normalize_header(column)
    if key == "gflops":
        return "GFLOPS"
    if key == "params":
        return "Params (M)"
    return str(column)


def y_axis_label(column: str) -> str:
    key = normalize_header(column)
    names = {
        "foreground_miou": "FmIoU (%)",
        "fmiou": "FmIoU (%)",
        "miou": "mIoU (%)",
        "iou": "IoU (%)",
        "f1": "F1-score (%)",
        "oa": "Overall accuracy (%)",
    }
    return names.get(key, f"{column} (%)")


if __name__ == "__main__":
    main()
