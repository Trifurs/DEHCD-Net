from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import XMLConfigParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore heterogeneous disaster CD dataset.")
    parser.add_argument("--config", default="configs/config.xml", help="XML config path.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--max-samples", type=int, default=8, help="Samples per split for metadata probing.")
    parser.add_argument("--output-dir", default="outputs/exploration", help="Report output directory.")
    return parser.parse_args()


def choose_data_root(config: Dict[str, Any], override: Optional[str]) -> Tuple[Path, List[str]]:
    tried: List[str] = []
    if override:
        candidates = [override]
    else:
        dataset_cfg = config.get("dataset", {})
        candidates = []
        root = dataset_cfg.get("root")
        if root:
            candidates.append(root)
        candidates.extend(dataset_cfg.get("candidate_roots") or [])

    seen = set()
    unique_candidates = []
    for item in candidates:
        if item and item not in seen:
            unique_candidates.append(item)
            seen.add(item)

    for item in unique_candidates:
        tried.append(str(item))
        path = Path(str(item)).expanduser()
        try:
            if path.exists() and path.is_dir():
                return path, tried
        except PermissionError:
            return path, tried
    return Path(str(unique_candidates[0] if unique_candidates else ".")).expanduser(), tried


def safe_listdir(path: Path) -> Tuple[List[Path], Optional[str]]:
    try:
        return sorted(path.iterdir(), key=lambda p: p.name), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def count_extensions(root: Path, allowed: Iterable[str]) -> Tuple[Dict[str, int], Optional[str]]:
    counter: Counter[str] = Counter()
    allowed_set = {ext.lower() for ext in allowed}
    try:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                suffix = Path(name).suffix.lower()
                if not allowed_set or suffix in allowed_set:
                    counter[suffix or "<no_ext>"] += 1
    except Exception as exc:
        return dict(counter), f"{type(exc).__name__}: {exc}"
    return dict(counter), None


def strip_suffix(name: str, suffix: str) -> str:
    path = Path(name)
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    suffix_stem = Path(suffix).stem if suffix else ""
    if suffix_stem and path.stem.endswith(suffix_stem):
        return path.stem[: -len(suffix_stem)]
    return path.stem


def infer_split_dirs(root: Path, configured_splits: List[str]) -> List[str]:
    children, error = safe_listdir(root)
    if error:
        return configured_splits
    child_dirs = [child.name for child in children if child.is_dir()]
    hits = [split for split in configured_splits if split in child_dirs]
    return hits or child_dirs


def find_files(directory: Path, extensions: Iterable[str]) -> List[Path]:
    ext_set = {ext.lower() for ext in extensions}
    files, error = safe_listdir(directory)
    if error:
        return []
    return [path for path in files if path.is_file() and path.suffix.lower() in ext_set]


def collect_samples(root: Path, dataset_cfg: Dict[str, Any], splits: List[str]) -> Dict[str, Any]:
    extensions = dataset_cfg.get("image_extensions") or [".tif", ".tiff", ".png"]
    pre_dir = dataset_cfg.get("optical_dir", dataset_cfg.get("pre_dir", "pre-event"))
    sar_dirs = list(dataset_cfg.get("sar_dirs") or [dataset_cfg.get("post_dir", "post-event")])
    label_dir = dataset_cfg.get("label_dir", "target")
    pre_suffix = dataset_cfg.get("optical_suffix", dataset_cfg.get("pre_suffix", "_pre_disaster.tif"))
    sar_suffixes = list(dataset_cfg.get("sar_suffixes") or [dataset_cfg.get("post_suffix", "_post_disaster.tif")] * len(sar_dirs))
    label_suffix = dataset_cfg.get("label_suffix", "_building_damage.tif")

    result: Dict[str, Any] = {}
    for split in splits:
        split_root = root / split
        split_children, split_error = safe_listdir(split_root)
        subdirs = [p.name for p in split_children if p.is_dir()]
        item = {
            "path": str(split_root),
            "read_error": split_error,
            "subdirs": subdirs,
            "counts": {},
            "samples": [],
            "missing_pairs": [],
        }
        if split_error:
            result[split] = item
            continue

        pre_files = find_files(split_root / pre_dir, extensions)
        sar_files = [find_files(split_root / sar_dir, extensions) for sar_dir in sar_dirs]
        label_files = find_files(split_root / label_dir, extensions)
        item["counts"] = {pre_dir: len(pre_files), label_dir: len(label_files)}
        item["counts"].update({sar_dir: len(files) for sar_dir, files in zip(sar_dirs, sar_files)})

        pre_map = {strip_suffix(path.name, pre_suffix): path for path in pre_files}
        sar_maps = [
            {strip_suffix(path.name, sar_suffixes[idx] if idx < len(sar_suffixes) else ""): path for path in files}
            for idx, files in enumerate(sar_files)
        ]
        label_map = {strip_suffix(path.name, label_suffix): path for path in label_files}

        ids = set(pre_map) | set(label_map)
        for mapping in sar_maps:
            ids |= set(mapping)
        ids = sorted(ids)
        for sample_id in ids:
            record = {
                "id": sample_id,
                "pre": str(pre_map[sample_id]) if sample_id in pre_map else None,
                "sar": [str(mapping[sample_id]) if sample_id in mapping else None for mapping in sar_maps],
                "label": str(label_map[sample_id]) if sample_id in label_map else None,
            }
            if not record["pre"] or not record["label"] or not all(record["sar"]):
                item["missing_pairs"].append(record)
            item["samples"].append(record)
        result[split] = item
    return result


def read_metadata(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[str]]:
    raster_error = None
    try:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read()
            meta = {
                "path": path,
                "driver": src.driver,
                "bands": src.count,
                "height": src.height,
                "width": src.width,
                "dtype": str(array.dtype),
                "crs": str(src.crs) if src.crs else None,
                "resolution": tuple(float(v) for v in src.res) if src.res else None,
                "nodata": src.nodata,
            }
            return meta, array, None
    except ImportError as exc:
        raster_error = f"Missing dependency rasterio: {exc}"
    except Exception as exc:
        raster_error = f"rasterio error: {exc}"

    try:
        import numpy as np
        from PIL import Image

        image = Image.open(path)
        array = np.asarray(image)
        bands = 1 if array.ndim == 2 else int(array.shape[2])
        meta = {
            "path": path,
            "driver": "PIL",
            "bands": bands,
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "dtype": str(array.dtype),
            "crs": None,
            "resolution": None,
            "nodata": None,
        }
        return meta, array, None
    except Exception as pil_exc:
        return None, None, f"{raster_error}; PIL fallback error: {pil_exc}"


def label_value_summary(array: Any, max_values: int = 32) -> Dict[str, Any]:
    import numpy as np

    if array.ndim == 3:
        array = array[0]
    values, counts = np.unique(array, return_counts=True)
    pairs = [
        {"value": _python_scalar(v), "count": int(c)}
        for v, c in zip(values[:max_values], counts[:max_values])
    ]
    return {
        "unique_count": int(values.size),
        "first_values": pairs,
        "is_binary_like": bool(values.size <= 2 or set(values.tolist()).issubset({0, 1, 255})),
    }


def _python_scalar(value: Any) -> Any:
    try:
        return value.item()
    except AttributeError:
        return value


def probe_metadata(samples: Dict[str, Any], max_samples: int) -> Dict[str, Any]:
    probed: Dict[str, Any] = {}
    dependency_errors: Counter[str] = Counter()
    for split, split_info in samples.items():
        split_records = []
        aligned = []
        label_modes = []
        for record in split_info.get("samples", [])[:max_samples]:
            sample_probe: Dict[str, Any] = {"id": record["id"], "modalities": {}}
            arrays = {}
            modality_paths = {"pre": record.get("pre"), "label": record.get("label")}
            for idx, path in enumerate(record.get("sar", [])):
                modality_paths[f"sar_{idx}"] = path
            for key, path in modality_paths.items():
                if not path:
                    continue
                meta, array, error = read_metadata(path)
                if error:
                    dependency_errors[error] += 1
                    sample_probe["modalities"][key] = {"path": path, "error": error}
                    continue
                sample_probe["modalities"][key] = meta
                arrays[key] = array
                if key == "label":
                    try:
                        label_summary = label_value_summary(array)
                        sample_probe["label_values"] = label_summary
                        label_modes.append(label_summary)
                    except Exception as exc:
                        sample_probe["label_values_error"] = f"{type(exc).__name__}: {exc}"

            pre_meta = sample_probe["modalities"].get("pre", {})
            post_meta = sample_probe["modalities"].get("sar_0", {})
            label_meta = sample_probe["modalities"].get("label", {})
            same_shape = (
                pre_meta.get("height"),
                pre_meta.get("width"),
            ) == (
                post_meta.get("height"),
                post_meta.get("width"),
            ) == (
                label_meta.get("height"),
                label_meta.get("width"),
            )
            same_res = pre_meta.get("resolution") == post_meta.get("resolution")
            sample_probe["alignment"] = {
                "same_height_width": bool(same_shape),
                "same_pre_post_resolution": bool(same_res),
            }
            aligned.append(sample_probe["alignment"])
            split_records.append(sample_probe)

        probed[split] = {
            "records": split_records,
            "all_probed_shapes_aligned": all(item["same_height_width"] for item in aligned) if aligned else None,
            "label_binary_like": all(item.get("is_binary_like", False) for item in label_modes) if label_modes else None,
        }
    probed["dependency_errors"] = dict(dependency_errors)
    return probed


def summarize(report: Dict[str, Any]) -> str:
    lines = [
        "# Heterogeneous Disaster Data Exploration",
        "",
        f"- Data root: `{report['data_root']}`",
        f"- Root status: `{report['root_status']}`",
        f"- Candidate roots tried: `{report['candidate_roots_tried']}`",
        f"- Extension counts: `{report.get('extension_counts', {})}`",
        f"- Configured optical band indices: `{report.get('configured_band_indices', {}).get('optical')}`",
        f"- Configured SAR band indices: `{report.get('configured_band_indices', {}).get('sar')}`",
        "",
        "## Structure",
    ]
    for split, info in report.get("samples", {}).items():
        lines.append(
            f"- {split}: subdirs={info.get('subdirs', [])}, counts={info.get('counts', {})}, "
            f"paired_or_partial_samples={len(info.get('samples', []))}, missing_pairs={len(info.get('missing_pairs', []))}"
        )

    lines.extend(["", "## Metadata Probe"])
    for split, info in report.get("metadata", {}).items():
        if split == "dependency_errors":
            continue
        lines.append(
            f"- {split}: shapes_aligned={info.get('all_probed_shapes_aligned')}, "
            f"label_binary_like={info.get('label_binary_like')}"
        )
        for record in info.get("records", [])[:3]:
            modalities = record.get("modalities", {})
            pre = modalities.get("pre", {})
            label = modalities.get("label", {})
            sar_parts = []
            for key in sorted(k for k in modalities if k.startswith("sar_")):
                sar = modalities.get(key, {})
                sar_parts.append(
                    f"{key}={sar.get('bands')} bands/{sar.get('dtype')}/{sar.get('height')}x{sar.get('width')}"
                )
            sar_text = ", ".join(sar_parts)
            lines.append(
                f"  - {record.get('id')}: pre={pre.get('bands')} bands/{pre.get('dtype')}/"
                f"{pre.get('height')}x{pre.get('width')}, {sar_text}, "
                f"label_values={record.get('label_values')}"
            )

    dep_errors = report.get("metadata", {}).get("dependency_errors", {})
    if dep_errors:
        lines.extend(["", "## Dependency Or Read Errors"])
        for error, count in dep_errors.items():
            lines.append(f"- {count}x {error}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = XMLConfigParser(args.config).parse()
    dataset_cfg = config.get("dataset", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root, tried = choose_data_root(config, args.data_root)
    report: Dict[str, Any] = {
        "data_root": str(root),
        "candidate_roots_tried": tried,
        "root_status": "unknown",
        "configured_band_indices": {
            "optical": dataset_cfg.get("optical_band_indices", dataset_cfg.get("optical_bands")),
            "sar": dataset_cfg.get("sar_band_indices", dataset_cfg.get("sar_bands")),
        },
    }

    try:
        children, root_error = safe_listdir(root)
        if root_error:
            report["root_status"] = f"unreadable: {root_error}"
        elif not root.exists():
            report["root_status"] = "missing"
        else:
            report["root_status"] = "readable"
            report["top_level_entries"] = [child.name for child in children]
            extensions, extension_error = count_extensions(
                root, dataset_cfg.get("image_extensions") or []
            )
            report["extension_counts"] = extensions
            if extension_error:
                report["extension_count_error"] = extension_error

            splits = infer_split_dirs(root, dataset_cfg.get("splits") or ["train", "val", "test"])
            report["splits"] = splits
            samples = collect_samples(root, dataset_cfg, splits)
            report["samples"] = samples
            report["metadata"] = probe_metadata(samples, args.max_samples)
    except Exception as exc:
        report["root_status"] = f"failed: {type(exc).__name__}: {exc}"

    dataset_name = str(dataset_cfg.get("name", root.name)).lower()
    json_path = output_dir / f"{dataset_name}_exploration.json"
    md_path = output_dir / f"{dataset_name}_exploration.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(summarize(report), encoding="utf-8")
    print(f"Wrote exploration JSON: {json_path}")
    print(f"Wrote exploration summary: {md_path}")
    print(summarize(report))


if __name__ == "__main__":
    main()
