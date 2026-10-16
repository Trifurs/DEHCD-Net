"""Scene-disjoint transfer protocols and auditable label budgets."""
from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from utils.protocol import atomic_json, digest

EVENTS = ("turkey-earthquake", "haiti-earthquake", "morocco-earthquake", "noto-earthquake")
MANIFEST_FIELDS = ("id", "source_split", "split", "group", "event")


def scan_records(data_root, workers=8, events=None):
    """Read actual label IDs, without predictions or test metric feedback."""
    from concurrent.futures import ThreadPoolExecutor
    from utils.raster import read_raster

    root = Path(data_root).expanduser().resolve()
    with (root / "metadata/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    if events is not None:
        original = [row for row in original if row["event"] in events]

    def inspect(row):
        source_split = row["split"]
        label = root / source_split / "target" / f"{row['id']}_building_damage.tif"
        values, _ = read_raster(label)
        valid = values != 255
        if not np.isin(values[valid], [0, 1, 2, 3]).all():
            raise ValueError(f"Invalid four-class label IDs: {label}")
        counts = np.bincount(values[valid].astype(np.int64), minlength=4).tolist()
        files = [root / source_split / "pre-event" / f"{row['id']}_pre_disaster.tif",
                 root / source_split / "post-event" / f"{row['id']}_post_disaster.tif", label]
        stamps = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]
        return {"id": row["id"], "source_split": source_split, "group": row["group"],
                "event": row["event"], "class_pixels": counts, "files": stamps}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(inspect, original))
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate source sample IDs")
    return sorted(records, key=lambda row: row["id"])


def keyed_order(values, seed, tag):
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{tag}:{value}".encode()).hexdigest())


def stratified_groups(records, fractions, seed):
    """Allocate whole scenes once, reserving each available class per split.

    Existing reference labels are used for benchmark stratification only.
    No prediction, loss, or evaluation score is an input to this function.
    """
    grouped = defaultdict(list)
    for row in records:
        grouped[row["group"]].append(row)
    if len(fractions) < 2 or not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to one")
    parts = list(fractions)
    n = len(grouped)
    quotas = {part: max(1, math.ceil(n * fractions[part])) for part in parts[:-1]}
    quotas[parts[-1]] = n - sum(quotas.values())
    if min(quotas.values()) < 1:
        raise ValueError(f"Insufficient scenes for split quotas: {quotas}")
    hist = {group: np.sum([r["class_pixels"] for r in rows], axis=0) for group, rows in grouped.items()}
    total_pixels = np.sum(list(hist.values()), axis=0).astype(float)
    present = {group: set(np.flatnonzero(counts)) for group, counts in hist.items()}
    available = {cls: [group for group in grouped if cls in present[group]] for cls in range(4)}
    if any(len(groups) < len(parts) for groups in available.values()):
        raise ValueError("Too few class-positive scenes for stratified disjoint splits; increase data or change protocol explicitly")
    order = keyed_order(grouped, seed, records[0]["event"])
    assignments, coverage = {part: [] for part in parts}, {part: set() for part in parts}
    used = set()
    for cls in sorted(available, key=lambda cls: len(available[cls])):
        for part in parts:
            if cls in coverage[part]:
                continue
            candidates = [g for g in order if g not in used and cls in present[g]]
            if not candidates or len(assignments[part]) >= quotas[part]:
                raise ValueError(f"Cannot cover class {cls} in {part} at the specified scene budget")
            expected_positive_scenes = max(1, math.ceil(len(available[cls]) * fractions[part]))
            desired_pixels = total_pixels[cls] * fractions[part] / expected_positive_scenes
            chosen = min(candidates, key=lambda g: abs(math.log((hist[g][cls] + 1) / (desired_pixels + 1)))
                         - .05 * len(present[g] - coverage[part]))
            assignments[part].append(chosen)
            coverage[part].update(present[chosen])
            used.add(chosen)
    remaining = [g for g in order if g not in used]
    for part in parts:
        current = np.sum([hist[g] for g in assignments[part]], axis=0).astype(float)
        target = total_pixels * quotas[part] / n
        scales = np.maximum(total_pixels / n, 1.)
        while len(assignments[part]) < quotas[part]:
            desired = target * (len(assignments[part]) + 1) / quotas[part]
            chosen = min(remaining, key=lambda g: float(np.square((current + hist[g] - desired) / scales).sum()))
            assignments[part].append(chosen)
            current += hist[chosen]
            remaining.remove(chosen)
    if remaining:
        raise AssertionError("Unassigned scenes")
    return {part: [row for group in groups for row in sorted(grouped[group], key=lambda row: row["id"])]
            for part, groups in assignments.items()}


def validate_partition(partition, held_out_event=None, require_event_disjoint=False):
    seen_ids, seen_groups = {}, {}
    report = {}
    for split, rows in partition.items():
        if not rows:
            raise ValueError(f"Empty {split} split")
        for row in rows:
            for field, seen in (("id", seen_ids), ("group", seen_groups)):
                value = row[field]
                if value in seen and seen[value] != split:
                    raise ValueError(f"{field} leakage across {seen[value]} and {split}: {value}")
                seen[value] = split
        pixels = np.sum([row["class_pixels"] for row in rows], axis=0)
        report[split] = {"samples": len(rows), "scenes": len({r["group"] for r in rows}),
            "events": dict(Counter(row["event"] for row in rows)), "class_pixels": pixels.tolist(),
            "class_positive_samples": np.sum(np.asarray([r["class_pixels"] for r in rows]) > 0, axis=0).tolist()}
    if require_event_disjoint:
        if any(row["event"] == held_out_event for split in ("train", "val") for row in partition[split]):
            raise ValueError("Held-out event entered source training/validation")
        if {row["event"] for row in partition["test"]} != {held_out_event}:
            raise ValueError("Test must contain exactly the held-out event")
    return report


def loeo_partition(records, held_out_event, val_fraction, seed):
    partition = {"train": [], "val": [], "test": []}
    for event in EVENTS:
        event_rows = sorted([r for r in records if r["event"] == event], key=lambda row: row["id"])
        if event == held_out_event:
            partition["test"] = event_rows
        elif event == "turkey-earthquake" and {r["source_split"] for r in event_rows} == {"train", "val"}:
            # Keep the donor checkpoint's original train/validation boundary:
            # a previously trained-on scene must never become validation.
            for split in ("train", "val"):
                partition[split].extend(r for r in event_rows if r["source_split"] == split)
        else:
            selected = stratified_groups(event_rows, {"val": val_fraction, "train": 1 - val_fraction}, seed)
            partition["val"].extend(selected["val"])
            partition["train"].extend(selected["train"])
    validate_partition(partition, held_out_event, require_event_disjoint=True)
    return partition


def fewshot_partition(records, event, train_fraction, val_fraction, seed):
    if not 0 < train_fraction < 1 - val_fraction:
        raise ValueError("Few-shot train and validation fractions must leave a nonempty test set")
    selected = stratified_groups([r for r in records if r["event"] == event],
        {"train": train_fraction, "val": val_fraction, "test": 1 - train_fraction - val_fraction}, seed)
    validate_partition(selected)
    return selected


def write_partition(path, partition):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**{field: row[field] for field in MANIFEST_FIELDS if field != "split"}, "split": split}
            for split, values in partition.items() for row in sorted(values, key=lambda r: r["id"])]
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            if list(csv.DictReader(stream)) != rows:
                raise ValueError(f"An incompatible split manifest already exists: {path}; select a new output directory")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def source_class_weights(rows, cap=12.0):
    pixels = np.sum([row["class_pixels"] for row in rows], axis=0)
    if (pixels[1:] == 0).any():
        raise ValueError("Training split is missing a damage class")
    subclass = np.sqrt(pixels[1] / pixels[1:].astype(float)).clip(1.0, cap)
    return [1.0, *subclass.tolist()]


def seal_json(path, value):
    path = Path(path)
    if path.exists():
        import json
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError(f"Existing study contract differs: {path}; use a new output directory")
    else:
        atomic_json(path, value)
