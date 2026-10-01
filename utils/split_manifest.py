"""Explicit scene/event splits. CSV columns: id,source_split,split,group,event."""
from __future__ import annotations

import csv
from pathlib import Path


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"id", "source_split", "split", "group", "event"} <= rows[0].keys():
        raise ValueError("Manifest needs id,source_split,split,group,event columns and at least one row")
    seen, groups = set(), {}
    for row in rows:
        if row["split"] not in {"train", "val", "test"} or row["source_split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid split in manifest")
        if not all(row[k].strip() for k in ("id", "group", "event")):
            raise ValueError("Manifest ids, source groups and events must be explicit")
        key = (row["source_split"], row["id"])
        if key in seen:
            raise ValueError(f"Duplicate sample assignment: {key}")
        seen.add(key)
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError(f"Source scene/group crosses splits: {row['group']}")
        groups[row["group"]] = row["split"]
    return rows


def load_manifest_index(dataset, path):
    rows = [r for r in read_manifest(path) if r["split"] == dataset.split]
    if not rows:
        raise ValueError(f"Manifest contains no {dataset.split} samples")
    source_indices = {}
    original_root = dataset.split_root
    try:
        for split in sorted({r["source_split"] for r in rows}):
            dataset.split_root = dataset.root / split
            source_indices[split] = {r["id"]: r for r in dataset._build_index()}
    finally:
        dataset.split_root = original_root
    samples = []
    for row in rows:
        sample = source_indices[row["source_split"]].get(row["id"])
        if sample is None:
            raise ValueError(f"Manifest sample not found: {row}")
        samples.append({**sample, "group": row["group"], "event": row["event"]})
    return samples
