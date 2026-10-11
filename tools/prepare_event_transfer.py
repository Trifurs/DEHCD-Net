#!/usr/bin/env python3
"""Create an event-disjoint earthquake transfer dataset from paired imagery."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
SUFFIXES = {
    "pre": "_pre_disaster.tif",
    "post": "_post_disaster.tif",
    "label": "_building_damage.tif",
}
EVENT_PATTERN = re.compile(r"^(?P<event>.+?)_(?P<scene>\d{8})(?:_.*)?$")
GROUP_PATTERN = re.compile(r"^(?P<group>.+?_\d{8})_r\d+_c\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-event", default="turkey-earthquake")
    parser.add_argument(
        "--test-events",
        nargs="+",
        default=["haiti-earthquake", "morocco-earthquake", "noto-earthquake"],
    )
    parser.add_argument("--replace", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def event_and_group(sample_id: str) -> tuple[str, str]:
    event_match = EVENT_PATTERN.match(sample_id)
    group_match = GROUP_PATTERN.match(sample_id)
    event = event_match.group("event") if event_match else sample_id
    group = group_match.group("group") if group_match else sample_id
    return event, group


def index_split(root: Path, split: str) -> dict[str, dict[str, Path]]:
    folders = {
        "pre": root / split / "pre-event",
        "post": root / split / "post-event",
        "label": root / split / "target",
    }
    for folder in folders.values():
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing required directory: {folder}")
    maps = {}
    for kind, folder in folders.items():
        suffix = SUFFIXES[kind]
        maps[kind] = {
            path.name[: -len(suffix)]: path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.name.endswith(suffix)
        }
    ids = set(maps["pre"]) & set(maps["post"]) & set(maps["label"])
    all_ids = set().union(*(set(mapping) for mapping in maps.values()))
    missing = all_ids - ids
    if missing:
        raise ValueError(f"Unpaired samples in {split}: {sorted(missing)[:5]} (total={len(missing)})")
    return {sample_id: {kind: maps[kind][sample_id] for kind in maps} for sample_id in sorted(ids)}


def destination_split(event: str, source_split: str, train_event: str, test_events: set[str]) -> str | None:
    if event in test_events:
        return "test"
    if event == train_event:
        if source_split in {"train", "val"}:
            return source_split
        raise ValueError(f"The training event appears in source {source_split}; no safe destination is defined")
    return None


def copy_sample(sample: dict[str, Path], output: Path, split: str) -> None:
    destinations = {
        "pre": output / split / "pre-event" / f"{sample['id']}{SUFFIXES['pre']}",
        "post": output / split / "post-event" / f"{sample['id']}{SUFFIXES['post']}",
        "label": output / split / "target" / f"{sample['id']}{SUFFIXES['label']}",
    }
    for kind, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample[kind], destination)


def main() -> int:
    args = parse_args()
    source = args.source_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    train_event = str(args.train_event)
    test_events = set(args.test_events)
    if not source.is_dir():
        raise FileNotFoundError(f"Source root not found: {source}")
    if train_event in test_events or not test_events:
        raise ValueError("The train event must be distinct from a nonempty test-event set")
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"Output exists; use --replace explicitly: {output}")
        shutil.rmtree(output)

    rows = []
    counts = Counter()
    event_counts = defaultdict(Counter)
    processed = 0
    for source_split in SPLITS:
        for sample_id, sample in index_split(source, source_split).items():
            event, group = event_and_group(sample_id)
            destination = destination_split(event, source_split, train_event, test_events)
            if destination is None:
                continue
            copy_sample({**sample, "id": sample_id}, output, destination)
            rows.append({
                "id": sample_id,
                "source_split": destination,
                "split": destination,
                "group": group,
                "event": event,
                "origin_split": source_split,
            })
            counts[destination] += 1
            event_counts[destination][event] += 1
            processed += 1
            if processed % 1000 == 0:
                print(f"Copied {processed} samples", flush=True)

    if not rows:
        raise RuntimeError("No samples matched the requested events")
    groups = defaultdict(set)
    for row in rows:
        groups[row["group"]].add(row["split"])
    crossed = {group: sorted(splits) for group, splits in groups.items() if len(splits) != 1}
    if crossed:
        raise ValueError(f"Source groups cross destination splits: {list(crossed.items())[:5]}")

    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "source_split", "split", "group", "event", "origin_split"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["split"], row["event"], row["id"])))

    summary = {
        "schema_version": 1,
        "dataset": "event_transfer_earthquake",
        "task": "Turkey-event training with held-out earthquake-event transfer",
        "train_event": train_event,
        "test_events": sorted(test_events),
        "splits": {split: {"samples": counts[split], "events": dict(sorted(event_counts[split].items()))}
                   for split in SPLITS},
        "total_samples": sum(counts.values()),
        "manifest": str(manifest_path),
    }
    (metadata / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
