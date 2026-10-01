"""Create an event-disjoint split CSV from explicit event/source-scene metadata."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.split_manifest import read_manifest


def assign_events(rows, test_events, val_events):
    if not test_events or not val_events or test_events & val_events:
        raise ValueError("Test and validation events must be explicit, nonempty, and disjoint")
    available = {r["event"] for r in rows}
    if (test_events | val_events) - available:
        raise ValueError("Requested held-out event is absent from the input metadata")
    result = [{**r, "split": "test" if r["event"] in test_events else
               "val" if r["event"] in val_events else "train"} for r in rows]
    if {r["split"] for r in result} != {"train", "val", "test"}:
        raise ValueError("Event assignment must leave nonempty train, val and test splits")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV with id,source_split,split,group,event")
    p.add_argument("--output", required=True)
    p.add_argument("--test-events", nargs="+", required=True)
    p.add_argument("--val-events", nargs="+", required=True)
    a = p.parse_args()
    rows = assign_events(read_manifest(a.input), set(a.test_events), set(a.val_events))
    destination = Path(a.output); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "source_split", "split", "group", "event"])
        writer.writeheader(); writer.writerows(rows)
    read_manifest(destination)
    print(f"Saved {len(rows)} assignments without moving imagery: {destination}")


if __name__ == "__main__":
    main()
