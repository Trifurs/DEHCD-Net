"""Canonical definitions and ID-only views of shared experiment results."""
from __future__ import annotations

import json
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
                     ("id", "canonical_id", "dataset", "suite", "model_variant", "roles", "uses", "comparison")})
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


def write_catalog_views(root, rows):
    root = Path(root)
    atomic_json(root / "configs/experiments/catalog.json", {
        "schema_version": 3, "experiments": rows})
    atomic_json(root / "configs/experiments/groups.json", {
        "schema_version": 1, "groups": analysis_groups(rows)})
