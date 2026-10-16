"""Optional workstation paths, kept outside shared experiment definitions."""
from __future__ import annotations

import json
import os
from pathlib import Path

ENVIRONMENT_KEYS = {
    "data_base": "DEHCD_DATA_BASE", "output": "DEHCD_OUTPUT",
    "event_data_root": "DEHCD_EVENT_DATA_ROOT", "event_output": "DEHCD_EVENT_OUTPUT",
    "device": "DEHCD_DEVICE", "runtime_profile": "DEHCD_RUNTIME_PROFILE",
}
PATH_KEYS = {"data_base", "output", "event_data_root", "event_output"}


def launch_settings(root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    values = {"data_base": str(root / "data"), "output": str(root / "runs/campaign"),
              "event_data_root": str(root / "data/event_transfer_earthquake"),
              "event_output": str(root / "runs/event_transfer_earthquake"),
              "device": "auto", "runtime_profile": "none"}
    path = root / "local/settings.json"
    if path.exists():
        local = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(local, dict) or set(local) - set(values):
            raise ValueError(f"Invalid workstation settings keys: {path}")
        values.update(local)
    for name, variable in ENVIRONMENT_KEYS.items():
        if variable in os.environ:
            values[name] = os.environ[variable]
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Workstation setting {name!r} must be a nonempty string")
        if name in PATH_KEYS:
            path = Path(os.path.expandvars(value)).expanduser()
            values[name] = str(path if path.is_absolute() else root / path)
    if values["runtime_profile"] not in {"none", "rtx5090"}:
        raise ValueError("runtime_profile must be 'none' or 'rtx5090'")
    return values


def local_source_reviews(root=None):
    """Read exact, explicitly checked source transitions; never infer compatibility."""
    root = Path(root or Path(__file__).resolve().parents[1])
    path = root / "local/source_compatibility.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("source_reviews") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid source compatibility records: {path}")
    return rows
