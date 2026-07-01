from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def create_run_dir(root_dir: str | Path, stage: str, run_name: str) -> Path:
    root = Path(root_dir).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_name(run_name or "experiment")
    run_dir = root / stage / f"{timestamp}_{safe_name}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / stage / f"{timestamp}_{safe_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")


def save_config_snapshot(config: Dict[str, Any], run_dir: str | Path) -> None:
    path = Path(run_dir) / "config_snapshot.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
