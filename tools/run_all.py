"""Local four-dataset, three-seed launcher with the measured RTX 5090 profile."""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-base", default=str(Path.home() / "桌面/myData/Hete_CD"))
    parser.add_argument("--output", default=str(Path.home() / "桌面/myResult/DEHCD-Net"))
    parser.add_argument("--datasets", nargs="+", choices=["bright", "haiti", "xbd", "cau_flood"],
                        default=["bright", "haiti", "xbd", "cau_flood"])
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1051, 2060])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-workers", type=int, default=8)
    parser.add_argument("--cache", choices=["normalized", "none"], default="normalized")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    folders = {"bright": "BRIGHT1", "haiti": "Haiti1", "cau_flood": "CAU1", "xbd": "xBD1"}
    command = [str(ROOT / "tools/run_multiseed.py"), "--groups", *args.groups, "--datasets", *args.datasets,
               "--seeds", *map(str, args.seeds), "--runtime-profile", "rtx5090", "--device", args.device,
               "--output", args.output, "--cache", args.cache, "--prepare-workers", str(args.prepare_workers),
               "--quiet-warnings", "--resume"]
    for dataset in args.datasets:
        command += ["--data-root", f"{dataset}={Path(args.data_base).expanduser() / folders[dataset]}"]
    for mode in ("audit_only", "dry_run", "preflight_only", "prepare_only"):
        if getattr(args, mode): command.append("--" + mode.replace("_", "-"))
    sys.argv = command
    from tools.run_multiseed import main as run
    run()


if __name__ == "__main__":
    main()
