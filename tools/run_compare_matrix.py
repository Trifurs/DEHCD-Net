from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparison-model training configs.")
    parser.add_argument("--config-dir", default="configs/5090_x1/compare", help="Folder containing comparison XML configs.")
    parser.add_argument("--dataset", choices=["bright", "cau", "haiti"], default=None, help="Optional dataset prefix filter.")
    parser.add_argument(
        "--model",
        choices=["icif_net", "dminet", "hfa_panet", "wavehfg", "hrsicd", "haff"],
        default=None,
        help="Optional comparison model suffix filter.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--epochs", type=int, default=None, help="Forwarded to tools/train.py.")
    parser.add_argument("--batch-size", type=int, default=None, help="Forwarded to tools/train.py.")
    parser.add_argument("--patch-size", type=int, default=None, help="Forwarded to tools/train.py.")
    parser.add_argument("--num-workers", type=int, default=None, help="Forwarded to tools/train.py.")
    parser.add_argument("--device", default=None, help="Forwarded to tools/train.py.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Forwarded to tools/train.py.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Forwarded to tools/train.py.")
    parser.add_argument("--no-amp", action="store_true", help="Forwarded to tools/train.py.")
    return parser.parse_args()


def select_configs(config_dir: Path, dataset: str | None, model: str | None) -> list[Path]:
    configs = sorted(config_dir.glob("*.xml"))
    if dataset:
        configs = [path for path in configs if path.name.startswith(f"{dataset}_")]
    if model:
        configs = [path for path in configs if path.stem.endswith(model)]
    if not configs:
        raise FileNotFoundError(f"No comparison configs matched in {config_dir}")
    return configs


def build_command(config: Path, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "tools/train.py", "--config", str(config)]
    for option in ["epochs", "batch_size", "patch_size", "num_workers", "device", "max_train_batches", "max_val_batches"]:
        value = getattr(args, option)
        if value is not None:
            command.extend([f"--{option.replace('_', '-')}", str(value)])
    if args.no_amp:
        command.append("--no-amp")
    return command


def main() -> None:
    args = parse_args()
    config_dir = (PROJECT_ROOT / args.config_dir).resolve()
    configs = select_configs(config_dir, args.dataset, args.model)
    for config in configs:
        command = build_command(config.relative_to(PROJECT_ROOT), args)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
