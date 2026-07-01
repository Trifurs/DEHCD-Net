from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "dehcd_net",
    log_dir: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_dir:
        log_path = (Path(log_dir).expanduser() / "run.log").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path for handler in logger.handlers):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class ExperimentWriters:
    """Thin optional wrapper around TensorBoard and WandB."""

    def __init__(
        self,
        log_dir: str,
        enable_tensorboard: bool = True,
        enable_wandb: bool = False,
        wandb_project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.tb_writer = None
        self.wandb = None

        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=log_dir)
            except Exception as exc:  # pragma: no cover - depends on optional package
                logging.getLogger("dehcd_net").warning(
                    "TensorBoard disabled: %s", exc
                )

        if enable_wandb:
            try:
                import wandb

                self.wandb = wandb
                self.wandb.init(project=wandb_project, name=run_name, config=config)
            except Exception as exc:  # pragma: no cover - depends on optional package
                logging.getLogger("dehcd_net").warning("WandB disabled: %s", exc)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log({key: value, "step": step})

    def log_image(self, key: str, image, step: int) -> None:
        if self.tb_writer is not None:
            patch_pillow_antialias()
            self.tb_writer.add_image(key, image, step)
        if self.wandb is not None:
            self.wandb.log({key: self.wandb.Image(image.permute(1, 2, 0).detach().cpu().numpy()), "step": step})

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb is not None:
            self.wandb.finish()


def patch_pillow_antialias() -> None:
    """Keep torch 1.12 TensorBoard image logging working with Pillow 10+."""

    try:
        from PIL import Image
    except Exception:
        return
    if hasattr(Image, "ANTIALIAS"):
        return
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None and hasattr(resampling, "LANCZOS"):
        Image.ANTIALIAS = resampling.LANCZOS
