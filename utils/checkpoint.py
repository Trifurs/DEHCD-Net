from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch
import os
import tempfile


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def load_model_state(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    unwrap_model(model).load_state_dict(clean_state_dict(state_dict))


def save_checkpoint(path: str | Path, model: torch.nn.Module, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["model"] = unwrap_model(model).state_dict()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cpu_snapshot(value):
    import copy
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items(): result[key] = _cpu_snapshot(item)
        return result
    if isinstance(value, list): return [_cpu_snapshot(v) for v in value]
    if isinstance(value, tuple): return tuple(_cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def _write_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class CheckpointWriter:
    """One immutable CPU snapshot and at most one pending disk writer."""
    def __init__(self, asynchronous=False, compact_best=False):
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.pending = None
        self.compact_best = compact_best

    def flush(self):
        if self.pending is not None:
            self.pending.result()  # Never turn disk-full or I/O failure into success.
            self.pending = None

    def submit(self, directory, model, payload, improved=False, archive=False):
        self.flush()
        payload = _cpu_snapshot({**payload, "model": unwrap_model(model).state_dict(), "checkpoint_kind": "training"})
        directory = Path(directory)
        def write():
            _write_payload(directory / "last.pth", payload)
            if improved:
                best = payload
                if self.compact_best:
                    excluded = {"optimizer", "scheduler", "scaler", "rng_state"}
                    best = {k: v for k, v in payload.items() if k not in excluded}
                    best["checkpoint_kind"] = "inference"
                if self.compact_best:
                    _write_payload(directory / "best.pth", best)
                else:
                    _link_checkpoint(directory / "last.pth", directory / "best.pth")
            if archive:
                _link_checkpoint(directory / "last.pth", directory / f"epoch_{payload['epoch']:03d}.pth")
        if self.pool is not None: self.pending = self.pool.submit(write)
        else: write()

    def close(self):
        try: self.flush()
        finally:
            if self.pool is not None: self.pool.shutdown(wait=True)


def _link_checkpoint(source, destination):
    """Atomic immutable alias; replacing last.pth never mutates earlier best/epoch files."""
    import shutil
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(fd)
    os.unlink(temporary)
    try:
        try: os.link(source, temporary)
        except OSError: shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
