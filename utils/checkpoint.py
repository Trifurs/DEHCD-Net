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


def validate_training_checkpoint(checkpoint):
    """A weights-only artifact cannot silently restart optimizer/RNG on resume."""
    required = {'model', 'config', 'optimizer', 'scheduler', 'scaler', 'rng_state', 'epoch',
                'best_metric', 'best_metric_name', 'best_epoch', 'early_stop_reference',
                'epochs_without_improvement', 'collapse_count', 'foreground_stall_count'}
    if not isinstance(checkpoint, dict) or checkpoint.get('checkpoint_kind') == 'inference':
        raise ValueError('Resume requires a complete local training checkpoint, not inference weights')
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError('Incomplete training checkpoint; missing: ' + ', '.join(sorted(missing)))
    for name in ('model', 'config', 'optimizer', 'scheduler', 'scaler', 'rng_state'):
        if not isinstance(checkpoint[name], dict) or (name != 'scaler' and not checkpoint[name]):
            raise ValueError(f'Incomplete training checkpoint: invalid {name}')
    if not {'python', 'numpy', 'torch', 'cuda'} <= checkpoint['rng_state'].keys():
        raise ValueError('Incomplete training checkpoint RNG state')
    if not checkpoint['optimizer'].get('param_groups'):
        raise ValueError('Incomplete training checkpoint optimizer parameter groups')
    import math
    for name in ('best_metric', 'early_stop_reference'):
        if not math.isfinite(float(checkpoint[name])):
            raise ValueError(f'Non-finite training checkpoint {name}')
    for name in ('epoch', 'best_epoch', 'epochs_without_improvement', 'collapse_count', 'foreground_stall_count'):
        value = checkpoint[name]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f'Invalid training checkpoint {name}')
    if checkpoint['best_epoch'] > checkpoint['epoch']:
        raise ValueError('Training checkpoint best epoch exceeds committed epoch')
    if checkpoint['scheduler'].get('last_epoch') != checkpoint['epoch']:
        raise ValueError('Training checkpoint scheduler last_epoch differs from committed epoch')

    def finite_state(value, path):
        from numbers import Real
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Non-finite training checkpoint {path}')
        elif isinstance(value, Real):
            if not math.isfinite(value):
                raise ValueError(f'Non-finite training checkpoint {path}')
        elif isinstance(value, dict):
            for key, item in value.items(): finite_state(item, f'{path}.{key}')
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value): finite_state(item, f'{path}[{index}]')
    for name in ('model', 'optimizer', 'scheduler', 'scaler'):
        finite_state(checkpoint[name], name)

    # Validate on independent generators: never alter the process-global streams
    # while deciding whether this checkpoint is admissible for continuation.
    import random
    import numpy as np
    rng = checkpoint['rng_state']
    try:
        random.Random(0).setstate(rng['python'])
        state = rng['numpy']
        if len(state) != 5 or state[0] != 'MT19937':
            raise ValueError('Expected the recorded NumPy MT19937 state')
        raw_keys = np.asarray(state[1])
        if raw_keys.shape != (624,) or raw_keys.dtype.kind not in 'iu' or np.any(raw_keys < 0) or np.any(raw_keys > 2**32 - 1):
            raise ValueError('Invalid NumPy RNG key array')
        if not isinstance(state[2], int) or not 0 <= state[2] <= 624 or state[3] not in (0, 1) or not math.isfinite(state[4]):
            raise ValueError('Invalid NumPy RNG position/cache')
        np.random.RandomState(0).set_state((state[0], raw_keys.astype(np.uint32), state[2], state[3], state[4]))
        torch.Generator(device='cpu').set_state(rng['torch'].cpu())
        if not isinstance(rng['cuda'], (list, tuple)):
            raise ValueError('CUDA RNG states must be a sequence')
        if rng['cuda'] and (not torch.cuda.is_available() or len(rng['cuda']) != torch.cuda.device_count()):
            raise ValueError('The saved CUDA RNG states require the same available CUDA devices')
        for index, state in enumerate(rng['cuda']):
            torch.Generator(device=f'cuda:{index}').set_state(state.cpu())
    except (TypeError, ValueError, RuntimeError, AttributeError, IndexError) as exc:
        raise ValueError(f'Invalid or unavailable training checkpoint RNG state: {exc}') from exc


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
