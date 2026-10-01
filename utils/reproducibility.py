"""Randomness controls shared by training and repeat experiments."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    # CUDA grid_sample backward (flow alignment) has no deterministic implementation.
    # Warn rather than pretending that aligned CUDA training is bitwise reproducible.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def seed_epoch(loader, seed: int, epoch: int, deterministic: bool) -> None:
    """Epoch boundary restart is independent of a previous process's worker state."""
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    epoch_seed = (seed + 1000003 * epoch) % 2**32
    set_seed(epoch_seed, deterministic)
    if loader.generator is not None:
        loader.generator.manual_seed(epoch_seed)
    generator = getattr(loader.sampler, "generator", None)
    if generator is not None:
        generator.manual_seed((epoch_seed + 1) % 2**32)


def capture_rng_state() -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class BestTracker:
    """Strict best checkpoint and early-stop progress have distinct thresholds."""
    def __init__(self, best: float = -1.0, reference: float = -1.0, stale: int = 0):
        self.best, self.reference, self.stale = best, reference, stale

    def update(self, value: float, min_delta: float) -> bool:
        import math
        if not math.isfinite(value):
            raise ValueError("Validation metric is not finite")
        improved = value > self.best
        self.best = max(self.best, value)
        if value > self.reference + min_delta:
            self.reference, self.stale = value, 0
        else:
            self.stale += 1
        return improved
