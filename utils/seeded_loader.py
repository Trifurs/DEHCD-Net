"""Seed each sampled occurrence, making persistent workers restartable.

Indices carry their augmentation seed into the worker. Worker scheduling,
repeated weighted-sampler indices and a restarted process cannot reuse an old
epoch's augmentation stream. Main-process RNG state is restored (workers=0).
"""
from __future__ import annotations

import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class DrawSeededDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, request):
        index, seed = request
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Do not seed CUDA or change model/dropout RNG in the parent process.
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


class DrawSeededSampler(Sampler):
    def __init__(self, sampler, seed):
        self.sampler, self.seed, self.epoch = sampler, int(seed), 0

    @property
    def generator(self):
        return getattr(self.sampler, "generator", None)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for draw, index in enumerate(self.sampler):
            # Hash the tuple: adding seed + epoch + draw would make nearby run
            # seeds reuse one another's augmentation stream at a shifted index.
            key = f"{self.seed}:{self.epoch}:{draw}".encode("ascii")
            seed = int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "little")
            yield int(index), seed
