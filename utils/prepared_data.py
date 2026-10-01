"""Lossless shared inputs before random cropping/augmentation, with sealed identities."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.protocol import atomic_json, digest, file_digest

CACHE_VERSION = 1
# These fields are used AFTER loading the deterministic base tensors, or are
# display/root-discovery settings. Quality masks, bands and label semantics stay.
POST_LOAD_FIELDS = {"patch_size", "train_random_crop", "positive_crop_prob", "rare_crop_prob",
                    "rare_crop_classes", "crop_candidate_count", "eval_full_image",
                    "candidate_roots", "name", "splits"}


def input_signature(dataset):
    cfg = {k: v for k, v in dataset.dataset_cfg.items() if k not in POST_LOAD_FIELDS}
    cfg["root"] = str(dataset.root.resolve())
    return digest({"class": type(dataset).__name__, "dataset": cfg,
                   "normalization": dataset.normalization_cfg, "split": dataset.split})


@functools.lru_cache(maxsize=1)
def preprocessing_identity():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "datasets").glob("*.py")) + [root / "utils/raster.py", Path(__file__)]
    return digest({str(p.relative_to(root)): file_digest(p) for p in paths})


@functools.lru_cache(maxsize=4)
def _read_index(path):
    return json.loads(Path(path).read_text())


def attach_prepared_split(dataset):
    path = os.environ.get("DEHCD_PREPARED_INDEX")
    if not path:
        return False
    index = _read_index(path)
    if index["preprocessing_sha256"] != preprocessing_identity():
        raise ValueError("Prepared inputs were created with different preprocessing code")
    signature = input_signature(dataset)
    manifest_path = index["splits"].get(signature)
    if manifest_path is None:
        raise ValueError(f"No prepared input entry for {dataset.root}/{dataset.split}; prepare this configuration first")
    prepared = PreparedSplit(manifest_path, signature)
    manifest = prepared.manifest
    dataset.samples = manifest["samples"]
    dataset.num_optical_channels = manifest["channels"]["optical"]
    dataset.num_sar_channels = manifest["channels"]["sar"]
    dataset.sar_channel_counts = manifest["channels"]["sar_files"]
    dataset._prepared_split = prepared
    dataset._prepared_identities = manifest["identities"]
    return True


class PreparedSplit:
    def __init__(self, path, signature):
        self.path = Path(path)
        expected_hash = (self.path.parent / "manifest.sha256").read_text().strip()
        if file_digest(self.path) != expected_hash:
            raise ValueError("Prepared input manifest checksum mismatch")
        self.manifest = json.loads(self.path.read_text())
        if self.manifest["version"] != CACHE_VERSION or self.manifest["signature"] != signature:
            raise ValueError("Prepared input signature/version mismatch")
        for name, info in self.manifest["files"].items():
            stat = (self.path.parent / name).stat()
            if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
                raise ValueError(f"Prepared input file changed: {self.path.parent / name}")
        self._arrays = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = None
        return state

    def get(self, index):
        if self._arrays is None:
            self._arrays = {name: np.memmap(self.path.parent / f"{name}.bin", mode="r", dtype=info["dtype"])
                            for name, info in self.manifest["tensors"].items()}
        row = self.manifest["offsets"][index]
        item = {"id": self.manifest["samples"][index]["id"]}
        for name, array in self._arrays.items():
            offset, shape = row[name]
            # Own writable memory: no augmentation can mutate the shared cache.
            value = np.array(array[offset:offset + int(np.prod(shape))].reshape(shape), copy=True)
            tensor = torch.from_numpy(value)
            item[name] = tensor.long() if name == "label" else tensor
        return item

    def label_distribution(self, num_classes, ignore_index, max_samples=0):
        if [num_classes, ignore_index] != self.manifest["label_settings"]:
            raise ValueError("Prepared label statistics do not match the task")
        path = self.path.parent / "label_counts.npy"
        if not path.exists():
            return None
        counts = np.load(path, allow_pickle=False)
        if max_samples > 0:
            counts = counts[:max_samples]
        total = counts.sum(axis=0)
        frequencies = total / max(float(total.sum()), 1.0)
        return {"sample_counts": list(counts), "counts": total,
                "class_frequencies": {i: float(frequencies[i]) for i in range(num_classes)},
                "foreground_ratio": float(frequencies[1:].sum()) if num_classes > 1 else float(frequencies[0]),
                "sample_count": len(counts)}


class _PreparationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        dataset = self.dataset
        item = dataset._load_base_sample(index)
        stats = None
        if dataset.split == "train":
            # Match the existing label-statistics path exactly, including Haiti
            # quality masks at label resolution (which may differ from imagery).
            label = dataset.load_label_for_stats(dataset.samples[index]).numpy()
            valid = (label != dataset.ignore_index) & (label >= 0) & (label < dataset.num_classes)
            stats = np.bincount(label[valid].astype(np.int64).reshape(-1),
                                minlength=dataset.num_classes).astype(np.float64)
        return index, item, stats


def _identity_collate(value):
    return value


def prepare_split(dataset, identities, cache_root, workers=8, progress=None):
    if dataset.return_metadata:
        raise ValueError("Prepared tensor caches require dataset.return_metadata=false; use --cache none for metadata workflows")
    signature = input_signature(dataset)
    key = digest({"version": CACHE_VERSION, "signature": signature,
                  "data": identities, "preprocessing": preprocessing_identity()})
    target = Path(cache_root) / key
    manifest_path = target / "manifest.json"
    if target.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            valid = (manifest["signature"] == signature and manifest["identities"] == identities
                     and manifest["preprocessing_sha256"] == preprocessing_identity()
                     and file_digest(manifest_path) == (target / "manifest.sha256").read_text().strip()
                     and all((target / name).is_file() and (target / name).stat().st_size == info["bytes"]
                             and (target / name).stat().st_mtime_ns == info["mtime_ns"]
                             and file_digest(target / name) == info["sha256"]
                             for name, info in manifest["files"].items()))
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if valid:
            if progress: progress(len(dataset), len(dataset), True)
            return str(manifest_path)
        # Only rebuild this content-addressed generated cache, never source data.
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=target.parent))
    label_dtype = "uint8" if 0 <= dataset.ignore_index <= 255 and dataset.num_classes <= 256 else "int64"
    dtypes = {"optical": "float32", "sar": "float32", "label": label_dtype}
    offsets, statistics = [], []
    positions = dict.fromkeys(dtypes, 0)
    hashes = {name: hashlib.sha256() for name in dtypes}
    streams = {}
    try:
        streams = {name: (temporary / f"{name}.bin").open("wb", buffering=1024*1024) for name in dtypes}
        loader = DataLoader(_PreparationDataset(dataset), batch_size=None, shuffle=False,
                            num_workers=max(int(workers), 0), collate_fn=_identity_collate,
                            generator=torch.Generator().manual_seed(0))
        for index, item, stats in loader:
            row = {}
            for name, dtype in dtypes.items():
                tensor = item[name]
                if name != "label" and not np.isfinite(tensor.numpy()).all():
                    raise FloatingPointError(f"Non-finite prepared {name}: {dataset.samples[index]['id']}")
                if name == "label" and dtype == "uint8" and bool(((tensor.numpy() < 0) | (tensor.numpy() > 255)).any()):
                    raise ValueError("Label cannot be stored losslessly in uint8")
                array = np.ascontiguousarray(tensor.numpy(), dtype=dtype)
                buffer = memoryview(array).cast("B")
                streams[name].write(buffer)
                hashes[name].update(buffer)
                row[name] = [positions[name], list(array.shape)]
                positions[name] += array.size
            offsets.append(row)
            if stats is not None: statistics.append(stats)
            if progress: progress(index + 1, len(dataset), False)
        from utils.protocol import dataset_identity
        if dataset_identity(dataset) != identities["stat"]:
            raise ValueError("Source data changed while preparing inputs; no cache was published")
        for stream in streams.values():
            stream.flush(); os.fsync(stream.fileno()); stream.close()
        if statistics:
            np.save(temporary / "label_counts.npy", np.asarray(statistics), allow_pickle=False)
        files = {}
        for path in temporary.iterdir():
            stat = path.stat()
            files[path.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "sha256": hashes[path.stem].hexdigest() if path.suffix == ".bin" else file_digest(path)}
        manifest = {"version": CACHE_VERSION, "signature": signature,
                    "preprocessing_sha256": preprocessing_identity(), "identities": identities,
                    "samples": dataset.samples, "offsets": offsets, "files": files,
                    "tensors": {name: {"dtype": dtype} for name, dtype in dtypes.items()},
                    "channels": {"optical": dataset.num_optical_channels, "sar": dataset.num_sar_channels,
                                 "sar_files": dataset.sar_channel_counts},
                    "label_settings": [dataset.num_classes, dataset.ignore_index]}
        atomic_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(file_digest(temporary / "manifest.json") + "\n")
        temporary.rename(target)
        return str(manifest_path)
    finally:
        for stream in streams.values():
            if not stream.closed: stream.close()
        if temporary.exists(): shutil.rmtree(temporary)
