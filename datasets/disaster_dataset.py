from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.raster import read_raster


class BaseHeterogeneousDisasterDataset(Dataset):
    """Base optical + one/multiple SAR disaster CD dataset.

    Expected config keys:
    - optical_dir: directory relative to split root.
    - sar_dirs: list of SAR directories relative to split root.
    - label_dir: label directory relative to split root.
    - suffix fields are optional; exact stems are used when suffixes are empty.
    """

    DATASET_NAME = "generic_heterogeneous"
    DEFAULT_DATASET_CFG: Dict[str, Any] = {}
    DEFAULT_NORMALIZATION_CFG: Dict[str, Any] = {}
    DEFAULT_AUGMENTATION_CFG: Dict[str, Any] = {}
    EXPECTED_NUM_CLASSES: Optional[int] = None
    EXPECTED_LABEL_MODES: tuple[str, ...] = ()

    def __init__(
        self,
        root: str,
        split: str = "train",
        dataset_cfg: Optional[Dict[str, Any]] = None,
        normalization_cfg: Optional[Dict[str, Any]] = None,
        augmentation_cfg: Optional[Dict[str, Any]] = None,
        training: bool = False,
    ):
        self.root = Path(root).expanduser()
        self.split = split
        self.dataset_cfg = dataset_cfg or {}
        self.normalization_cfg = normalization_cfg or {}
        self.augmentation_cfg = augmentation_cfg or {}
        self.training = training

        self.optical_dir = self.dataset_cfg.get("optical_dir", self.dataset_cfg.get("pre_dir", "pre-event"))
        self.sar_dirs = list(self.dataset_cfg.get("sar_dirs") or [self.dataset_cfg.get("post_dir", "post-event")])
        self.label_dir = self.dataset_cfg.get("label_dir", "target")
        self.optical_suffix = str(self.dataset_cfg.get("optical_suffix", self.dataset_cfg.get("pre_suffix", "")) or "")
        self.sar_suffixes = list(self.dataset_cfg.get("sar_suffixes") or [self.dataset_cfg.get("post_suffix", "")] * len(self.sar_dirs))
        self.label_suffix = str(self.dataset_cfg.get("label_suffix", "") or "")
        self.optical_band_indices = normalize_band_indices(
            self.dataset_cfg.get("optical_band_indices", self.dataset_cfg.get("optical_bands")),
            name="optical_band_indices",
        )
        self.sar_band_indices = normalize_sar_band_indices(
            self.dataset_cfg.get("sar_band_indices", self.dataset_cfg.get("sar_bands")),
            sar_count=len(self.sar_dirs),
        )
        self.extensions = {ext.lower() for ext in self.dataset_cfg.get("image_extensions", [".tif", ".tiff", ".png"])}
        self.label_mode = str(self.dataset_cfg.get("label_mode", "binary_change")).lower()
        self.ignore_index = int(self.dataset_cfg.get("ignore_index", 255))
        self.num_classes = int(self.dataset_cfg.get("num_classes", 0) or 0)
        self.label_ignore_values = [int(item) for item in self.dataset_cfg.get("label_ignore_values", [])]
        self.patch_size = int(self.dataset_cfg.get("patch_size", 0) or 0)
        self.train_random_crop = bool(self.dataset_cfg.get("train_random_crop", True))
        self.positive_crop_prob = float(self.dataset_cfg.get("positive_crop_prob", 0.0) or 0.0)
        self.rare_crop_prob = float(self.dataset_cfg.get("rare_crop_prob", 0.0) or 0.0)
        self.rare_crop_classes = [int(item) for item in self.dataset_cfg.get("rare_crop_classes", [])]
        self.crop_candidate_count = max(int(self.dataset_cfg.get("crop_candidate_count", 1) or 1), 1)
        self.eval_full_image = bool(self.dataset_cfg.get("eval_full_image", False))
        self.align_to_optical = bool(self.dataset_cfg.get("align_to_optical", self.dataset_cfg.get("align_to_pre_event", True)))
        self.return_metadata = bool(self.dataset_cfg.get("return_metadata", False))

        self.split_root = self.root / split
        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.split_root}")
        self.num_optical_channels, self.num_sar_channels, self.sar_channel_counts = self._infer_channels()

    @classmethod
    def from_config(cls, config: Dict[str, Any], split: str, training: bool = False) -> "BaseHeterogeneousDisasterDataset":
        dataset_cfg = merge_dicts(cls.DEFAULT_DATASET_CFG, config.get("dataset", {}))
        dataset_cfg.setdefault("ignore_index", config.get("task", {}).get("ignore_index", 255))
        dataset_cfg.setdefault("num_classes", config.get("task", {}).get("num_classes", cls.EXPECTED_NUM_CLASSES or 0))
        dataset_cfg = cls.normalize_dataset_cfg(dataset_cfg, config)
        cls.validate_config(config, dataset_cfg)
        root = resolve_dataset_root(dataset_cfg)
        return cls(
            root=root,
            split=split,
            dataset_cfg=dataset_cfg,
            normalization_cfg=merge_dicts(cls.DEFAULT_NORMALIZATION_CFG, config.get("normalization", {})),
            augmentation_cfg=merge_dicts(cls.DEFAULT_AUGMENTATION_CFG, config.get("augmentation", {})),
            training=training,
        )

    @classmethod
    def normalize_dataset_cfg(cls, dataset_cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return dataset_cfg

    @classmethod
    def validate_config(cls, config: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> None:
        task_cfg = config.get("task", {})
        if cls.EXPECTED_NUM_CLASSES is not None and not bool(dataset_cfg.get("allow_num_classes_override", False)):
            num_classes = int(task_cfg.get("num_classes", cls.EXPECTED_NUM_CLASSES))
            if num_classes != cls.EXPECTED_NUM_CLASSES:
                raise ValueError(
                    f"{cls.__name__} expects task.num_classes={cls.EXPECTED_NUM_CLASSES}, got {num_classes}. "
                    "Set dataset.allow_num_classes_override=true only for a deliberate ablation."
                )
        if cls.EXPECTED_LABEL_MODES:
            label_mode = str(dataset_cfg.get("label_mode", "")).lower()
            if label_mode not in cls.EXPECTED_LABEL_MODES:
                expected = ", ".join(cls.EXPECTED_LABEL_MODES)
                raise ValueError(f"{cls.__name__} expects label_mode in {{{expected}}}, got '{label_mode}'.")

    def _build_index(self) -> List[Dict[str, Any]]:
        optical_root = self.split_root / self.optical_dir
        sar_roots = [self.split_root / item for item in self.sar_dirs]
        label_root = self.split_root / self.label_dir
        for required in [optical_root, label_root, *sar_roots]:
            if not required.exists():
                raise FileNotFoundError(f"Required dataset directory not found: {required}")

        optical_map = index_files(optical_root, self.optical_suffix, self.extensions)
        sar_maps = [
            index_files(root, self.sar_suffixes[idx] if idx < len(self.sar_suffixes) else "", self.extensions)
            for idx, root in enumerate(sar_roots)
        ]
        label_map = index_files(label_root, self.label_suffix, self.extensions)

        paired_ids = set(optical_map) & set(label_map)
        for mapping in sar_maps:
            paired_ids &= set(mapping)
        paired_ids = sorted(paired_ids, key=natural_key)

        all_ids = set(optical_map) | set(label_map)
        for mapping in sar_maps:
            all_ids |= set(mapping)
        missing = sorted(all_ids - set(paired_ids), key=natural_key)
        if missing:
            print(f"[{self.__class__.__name__}] Warning: skipped {len(missing)} incomplete sample ids, e.g. {missing[:5]}")

        return [
            {
                "id": sample_id,
                "optical": str(optical_map[sample_id]),
                "sar": [str(mapping[sample_id]) for mapping in sar_maps],
                "label": str(label_map[sample_id]),
            }
            for sample_id in paired_ids
        ]

    def _infer_channels(self) -> tuple[int, int, List[int]]:
        sample = self.samples[0]
        optical, _ = read_raster(sample["optical"])
        optical = select_bands(optical, self.optical_band_indices, source=sample["optical"])
        sar_channels = 0
        sar_channel_counts: List[int] = []
        for sar_idx, path in enumerate(sample["sar"]):
            sar, _ = read_raster(path)
            sar = select_bands(sar, self.sar_band_indices[sar_idx], source=path)
            channel_count = int(sar.shape[0])
            sar_channel_counts.append(channel_count)
            sar_channels += channel_count
        return int(optical.shape[0]), sar_channels, sar_channel_counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        optical_np, optical_meta = read_raster(sample["optical"])
        optical_np = select_bands(optical_np, self.optical_band_indices, source=sample["optical"])
        sar_arrays = []
        sar_meta = []
        for sar_idx, path in enumerate(sample["sar"]):
            array, meta = read_raster(path)
            array = select_bands(array, self.sar_band_indices[sar_idx], source=path)
            sar_cfg = slice_sar_normalization_cfg(
                self.normalization_cfg.get("sar", {}),
                sar_idx=sar_idx,
                channel_counts=self.sar_channel_counts,
            )
            sar_arrays.append(torch.from_numpy(normalize_array(array, normalization_with_nodata(sar_cfg, meta))).float())
            sar_meta.append(meta)
        label_np, label_meta = read_raster(sample["label"])

        optical = torch.from_numpy(normalize_array(optical_np, normalization_with_nodata(self.normalization_cfg.get("optical", {}), optical_meta))).float()
        sar = torch.cat(sar_arrays, dim=0)
        label = torch.from_numpy(label_np[0] if label_np.ndim == 3 else label_np)
        label = prepare_label(
            label,
            mode=self.label_mode,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
            extra_ignore_values=self.label_ignore_values,
        )

        if self.align_to_optical:
            height, width = optical.shape[-2:]
            sar = resize_tensor(sar, (height, width), mode="bilinear")
            label = resize_label(label, (height, width))

        if self.patch_size > 0 and (self.training or not self.eval_full_image):
            crop_mode = "random" if self.training and self.train_random_crop else "center"
            optical, sar, label = crop_or_pad_sample(
                optical,
                sar,
                label,
                crop_size=self.patch_size,
                mode=crop_mode,
                ignore_index=self.ignore_index,
                positive_crop_prob=self.positive_crop_prob if self.training else 0.0,
                rare_crop_prob=self.rare_crop_prob if self.training else 0.0,
                rare_crop_classes=self.rare_crop_classes,
                crop_candidate_count=self.crop_candidate_count if self.training else 1,
            )

        if self.training and bool(self.augmentation_cfg.get("enabled", True)):
            optical, sar, label = apply_augmentation(
                optical,
                sar,
                label,
                random_flip=bool(self.augmentation_cfg.get("random_flip", True)),
                random_rotate90=bool(self.augmentation_cfg.get("random_rotate90", True)),
                optical_scale_jitter=float(self.augmentation_cfg.get("optical_scale_jitter", 0.0) or 0.0),
                optical_shift_jitter=float(self.augmentation_cfg.get("optical_shift_jitter", 0.0) or 0.0),
                sar_scale_jitter=float(self.augmentation_cfg.get("sar_scale_jitter", 0.0) or 0.0),
                sar_shift_jitter=float(self.augmentation_cfg.get("sar_shift_jitter", 0.0) or 0.0),
                optical_noise_std=float(self.augmentation_cfg.get("optical_noise_std", 0.0) or 0.0),
                sar_noise_std=float(self.augmentation_cfg.get("sar_noise_std", 0.0) or 0.0),
            )

        item: Dict[str, Any] = {"id": sample["id"], "optical": optical, "sar": sar, "label": label}
        if self.return_metadata:
            item["metadata"] = {
                "optical": optical_meta,
                "sar": sar_meta,
                "label": label_meta,
                "paths": sample,
            }
        return item


def resolve_dataset_root(dataset_cfg: Dict[str, Any]) -> str:
    candidates: List[str] = []
    if dataset_cfg.get("root"):
        candidates.append(str(dataset_cfg["root"]))
    candidates.extend(str(item) for item in dataset_cfg.get("candidate_roots", []) if item)
    for item in candidates:
        path = Path(item).expanduser()
        if path.exists() and path.is_dir():
            return str(path)
    if candidates:
        return candidates[0]
    raise ValueError("dataset.root or dataset.candidate_roots must be configured")


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def index_files(directory: Path, suffix: str, extensions: set[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda p: natural_key(p.stem)):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        sample_id = strip_suffix(path, suffix)
        mapping[sample_id] = path
    return mapping


def strip_suffix(path: Path, suffix: str) -> str:
    if suffix and path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    suffix_stem = Path(suffix).stem if suffix else ""
    if suffix_stem and path.stem.endswith(suffix_stem):
        return path.stem[: -len(suffix_stem)]
    return path.stem


def normalize_band_indices(value: Any, name: str) -> Optional[List[int]]:
    """Normalize optional zero-based band indices from XML/list configs."""
    if value in (None, "", "auto"):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of zero-based band indices, got {type(value).__name__}")
    indices = [int(item) for item in value]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate indices: {indices}")
    if any(idx < 0 for idx in indices):
        raise ValueError(f"{name} must contain non-negative zero-based indices: {indices}")
    return indices


def normalize_sar_band_indices(value: Any, sar_count: int) -> List[Optional[List[int]]]:
    """Return one optional zero-based band-index list per SAR file.

    A flat list such as [0, 1] is applied to every configured SAR directory,
    while a nested list such as [[0, 1], [0, 1]] can select per-SAR bands.
    """
    if value in (None, "", "auto") or (isinstance(value, (list, tuple)) and len(value) == 0):
        return [None] * sar_count
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"sar_band_indices must be a list, got {type(value).__name__}")
    if all(not isinstance(item, (list, tuple)) for item in value):
        shared = normalize_band_indices(value, name="sar_band_indices")
        return [shared] * sar_count
    if len(value) != sar_count:
        raise ValueError(f"sar_band_indices must have {sar_count} per-SAR entries, got {len(value)}")
    return [normalize_band_indices(item, name=f"sar_band_indices[{idx}]") for idx, item in enumerate(value)]


def select_bands(array: np.ndarray, indices: Optional[List[int]], source: str | Path) -> np.ndarray:
    if indices is None:
        return array
    channel_count = int(array.shape[0])
    if any(idx >= channel_count for idx in indices):
        raise ValueError(f"Band index out of range for {source}: requested {indices}, available channels={channel_count}")
    return array[np.asarray(indices, dtype=np.int64), ...]


def natural_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def normalize_array(array: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    method = str(cfg.get("method", "percentile_minmax")).lower()
    nodata_values = cfg.get("nodata_values") or []
    nodata_mask = ~np.isfinite(arr)
    for value in nodata_values:
        if value is None:
            continue
        nodata_mask |= arr == float(value)
    if bool(np.any(nodata_mask)):
        arr = arr.copy()
        arr[nodata_mask] = np.nan

    if method == "standard" and cfg.get("mean") and cfg.get("std"):
        mean = np.asarray(cfg["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(cfg["std"], dtype=np.float32).reshape(-1, 1, 1)
        out = (arr - mean) / np.maximum(std, 1e-6)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    if bool(cfg.get("log_transform", False)):
        min_value = safe_nanmin(arr)
        if min_value >= 0:
            arr = np.log1p(arr)

    if method in {"fixed_minmax", "static_minmax"}:
        min_value = cfg.get("min_values", cfg.get("min_value", cfg.get("min", 0.0)))
        max_value = cfg.get("max_values", cfg.get("max_value", cfg.get("max", 1.0)))
        low = reshape_channel_value(min_value, arr.shape[0])
        high = reshape_channel_value(max_value, arr.shape[0])
        arr = np.clip(arr, low, high)
        out = (arr - low) / np.maximum(high - low, 1e-6)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    lower = float(cfg.get("lower_percentile", 2.0))
    upper = float(cfg.get("upper_percentile", 98.0))
    low = safe_channel_nanpercentile(arr, lower)
    high = safe_channel_nanpercentile(arr, upper)
    arr = np.clip(arr, low, high)

    if method in {"robust_zscore", "zscore"}:
        mean = safe_channel_nanmean(arr)
        std = safe_channel_nanstd(arr)
        out = (arr - mean) / np.maximum(std, 1e-6)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    out = (arr - low) / np.maximum(high - low, 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def normalization_with_nodata(cfg: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg or {})
    values = list(merged.get("nodata_values") or [])
    meta_nodata = meta.get("nodata") if meta else None
    if meta_nodata is not None and meta_nodata not in values:
        values.append(meta_nodata)
    merged["nodata_values"] = values
    return merged


def slice_sar_normalization_cfg(cfg: Dict[str, Any], sar_idx: int, channel_counts: List[int]) -> Dict[str, Any]:
    """Select per-file values from a concatenated SAR normalization config."""
    merged = dict(cfg or {})
    total_channels = sum(channel_counts)
    start = sum(channel_counts[:sar_idx])
    end = start + channel_counts[sar_idx]
    for key in ("min_values", "max_values", "mean", "std"):
        value = merged.get(key)
        if isinstance(value, (list, tuple)) and len(value) == total_channels:
            merged[key] = list(value[start:end])
    return merged


def safe_nanmin(arr: np.ndarray) -> float:
    if not bool(np.isfinite(arr).any()):
        return 0.0
    return float(np.nanmin(arr))


def safe_channel_nanpercentile(arr: np.ndarray, percentile: float) -> np.ndarray:
    value = np.zeros((arr.shape[0], 1, 1), dtype=np.float32)
    for channel_idx in range(arr.shape[0]):
        channel = arr[channel_idx]
        finite = np.isfinite(channel)
        if bool(finite.any()):
            value[channel_idx, 0, 0] = float(np.nanpercentile(channel[finite], percentile))
    return value


def safe_channel_nanmean(arr: np.ndarray) -> np.ndarray:
    value = np.zeros((arr.shape[0], 1, 1), dtype=np.float32)
    for channel_idx in range(arr.shape[0]):
        channel = arr[channel_idx]
        finite = np.isfinite(channel)
        if bool(finite.any()):
            value[channel_idx, 0, 0] = float(channel[finite].mean())
    return value


def safe_channel_nanstd(arr: np.ndarray) -> np.ndarray:
    value = np.ones((arr.shape[0], 1, 1), dtype=np.float32)
    for channel_idx in range(arr.shape[0]):
        channel = arr[channel_idx]
        finite = np.isfinite(channel)
        if bool(finite.any()):
            std = float(channel[finite].std())
            value[channel_idx, 0, 0] = max(std, 1e-6)
    return value


def reshape_channel_value(value: Any, channels: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full((channels,), float(arr), dtype=np.float32)
    if arr.size == 1:
        arr = np.full((channels,), float(arr.reshape(-1)[0]), dtype=np.float32)
    if arr.size != channels:
        raise ValueError(f"Expected scalar or {channels} normalization values, got {arr.size}")
    return arr.reshape(channels, 1, 1)


def prepare_label(
    label: torch.Tensor,
    mode: str,
    ignore_index: int = 255,
    num_classes: int = 0,
    extra_ignore_values: Optional[List[int]] = None,
) -> torch.Tensor:
    if label.is_floating_point():
        finite_mask = torch.isfinite(label)
        safe_label = torch.where(finite_mask, label, torch.zeros_like(label)).round().long()
        ignore_mask = ~finite_mask
    else:
        safe_label = label.long()
        ignore_mask = torch.zeros_like(safe_label, dtype=torch.bool)
    ignore_mask |= safe_label == ignore_index
    for value in extra_ignore_values or []:
        ignore_mask |= safe_label == int(value)
    if mode in {"binary", "binary_change", "change"}:
        out = (safe_label > 0).long()
        out[ignore_mask] = ignore_index
        return out
    if mode in {"multiclass", "multiclass_damage", "damage_levels"}:
        out = safe_label.long()
        if num_classes > 0:
            ignore_mask |= (out < 0) | (out >= num_classes)
        out[ignore_mask] = ignore_index
        return out
    raise ValueError(f"Unsupported label_mode: {mode}")


def resize_tensor(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, **kwargs).squeeze(0)


def resize_label(label: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(label.shape[-2:]) == tuple(size):
        return label
    resized = F.interpolate(label[None, None].float(), size=size, mode="nearest")
    return resized.squeeze(0).squeeze(0).long()


def crop_or_pad_sample(
    optical: torch.Tensor,
    sar: torch.Tensor,
    label: torch.Tensor,
    crop_size: int,
    mode: str,
    ignore_index: int,
    positive_crop_prob: float = 0.0,
    rare_crop_prob: float = 0.0,
    rare_crop_classes: Optional[List[int]] = None,
    crop_candidate_count: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = label.shape[-2:]
    pad_h = max(crop_size - height, 0)
    pad_w = max(crop_size - width, 0)
    if pad_h or pad_w:
        pad = (0, pad_w, 0, pad_h)
        optical = F.pad(optical, pad, mode="reflect")
        sar = F.pad(sar, pad, mode="reflect")
        label = F.pad(label, pad, mode="constant", value=ignore_index)
        height, width = label.shape[-2:]

    if height == crop_size and width == crop_size:
        return optical, sar, label

    max_top = height - crop_size
    max_left = width - crop_size
    if mode == "random":
        top, left = choose_random_crop_origin(
            label,
            crop_size=crop_size,
            max_top=max_top,
            max_left=max_left,
            ignore_index=ignore_index,
            positive_crop_prob=positive_crop_prob,
            rare_crop_prob=rare_crop_prob,
            rare_crop_classes=rare_crop_classes or [],
            crop_candidate_count=crop_candidate_count,
        )
    elif mode == "center":
        top = max_top // 2
        left = max_left // 2
    else:
        raise ValueError(f"Unsupported crop mode: {mode}")

    bottom = top + crop_size
    right = left + crop_size
    return (
        optical[:, top:bottom, left:right].contiguous(),
        sar[:, top:bottom, left:right].contiguous(),
        label[top:bottom, left:right].contiguous(),
    )


def choose_random_crop_origin(
    label: torch.Tensor,
    crop_size: int,
    max_top: int,
    max_left: int,
    ignore_index: int,
    positive_crop_prob: float,
    rare_crop_prob: float = 0.0,
    rare_crop_classes: Optional[List[int]] = None,
    crop_candidate_count: int = 1,
) -> tuple[int, int]:
    rare_crop_classes = rare_crop_classes or []
    crop_candidate_count = max(int(crop_candidate_count or 1), 1)
    if rare_crop_prob > 0 and rare_crop_classes and random.random() < rare_crop_prob:
        rare = torch.zeros_like(label, dtype=torch.bool)
        for class_idx in rare_crop_classes:
            rare |= label == int(class_idx)
        rare &= label != ignore_index
        if bool(rare.any()):
            return best_crop_origin_around_mask(
                rare,
                label=label,
                crop_size=crop_size,
                max_top=max_top,
                max_left=max_left,
                candidate_count=crop_candidate_count,
                positive_mask=(label > 0) & (label != ignore_index),
            )

    if positive_crop_prob > 0 and random.random() < positive_crop_prob:
        positive = (label > 0) & (label != ignore_index)
        if bool(positive.any()):
            return best_crop_origin_around_mask(
                positive,
                label=label,
                crop_size=crop_size,
                max_top=max_top,
                max_left=max_left,
                candidate_count=crop_candidate_count,
                positive_mask=positive,
            )
    return random.randint(0, max_top), random.randint(0, max_left)


def crop_origin_around_mask(mask: torch.Tensor, crop_size: int, max_top: int, max_left: int) -> tuple[int, int]:
    coords = torch.nonzero(mask, as_tuple=False)
    y, x = coords[random.randrange(coords.shape[0])].tolist()
    top_min = max(0, y - crop_size + 1)
    top_max = min(y, max_top)
    left_min = max(0, x - crop_size + 1)
    left_max = min(x, max_left)
    if top_min <= top_max and left_min <= left_max:
        return random.randint(top_min, top_max), random.randint(left_min, left_max)
    return random.randint(0, max_top), random.randint(0, max_left)


def best_crop_origin_around_mask(
    mask: torch.Tensor,
    label: torch.Tensor,
    crop_size: int,
    max_top: int,
    max_left: int,
    candidate_count: int,
    positive_mask: torch.Tensor,
) -> tuple[int, int]:
    best_origin = crop_origin_around_mask(mask, crop_size, max_top, max_left)
    best_score = crop_score(label, positive_mask, mask, best_origin, crop_size)
    for _ in range(max(int(candidate_count) - 1, 0)):
        origin = crop_origin_around_mask(mask, crop_size, max_top, max_left)
        score = crop_score(label, positive_mask, mask, origin, crop_size)
        if score > best_score:
            best_origin = origin
            best_score = score
    return best_origin


def crop_score(
    label: torch.Tensor,
    positive_mask: torch.Tensor,
    target_mask: torch.Tensor,
    origin: tuple[int, int],
    crop_size: int,
) -> tuple[int, int]:
    top, left = origin
    bottom = top + crop_size
    right = left + crop_size
    target_pixels = int(target_mask[top:bottom, left:right].sum().item())
    positive_pixels = int(positive_mask[top:bottom, left:right].sum().item())
    return target_pixels, positive_pixels


def apply_augmentation(
    optical: torch.Tensor,
    sar: torch.Tensor,
    label: torch.Tensor,
    random_flip: bool = True,
    random_rotate90: bool = True,
    optical_scale_jitter: float = 0.0,
    optical_shift_jitter: float = 0.0,
    sar_scale_jitter: float = 0.0,
    sar_shift_jitter: float = 0.0,
    optical_noise_std: float = 0.0,
    sar_noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if random_flip and random.random() < 0.5:
        optical = torch.flip(optical, dims=(-1,))
        sar = torch.flip(sar, dims=(-1,))
        label = torch.flip(label, dims=(-1,))
    if random_flip and random.random() < 0.5:
        optical = torch.flip(optical, dims=(-2,))
        sar = torch.flip(sar, dims=(-2,))
        label = torch.flip(label, dims=(-2,))
    if random_rotate90:
        k = random.randint(0, 3)
        if k:
            optical = torch.rot90(optical, k, dims=(-2, -1))
            sar = torch.rot90(sar, k, dims=(-2, -1))
            label = torch.rot90(label, k, dims=(-2, -1))
    optical = apply_radiometric_jitter(
        optical,
        scale_jitter=optical_scale_jitter,
        shift_jitter=optical_shift_jitter,
        noise_std=optical_noise_std,
    )
    sar = apply_radiometric_jitter(
        sar,
        scale_jitter=sar_scale_jitter,
        shift_jitter=sar_shift_jitter,
        noise_std=sar_noise_std,
    )
    return optical.contiguous(), sar.contiguous(), label.contiguous()


def apply_radiometric_jitter(
    tensor: torch.Tensor,
    scale_jitter: float = 0.0,
    shift_jitter: float = 0.0,
    noise_std: float = 0.0,
) -> torch.Tensor:
    if scale_jitter > 0:
        low = max(1.0 - scale_jitter, 0.0)
        high = 1.0 + scale_jitter
        scale = tensor.new_empty((tensor.shape[0], 1, 1)).uniform_(low, high)
        tensor = tensor * scale
    if shift_jitter > 0:
        shift = tensor.new_empty((tensor.shape[0], 1, 1)).uniform_(-shift_jitter, shift_jitter)
        tensor = tensor + shift
    if noise_std > 0:
        tensor = tensor + torch.randn_like(tensor) * noise_std
    return tensor


class HeterogeneousDisasterDataset(BaseHeterogeneousDisasterDataset):
    """Backward-compatible generic dataset class.

    New experiments should use BRIGHTDataset or CAU1Dataset via
    dataset.type, while this class remains available for custom datasets.
    """
