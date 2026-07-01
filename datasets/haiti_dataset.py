from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from utils.raster import read_raster

from .disaster_dataset import (
    BaseHeterogeneousDisasterDataset,
    apply_augmentation,
    crop_or_pad_sample,
    normalization_with_nodata,
    prepare_label,
    reshape_channel_value,
    resize_label,
    resize_tensor,
    safe_channel_nanmean,
    safe_channel_nanpercentile,
    safe_channel_nanstd,
    safe_nanmin,
    select_bands,
    slice_sar_normalization_cfg,
)


class HaitiDataset(BaseHeterogeneousDisasterDataset):
    """Haiti 2021 multimodal landslide change-detection dataset.

    Haiti1 stores imagery and quality masks together in the same GeoTIFFs:
    Sentinel-2 has three optical bands plus a cloud/cloud-shadow mask, while
    Sentinel-1 has two SAR bands plus a layover-shadow mask. The mask bands are
    never exposed as ordinary model inputs here. They are used to suppress
    unreliable pixels in the input tensors and to mark untrainable label pixels
    with ``ignore_index``.
    """

    DATASET_NAME = "Haiti1"
    EXPECTED_NUM_CLASSES = 4
    EXPECTED_LABEL_MODES = ("multiclass", "multiclass_damage", "damage_levels")
    DEFAULT_DATASET_CFG: Dict[str, Any] = {
        "type": "haiti",
        "name": "Haiti1",
        "splits": ["train", "val", "test"],
        "optical_dir": "Pre_event/S2_20210804",
        "sar_dirs": ["Post_event/S1_ASC_20210817", "Post_event/S1_DESC_20210815"],
        "label_dir": "Annotations",
        "optical_band_indices": [0, 1, 2],
        "sar_band_indices": [[0, 1], [0, 1]],
        "optical_mask_band_index": 3,
        "sar_mask_band_indices": [2, 2],
        "optical_mask_invalid_values": [1],
        "sar_mask_valid_values": [0],
        "label_quality_ignore_policy": "all_modalities",
        "mask_normalization_stats": True,
        "masked_input_policy": "keep",
        "optical_suffix": "",
        "sar_suffixes": ["", ""],
        "label_suffix": "",
        "image_extensions": [".tif", ".tiff"],
        "label_mode": "multiclass_damage",
        "label_ignore_values": [],
        "patch_size": 128,
        "train_random_crop": True,
        "positive_crop_prob": 0.75,
        "rare_crop_prob": 0.75,
        "rare_crop_classes": [2, 3],
        "crop_candidate_count": 6,
        "eval_full_image": False,
        "align_to_optical": True,
        "return_metadata": False,
    }
    DEFAULT_NORMALIZATION_CFG: Dict[str, Any] = {
        "optical": {
            "method": "percentile_minmax",
            "lower_percentile": 2.0,
            "upper_percentile": 98.0,
        },
        "sar": {
            "method": "robust_zscore",
            "log_transform": False,
            "lower_percentile": 2.0,
            "upper_percentile": 98.0,
        },
    }

    @classmethod
    def normalize_dataset_cfg(cls, dataset_cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        dataset_cfg["label_mode"] = "multiclass_damage"
        dataset_cfg["num_classes"] = int(config.get("task", {}).get("num_classes", cls.EXPECTED_NUM_CLASSES))
        dataset_cfg["label_ignore_values"] = [int(item) for item in dataset_cfg.get("label_ignore_values", [])]
        dataset_cfg["rare_crop_classes"] = [int(item) for item in dataset_cfg.get("rare_crop_classes", [2, 3])]
        dataset_cfg["positive_crop_prob"] = min(max(float(dataset_cfg.get("positive_crop_prob", 0.0) or 0.0), 0.0), 1.0)
        dataset_cfg["rare_crop_prob"] = min(max(float(dataset_cfg.get("rare_crop_prob", 0.0) or 0.0), 0.0), 1.0)
        dataset_cfg["crop_candidate_count"] = max(int(dataset_cfg.get("crop_candidate_count", 1) or 1), 1)
        dataset_cfg["sar_mask_band_indices"] = _normalize_mask_band_list(
            dataset_cfg.get("sar_mask_band_indices", [2] * len(dataset_cfg.get("sar_dirs", []))),
            expected=len(dataset_cfg.get("sar_dirs", [])),
        )
        return dataset_cfg

    @classmethod
    def validate_config(cls, config: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> None:
        super().validate_config(config, dataset_cfg)
        sar_dirs = list(dataset_cfg.get("sar_dirs") or [])
        if len(sar_dirs) != 2 and not bool(dataset_cfg.get("allow_sar_dirs_override", False)):
            raise ValueError(
                "HaitiDataset expects the two post-event Sentinel-1 passes. "
                "Set dataset.allow_sar_dirs_override=true only for an intentional ablation."
            )
        optical_mask_band = int(dataset_cfg.get("optical_mask_band_index", 3))
        optical_bands = set(dataset_cfg.get("optical_band_indices") or [])
        if optical_mask_band in optical_bands:
            raise ValueError("Haiti optical cloud mask band must not be listed in optical_band_indices.")
        sar_band_indices = dataset_cfg.get("sar_band_indices") or []
        for idx, mask_band in enumerate(dataset_cfg.get("sar_mask_band_indices", [])):
            if sar_band_indices and all(not isinstance(item, (list, tuple)) for item in sar_band_indices):
                sar_bands = set(sar_band_indices)
            else:
                sar_bands = set(sar_band_indices[idx] or []) if idx < len(sar_band_indices) else set()
            if int(mask_band) in sar_bands:
                raise ValueError(f"Haiti SAR mask band for pass {idx} must not be listed in sar_band_indices.")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        optical_np, optical_meta = read_raster(sample["optical"])
        optical_mask = self._mask_from_band(
            optical_np,
            int(self.dataset_cfg.get("optical_mask_band_index", 3)),
            invalid_values=self.dataset_cfg.get("optical_mask_invalid_values", [1]),
        )
        optical_np = select_bands(optical_np, self.optical_band_indices, source=sample["optical"])
        optical = torch.from_numpy(
            _normalize_haiti_array(
                optical_np,
                optical_mask,
                normalization_with_nodata(self.normalization_cfg.get("optical", {}), optical_meta),
                mask_stats=bool(self.dataset_cfg.get("mask_normalization_stats", True)),
                masked_input_policy=str(self.dataset_cfg.get("masked_input_policy", "keep")),
            )
        ).float()

        sar_arrays: List[torch.Tensor] = []
        sar_invalid_masks: List[torch.Tensor] = []
        sar_meta = []
        for sar_idx, path in enumerate(sample["sar"]):
            sar_np, meta = read_raster(path)
            sar_invalid = self._mask_from_band(
                sar_np,
                int(self.dataset_cfg["sar_mask_band_indices"][sar_idx]),
                valid_values=self.dataset_cfg.get("sar_mask_valid_values", [0]),
            )
            sar_np = select_bands(sar_np, self.sar_band_indices[sar_idx], source=path)
            sar_cfg = slice_sar_normalization_cfg(
                self.normalization_cfg.get("sar", {}),
                sar_idx=sar_idx,
                channel_counts=self.sar_channel_counts,
            )
            sar_arrays.append(
                torch.from_numpy(
                    _normalize_haiti_array(
                        sar_np,
                        sar_invalid,
                        normalization_with_nodata(sar_cfg, meta),
                        mask_stats=bool(self.dataset_cfg.get("mask_normalization_stats", True)),
                        masked_input_policy=str(self.dataset_cfg.get("masked_input_policy", "keep")),
                    )
                ).float()
            )
            sar_invalid_masks.append(torch.from_numpy(sar_invalid))
            sar_meta.append(meta)

        label_np, label_meta = read_raster(sample["label"])
        sar = torch.cat(sar_arrays, dim=0)
        label = torch.from_numpy(label_np[0] if label_np.ndim == 3 else label_np)
        label = prepare_label(
            label,
            mode=self.label_mode,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
            extra_ignore_values=self.label_ignore_values,
        )

        optical_invalid = torch.from_numpy(optical_mask)
        if self.align_to_optical:
            height, width = optical.shape[-2:]
            sar = resize_tensor(sar, (height, width), mode="bilinear")
            label = resize_label(label, (height, width))
            optical_invalid = _resize_bool_mask(optical_invalid, (height, width))
            sar_invalid_masks = [_resize_bool_mask(mask, (height, width)) for mask in sar_invalid_masks]

        label = self._apply_quality_ignore(label, optical_invalid, sar_invalid_masks)

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
                "quality_policy": {
                    "optical_cloud_invalid": "optical mask band in optical_mask_invalid_values",
                    "label_quality_ignore_policy": self.dataset_cfg.get("label_quality_ignore_policy", "all_modalities"),
                    "masked_input_policy": self.dataset_cfg.get("masked_input_policy", "keep"),
                },
            }
        return item

    def load_label_for_stats(self, sample: Dict[str, Any]) -> torch.Tensor:
        label_np, _ = read_raster(sample["label"])
        label = torch.from_numpy(label_np[0] if label_np.ndim == 3 else label_np)
        label = prepare_label(
            label,
            mode=self.label_mode,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
            extra_ignore_values=self.label_ignore_values,
        )
        optical_np, _ = read_raster(sample["optical"])
        optical_invalid = torch.from_numpy(
            self._mask_from_band(
                optical_np,
                int(self.dataset_cfg.get("optical_mask_band_index", 3)),
                invalid_values=self.dataset_cfg.get("optical_mask_invalid_values", [1]),
            )
        )
        sar_invalid_masks = []
        for sar_idx, path in enumerate(sample["sar"]):
            sar_np, _ = read_raster(path)
            sar_invalid_masks.append(
                torch.from_numpy(
                    self._mask_from_band(
                        sar_np,
                        int(self.dataset_cfg["sar_mask_band_indices"][sar_idx]),
                        valid_values=self.dataset_cfg.get("sar_mask_valid_values", [0]),
                    )
                )
            )
        if tuple(optical_invalid.shape[-2:]) != tuple(label.shape[-2:]):
            optical_invalid = _resize_bool_mask(optical_invalid, label.shape[-2:])
        sar_invalid_masks = [
            _resize_bool_mask(mask, label.shape[-2:]) if tuple(mask.shape[-2:]) != tuple(label.shape[-2:]) else mask
            for mask in sar_invalid_masks
        ]
        return self._apply_quality_ignore(label, optical_invalid, sar_invalid_masks)

    def _apply_quality_ignore(
        self,
        label: torch.Tensor,
        optical_invalid: torch.Tensor,
        sar_invalid_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        policy = str(self.dataset_cfg.get("label_quality_ignore_policy", "all_modalities")).lower()
        optical_invalid = optical_invalid.bool()
        invalid = torch.zeros_like(optical_invalid, dtype=torch.bool)
        sar_all_invalid = torch.zeros_like(optical_invalid, dtype=torch.bool)
        sar_any_invalid = torch.zeros_like(optical_invalid, dtype=torch.bool)
        if sar_invalid_masks:
            stacked = torch.stack([mask.bool() for mask in sar_invalid_masks], dim=0)
            sar_all_invalid = stacked.all(dim=0)
            sar_any_invalid = stacked.any(dim=0)
        if policy in {"none", "off"}:
            pass
        elif policy in {"all_modalities", "any_valid"}:
            invalid = optical_invalid & sar_all_invalid
        elif policy in {"required_pair", "optical_or_all_sar"}:
            invalid = optical_invalid | sar_all_invalid
        elif policy in {"all_sar", "sar_all"}:
            invalid = sar_all_invalid
        elif policy == "optical":
            invalid = optical_invalid
        elif policy == "any":
            invalid = optical_invalid | sar_any_invalid
        else:
            raise ValueError(f"Unsupported label_quality_ignore_policy: {policy}")
        label = label.clone()
        label[invalid] = self.ignore_index
        return label

    @staticmethod
    def _mask_from_band(
        array: np.ndarray,
        band_index: int,
        invalid_values: List[int] | None = None,
        valid_values: List[int] | None = None,
    ) -> np.ndarray:
        if band_index < 0 or band_index >= array.shape[0]:
            raise ValueError(f"Mask band index {band_index} is out of range for {array.shape[0]}-band Haiti raster.")
        mask_band = array[band_index]
        finite = np.isfinite(mask_band)
        if valid_values is not None:
            valid = np.zeros(mask_band.shape, dtype=bool)
            for value in _as_list(valid_values):
                valid |= np.isclose(mask_band, float(value))
            return (~valid) | (~finite)
        invalid = ~finite
        for value in _as_list(invalid_values):
            invalid |= np.isclose(mask_band, float(value))
        return invalid


def _normalize_mask_band_list(value: Any, expected: int) -> List[int]:
    if value in (None, "", "auto"):
        return [2] * expected
    if not isinstance(value, (list, tuple)):
        return [int(value)] * expected
    if len(value) != expected:
        raise ValueError(f"sar_mask_band_indices must contain {expected} entries, got {len(value)}")
    return [int(item) for item in value]


def _as_list(value: Any) -> List[Any]:
    if value is None or (isinstance(value, str) and value == ""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_haiti_array(
    array: np.ndarray,
    invalid_mask: np.ndarray,
    cfg: Dict[str, Any],
    mask_stats: bool = True,
    masked_input_policy: str = "keep",
) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    data = arr.copy()
    stats = arr.copy()

    nodata_values = cfg.get("nodata_values") or []
    nodata_mask = ~np.isfinite(arr)
    for value in nodata_values:
        if value is not None:
            nodata_mask |= arr == float(value)
    data[nodata_mask] = np.nan
    stats[nodata_mask] = np.nan
    if mask_stats and bool(np.any(invalid_mask)):
        stats[:, invalid_mask] = np.nan

    method = str(cfg.get("method", "percentile_minmax")).lower()
    if method == "standard" and cfg.get("mean") and cfg.get("std"):
        mean = np.asarray(cfg["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(cfg["std"], dtype=np.float32).reshape(-1, 1, 1)
        out = (data - mean) / np.maximum(std, 1e-6)
        return _apply_masked_input_policy(out, invalid_mask, masked_input_policy)

    if bool(cfg.get("log_transform", False)):
        if safe_nanmin(stats) >= 0:
            data = np.log1p(data)
            stats = np.log1p(stats)

    if method in {"fixed_minmax", "static_minmax"}:
        min_value = cfg.get("min_values", cfg.get("min_value", cfg.get("min", 0.0)))
        max_value = cfg.get("max_values", cfg.get("max_value", cfg.get("max", 1.0)))
        low = reshape_channel_value(min_value, data.shape[0])
        high = reshape_channel_value(max_value, data.shape[0])
        out = (np.clip(data, low, high) - low) / np.maximum(high - low, 1e-6)
        return _apply_masked_input_policy(out, invalid_mask, masked_input_policy)

    lower = float(cfg.get("lower_percentile", 2.0))
    upper = float(cfg.get("upper_percentile", 98.0))
    low = safe_channel_nanpercentile(stats, lower)
    high = safe_channel_nanpercentile(stats, upper)
    clipped_data = np.clip(data, low, high)
    clipped_stats = np.clip(stats, low, high)

    if method in {"robust_zscore", "zscore"}:
        mean = safe_channel_nanmean(clipped_stats)
        std = safe_channel_nanstd(clipped_stats)
        out = (clipped_data - mean) / np.maximum(std, 1e-6)
    else:
        out = (clipped_data - low) / np.maximum(high - low, 1e-6)
    return _apply_masked_input_policy(out, invalid_mask, masked_input_policy)


def _apply_masked_input_policy(array: np.ndarray, invalid_mask: np.ndarray, policy: str) -> np.ndarray:
    out = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if not bool(np.any(invalid_mask)):
        return out
    policy = str(policy).lower()
    if policy in {"keep", "raw", "none", "off"}:
        return out
    if policy in {"zero", "zero_after_norm"}:
        out = out.copy()
        out[:, invalid_mask] = 0.0
        return out
    if policy in {"channel_mean", "mean"}:
        out = out.copy()
        valid = ~invalid_mask
        for channel_idx in range(out.shape[0]):
            channel = out[channel_idx]
            channel[invalid_mask] = float(channel[valid].mean()) if bool(valid.any()) else 0.0
        return out
    raise ValueError(f"Unsupported masked_input_policy: {policy}")


def _resize_bool_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    resized = resize_label(mask.long(), size)
    return resized.bool()
