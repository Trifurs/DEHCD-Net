from __future__ import annotations

from typing import Any, Dict

from .disaster_dataset import BaseHeterogeneousDisasterDataset


class CAU1Dataset(BaseHeterogeneousDisasterDataset):
    """CAU1 binary flood/change detection dataset.

    The configured inputs are pre-event optical images from `opt` and post-event
    SAR VV images from `vv`; labels in `flood_vv` are binary.
    """

    DATASET_NAME = "CAU1"
    EXPECTED_NUM_CLASSES = 2
    EXPECTED_LABEL_MODES = ("binary_change", "binary", "change")
    DEFAULT_DATASET_CFG: Dict[str, Any] = {
        "type": "cau1",
        "name": "CAU1",
        "splits": ["train", "val", "test"],
        "optical_dir": "opt",
        "sar_dirs": ["vv"],
        "label_dir": "flood_vv",
        "optical_suffix": "",
        "sar_suffixes": [""],
        "label_suffix": "",
        "image_extensions": [".png"],
        "label_mode": "binary_change",
        "label_ignore_values": [],
        "patch_size": 256,
        "train_random_crop": True,
        "positive_crop_prob": 0.80,
        "rare_crop_prob": 0.0,
        "rare_crop_classes": [],
        "eval_full_image": False,
        "align_to_optical": True,
        "return_metadata": False,
    }

    @classmethod
    def normalize_dataset_cfg(cls, dataset_cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        dataset_cfg["label_mode"] = "binary_change"
        dataset_cfg["rare_crop_prob"] = 0.0
        dataset_cfg["rare_crop_classes"] = []
        dataset_cfg["train_random_crop"] = True
        dataset_cfg["positive_crop_prob"] = max(float(dataset_cfg.get("positive_crop_prob", 0.0) or 0.0), 0.80)
        return dataset_cfg
