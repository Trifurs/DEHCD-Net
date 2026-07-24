from __future__ import annotations

from typing import Any, Dict

from .disaster_dataset import BaseHeterogeneousDisasterDataset


class XBDDataset(BaseHeterogeneousDisasterDataset):
    """xBD post-disaster building-damage dataset.

    The prepared xBD1 layout mirrors the shared pre/post/target structure used
    by the other disaster datasets. Unlike BRIGHT/Haiti, the second branch is a
    post-disaster Optical image rather than SAR; it is still passed through the
    shared ``sar`` tensor slot so the common training, testing, and comparison
    model code can be reused without branching.

    Label convention:
    0 background, 1 no-damage building, 2 minor damage, 3 major damage,
    4 destroyed. xBD's un-classified label value 5 is ignored by default.
    """

    DATASET_NAME = "xBD1"
    EXPECTED_NUM_CLASSES = 5
    EXPECTED_LABEL_MODES = ("multiclass_damage", "multiclass", "damage_levels")
    DEFAULT_DATASET_CFG: Dict[str, Any] = {
        "type": "xbd1",
        "name": "xBD1",
        "splits": ["train", "val", "test"],
        "optical_dir": "pre-event",
        "sar_dirs": ["post-event"],
        "label_dir": "target",
        "optical_suffix": "_pre_disaster.png",
        "sar_suffixes": ["_post_disaster.png"],
        "label_suffix": "_building_damage.png",
        "image_extensions": [".png"],
        "source_label_mode": "xbd_damage_levels",
        "label_mode": "multiclass_damage",
        "label_ignore_values": [5],
        "class_names": ["background", "no damage", "minor damage", "major damage", "destroyed"],
        "second_modality_name": "Post-event Optical",
        "second_modality_rgb": True,
        "patch_size": 256,
        "train_random_crop": True,
        "positive_crop_prob": 0.70,
        "rare_crop_prob": 0.75,
        "rare_crop_classes": [3, 4],
        "crop_candidate_count": 4,
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
            "method": "percentile_minmax",
            "lower_percentile": 2.0,
            "upper_percentile": 98.0,
        },
    }

    @classmethod
    def normalize_dataset_cfg(cls, dataset_cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        dataset_cfg["label_mode"] = "multiclass_damage"
        dataset_cfg["num_classes"] = int(config.get("task", {}).get("num_classes", cls.EXPECTED_NUM_CLASSES))
        dataset_cfg["label_ignore_values"] = [int(item) for item in dataset_cfg.get("label_ignore_values", [5])]
        dataset_cfg["rare_crop_classes"] = [int(item) for item in dataset_cfg.get("rare_crop_classes", [3, 4])]
        dataset_cfg["positive_crop_prob"] = min(max(float(dataset_cfg.get("positive_crop_prob", 0.0) or 0.0), 0.0), 1.0)
        dataset_cfg["rare_crop_prob"] = min(max(float(dataset_cfg.get("rare_crop_prob", 0.0) or 0.0), 0.0), 1.0)
        dataset_cfg["crop_candidate_count"] = max(int(dataset_cfg.get("crop_candidate_count", 1) or 1), 1)
        dataset_cfg["second_modality_rgb"] = bool(dataset_cfg.get("second_modality_rgb", True))
        return dataset_cfg

    @classmethod
    def validate_config(cls, config: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> None:
        super().validate_config(config, dataset_cfg)
        sar_dirs = list(dataset_cfg.get("sar_dirs") or [])
        if len(sar_dirs) != 1 and not bool(dataset_cfg.get("allow_sar_dirs_override", False)):
            raise ValueError(
                "XBDDataset expects one post-event Optical branch. "
                "Set dataset.allow_sar_dirs_override=true only for an intentional ablation."
            )
