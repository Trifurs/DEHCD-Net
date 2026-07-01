from __future__ import annotations

from typing import Any, Dict

from .disaster_dataset import BaseHeterogeneousDisasterDataset


class BRIGHTDataset(BaseHeterogeneousDisasterDataset):
    """BRIGHT1 multiclass building-damage change detection dataset.

    Pre-disaster imagery is optical, post-disaster imagery is SAR, and labels
    are damage levels 0/1/2/3. Class 2 and 3 are rare, so the default config
    enables rare-class crop targeting for patch training.
    """

    DATASET_NAME = "BRIGHT1"
    EXPECTED_NUM_CLASSES = 4
    EXPECTED_LABEL_MODES = ("multiclass_damage", "multiclass", "damage_levels")
    DEFAULT_DATASET_CFG: Dict[str, Any] = {
        "type": "bright1",
        "name": "BRIGHT1",
        "splits": ["train", "val", "test"],
        "optical_dir": "pre-event",
        "sar_dirs": ["post-event"],
        "label_dir": "target",
        "optical_suffix": "_pre_disaster.tif",
        "sar_suffixes": ["_post_disaster.tif"],
        "label_suffix": "_building_damage.tif",
        "image_extensions": [".tif", ".tiff", ".png", ".jpg", ".jpeg"],
        "source_label_mode": "damage_levels",
        "label_mode": "multiclass_damage",
        "patch_size": 256,
        "train_random_crop": True,
        "positive_crop_prob": 0.70,
        "rare_crop_prob": 0.75,
        "rare_crop_classes": [2, 3],
        "eval_full_image": False,
        "align_to_optical": True,
        "return_metadata": False,
    }
