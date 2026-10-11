from __future__ import annotations

import re
from typing import Any, Dict

from .disaster_dataset import BaseHeterogeneousDisasterDataset


class EventTransferDataset(BaseHeterogeneousDisasterDataset):
    """Event-disjoint four-class optical-SAR transfer dataset."""

    DATASET_NAME = "EventTransferEarthquake"
    EXPECTED_NUM_CLASSES = 4
    EXPECTED_LABEL_MODES = ("multiclass_damage", "multiclass", "damage_levels")
    EVENT_PATTERN = re.compile(r"^(?P<event>.+?)_(?P<scene>\d{8})(?:_.*)?$")
    GROUP_PATTERN = re.compile(r"^(?P<group>.+?_\d{8})_r\d+_c\d+$")
    DEFAULT_DATASET_CFG: Dict[str, Any] = {
        "type": "event_transfer",
        "name": "EventTransferEarthquake",
        "splits": ["train", "val", "test"],
        "optical_dir": "pre-event",
        "sar_dirs": ["post-event"],
        "label_dir": "target",
        "optical_suffix": "_pre_disaster.tif",
        "sar_suffixes": ["_post_disaster.tif"],
        "label_suffix": "_building_damage.tif",
        "image_extensions": [".tif", ".tiff"],
        "source_label_mode": "damage_levels",
        "label_mode": "multiclass_damage",
        "patch_size": 256,
        "train_random_crop": True,
        "positive_crop_prob": 0.70,
        "rare_crop_prob": 0.75,
        "rare_crop_classes": [2, 3],
        "crop_candidate_count": 4,
        "eval_full_image": False,
        "align_to_optical": True,
        "return_metadata": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        event_filter = self.dataset_cfg.get("event_filter")
        if not event_filter:
            return
        if self.split != "test":
            raise ValueError("event_filter is only valid for the test split")
        original_samples = list(self.samples)
        selected_indices = [
            index for index, sample in enumerate(original_samples)
            if sample.get("event") == event_filter
        ]
        if not selected_indices:
            available = sorted({sample.get("event") for sample in original_samples})
            raise ValueError(f"No samples found for event '{event_filter}'. Available events: {available}")
        self.samples = [original_samples[index] for index in selected_indices]
        if getattr(self, "_prepared_split", None) is not None:
            self._prepared_sample_indices = selected_indices
            # The prepared identity covers the full test split; this filtered
            # evaluation must publish the identity of the selected event only.
            self._prepared_identities = {}

    def _build_index(self):
        samples = super()._build_index()
        for sample in samples:
            event_match = self.EVENT_PATTERN.match(sample["id"])
            group_match = self.GROUP_PATTERN.match(sample["id"])
            sample["event"] = event_match.group("event") if event_match else sample["id"]
            sample["group"] = group_match.group("group") if group_match else sample["id"]
        return samples
