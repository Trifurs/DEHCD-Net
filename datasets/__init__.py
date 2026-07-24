from .bright_dataset import BRIGHTDataset
from .cau_dataset import CAU1Dataset
from .disaster_dataset import BaseHeterogeneousDisasterDataset, HeterogeneousDisasterDataset
from .haiti_dataset import HaitiDataset
from .xbd_dataset import XBDDataset

__all__ = [
    "BaseHeterogeneousDisasterDataset",
    "BRIGHTDataset",
    "CAU1Dataset",
    "HaitiDataset",
    "HeterogeneousDisasterDataset",
    "XBDDataset",
    "build_dataset",
]


DATASET_REGISTRY = {
    "bright": BRIGHTDataset,
    "bright1": BRIGHTDataset,
    "cau": CAU1Dataset,
    "cau1": CAU1Dataset,
    "haiti": HaitiDataset,
    "haiti1": HaitiDataset,
    "xbd": XBDDataset,
    "xbd1": XBDDataset,
    "generic_heterogeneous": HeterogeneousDisasterDataset,
    "heterogeneous_disaster": HeterogeneousDisasterDataset,
}


def build_dataset(config, split: str, training: bool = False):
    dataset_cfg = config.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "bright1")).lower()
    dataset_cls = DATASET_REGISTRY.get(dataset_type)
    if dataset_cls is not None:
        return dataset_cls.from_config(config, split=split, training=training)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
