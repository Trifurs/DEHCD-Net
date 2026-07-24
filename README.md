# DEHCD-Net

This repository provides the implementation of **DEHCD-Net** for the paper
**Difference-Enhanced Optical-SAR Heterogeneous Change Detection for Multi-Class Disaster Mapping**.

DEHCD-Net targets rapid disaster mapping from asymmetric observations, especially
pre-disaster optical imagery and post-disaster SAR imagery. The network is built
around explicit difference enhancement: modality-induced pseudo-differences are
suppressed, while disaster-induced structural and semantic changes are preserved
for binary and multi-class prediction.

## Highlights

- Optical-SAR heterogeneous change detection for disaster response.
- Unified support for binary affected-area detection and multi-class disaster mapping.
- HOG structural-prior modulation for stable cross-modal geometric cues.
- Difference Perception Module (DPM) for alignment and difference-aware fusion.
- Bidirectional Cross-Scale Fusion (BiCSF) for multi-scale feature interaction.
- Iterative Refinement Block (IRB) for semantic correction at the bottleneck.
- Compact S/M/L variants using the same training, evaluation, and inference tools.

## Method Overview

Given a pre-event image and a post-event image, DEHCD-Net predicts a pixel-level
change or damage map. The implementation contains:

- dual modality encoders for hierarchical optical and post-event feature extraction;
- shallow HOG feature modulation to introduce structural priors;
- difference-aware fusion with bounded alignment and gated change evidence;
- global-context guided cross-scale fusion;
- iterative residual refinement before decoding;
- a task-specific decoder and segmentation head.

The provided model variants are:

| Variant | Backbone name | Typical use |
| --- | --- | --- |
| DEHCD-Net-S | `dehcd_s` | Fast debugging and ablation |
| DEHCD-Net-M | `dehcd_m` | Balanced experiments |
| DEHCD-Net-L | `dehcd_l` | Main reported setting |

## Datasets

The code supports the main optical-SAR multi-class experiments and auxiliary
generalization experiments used by the project.

| Dataset | Setting | Task | Modalities | Classes | Patch size |
| --- | --- | --- | --- | --- | --- |
| BRIGHT | Primary | Building damage mapping | Optical + SAR | 4 | 256 |
| Haiti | Primary | Landslide mapping | Optical + SAR | 4 | 128 |
| CAU-Flood | Auxiliary | Flood extraction | Optical + SAR | 2 | 256 |
| xBD | Auxiliary | Building damage mapping | Optical + Optical | 5 | 256 |

Dataset roots in the XML files are relative placeholders such as `data/BRIGHT`.
Edit the corresponding file under `configs/datasets/` to match your prepared data.

## Repository Structure

```text
configs/
  base.xml              Shared model, optimization, logging, and inference defaults
  config.xml            Default experiment reference
  datasets/             Dataset-specific task and preprocessing settings
  dehcd/                S/M/L DEHCD-Net experiment configs
datasets/               Dataset loaders
models/                 DEHCD-Net, backbones, encoders, and fusion modules
compare/                Comparison-model wrappers
tools/                  Training, testing, inference, and dataset utilities
utils/                  Config parsing, losses, metrics, logging, and raster I/O
```

## Installation

Create an environment with Python 3.10 or later, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

For machines with a different CUDA or CPU-only setup, install the matching PyTorch
build first, then install the remaining dependencies.

## Configuration

The XML configuration tree is intentionally compact:

- `configs/base.xml` stores shared defaults.
- `configs/datasets/*.xml` stores dataset, task, normalization, loss, and sampling settings.
- `configs/dehcd/*.xml` stores only the dataset reference, model size, and run name.

Example experiment files:

```text
configs/dehcd/bright_l.xml
configs/dehcd/haiti_l.xml
configs/dehcd/cau_flood_l.xml
configs/dehcd/xbd_m.xml
```

`configs/config.xml` points to the default experiment.

## Training

Train with the default config:

```bash
python tools/train.py
```

Train a specific experiment:

```bash
python tools/train.py --config configs/dehcd/bright_l.xml
python tools/train.py --config configs/dehcd/haiti_l.xml
python tools/train.py --config configs/dehcd/cau_flood_l.xml
python tools/train.py --config configs/dehcd/xbd_m.xml
```

Run a short debugging job:

```bash
python tools/train.py --config configs/dehcd/bright_s.xml --epochs 1 --max-train-batches 20 --max-val-batches 5
```

## Evaluation and Inference

Evaluate a checkpoint:

```bash
python tools/evaluate.py --config configs/dehcd/bright_l.xml --checkpoint <checkpoint.pth> --split test
```

Run testing and export predictions:

```bash
python tools/test.py --train-root runs/train --runs <run_name> --checkpoint best --split test --save-predictions
```

Run inference on a split:

```bash
python tools/infer.py --config configs/dehcd/bright_l.xml --checkpoint <checkpoint.pth> --split test
```

## Dataset Utilities

Inspect a configured dataset:

```bash
python tools/explore_data.py --config configs/dehcd/bright_l.xml --max-samples 12
```

Audit label values:

```bash
python tools/audit_dataset_labels.py --configs configs/dehcd/bright_l.xml --splits train
```

Prepare supported dataset layouts:

```bash
python tools/dataset_tools/bright_split.py --src-root data/raw/BRIGHT --dst-root data/BRIGHT
python tools/dataset_tools/bright_crop_1024_to_256.py --root data/BRIGHT --replace-root
python tools/dataset_tools/cau_split.py --src-root data/raw/CAU-Flood --dst-root data/CAU-Flood
python tools/dataset_tools/xbd_split_crop_1024_to_256.py --src-root data/raw/xBD --dst-root data/xBD
```

## Metrics

The project reports:

- `OA`: overall accuracy.
- `P`: foreground precision.
- `R`: foreground recall.
- `F1`: foreground F1 score.
- `mIoU`: mean IoU over all classes.
- `FmIoU`: foreground mean IoU over non-background classes.

For binary change detection, `FmIoU` is the IoU of the foreground change class.
For multi-class disaster mapping, it is the mean IoU over foreground disaster classes.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@misc{liu2026dehcdnet,
  title  = {Difference-Enhanced Optical-SAR Heterogeneous Change Detection for Multi-Class Disaster Mapping},
  author = {Liu, Bo and Li, Deren and Xiao, Xiongwu and Shao, Zhenfeng and Li, Yingbing and Duan, Yueming and Luo, Zheng},
  year   = {2026}
}
```

## License

Please check the project license and the licenses of included comparison-model
implementations before redistribution or commercial use.
