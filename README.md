# DEHCD-Net

Official implementation of **DEHCD-Net**: **Difference-Enhanced Heterogeneous Change Detection Network** for disaster-scene optical-SAR heterogeneous change detection.

DEHCD-Net is designed for asymmetric disaster observations, where pre-event optical imagery provides structural and semantic reference information and post-event SAR imagery provides all-weather disaster response observations. The model focuses on suppressing cross-modal pseudo-differences while enhancing disaster-induced structural, semantic, and multi-scale changes.

## Highlights

- Difference-enhanced framework for optical-SAR heterogeneous change detection.
- HOG-based structural prior modulation for shallow edge and contour cues.
- Difference-aware heterogeneous fusion with bounded SAR-to-optical alignment and gated change evidence.
- Bidirectional cross-scale fusion for multi-level disaster evidence propagation.
- Lightweight iterative bottleneck refinement for stable high-level change semantics.
- Unified training, evaluation, inference, and comparison pipeline for binary and multi-class disaster change detection.

## Method

Given a pre-event optical image and a post-event SAR image, DEHCD-Net predicts a pixel-level change or damage map. The network contains:

- **Dual encoders** for optical and SAR feature extraction.
- **Structural prior modulation** using HOG-like orientation histograms.
- **Difference perception and heterogeneous fusion** for bounded alignment, shared-reference construction, and difference gating.
- **Bidirectional cross-scale fusion** for combining boundary details, semantic context, and spatial connectivity.
- **Iterative refinement** for deterministic residual correction of bottleneck features.
- **Decoder and prediction head** for full-resolution output.

The main model variants are:

- `DEHCD-Net-S`
- `DEHCD-Net-M`
- `DEHCD-Net-L`

## Repository Structure

```text
configs/      XML experiment configurations
datasets/     Dataset loaders and preprocessing utilities
models/       DEHCD-Net, backbones, fusion modules, and model builder
compare/      Adapted comparison models
tools/        Training, evaluation, inference, visualization, and analysis scripts
utils/        Config parsing, losses, metrics, logging, checkpoints, and raster I/O
```

## Installation

Create a Python environment and install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Data Preparation

The code supports the following disaster optical-SAR change detection datasets:

| Dataset | Task | Optical input | SAR input | Output |
| --- | --- | --- | --- | --- |
| BRIGHT | Building damage assessment | Pre-event optical | Post-event SAR | Multi-class damage map |
| CAU-Flood | Flood extraction | Pre-event optical | Post-event SAR VV | Binary flood map |
| Haiti | Landslide change detection | Pre-event optical | Post-event SAR | Multi-class landslide map |

Prepare each dataset according to its official release format, then set the dataset root and split-specific options in the corresponding XML configuration file.

A typical paired dataset layout is:

```text
dataset_root/
  train/
    pre-event/
    post-event/
    target/
  val/
    pre-event/
    post-event/
    target/
  test/
    pre-event/
    post-event/
    target/
```

Dataset-specific loaders handle channel selection, normalization, label sanitization, valid-mask handling, cropping, and augmentation.

## Training

Train with any XML configuration:

```bash
python tools/train.py --config <config.xml>
```

Short debugging run:

```bash
python tools/train.py --config <config.xml> --epochs 1 --max-train-batches 20 --max-val-batches 5
```

Smoke test:

```bash
python tools/smoke_test.py --config <config.xml> --batch-size 2 --backward
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python tools/evaluate.py --config <config.xml> --checkpoint <checkpoint.pth> --split test
```

Run inference and export visualizations:

```bash
python tools/infer.py --config <config.xml> --checkpoint <checkpoint.pth> --split test --max-samples 8
```

## Comparison Experiments

The repository includes wrappers for several representative comparison models:

- `ICIF-Net`
- `DMINet`
- `HFA-PANet`
- `WaveHFG`
- `HRSICD`
- `HAFF`

Run a comparison experiment with:

```bash
python tools/run_compare_matrix.py --dataset <dataset> --model <model>
```

Use the dry-run option to inspect scheduled commands:

```bash
python tools/run_compare_matrix.py --dry-run
```

## Metrics

The project reports confusion-matrix based metrics:

- `OA`: overall accuracy.
- `P`: foreground precision.
- `R`: foreground recall.
- `F1`: foreground F1 score.
- `mIoU`: mean intersection over union over all classes.
- `FmIoU`: foreground mean IoU over non-background classes.

For binary change detection, `FmIoU` is equivalent to the IoU of the change class. For multi-class damage or landslide mapping, `FmIoU` better reflects performance on disaster-related foreground categories.

## Outputs

Training outputs are saved under:

```text
runs/train/
```

Prediction outputs are saved under:

```text
runs/predict/
```

Each run stores logs, checkpoints, configuration snapshots, and exported prediction results when applicable.

## Citation

If this repository is useful for your research, please cite the corresponding paper:

```bibtex
To be continued...
```

## License

Please refer to the project license and the licenses of the included comparison-model implementations before redistribution or commercial use.
