# DEHCD-Net

PyTorch implementation of **Selective Difference Learning for Multi-Class Disaster Mapping from Pre-Disaster Optical and Post-Disaster SAR Imagery**.

DEHCD-Net combines HOG guidance, difference-aware multimodal fusion (DPM), bidirectional cross-scale fusion (BiCSF) and iterative refinement (IRB). S/M/L variants share the same training and evaluation interface. The repository includes comparison models, component and capacity controls, sensitivity experiments, event transfer and supervised few-shot adaptation.

## Installation

Use Python 3.10 or newer. Install a matching `torch`/`torchvision` pair for your device, then the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The verified workstation uses Python 3.10, PyTorch 2.11.0, torchvision 0.26.0 and CUDA 13.0. The requirements list direct project dependencies; the PyTorch installation supplies its own CUDA runtime dependencies. No environment upgrade is needed when using an existing working environment.

ChangeMamba uses a reference scan on CPU. CUDA training requires the bundled selective-scan extension, built against the same PyTorch environment:

```bash
python -m pip install -r requirements-baselines.txt
python -m pip install --no-build-isolation ./compare/kernels/selective_scan
```

If `nvcc` is outside the default location, set `CUDA_HOME` to the toolkit root before building. See [baseline sources and CUDA build](compare/UPSTREAM.md). `requirements-optional.txt` adds W&B and dataset-conversion utilities.

## Data

The main launcher accepts a common parent directory containing:

```text
data/
  BRIGHT1/{train,val,test}/{pre-event,post-event,target}/
  Haiti1/{train,val,test}/{Pre_event,Post_event,Annotations}/
  xBD1/{train,val,test}/
  CAU1/{train,val,test}/{opt,vv,flood_vv}/
```

Exact filenames, sensor bands, quality masks and label definitions are in [dataset configurations](configs/datasets/) and [data preparation](docs/DATA.md). Datasets and weights are supplied separately. BRIGHT preparation groups patches from the same source scene and rejects duplicate sample IDs across splits.

## Train and evaluate

Run from the project root:

```bash
python -W ignore tools/run_all.py --data-base /path/to/Hete_CD --output /path/to/results --device cuda:0 --runtime-profile rtx5090 --preflight-only
python -W ignore tools/run_all.py --data-base /path/to/Hete_CD --output /path/to/results --device cuda:0 --runtime-profile rtx5090
```

The main catalog has **80 unique configurations** and seeds **42, 1051, 2060**, giving **240 tasks**. It finishes all configurations for one seed before moving to the next. Each seed uses **BRIGHT → Haiti → xBD → CAU-Flood**, keeping xBD and CAU-Flood last. Verified completed results are reused; compatible interrupted training resumes from `last.pth`.

Early-stop patience is **30 validation checks** for BRIGHT, xBD and CAU-Flood, and **100** for Haiti. Validation runs every epoch. The best checkpoint is selected by strict maximum validation foreground mIoU; the early-stop minimum delta is tracked separately. Progress shows current and total progress, ETA, best validation score and stopping counters.

Public defaults use `data/`, `runs/campaign`, automatic device selection and no hardware profile. To keep workstation paths out of version control, copy `configs/local.example.json` to `local/settings.json` and edit it. Environment variables override that file; command-line options override both. The supported variables are `DEHCD_DATA_BASE`, `DEHCD_OUTPUT`, `DEHCD_EVENT_DATA_ROOT`, `DEHCD_EVENT_OUTPUT`, `DEHCD_DEVICE` and `DEHCD_RUNTIME_PROFILE`.

After configuring these paths, the full command is:

```bash
python -W ignore tools/run_all.py
```

See [workflow and output layout](docs/RUN_WORKFLOW.md), [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [controlled comparisons](docs/FAIR_COMPARISONS.md), [runtime profile](docs/RUNTIME_PROFILE.md) and the [generated catalog](docs/EXPERIMENT_CATALOG.md). `--audit-only` is read-only; `--dry-run` writes a plan; `--preflight-only` checks data and evidence without training.

## Models and comparisons

| Variant | Channels | Stage depths | Expansion |
| --- | --- | --- | ---: |
| S | 16,32,64,96 | 1,1,2,1 | 2 |
| M | 24,48,96,144 | 1,2,3,2 | 2 |
| L | 32,64,128,192 | 2,2,4,2 | 3 |

The two modality stems and levels 0–1 are independent; levels 2–3 share parameter objects. HOG modulates the first two levels by default with six orientation bins. DPM combines bounded flow alignment, modality weighting and difference gating. BiCSF contains GCBM and bidirectional WASM. IRB reuses one denoiser for three iterations; its bounded residual does not independently bound learned step scalars.

| Dataset | Inputs | Classes | Patch | Configurations |
| --- | --- | ---: | ---: | ---: |
| BRIGHT | Optical/SAR | 4 | 256 | 39 |
| Haiti | Optical/SAR | 4 | 128 | 23 |
| xBD | Optical/Optical | 5 | 256 | 9 |
| CAU-Flood | Optical/SAR | 2 | 256 | 9 |

ICIF-Net, DMINet, HFA-PANet, WaveHFG, HRSICD and HAFF cover all four datasets. BRIGHT and Haiti also include ChangeOS-R50, DamageFormer and ChangeMamba. These are adapted architectures under a common scratch-training recipe. Data, supervision, sampling, optimizer, budget, precision and inference rules are held common within each dataset. Baseline-specific internal restrictions are described in [comparison models](compare/README.md).

## Event transfer and few-shot adaptation

```bash
python -W ignore tools/run_event_transfer.py --data-root /path/to/event_data --output /path/to/event_results --seeds 42 1051 2060
python -W ignore tools/run_transfer_study.py --protocols fewshot --sizes m --events noto-earthquake --seeds 42 1051 2060 --source-root /path/to/event_results --source-seed 3402 --output /path/to/fewshot_results
```

The second command requires the specified source checkpoint. It compares target-only scratch training, weight-initialized fine-tuning and a fixed zero-shot reference on the same held-out scenes. Target training and validation labels are reported separately. A fixed source checkpoint is not counted as multiple independent zero-shot runs. See [event transfer](docs/event_transfer.md) and [transfer-study protocols](configs/event_transfer/README.md).

## Results

Each run retains resolved configuration, data/source fingerprints, training history, best/last checkpoints and full test evidence. Summaries report per-seed metrics, mean, sample SD (`ddof=1`), n and completeness. Missing seeds remain missing. P/R/F1 are foreground classwise aggregates; mIoU includes background and FmIoU excludes it. Metric relationships are derived from raw confusion counts.

Evaluation, feature export and efficiency profiling reuse saved checkpoints. Latency and throughput require matched device, input size, batch, precision and scan backend. Partial operation estimates are not complete FLOPs. Prediction panels and feature responses require real inference outputs.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/check_source.py
```

Public source includes code, configurations, tests, documentation and retained third-party licenses. Workstation settings, data, results, weights, caches and editor files are ignored by Git. [Validation commands](docs/VALIDATION.md) describe CPU and CUDA checks.

## Citation and licenses

```bibtex
@misc{liu2026dehcdnet,
  title = {Selective Difference Learning for Multi-Class Disaster Mapping from Pre-Disaster Optical and Post-Disaster SAR Imagery},
  author = {Liu, Bo and Li, Deren and Xiao, Xiongwu and Shao, Zhenfeng and Li, Yingbing and Duan, Yueming and Luo, Zheng},
  year = {2026}
}
```

A project-wide license has not been specified. Included third-party implementations retain their source notices and licenses; see [baseline sources](compare/UPSTREAM.md).
