# DEHCD-Net

Implementation for **Selective Difference Learning for Multi-Class Disaster Mapping from Pre-Disaster Optical and Post-Disaster SAR Imagery**.

The formal collection contains **80 unique configurations**, each with seeds **42, 1051, 2060**, for **240 target tasks**. Compatible existing results count toward this total; only missing or incompatible work is scheduled. The [generated catalog and figure coverage](docs/EXPERIMENT_CATALOG.md) derive from configuration metadata, including shared references and default sensitivity points.

## Run the local campaign

Use the existing training Python environment. On the configured workstation:

```bash
python -W ignore tools/run_all.py --audit-only
python -W ignore tools/run_all.py --preflight-only
python -W ignore tools/run_all.py
```

The first command only inspects. The second validates and records the plan/data/reuse evidence without training. The third completes the outstanding work; repeat it after interruption to continue. The default schedule finishes all configurations for **seed 42**, then **1051**, then **2060**. Within each seed, dataset order is **BRIGHT → Haiti → xBD → CAU-Flood**. Completed results are skipped; necessary reevaluation and compatible continuation precede new training only within the same seed and dataset. An explicit `--seeds` list sets the seed order.

Defaults are `~/桌面/myData/Hete_CD/{BRIGHT1,Haiti1,xBD1,CAU1}` for data and `~/桌面/myResult/DEHCD-Net` for all results, with `cuda:0` and the RTX5090 profile. `--data-base` and `--output` override paths. Console progress includes total/current progress, provisional ETA, best validation foreground mIoU and epoch, early-stop status and foreground-protection counters. Ordinary early stopping is disabled in formal experiments. Python warnings are hidden; errors remain visible.

```bash
python -W ignore tools/run_all.py --groups main ablation sensitivity scaling --dry-run
python -W ignore tools/compare_results.py --campaign "$HOME/桌面/myResult/DEHCD-Net" --groups main ablation sensitivity scaling --output "$HOME/桌面/myResult/DEHCD-Net/summary/analysis"
```

Selecting overlapping groups does not duplicate a canonical ID/seed task. Reports keep incomplete groups explicitly incomplete. See [commands, resume and output layout](docs/RUN_WORKFLOW.md), [protocol](docs/EXPERIMENT_PROTOCOL.md), [comparison factors](docs/FAIR_COMPARISONS.md), and [hardware settings](docs/RUNTIME_PROFILE.md).

## Network

The custom four-level convolutional backbone uses depthwise local/dilated branches, channel expansion, GRN and context gates; it is not an unnamed external pretrained backbone. The two modality stems and levels 0–1 are independent. Levels 2–3 and their downsampling modules share parameter objects. Both binary and multiclass tasks use the same three-stage decoder structure; only the output class count changes.

| Variant | Channels | Stage depths | Expansion |
| --- | --- | --- | ---: |
| S | 16,32,64,96 | 1,1,2,1 | 2 |
| M | 24,48,96,144 | 1,2,3,2 | 2 |
| L | 32,64,128,192 | 2,2,4,2 | 3 |

HOG means Histogram of Oriented Gradients. Its affine modulation guides the first two levels by default with six bins. DPM combines bounded flow alignment, modality weighting and difference gating. BiCSF contains both Global Context Bridge Modulation (GCBM) and bidirectional Weighted Adjacent-Scale Merge (WASM); WASM gates the **absolute** difference of neighboring features. IRB reuses one denoiser for three iterations by default. `tanh` bounds its residual, while learned step scalars remain unconstrained; it does not bound the full update independently of those scalars.

The architecture aims to select useful heterogeneous evidence. Feature heatmaps or non-additive ablation scores alone do not prove physical separation of sensor effects and true changes or causal synergy.

## Data and experiments

| Dataset | Inputs | Classes | Patch | Unique configurations |
| --- | --- | ---: | ---: | ---: |
| BRIGHT | Optical/SAR | 4 | 256 | 39 |
| Haiti | Optical/SAR | 4 | 128 | 23 |
| xBD | Optical/Optical | 5 | 256 | 9 |
| CAU-Flood | Optical/SAR | 2 | 256 | 9 |

All four datasets retain S/M/L and ICIF-Net, DMINet, HFA-PANet, WaveHFG, HRSICD and HAFF; BRIGHT/Haiti also include ChangeOS-R50, DamageFormer and ChangeMamba. These are adapted architectures under the shared scratch-training strategy, not claimed reproductions of official best benchmark scores. See [baseline adaptations and licenses](compare/README.md) and [pinned sources/CUDA scan](compare/UPSTREAM.md).

Formal comparisons use primary supervision only, fixed dataset-specific budgets, strict best validation `foreground_miou`, full-test primary argmax and no TTA. RTX5090 uses FP32 tensors/losses with TF32 math enabled. Physical batch and accumulation are identical within a dataset; no per-model hidden fallback is allowed. The protocol discloses the BRIGHT compound loss/class weights/sampling and Haiti hierarchical loss. Only BRIGHT receives the additional capacity, DPM-part and recipe controls.

Shared templates live in configs/base.xml, configs/datasets/ and configs/dehcd/. The executable formal definitions live once each in configs/experiments/. Templates are not extra experiment tasks; resolved per-seed snapshots are necessary evidence. `python tools/build_experiment_configs.py` regenerates the formal catalog, groups and usage documentation.

## Results and postprocessing

Each seed retains its snapshot, protocol, history, full best/last checkpoint and complete test evidence. Summaries report raw per-seed metrics, mean, sample SD (ddof=1), n, expected_n=3 and completeness. A shared reference appearing in several figures still has only three independent training repeats. Missing seeds are never filled with zero. Three-seed exact two-sided sign-flip tests cannot produce p<0.05 (minimum 0.25).

OA is overall accuracy. Foreground P/R/F1 are classwise foreground aggregates; mean_iou includes background and foreground_miou excludes it, using classes with a valid union. Raw confusion counts and per-class denominators support confusion analysis. Row-normalized diagonal values are class recall, not OA. Unsupervised localization heads have formal localization_f1=null and are excluded from localization rankings.

Prediction panels, feature responses, confusion matrices, parameter counts and efficiency measurements reuse checkpoints. Display seed and samples are fixed before comparing scores. Partial hook operation counts are not complete FLOPs; efficiency reports use matched-device latency/throughput settings. In-domain results do not establish unseen-event generalization.

## Environment and verification

The existing hlcd environment includes PyTorch and the selective-scan CUDA extension; do not rebuild it for ordinary runs. For a new environment, install a suitable PyTorch build then dependencies from requirements.txt and follow the CUDA build instructions in compare/UPSTREAM.md.

```bash
python -m unittest discover -s tests -v
```

Tests and bounded smoke checks validate execution and evidence handling, not completed scientific results or convergence. See [validation](docs/VALIDATION.md).

## Citation

```bibtex
@misc{liu2026dehcdnet,
  title = {Selective Difference Learning for Multi-Class Disaster Mapping from Pre-Disaster Optical and Post-Disaster SAR Imagery},
  author = {Liu, Bo and Li, Deren and Xiao, Xiongwu and Shao, Zhenfeng and Li, Yingbing and Duan, Yueming and Luo, Zheng},
  year = {2026}
}
```

Check the project license and the retained licenses of included baseline implementations before redistribution or commercial use.
