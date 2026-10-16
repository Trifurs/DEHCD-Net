# DEHCD-Net

PyTorch tools for multimodal change detection and disaster mapping. The code provides DEHCD-Net, comparison-model adapters, configurable training, checkpoint recovery, evaluation, inference and profiling.

## Installation

Use Python 3.10 or newer. Install a matching PyTorch/torchvision build for the target CPU or CUDA device, then:

```bash
python -m pip install -r requirements.txt
```

ChangeMamba CUDA execution requires its selective-scan extension:

```bash
python -m pip install -r requirements-baselines.txt
python -m pip install --no-build-isolation ./compare/kernels/selective_scan
```

Build against the active PyTorch environment and CUDA toolkit. Set `CUDA_HOME` when `nvcc` is outside the default location. See [extension and upstream details](compare/UPSTREAM.md). Optional tracking and conversion dependencies are in `requirements-optional.txt`.

## Configuration and data

XML configurations describe the model, dataset layout, preprocessing, optimization and inference settings. Dataset adapters cover BRIGHT, Haiti, xBD and CAU-Flood. Data and trained weights are supplied separately.

Use CLI paths or copy `configs/local.example.json` to `local/settings.json`. Machine-specific settings are ignored by Git. Environment variables override that file; CLI arguments take precedence. See [configuration](docs/CONFIGURATION.md) and [data preparation](docs/DATA.md).

## Run

From the repository root, inspect a batch plan and then launch it:

```bash
python -W ignore tools/run_all.py --data-base /path/to/data --output /path/to/runs --dry-run
python -W ignore tools/run_all.py --data-base /path/to/data --output /path/to/runs --device cuda:0
```

Use `--groups`, `--datasets` or `--seeds` to select configurations and repetitions. Compatible completed runs are reused. `--preflight-only` validates data and existing evidence; `--prepare-only` prepares reusable inputs. A hardware profile may be selected explicitly with `--runtime-profile`.

For individual tools:

```bash
python tools/train.py --help
python tools/test.py --help
python tools/infer.py --help
python tools/export_features.py --help
python tools/benchmark.py --help
python tools/compare_results.py --help
python tools/run_event_transfer.py --help
python tools/run_transfer_study.py --help
```

[Running tools](docs/RUN_WORKFLOW.md) describes continuation, output files and command options. [Transfer tools](configs/event_transfer/README.md) describes source-weight initialization and scene-disjoint adaptation.

## Model interface

`models.build_model` constructs the configured model. Models accept optical and second-modality tensors; `utils.model_outputs.extract_logits` selects the primary logits. DEHCD-Net exposes HOG guidance, DPM, BiCSF and IRB switches. BiCSF includes both GCBM and bidirectional WASM. Architecture and normalization settings are shared through the configuration parser.

Comparison adapters support ICIF-Net, DMINet, HFA-PANet, WaveHFG, HRSICD, HAFF, ChangeOS, DamageFormer and ChangeMamba. See [adapter interfaces and source notices](compare/README.md).

## Development

```bash
python scripts/check_source.py
python -m unittest discover -s tests -v
```

The source check validates code/configuration syntax and the actual Git candidate paths. Machine settings, data, run outputs, result tables, weights and caches are excluded from publication. Tests create temporary fixtures; CUDA-specific checks require a device and matching extension. See [validation](docs/VALIDATION.md).

## Citation and licenses

```bibtex
@misc{liu2026dehcdnet,
  title = {Selective Difference Learning for Multi-Class Disaster Mapping from Pre-Disaster Optical and Post-Disaster SAR Imagery},
  author = {Liu, Bo and Li, Deren and Xiao, Xiongwu and Shao, Zhenfeng and Li, Yingbing and Duan, Yueming and Luo, Zheng},
  year = {2026}
}
```

A project-wide license has not been specified. Third-party source notices and licenses are retained beside their implementations.
