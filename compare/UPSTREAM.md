# New building-damage baselines: sources and execution contract

These are trainable ports of the authors' public architectures with explicitly declared DEHCD-Net input/task adaptations. They are not a claim to reproduce a published benchmark score. No network access or pretrained weight download occurs when constructing a model.

| Registry name | Pinned source | Architecture retained |
|---|---|---|
| `changeos` | [Z-Zheng/pytorch-change-models](https://github.com/Z-Zheng/pytorch-change-models/tree/190b07a2eb4c389a5dc3f72802c67f428cdfd6b7), `torchange/models/changeos.py` and `cos_r50.py`; [EVER](https://github.com/Z-Zheng/ever/tree/edfe743fb1ebba5fa7baaedbacba37f2ee961ca3) FPN, asymmetric decoder, SE and deep heads | Shared ResNet-50 (torchvision, matching the standard EVER ResNet stride layout), separate localization/damage FPN decoders, residual SE fusion, both deep heads |
| `damageformer` | [BRIGHT benchmark](https://github.com/ChenHongruixuan/BRIGHT/tree/59269142f3a3550320513e362692732f46486985/bda_benchmark), `model/DamageFormer.py` | Swin-T [2,2,6,2], four-level concatenation, three residual fusion blocks, localization/damage heads |
| `changemamba` | [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba/tree/9ce9cec13f9ea14bc0ad91f071577ec9b3a97983) | Published Tiny configuration: dims 96, depths [2,2,4,2], state 1, ratio 2, v3noz; original semantic/change decoders |

ChangeOS's separate `Z-Zheng/ChangeOS` package supplies released TorchScript inference models rather than the trainable core. This port uses the author's maintained TorChange training implementation. The registry's ChangeOS variant is **R50 with deep heads**, not every ChangeOS variant.

The formal BRIGHT/Haiti configurations use ChangeMamba **MMBDA** (two encoders and the BRIGHT concatenation decoder) and two-encoder DamageFormer. The retained implementation can construct BDA/shared-encoder variants, but these are not additional formal tasks. Parameter counts refer to the declared variant, input and class count.

The CUDA reverse-scan header replaces removed CCCL 3 `cub::LaneId`/`cub::CTA_SYNC` calls with the equivalent linear-thread warp-lane index and native `__syncthreads`; the recurrence and scan algebra are unchanged.

Original files and their SHA256 values are listed in `upstream_manifest.json`; modifications are marked in the relevant files. Licenses are retained alongside each source tree. The manifest hashes refer to upstream files before adaptation, not to the final modified file. Changes to the TorchChange/EVER extracted modules are described by their file headers and this document.

## Interface and supervision

`forward(optical, second_modality)` returns primary damage logits and two-class localization logits in both train and eval modes. ChangeOS converts its one-channel localization logit z to [0,z], preserving the sigmoid probability. Learned three-channel input stems support the project sensors; padding to multiples of 32 is cropped back to the original extent.

The formal main comparison trains from scratch with the same primary-task loss, data, sampling, budget and optimizer/schedule as DEHCD-Net on each dataset. Core BN is adapted to GN. Localization, deep-supervision and feature-pair auxiliary weights are **zero**, preserving dual-head structure/checkpoint compatibility without adding supervision. Localization targets used for diagnostics are label>0 with ignore=255; this is not a separate pre-event building-mask benchmark. Formal localization_f1 is null when localization_supervised=false; any diagnostic score is excluded from localization ranking.

The retained heads remain part of the original architecture. Unused classification layers were removed from DamageFormer, while required model heads remain. Core gradients and selective-scan behavior are tested. Package-local VMamba imports do not require optional profilers or Triton for the retained v3 path.

Shared losses run in FP32 and handle ignored/all-ignore labels. Full-test primary-logit argmax with no TTA is the formal prediction rule in test/evaluate/infer. Conditional damage harmonic F1, foreground mIoU and full-map mIoU have different denominators and must not be interchanged. No postprocessing rule is selected after inspecting test scores.

All formal configurations reject encoder pretraining; test/inference use a full trained task checkpoint with its saved configuration. Source hashes, initialization, classes, normalization and scan backend are recorded. Published upstream recipes remain source context, not runnable extra comparison suites in this collection.

## ChangeMamba CUDA scan

The reference path implements the same selective state-space recurrence, projections and four spatial directions with ordinary autograd. It is not a convolution substitute. It is intended for correctness checks and portability; its speed must not be quoted as the fused model's speed.

- `selective_scan_backend=auto`: reference on CPU, official `selective_scan_cuda_oflex` on CUDA; fails clearly if the CUDA extension is absent.
- `torch`: explicit reference recurrence on either device.
- `cuda`: require the official compiled extension.

Build the bundled extension **in the same Python/PyTorch environment used for training**, with a matching CUDA toolkit/nvcc and C++ compiler:

```bash
python -m pip install ninja
python -m pip install --no-build-isolation ./compare/kernels/selective_scan
python -m unittest discover -s tests -p 'test_damage_baselines.py' -v
```

The builder compiles only the used `oflex` variant. It uses PyTorch's visible-GPU architecture detection or `TORCH_CUDA_ARCH_LIST`, rather than upstream hard-coded sm_70/80/90 flags. A CUDA runtime bundled with PyTorch is not by itself a CUDA compiler; see the [NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). CUDA tests compare the extension's outputs **and gradients** with the reference. The test is explicitly skipped when the device/extension is absent.

Local validation (2026-09-29) built and installed the extension in `hlcd` for Python 3.10 / PyTorch 2.11+cu130 / RTX 5090 (sm_120), using CUDA 13.0.88 and GCC 13.4. All forward/backward equivalence tests, including lengths 9, 257 and 1025, passed. This is an environment-specific build; other Torch/CUDA/GPU combinations should rebuild from source.

All generated experiment overlays set `training.amp=false`, including DEHCD-Net, because the nine baseline cores enforce FP32. Their recorded metadata exposes that policy. The benchmark rejects `--amp` for fixed-FP32 cores instead of presenting a partial-autocast run as AMP efficiency; compare all models with its default FP32 mode.

The original module-hook complexity counter misses attention, functional projections and selective scans. Its output is now explicitly a partial operation estimate and total FLOPs fields are null. Use `tools/benchmark.py` on the same GPU, shape, precision, batch size and backend; it records backend identity and measured latency/memory.
