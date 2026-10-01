# New building-damage baselines: sources and execution contract

These are trainable ports of the authors' public architectures with explicitly declared DEHCD-Net input/task adaptations. They are not a claim to reproduce a published benchmark score. No network access or pretrained weight download occurs when constructing a model.

| Registry name | Pinned source | Architecture retained |
|---|---|---|
| `changeos` | [Z-Zheng/pytorch-change-models](https://github.com/Z-Zheng/pytorch-change-models/tree/190b07a2eb4c389a5dc3f72802c67f428cdfd6b7), `torchange/models/changeos.py` and `cos_r50.py`; [EVER](https://github.com/Z-Zheng/ever/tree/edfe743fb1ebba5fa7baaedbacba37f2ee961ca3) FPN, asymmetric decoder, SE and deep heads | Shared ResNet-50 (torchvision, matching the standard EVER ResNet stride layout), separate localization/damage FPN decoders, residual SE fusion, both deep heads |
| `damageformer` | [BRIGHT benchmark](https://github.com/ChenHongruixuan/BRIGHT/tree/59269142f3a3550320513e362692732f46486985/bda_benchmark), `model/DamageFormer.py` | Swin-T [2,2,6,2], four-level concatenation, three residual fusion blocks, localization/damage heads |
| `changemamba` | [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba/tree/9ce9cec13f9ea14bc0ad91f071577ec9b3a97983) | Published Tiny configuration: dims 96, depths [2,2,4,2], state 1, ratio 2, v3noz; original semantic/change decoders |

ChangeOS's separate `Z-Zheng/ChangeOS` package supplies released TorchScript inference models rather than the trainable core. This port uses the author's maintained TorChange training implementation. The registry's ChangeOS variant is **R50 with deep heads**, not every ChangeOS variant.

ChangeMamba uses **MMBDA** (two encoders and the BRIGHT concatenation decoder) for optical/SAR, and **BDA** (shared encoder, concatenated/interleaved/sequential temporal fusion) for xBD. DamageFormer uses two encoders for heterogeneous inputs and a shared encoder for xBD. These choices are explicit in XML. Do not pool their parameter counts as one universal architecture.

The CUDA reverse-scan header replaces removed CCCL 3 `cub::LaneId`/`cub::CTA_SYNC` calls with the equivalent linear-thread warp-lane index and native `__syncthreads`; the recurrence and scan algebra are unchanged.

Original files and their SHA256 values are listed in `upstream_manifest.json`; modifications are marked in the relevant files. Licenses are retained alongside each source tree. The manifest hashes refer to upstream files before adaptation, not to the final modified file. Changes to the TorchChange/EVER extracted modules are described by their file headers and this document.

## Interface and supervision

`forward(optical, second_modality)` returns `{"logits": damage_logits, "localization_logits": two_class_localization_logits}` in both train and eval modes. ChangeOS's one-channel localization logit `z` becomes `[0,z]`, preserving its sigmoid probability. Learned 3-channel input stems handle 1/3/4-channel sensors. The decoder input is padded to a multiple of 32 and the outputs are cropped back, without changing normal 256×256 images.

Unused ImageNet classifiers/final unused Swin normalization were removed from DamageFormer. Real core parameters remain trainable; the tests check their backward gradients. VMamba imports are package-local; optional FLOP profilers/Triton cross-scan imports are not required for the published v3 path. The backbone no longer silently ignores a failed or incomplete pretrained checkpoint.

| Suite | Primary/head loss | Normalization and sampling |
|---|---|---|
| `main` | Same configured damage-task loss as DEHCD-Net; localization CE+Dice, weight 1 | Shared project data/epoch budget, FP32 for every model, learned input stems, core BN→GN; new runs initialized from scratch by default |
| `adapter` | Same shared loss | Original core BN; learned input stems still contain GN |
| `head_recipe` | ChangeOS: BCE + Tversky for localization, CE + all-class Dice for damage. DamageFormer/Mamba: CE for both heads + 0.5 localization Lovasz + 0.75 damage Lovasz | Original core BN, uniform sampling/cropping, same project split/normalization/epoch budget |
| `optimizer_recipe` | DamageFormer/Mamba head recipe | Additionally AdamW lr=1e-4, weight_decay=5e-3, constant lr, best validation mean_iou, following the pinned scripts; project split, crop size and epoch budget remain different |

The local labels provide one damage/change raster. Localization supervision is therefore **`label > 0`, ignoring 255**, including on xBD. If reproducing a benchmark with a separate pre-event building mask, that native mask needs a separate data protocol. Binary CAU and Haiti landslide localization are adapted auxiliary tasks, not native building-localization experiments. Optional DEHCD deep supervision is distinct from these necessary dual heads.

FP32 losses mask ignored pixels for both heads. Lovasz uses the whole valid batch as upstream; ChangeOS Dice/Tversky use smooth=1. All-ignore batches return graph-connected zero loss. TTA averages both heads after inverse spatial transforms. Raw damage argmax is the default primary prediction, including BRIGHT's full-map mIoU. Test/evaluate additionally report localization F1 and damage harmonic F1 restricted to labeled foreground, with raw confusion matrices. **Conditional damage F1, foreground mIoU and full-map mIoU are different endpoints.**

`inference.prediction_rule` can explicitly select `localization_gated` or `changeos_object`. Object voting uses eight-connected components; original xBD weights [8,38,25,11] apply only to five-class xBD. Other taxonomies require an explicit weight per foreground class. Decode rules are recorded and cannot be mixed during campaign aggregation/comparison. Do not introduce postprocessing only after observing test scores.

## Pretraining

All released experiment XMLs start from scratch, so external weights cannot silently benefit one model. To run a separately labeled pretrained comparison:

```bash
python tools/run_multiseed.py --experiments bright_damageformer_optimizer_recipe \
  --encoder-checkpoint damageformer=/absolute/path/to/torchvision_swin_t_state_dict.pth \
  --data-root bright=/absolute/path/to/BRIGHT1 \
  --output runs/experiments/damageformer_pretrained --preflight-only
```

Use torchvision ResNet-50 encoder weights for ChangeOS, torchvision Swin-T weights for DamageFormer, and **the matching VMamba Tiny [2,2,4,2]/v3noz encoder** for ChangeMamba. See the linked upstream repositories for weight provenance. Full task checkpoints belong in `--checkpoint`/`--resume`, not `--encoder-checkpoint`. Missing, mismatched or partially loaded encoders fail. SHA256, initialization, classes, core normalization and scan backend are recorded. Test/inference load the full trained checkpoint and do not require the original encoder file to still exist.

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
