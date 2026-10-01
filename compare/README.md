# Comparison Models

This folder now keeps the official model code from each comparison repository
under `compare/official/<model_name>/`. The project registry uses thin wrappers
from `official_adapters.py` and `damage_adapters.py`. All models expose
`forward(optical, sar)`; use `extract_logits(output)` for the primary prediction.
The new damage baselines also return localization logits. The controlled main comparison
uses the same primary-only objective for every model; separate auxiliary/native
head controls supervise localization. See [controlled comparisons](../docs/FAIR_COMPARISONS.md).

Only files required by the registered model entrypoints are retained. Broken or
unrelated repository utilities, such as HAFF's unused `ACnet.py` and WaveHFG's
standalone complexity script, are intentionally excluded from the runnable
project tree.

The wrappers do project-level adaptation so the official cores can train on the
supported disaster-mapping datasets rather than just receiving a raw RGB/RGB-style
call:

- learn a small optical/SAR pseudo-RGB stem for 1-channel CAU SAR, 1-channel
  BRIGHT SAR, and 4-channel Haiti SAR stacks;
- replace BatchNorm with GroupNorm in the main adapted baseline protocol;
  `configs/experiments/adapter/` contains original-BN controls for all four datasets;
- optionally expose multi-output heads as `aux_logits`; experiment main configurations
  disable optional deep supervision for the six legacy models; the new damage
  models retain their localization head, with its auxiliary weight explicitly zero
  in the main table and one in separate `auxiliary/` controls;
- retain HFA-PANet feature pairs for optional modality-alignment loss; it is disabled
  in the experiment main configurations;
- replace final prediction heads where needed so multiclass datasets use native
  `num_classes` logits;
- run official cores in FP32; all generated experiment configurations also disable AMP for DEHCD-Net, so the comparison uses one precision policy;
- resize HRSICD to its native 64x64 resolution, train it with raw logits, and
  upsample logits back;
- convert one-channel binary outputs to the project's two-logit convention;
- support explicit local encoder checkpoints outside the controlled scratch
  protocol; controlled experiments reject pretrained weights in the common-recipe comparison.

Source repositories:

- https://github.com/ZhengJianwei2/ICIF-Net
- https://github.com/ZhengJianwei2/DMINet
- https://github.com/TongfeiLiu/HFA-PANet-for-MCD
- https://github.com/songxy9037/WaveHFG
- https://github.com/Lucky-DW/HRSICD
- https://github.com/ImgSciGroup/HAFF

Losses, sampling, metrics, logging, checkpointing, and dataset preprocessing stay
in the shared project pipeline so BRIGHT, Haiti, CAU-Flood, and xBD remain comparable.

These are **adapted baselines**, not reproductions of the official training recipes.
Input stems, changed heads, normalization, resizing, precision and losses must be
disclosed with the results. **ChangeOS-R50, DamageFormer and ChangeMamba are now
registered.** See [pinned sources, variants, head recipes and CUDA setup](UPSTREAM.md)
and [experiment protocol](../docs/EXPERIMENT_PROTOCOL.md).
