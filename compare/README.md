# Comparison Models

This folder now keeps the official model code from each comparison repository
under `compare/official/<model_name>/`. The project registry uses thin wrappers
from `official_adapters.py` so every model exposes the same
`forward(optical, sar) -> logits` interface as DEHCD-Net.

Only files required by the registered model entrypoints are retained. Broken or
unrelated repository utilities, such as HAFF's unused `ACnet.py` and WaveHFG's
standalone complexity script, are intentionally excluded from the runnable
project tree.

The wrappers do project-level adaptation so the official cores can train on the
three heterogeneous datasets rather than just receiving a raw RGB/RGB-style
call:

- learn a small optical/SAR pseudo-RGB stem for 1-channel CAU SAR, 1-channel
  BRIGHT SAR, and 4-channel Haiti SAR stacks;
- replace BatchNorm in the official cores with GroupNorm for small-batch
  heterogeneous training;
- keep official multi-output heads as `aux_logits` so ICIF-Net, DMINet,
  HFA-PANet, and HAFF get deep supervision through the shared loss;
- keep HFA-PANet feature pairs for the optional lightweight modality-alignment
  loss enabled in its comparison XML files;
- replace final prediction heads where needed so multiclass datasets use native
  `num_classes` logits;
- run the official model core in FP32 so older paper-code operators remain safe under project-level AMP;
- resize HRSICD to its native 64x64 resolution, train it with raw logits, and
  upsample logits back;
- convert one-channel binary outputs to the project's two-logit convention;
- skip missing local pretrained weight files instead of hard failing.

Source repositories:

- https://github.com/ZhengJianwei2/ICIF-Net
- https://github.com/ZhengJianwei2/DMINet
- https://github.com/TongfeiLiu/HFA-PANet-for-MCD
- https://github.com/songxy9037/WaveHFG
- https://github.com/Lucky-DW/HRSICD
- https://github.com/ImgSciGroup/HAFF

Losses, sampling, metrics, logging, checkpointing, and dataset preprocessing stay
in the shared project pipeline so BRIGHT1, Haiti1, and CAU1 remain comparable.
