# Comparison models

Six original baselines (ICIF-Net, DMINet, HFA-PANet, WaveHFG, HRSICD, HAFF) run on all four datasets. ChangeOS-R50, DamageFormer and ChangeMamba are included on BRIGHT and Haiti. Sources reside in official/; official_adapters.py and damage_adapters.py expose forward(optical, second_modality). extract_logits(output) selects the primary prediction. Required heads and upstream licenses are retained.

Formal main comparisons train from scratch with the same primary-task loss, data, sampler, budget and validation foreground-mIoU selection as DEHCD-Net. Localization/deep-supervision/feature-pair weights are zero. Dual-head structures remain compatible; unsupervised localization has localization_supervised=false and formal localization_f1=null. Diagnostics do not enter localization rankings.

Declared adaptations:

- Learned input stems for three-channel core interfaces, including one-band BRIGHT/CAU SAR and four-band Haiti SAR.
- Core BatchNorm-to-GroupNorm adaptation and configured output classes.
- Shared preprocessing, label/ignore handling, full test split, primary-logit argmax and no TTA.
- FP32 tensors/losses; RTX5090 TF32 math explicitly enabled and recorded.
- HRSICD bilinear input resize to a 64×64 core and raw-logit resize back; this internal resolution restriction is disclosed.
- Probability-to-logit conversion and two-logit binary convention where required.

These are adapted architectures under a common strategy, not official best benchmark reproductions. No hidden model-specific batch fallback, resolution override or pretraining is allowed.

Original sources: [ICIF-Net](https://github.com/ZhengJianwei2/ICIF-Net), [DMINet](https://github.com/ZhengJianwei2/DMINet), [HFA-PANet](https://github.com/TongfeiLiu/HFA-PANet-for-MCD), [WaveHFG](https://github.com/songxy9037/WaveHFG), [HRSICD](https://github.com/Lucky-DW/HRSICD), [HAFF](https://github.com/ImgSciGroup/HAFF).

See [pinned sources/CUDA](UPSTREAM.md), [controlled comparisons](../docs/FAIR_COMPARISONS.md), and [protocol](../docs/EXPERIMENT_PROTOCOL.md).
