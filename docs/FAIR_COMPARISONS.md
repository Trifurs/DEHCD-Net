# Controlled comparisons

The 80 canonical configurations use controlled_comparison_v1 and seeds `[42,1051,2060]`. A reference serves several analyses without creating another training job. Resolved model/data/training/execution settings define compatibility; presentation metadata does not.

There are 42 main configurations: S/M/L and six original baselines on every dataset, plus ChangeOS-R50, DamageFormer and ChangeMamba on BRIGHT/Haiti. Splits, bands, masks, normalization, augmentation, sampling, crops, physical batch, accumulation, optimizer/schedule, budget, precision and evaluation rules are common within a dataset. S/M/L external dropout is 0.15. Baseline-internal layers/stochastic depth remain architectural properties.

All main models train from scratch with primary supervision only. Auxiliary localization, deep-supervision and feature-pair weights are zero. Ordinary early stopping is disabled; strict maximum validation foreground mIoU selects best. Complete testing uses primary argmax and no TTA. Numerical/foreground failures do not count as completion. FP32 tensors with TF32 arithmetic are recorded explicitly.

This is a common-recipe adapted-architecture comparison, not each baseline's best published benchmark. HRSICD resizes the common external input to its 64×64 core and restores output logits; internal resolutions therefore differ by architecture.

## Exact module semantics

BiCSF includes GCBM and bidirectional WASM. BRIGHT/Haiti L use:

| Suffix | HOG | DPM | GCBM + WASM | IRB |
| --- | ---: | ---: | ---: | ---: |
| dehcd_l | 1 | 1 | 1 | 1 |
| no_dpm | 1 | 0 | 1 | 1 |
| no_bicsf | 1 | 1 | 0 | 1 |
| no_irb | 1 | 1 | 1 | 0 |
| no_hog | 0 | 1 | 1 | 1 |
| no_hog_no_dpm | 0 | 0 | 1 | 1 |
| no_hog_no_dpm_no_bicsf | 0 | 0 | 0 | 1 |
| all_off | 0 | 0 | 0 | 0 |

DPM removal uses plain concatenation/convolution fusion. Full BiCSF removal disables both global_context and cross_scale_fusion. The internal bicsf_mode key controls WASM; complete capacity matching also sets gcb_mode=conv_matched.

| Analysis | Reference | Factor |
| --- | --- | --- |
| Main/scaling | same-dataset dehcd_l | declared architecture |
| Component/combination | same-dataset dehcd_l | declared switches/combination |
| BRIGHT capacity | bright_dehcd_l | HOG prior, DPM, whole BiCSF or IRB replacement |
| BRIGHT DPM parts | bright_dehcd_l | flow or difference gate |
| BRIGHT recipe | bright_dehcd_l | loss scheme, sampler or class weights |
| Haiti bins | haiti_dehcd_l | 2/4/6/8/10 |
| BRIGHT-M IRB | bright_dehcd_m | 0/1/2/3/4/5/6/7/8 |
| BRIGHT-M levels | bright_dehcd_m | first 1/2/3/4 levels, both modalities |

Default points reuse main configurations. No Cartesian product is formed. BRIGHT-M T=0 is different from BRIGHT-L no_irb. Parameter controls export actual reference/control counts and errors; all retained parameters participate. Counts do not match receptive field/FLOPs/inductive bias. Intensity control retains HOG modulators with intensity-bin input. Feedforward IRB control uses one denoiser and separate channel groups for the retained step scalars.

CE+Dice changes a loss bundle including smoothing, not a single causal factor. Sampling with replacement is not fixed sample duplication. Crop effects require measured spatial freedom and valid targets. No additional crop control or enlarged input is part of this plan.

Configurations declare comparison.reference/factor/allowed_changes; preflight checks actual differences and three data splits. Historical aliases require full evidence, not renaming, shape matching or equal parameter count. Report per seed and mean±sample SD, n and expected_n=3. Incomplete results stay incomplete. Pair matching seeds to the declared reference. Repeated reference appearances do not increase n. Non-additive scores and feature maps alone do not prove causal synergy or physical separation of sensor/disaster effects. In-domain splits do not establish unseen-event generalization.

See [protocol](EXPERIMENT_PROTOCOL.md) and [commands](RUN_WORKFLOW.md).
