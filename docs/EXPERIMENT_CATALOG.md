# Experiment catalog

Generated from the metadata in the formal XML definitions by `python tools/build_experiment_configs.py`.

There are **80 unique configurations** and **240 target tasks** with seeds `[42, 1051, 2060]`. Compatible completed tasks count toward this total; it is not the remaining training count.

Shared `base`, `datasets`, `dehcd`, and runtime files are templates, not additional scheduled tasks. Each formal configuration lives in exactly one location. Group views reuse the same canonical ID and seed results.

| Dataset | Unique configurations | Target tasks |
|---|---:|---:|
| bright | 39 | 117 |
| haiti | 23 | 69 |
| xbd | 9 | 27 |
| cau_flood | 9 | 27 |

## Coverage and analysis views

Main metrics use all three complete test seeds, sample SD (`ddof=1`), `n`, and `expected_n=3`. A shared reference still has only three independent runs. Qualitative displays use seed 42; parameter counting and fixed-device efficiency profiling reuse checkpoints and do not train extra models.

| Analysis group | Paper use | Canonical IDs (reference/default marked *) |
|---|---|---|
| main | Table 2, Table 3, Table 5, Table 4 | `bright_dehcd_s`<br>`bright_dehcd_m`<br>`bright_dehcd_l`<br>`bright_icif_net`<br>`bright_dminet`<br>`bright_hfa_panet`<br>`bright_wavehfg`<br>`bright_hrsicd`<br>`bright_haff`<br>`bright_changeos`<br>`bright_damageformer`<br>`bright_changemamba`<br>`haiti_dehcd_s`<br>`haiti_dehcd_m`<br>`haiti_dehcd_l`<br>`haiti_icif_net`<br>`haiti_dminet`<br>`haiti_hfa_panet`<br>`haiti_wavehfg`<br>`haiti_hrsicd`<br>`haiti_haff`<br>`haiti_changeos`<br>`haiti_damageformer`<br>`haiti_changemamba`<br>`xbd_dehcd_s`<br>`xbd_dehcd_m`<br>`xbd_dehcd_l`<br>`xbd_icif_net`<br>`xbd_dminet`<br>`xbd_hfa_panet`<br>`xbd_wavehfg`<br>`xbd_hrsicd`<br>`xbd_haff`<br>`cau_flood_dehcd_s`<br>`cau_flood_dehcd_m`<br>`cau_flood_dehcd_l`<br>`cau_flood_icif_net`<br>`cau_flood_dminet`<br>`cau_flood_hfa_panet`<br>`cau_flood_wavehfg`<br>`cau_flood_hrsicd`<br>`cau_flood_haff` |
| main_bright | Table 2 | `bright_dehcd_s`<br>`bright_dehcd_m`<br>`bright_dehcd_l`<br>`bright_icif_net`<br>`bright_dminet`<br>`bright_hfa_panet`<br>`bright_wavehfg`<br>`bright_hrsicd`<br>`bright_haff`<br>`bright_changeos`<br>`bright_damageformer`<br>`bright_changemamba` |
| qualitative | Figs.8–11, Fig.14, Fig.13 | `bright_dehcd_s`<br>`bright_dehcd_m`<br>`bright_dehcd_l`<br>`bright_icif_net`<br>`bright_dminet`<br>`bright_hfa_panet`<br>`bright_wavehfg`<br>`bright_hrsicd`<br>`bright_haff`<br>`bright_changeos`<br>`bright_damageformer`<br>`bright_changemamba`<br>`haiti_dehcd_s`<br>`haiti_dehcd_m`<br>`haiti_dehcd_l`<br>`haiti_icif_net`<br>`haiti_dminet`<br>`haiti_hfa_panet`<br>`haiti_wavehfg`<br>`haiti_hrsicd`<br>`haiti_haff`<br>`haiti_changeos`<br>`haiti_damageformer`<br>`haiti_changemamba`<br>`xbd_dehcd_s`<br>`xbd_dehcd_m`<br>`xbd_dehcd_l`<br>`xbd_icif_net`<br>`xbd_dminet`<br>`xbd_hfa_panet`<br>`xbd_wavehfg`<br>`xbd_hrsicd`<br>`xbd_haff`<br>`cau_flood_dehcd_s`<br>`cau_flood_dehcd_m`<br>`cau_flood_dehcd_l`<br>`cau_flood_icif_net`<br>`cau_flood_dminet`<br>`cau_flood_hfa_panet`<br>`cau_flood_wavehfg`<br>`cau_flood_hrsicd`<br>`cau_flood_haff` |
| efficiency | Figs.15/16 | `bright_dehcd_s`<br>`bright_dehcd_m`<br>`bright_dehcd_l`<br>`bright_icif_net`<br>`bright_dminet`<br>`bright_hfa_panet`<br>`bright_wavehfg`<br>`bright_hrsicd`<br>`bright_haff`<br>`bright_changeos`<br>`bright_damageformer`<br>`bright_changemamba`<br>`haiti_dehcd_s`<br>`haiti_dehcd_m`<br>`haiti_dehcd_l`<br>`haiti_icif_net`<br>`haiti_dminet`<br>`haiti_hfa_panet`<br>`haiti_wavehfg`<br>`haiti_hrsicd`<br>`haiti_haff`<br>`haiti_changeos`<br>`haiti_damageformer`<br>`haiti_changemamba`<br>`xbd_dehcd_s`<br>`xbd_dehcd_m`<br>`xbd_dehcd_l`<br>`xbd_icif_net`<br>`xbd_dminet`<br>`xbd_hfa_panet`<br>`xbd_wavehfg`<br>`xbd_hrsicd`<br>`xbd_haff`<br>`cau_flood_dehcd_s`<br>`cau_flood_dehcd_m`<br>`cau_flood_dehcd_l`<br>`cau_flood_icif_net`<br>`cau_flood_dminet`<br>`cau_flood_hfa_panet`<br>`cau_flood_wavehfg`<br>`cau_flood_hrsicd`<br>`cau_flood_haff` |
| confusion | Figs.8–11 | `bright_dehcd_s`<br>`bright_dehcd_m`<br>`bright_dehcd_l`<br>`bright_icif_net`<br>`bright_dminet`<br>`bright_hfa_panet`<br>`bright_wavehfg`<br>`bright_hrsicd`<br>`bright_haff`<br>`bright_changeos`<br>`bright_damageformer`<br>`bright_changemamba`<br>`haiti_dehcd_s`<br>`haiti_dehcd_m`<br>`haiti_dehcd_l`<br>`haiti_icif_net`<br>`haiti_dminet`<br>`haiti_hfa_panet`<br>`haiti_wavehfg`<br>`haiti_hrsicd`<br>`haiti_haff`<br>`haiti_changeos`<br>`haiti_damageformer`<br>`haiti_changemamba` |
| scaling | Figs.15/16 | `bright_dehcd_s` (x=s)<br>`bright_dehcd_m` (x=m)<br>`bright_dehcd_l` (x=l)<br>`haiti_dehcd_s` (x=s)<br>`haiti_dehcd_m` (x=m)<br>`haiti_dehcd_l` (x=l)<br>`xbd_dehcd_s` (x=s)<br>`xbd_dehcd_m` (x=m)<br>`xbd_dehcd_l` (x=l)<br>`cau_flood_dehcd_s` (x=s)<br>`cau_flood_dehcd_m` (x=m)<br>`cau_flood_dehcd_l` (x=l) |
| scaling_bright | Figs.15/16 | `bright_dehcd_s` (x=s)<br>`bright_dehcd_m` (x=m)<br>`bright_dehcd_l` (x=l) |
| sensitivity | Figs.18/19, Fig.19, Fig.18 | `bright_dehcd_m` *<br>`haiti_dehcd_l` *<br>`haiti_hog_bins_2`<br>`haiti_hog_bins_4`<br>`haiti_hog_bins_8`<br>`haiti_hog_bins_10`<br>`bright_m_irb_steps_0`<br>`bright_m_irb_steps_1`<br>`bright_m_irb_steps_2`<br>`bright_m_irb_steps_4`<br>`bright_m_irb_steps_5`<br>`bright_m_irb_steps_6`<br>`bright_m_irb_steps_7`<br>`bright_m_irb_steps_8`<br>`bright_m_hog_levels_1`<br>`bright_m_hog_levels_3`<br>`bright_m_hog_levels_4` |
| irb_steps | Fig.18 | `bright_m_irb_steps_0` (x=0)<br>`bright_m_irb_steps_1` (x=1)<br>`bright_m_irb_steps_2` (x=2)<br>`bright_dehcd_m` * (x=3)<br>`bright_m_irb_steps_4` (x=4)<br>`bright_m_irb_steps_5` (x=5)<br>`bright_m_irb_steps_6` (x=6)<br>`bright_m_irb_steps_7` (x=7)<br>`bright_m_irb_steps_8` (x=8) |
| hog_levels | Fig.19 | `bright_m_hog_levels_1` (x=1)<br>`bright_dehcd_m` * (x=2)<br>`bright_m_hog_levels_3` (x=3)<br>`bright_m_hog_levels_4` (x=4) |
| data_statistics | Table 1 | `bright_dehcd_l`<br>`haiti_dehcd_l`<br>`xbd_dehcd_l`<br>`cau_flood_dehcd_l` |
| ablation | Fig.17 | `bright_dehcd_l` *<br>`haiti_dehcd_l` *<br>`bright_no_dpm`<br>`bright_no_bicsf`<br>`bright_no_irb`<br>`bright_no_hog`<br>`bright_no_hog_no_dpm`<br>`bright_no_hog_no_dpm_no_bicsf`<br>`bright_all_off`<br>`haiti_no_dpm`<br>`haiti_no_bicsf`<br>`haiti_no_irb`<br>`haiti_no_hog`<br>`haiti_no_hog_no_dpm`<br>`haiti_no_hog_no_dpm_no_bicsf`<br>`haiti_all_off` |
| fig17_bright | Fig.17 | `bright_dehcd_l` *<br>`bright_no_dpm`<br>`bright_no_bicsf`<br>`bright_no_irb`<br>`bright_no_hog`<br>`bright_no_hog_no_dpm`<br>`bright_no_hog_no_dpm_no_bicsf`<br>`bright_all_off` |
| features | Fig.12 | `bright_dehcd_l`<br>`haiti_dehcd_l` |
| capacity | Additional capacity comparison | `bright_dehcd_l` *<br>`bright_hog_intensity_matched`<br>`bright_dpm_conv_matched`<br>`bright_bicsf_conv_matched`<br>`bright_irb_feedforward_matched` |
| dpm | Additional dpm comparison | `bright_dehcd_l` *<br>`bright_no_flow`<br>`bright_no_difference_gate` |
| recipe | Additional recipe comparison | `bright_dehcd_l` *<br>`bright_ce_dice_full_sampling`<br>`bright_no_weighted_sampler`<br>`bright_no_class_weights` |
| main_haiti | Table 3 | `haiti_dehcd_s`<br>`haiti_dehcd_m`<br>`haiti_dehcd_l`<br>`haiti_icif_net`<br>`haiti_dminet`<br>`haiti_hfa_panet`<br>`haiti_wavehfg`<br>`haiti_hrsicd`<br>`haiti_haff`<br>`haiti_changeos`<br>`haiti_damageformer`<br>`haiti_changemamba` |
| scaling_haiti | Figs.15/16 | `haiti_dehcd_s` (x=s)<br>`haiti_dehcd_m` (x=m)<br>`haiti_dehcd_l` (x=l) |
| fig17_haiti | Fig.17 | `haiti_dehcd_l` *<br>`haiti_no_dpm`<br>`haiti_no_bicsf`<br>`haiti_no_irb`<br>`haiti_no_hog`<br>`haiti_no_hog_no_dpm`<br>`haiti_no_hog_no_dpm_no_bicsf`<br>`haiti_all_off` |
| hog_bins | Fig.19 | `haiti_hog_bins_2` (x=2)<br>`haiti_hog_bins_4` (x=4)<br>`haiti_dehcd_l` * (x=6)<br>`haiti_hog_bins_8` (x=8)<br>`haiti_hog_bins_10` (x=10) |
| main_xbd | Table 5 | `xbd_dehcd_s`<br>`xbd_dehcd_m`<br>`xbd_dehcd_l`<br>`xbd_icif_net`<br>`xbd_dminet`<br>`xbd_hfa_panet`<br>`xbd_wavehfg`<br>`xbd_hrsicd`<br>`xbd_haff` |
| scaling_xbd | Figs.15/16 | `xbd_dehcd_s` (x=s)<br>`xbd_dehcd_m` (x=m)<br>`xbd_dehcd_l` (x=l) |
| main_cau_flood | Table 4 | `cau_flood_dehcd_s`<br>`cau_flood_dehcd_m`<br>`cau_flood_dehcd_l`<br>`cau_flood_icif_net`<br>`cau_flood_dminet`<br>`cau_flood_hfa_panet`<br>`cau_flood_wavehfg`<br>`cau_flood_hrsicd`<br>`cau_flood_haff` |
| scaling_cau_flood | Figs.15/16 | `cau_flood_dehcd_s` (x=s)<br>`cau_flood_dehcd_m` (x=m)<br>`cau_flood_dehcd_l` (x=l) |

## Fixed factors

The Fig.17 order is HOG, DPM, BiCSF, IRB. Plain fusion replaces a disabled DPM; disabling BiCSF turns off both GCBM and WASM. Each dataset uses its existing L main model as full reference.

Haiti HOG bins use L at K=[2,4,6,8,10], with K=6 supplied by `haiti_dehcd_l`. BRIGHT IRB uses M at T=0–8, with T=3 supplied by `bright_dehcd_m`. BRIGHT HOG guidance uses M at G=[1,2,3,4], with G=2 supplied by the same M run. Guidance covers the first G levels of both branches. All scans retain the other default factors (K=6, G=2, T=3).

`bright_m_irb_steps_0` is an M model and is distinct from the L component control `bright_no_irb`. Former L scan IDs cannot be relabeled as M results. The M guidance assignment is supported by the historical definitions `Heterogeneous_LCD/configs/5090_x1/discussion_hog_levels_bright_m/bright_m_hog_levels_{1..4}.xml`, which inherit `bright_multiclass_hacf_m.xml`. Those definitions establish the variant only: their dropout=0.12, seed=1234 and other earlier settings do not make historical runs compatible with the current protocol.

The three BRIGHT training controls separately compare the loss recipe, weighted sampler, and class weights. CE+Dice replaces several loss terms and smoothing and is a recipe comparison. Capacity and DPM controls are limited to BRIGHT. Matched parameter counts do not imply identical receptive fields or operation counts.

Main architecture results share one from-scratch task objective and disable localization, feature-pair, and deep-supervision auxiliary losses. They measure adapted architectures under the common protocol, not each upstream implementation's best benchmark. Ordinary early stopping stays disabled; fixed budgets, validation foreground_miou selection, no TTA, and main-logits argmax remain in force.

Table 1 is computed from actual dataset evidence. Confusion plots retain raw counts and explicit denominators; row normalization is not overall accuracy. Efficiency axes use validated full counts or measured latency/throughput under common settings. Postprocessing does not create additional training samples or tasks.

## Unique definitions

| Canonical ID | Variant | Roles | Declared reference | Allowed scientific changes |
|---|---|---|---|---|
| `bright_dehcd_s` | s | main, qualitative, efficiency, confusion, scaling | bright_dehcd_l | model |
| `bright_dehcd_m` | m | main, qualitative, efficiency, confusion, scaling, sensitivity | bright_dehcd_l | model |
| `bright_dehcd_l` | l | main, qualitative, efficiency, confusion, scaling, data_statistics, ablation, features, capacity, dpm, recipe | dataset root | model |
| `bright_icif_net` | icif_net | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_dminet` | dminet | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_hfa_panet` | hfa_panet | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_wavehfg` | wavehfg | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_hrsicd` | hrsicd | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_haff` | haff | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_changeos` | changeos | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_damageformer` | damageformer | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `bright_changemamba` | changemamba | main, qualitative, efficiency, confusion | bright_dehcd_l | model |
| `haiti_dehcd_s` | s | main, qualitative, efficiency, confusion, scaling | haiti_dehcd_l | model |
| `haiti_dehcd_m` | m | main, qualitative, efficiency, confusion, scaling | haiti_dehcd_l | model |
| `haiti_dehcd_l` | l | main, qualitative, efficiency, confusion, scaling, data_statistics, ablation, features, sensitivity | dataset root | model |
| `haiti_icif_net` | icif_net | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_dminet` | dminet | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_hfa_panet` | hfa_panet | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_wavehfg` | wavehfg | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_hrsicd` | hrsicd | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_haff` | haff | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_changeos` | changeos | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_damageformer` | damageformer | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `haiti_changemamba` | changemamba | main, qualitative, efficiency, confusion | haiti_dehcd_l | model |
| `xbd_dehcd_s` | s | main, qualitative, efficiency, scaling | xbd_dehcd_l | model |
| `xbd_dehcd_m` | m | main, qualitative, efficiency, scaling | xbd_dehcd_l | model |
| `xbd_dehcd_l` | l | main, qualitative, efficiency, scaling, data_statistics | dataset root | model |
| `xbd_icif_net` | icif_net | main, qualitative, efficiency | xbd_dehcd_l | model |
| `xbd_dminet` | dminet | main, qualitative, efficiency | xbd_dehcd_l | model |
| `xbd_hfa_panet` | hfa_panet | main, qualitative, efficiency | xbd_dehcd_l | model |
| `xbd_wavehfg` | wavehfg | main, qualitative, efficiency | xbd_dehcd_l | model |
| `xbd_hrsicd` | hrsicd | main, qualitative, efficiency | xbd_dehcd_l | model |
| `xbd_haff` | haff | main, qualitative, efficiency | xbd_dehcd_l | model |
| `cau_flood_dehcd_s` | s | main, qualitative, efficiency, scaling | cau_flood_dehcd_l | model |
| `cau_flood_dehcd_m` | m | main, qualitative, efficiency, scaling | cau_flood_dehcd_l | model |
| `cau_flood_dehcd_l` | l | main, qualitative, efficiency, scaling, data_statistics | dataset root | model |
| `cau_flood_icif_net` | icif_net | main, qualitative, efficiency | cau_flood_dehcd_l | model |
| `cau_flood_dminet` | dminet | main, qualitative, efficiency | cau_flood_dehcd_l | model |
| `cau_flood_hfa_panet` | hfa_panet | main, qualitative, efficiency | cau_flood_dehcd_l | model |
| `cau_flood_wavehfg` | wavehfg | main, qualitative, efficiency | cau_flood_dehcd_l | model |
| `cau_flood_hrsicd` | hrsicd | main, qualitative, efficiency | cau_flood_dehcd_l | model |
| `cau_flood_haff` | haff | main, qualitative, efficiency | cau_flood_dehcd_l | model |
| `bright_no_dpm` | l | ablation | bright_dehcd_l | model.fusion_mode |
| `bright_no_bicsf` | l | ablation | bright_dehcd_l | model.global_context, model.cross_scale_fusion |
| `bright_no_irb` | l | ablation | bright_dehcd_l | model.diffusion_steps |
| `bright_no_hog` | l | ablation | bright_dehcd_l | model.use_hog |
| `bright_no_hog_no_dpm` | l | ablation | bright_dehcd_l | model.use_hog, model.fusion_mode |
| `bright_no_hog_no_dpm_no_bicsf` | l | ablation | bright_dehcd_l | model.use_hog, model.fusion_mode, model.global_context, model.cross_scale_fusion |
| `bright_all_off` | l | ablation | bright_dehcd_l | model.use_hog, model.fusion_mode, model.global_context, model.cross_scale_fusion, model.diffusion_steps |
| `haiti_no_dpm` | l | ablation | haiti_dehcd_l | model.fusion_mode |
| `haiti_no_bicsf` | l | ablation | haiti_dehcd_l | model.global_context, model.cross_scale_fusion |
| `haiti_no_irb` | l | ablation | haiti_dehcd_l | model.diffusion_steps |
| `haiti_no_hog` | l | ablation | haiti_dehcd_l | model.use_hog |
| `haiti_no_hog_no_dpm` | l | ablation | haiti_dehcd_l | model.use_hog, model.fusion_mode |
| `haiti_no_hog_no_dpm_no_bicsf` | l | ablation | haiti_dehcd_l | model.use_hog, model.fusion_mode, model.global_context, model.cross_scale_fusion |
| `haiti_all_off` | l | ablation | haiti_dehcd_l | model.use_hog, model.fusion_mode, model.global_context, model.cross_scale_fusion, model.diffusion_steps |
| `bright_hog_intensity_matched` | l | capacity | bright_dehcd_l | model.use_hog, model.hog_prior |
| `bright_dpm_conv_matched` | l | capacity | bright_dehcd_l | model.fusion_mode |
| `bright_bicsf_conv_matched` | l | capacity | bright_dehcd_l | model.gcb_mode, model.bicsf_mode |
| `bright_irb_feedforward_matched` | l | capacity | bright_dehcd_l | model.irb_mode |
| `bright_no_flow` | l | dpm | bright_dehcd_l | model.align_fusion |
| `bright_no_difference_gate` | l | dpm | bright_dehcd_l | model.difference_gate |
| `bright_ce_dice_full_sampling` | l | recipe | bright_dehcd_l | training.loss, training.ce_weight, training.dice_weight, training.focal_weight, training.lovasz_weight, training.tversky_weight, training.foreground_dice_weight, training.label_smoothing, training.aux_loss_weight, training.feature_pair_loss_weight |
| `bright_no_weighted_sampler` | l | recipe | bright_dehcd_l | training.class_balanced_sampler |
| `bright_no_class_weights` | l | recipe | bright_dehcd_l | training.class_weights |
| `haiti_hog_bins_2` | l | sensitivity | haiti_dehcd_l | model.hog_bins |
| `haiti_hog_bins_4` | l | sensitivity | haiti_dehcd_l | model.hog_bins |
| `haiti_hog_bins_8` | l | sensitivity | haiti_dehcd_l | model.hog_bins |
| `haiti_hog_bins_10` | l | sensitivity | haiti_dehcd_l | model.hog_bins |
| `bright_m_irb_steps_0` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_irb_steps_1` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_irb_steps_2` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_irb_steps_4` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_irb_steps_5` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_irb_steps_6` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_irb_steps_7` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_irb_steps_8` | m | sensitivity | bright_dehcd_m | model.diffusion_steps |
| `bright_m_hog_levels_1` | m | sensitivity | bright_dehcd_m | model.hog_modulation_levels |
| `bright_m_hog_levels_3` | m | sensitivity | bright_dehcd_m | model.hog_modulation_levels |
| `bright_m_hog_levels_4` | m | sensitivity | bright_dehcd_m | model.hog_modulation_levels |
