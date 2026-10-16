# Configuration reference

`utils.config.XMLConfigParser` loads typed XML, follows `base_param` inheritance and accepts saved JSON configurations. Child settings override the inherited values. A `config_ref` redirects to another configuration file.

| Section | Purpose |
| --- | --- |
| `task` | output classes, class names and ignore label |
| `dataset` | adapter, root paths, modality bands, masks, crops and pairing rules |
| `normalization` | optical and SAR input transforms |
| `augmentation` | synchronized spatial and modality-specific transforms |
| `model` | architecture, channels, module switches and initialization |
| `training` | optimizer, schedule, batch, numerical precision and stopping |
| `inference` | checkpoint, split, prediction and export settings |
| `logging` | log output, images and optional tracking |

Shared definitions are in `configs/base.xml`, `configs/datasets/` and `configs/dehcd/`. `configs/experiments/` contains executable presets and an ID-based registry for the batch launcher. Group selection takes the ID union before creating run-specific configurations. The registry stores configuration relationships, not measured results or a fixed repetition schedule.

Use `training.seed` for individual training or the launcher's `--seeds` option for repetitions. Defaults live in `utils/defaults.py`. Aggregate tools obtain the requested repetitions from the saved run plan rather than assuming a fixed count.

## Module controls

`use_hog` and `hog_modulation_levels` control HOG guidance. DPM uses `fusion_mode`, `align_fusion`, `difference_gate` and modality weighting. Disabling full BiCSF requires both `global_context=false` and `cross_scale_fusion=false`. `diffusion_steps` controls IRB; the refinement module reuses its denoiser parameters.

Comparison presets declare their reference and allowed setting changes. Validation detects undeclared differences in input preparation, labels, supervision and optimization. Adapted baseline limitations are described in [comparison models](../compare/README.md).

## Paths and device

`local/settings.json` may define `data_base`, `output`, `event_data_root`, `event_output`, `device` and `runtime_profile`. Relative paths resolve from the project root. Corresponding environment variables use the `DEHCD_` prefix and override the file. CLI arguments override both. `target_exposure_note` optionally records caller-declared prior inspection of transfer targets.

The public defaults use project-relative data/run directories and automatic device selection. A missing machine-settings file is valid. The [runtime profile](RUNTIME_PROFILE.md) changes execution settings uniformly across applicable presets.
