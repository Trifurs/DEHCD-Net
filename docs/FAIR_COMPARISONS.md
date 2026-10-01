# Controlled comparison protocol

This protocol applies to `configs/experiments/`, identified by
`experiment.protocol=controlled_comparison_v1`. Use a separate campaign directory
for each fixed combination of code, data, seeds and execution settings.

## BiCSF components

BiCSF includes GCBM and bidirectional WASM. The configuration names
now have the following exact meanings on both BRIGHT and Haiti:

| Control | GCBM | WASM / bidirectional cross-scale path |
|---|---|---|
| `no_bicsf` | removed | removed |
| `no_wasm` | retained | removed |
| `no_gcb` | removed | retained |
| `bicsf_conv_matched` | independent-scale conv control | independent-scale conv control |
| `wasm_conv_matched` | retained | independent-scale conv control |
| `gcb_conv_matched` | independent-scale conv control | retained |

The existing internal key `bicsf_mode` controls the WASM path; a complete
`bicsf_conv_matched` experiment therefore sets both `bicsf_mode` and `gcb_mode`.
Actual reference/control parameter counts and matching errors are recorded.
These controls do not promise identical FLOPs or receptive fields.

## Common main comparison

All 48 main configurations (S/M/L plus nine baselines, on four datasets) use:

- The same train/validation/test files within each dataset, bands, normalization,
  label/quality masks, ignore policy, cropping, augmentation and sampler settings.
- The same planned seeds, physical batch size, accumulation, epoch budget,
  optimizer, LR schedule, clipping and numerical-failure rules within a dataset.
  Full train/validation splits are required; no per-model batch-size fallback.
- Scratch initialization, FP32 forward/loss, no ordinary early stopping,
  validation foreground mIoU checkpoint selection, no TTA and primary-logit argmax.
  Failure/collapse protection remains enabled and failed seeds are not discarded.
- The same primary task objective. Localization, deep-supervision and feature-pair
  auxiliary loss weights are zero in the main comparison.
- A common DEHCD dropout of 0.15 across S/M/L so model size is not coupled to a
  different external dropout setting. Published baseline-internal stochastic
  depth/dropout and layer types remain part of their architecture.

Dual-head structures are retained. Their localization classifiers are not trained
by an auxiliary objective in the primary-only main table, and their localization
scores must not be presented as trained native-head benchmark results. The
`auxiliary` suite enables that objective in an otherwise identical experiment.
The full native-head objective has its own explicit control below.

This is a **common-recipe adapted-architecture comparison**, not proof that every
method has reached its best possible tuned performance. Input stems, GN
conversion and each method's architectural constraints must be disclosed.
In particular, the existing HRSICD adapter resizes to its native 64×64 core before
restoring the output size; the runtime metadata now records that transformation.
Do not describe its internal input resolution as identical to a 256×256 model.
This update does not redesign that baseline or the supplied datasets.

## Named references and controlled factors

Each configuration records `experiment.comparison.reference`, `kind`, `factor`
and exact `allowed_changes`. Preflight records the actual differing fields.

| Experiment | Required reference | Permitted change |
|---|---|---|
| Main S/M/L or baseline | same-dataset DEHCD-L | architecture, with the shared recipe fixed |
| Module/capacity/sensitivity control | same-dataset DEHCD-L | declared model switches only |
| `*_original_bn` (all nine baselines, four datasets) | that baseline's main configuration | BN-to-GN adaptation only |
| `*_localization_aux` (three dual-head models, four datasets) | that baseline's main configuration | localization loss weight 0 → 1 |
| `*_head_recipe` | that baseline's `*_original_bn` | native dual-head objective only; keep sampling and selection fixed |
| `*_optimizer_recipe` | that baseline's `*_head_recipe` | specified LR/weight decay/scheduler recipe; same selection metric |
| `haiti_pixel_mean_equal_weights` | `haiti_dehcd_l` | foreground/background CE reduction only |
| `haiti_legacy_pixel_mean` | `haiti_pixel_mean_equal_weights` | binary class weights only |

The basic CE+Dice recipe changes several declared settings together; it is labeled
`recipe_bundle`, not a single-parameter causal attribution. Likewise, removing
all architecture modules or swapping an optimizer recipe is a declared bundle.
Native-loss/optimizer controls still use the project data and budget; they do
not constitute full official-benchmark reproductions.

Other pre-existing audited limitations (including the effective range of online
cropping on pre-cropped tiles) are not changed by this update. A declared factor
does not prove it has a nonzero effect on a particular dataset.

## Running and reporting

The catalog contains 174 configurations, including 48 main experiments. Selecting
all suites with five seeds plans 870 jobs; generation and preflight do not train.

```bash
python tools/run_multiseed.py --suite main --datasets bright \
  --data-root bright=/path/to/BRIGHT1 \
  --output runs/controlled/bright_main --preflight-only
```

Inspect `protocol.json`: it includes the actual control differences, reference
configurations and all three split identities. To start, use the same arguments
and output, replacing `--preflight-only` with `--resume`. If batch size or
accumulation needs changing, apply a uniform override to the entire comparison
and use a new output. Comparisons on different event folds remain separate.

```bash
python tools/run_multiseed.py \
  --experiments bright_dehcd_l bright_no_bicsf bright_no_wasm bright_no_gcb \
                bright_bicsf_conv_matched bright_wasm_conv_matched bright_gcb_conv_matched \
  --data-root bright=/path/to/BRIGHT1 \
  --output runs/controlled/bright_bicsf --preflight-only

python tools/compare_results.py --campaign runs/controlled/bright_main \
  --reference bright_dehcd_l \
  --comparators bright_changeos bright_damageformer bright_changemamba \
  --output runs/controlled/bright_main/statistics
```

For auxiliary/BN/native-head/optimizer effects, include both configurations in
the campaign and compare the named reference pair. A native-head recipe cannot
be silently mixed with DEHCD-L in an architecture-only paired test. Explicit
invalid pairs fail; automatic selection records excluded non-comparable pairs.
Descriptive tables remain available for every completed experiment.
Legacy configurations without comparison declarations can still be summarized,
but cannot silently enter a controlled paired architecture comparison.

Checks run before training and during aggregation/statistical reporting. They
include the complete data transformations, training settings and all three
split identities, not just the test files. Within an experiment, settings must
be unchanged across seeds. Epoch, batch, optimizer, loss, augmentation, masks,
resolution, sampler, initialization, TTA and checkpoint-selection differences
cannot pass as an undeclared architecture gain.

For event catalogs generated by `tools/build_event_cv.py`, include each control's
reference in `--configs`; reference names are remapped to the same event fold.
Nothing in these guards establishes geographical independence without verified
source-scene/event metadata or replaces the need for formal experiments.
