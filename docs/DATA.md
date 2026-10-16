# Data preparation

Dataset roots are configurable through the launcher or `local/settings.json`. Dataset XML files define exact relative paths, input bands, masks and label mapping. Use one sealed train/validation/test partition for every comparison model and seed.

| Dataset | Classes | Notes |
| --- | --- | --- |
| BRIGHT | background, intact, damaged, destroyed | paired optical/SAR, 256-pixel patches |
| Haiti | background, C1, C2, C3 | optical plus ascending/descending SAR; preserve quality masks |
| xBD | background, no damage, minor damage, major damage, destroyed | paired optical images |
| CAU-Flood | background, flood | paired optical/SAR |

## BRIGHT

`tools/dataset_tools/bright_crop_1024_to_256.py` crops source tiles. `tools/dataset_tools/bright_split.py` creates a class-balanced partition using complete source-scene groups. Its default command writes a plan and reports without moving the dataset:

```bash
python tools/dataset_tools/bright_split.py --root /path/to/BRIGHT1 --report-dir /path/to/split_plan
```

Inspect the manifest, class supports and split audit before activating the same plan with `--apply`. The tool checks paired files, duplicate IDs, source identities and active readers, stages files, and retains the original source directory when applying. Do not repartition a dataset already used by an active campaign. `--unit patch` is available for explicitly patch-level partitions and does not establish scene isolation.

Use the current dataset's `metadata/manifest.csv` and distribution evidence with its results. Changing a partition requires a separate campaign and new data fingerprints. Duplicate train/validation/test samples and overlapping declared scene groups fail preflight.

## Other datasets and event data

The remaining tools in `tools/dataset_tools/` provide conversion and crop helpers. Run each tool with `--help` for its input and output arguments. NetCDF conversion requires `requirements-optional.txt`.

`tools/prepare_event_transfer.py` prepares earthquake event records. `tools/run_transfer_study.py` builds scene-disjoint manifests and audits for leave-one-event-out or few-shot protocols; see [transfer-study settings](../configs/event_transfer/README.md). Target validation annotations contribute to the few-shot label budget.

## Shared preprocessing

Campaign preflight seals source identities and checks labels, pairings, split overlap and sampler applicability. Normalized FP32 inputs and integer labels are prepared before random augmentation and reused across compatible models/seeds. Cache keys include preprocessing settings and source fingerprints; cached tensors do not replace the source data or its manifest.
