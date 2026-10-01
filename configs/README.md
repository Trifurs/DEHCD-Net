# Configuration Layout

The XML files use a small inheritance tree so most variables live in one place.

- `base.xml` stores shared model, optimization, logging, and inference defaults.
- `datasets/*.xml` store task definitions, dataset layout, normalization, loss, and sampling policy.
- `dehcd/*.xml` store only the DEHCD-Net size variant and run name.

Use any variant directly, for example:

```bash
python tools/train.py --config configs/dehcd/bright_l.xml
python tools/evaluate.py --config configs/dehcd/haiti_l.xml --checkpoint <checkpoint.pth>
```

Dataset roots are intentionally placeholders such as `data/BRIGHT`; edit the corresponding dataset base file or pass a modified copy for your environment.

## Controlled experiments

`experiments/catalog.json` indexes 174 fully resolved-through-inheritance experiments.
All generated experiment overlays use FP32 (`training.amp=false`) for every model.
Use `tools/run_multiseed.py` to select main, ablation, recipe, sensitivity, adapter, auxiliary, head_recipe or optimizer_recipe suites.
For model comparisons use these controlled overlays, not the ordinary single-run
presets above. Main experiments share primary-only supervision and each control
names its reference and permitted changes; see [fair comparisons](../docs/FAIR_COMPARISONS.md).
`tools/build_event_cv.py` generates additional cross_event catalogs from verified metadata.
See [the protocol](../docs/EXPERIMENT_PROTOCOL.md) before running.
`--runtime-profile rtx5090` applies the shared hardware settings in `runtime/rtx5090.json`;
see [execution settings](../docs/RUNTIME_PROFILE.md).
