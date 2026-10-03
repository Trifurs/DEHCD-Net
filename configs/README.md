# Configuration layout

`base.xml`, `datasets/*.xml`, `dehcd/*.xml` and `runtime/*.json` are shared templates. Only the 80 XML files indexed by `experiments/catalog.json` define formal canonical experiments. Templates may contain settings that the formal overlays replace; do not treat direct template runs as formal comparison results.

The catalog is derived from formal configuration metadata. `experiments/groups.json` references canonical IDs without copying scientific settings. Multiple selected groups take the ID union before expanding seeds `[42, 1051, 2060]`: 240 target tasks in total, including compatible existing work.

```bash
python -W ignore tools/run_all.py --dry-run
python -W ignore tools/run_all.py --preflight-only
python -W ignore tools/run_all.py
```

Use `--groups main ablation sensitivity scaling` to select shared views without duplicating their full reference. The default schedule completes all configurations for seed 42, then 1051, then 2060. Within each seed, dataset order is BRIGHT, Haiti, xBD, CAU-Flood. Compatible completed results are reused and interrupted runs continue from their last checkpoint. The local launcher uses the common RTX 5090 profile and data under `~/桌面/myData/Hete_CD`; `--data-base` and `--output` override locations.

See [generated coverage](../docs/EXPERIMENT_CATALOG.md), [protocol](../docs/EXPERIMENT_PROTOCOL.md), and [runtime settings](../docs/RUNTIME_PROFILE.md). FP32 tensors and TF32 math are recorded separately. Fixed training budgets, foreground_miou selection and disabled ordinary early stopping apply to every formal experiment.
