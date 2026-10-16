# Configuration layout

`base.xml`, `datasets/*.xml`, `dehcd/*.xml` and `runtime/*.json` are shared templates. The 80 XML files indexed by `experiments/catalog.json` define the main canonical collection. `event_transfer/` contains separate event-transfer and adaptation settings.

The catalog derives from configuration metadata. `experiments/groups.json` references canonical IDs; the union of selected groups is taken before expanding seeds `[42,1051,2060]`, giving 240 main tasks. Shared references and default sensitivity points train once.

```bash
python -W ignore tools/run_all.py --dry-run
python -W ignore tools/run_all.py --preflight-only
python -W ignore tools/run_all.py
```

Configure paths using `local/settings.json`, environment variables or CLI arguments as described in the root README. CLI settings take precedence. The default seed rounds are 42, 1051 and 2060; each uses BRIGHT, Haiti, xBD and CAU-Flood in that order. Compatible completed results are reused.

Main early stopping uses patience 30 for BRIGHT/xBD/CAU-Flood and 100 for Haiti. The best checkpoint uses strict maximum validation foreground mIoU. All models within each dataset use the same stopping rule, budget and runtime settings.

`python tools/build_experiment_configs.py` regenerates canonical definitions, catalog, groups and [coverage documentation](../docs/EXPERIMENT_CATALOG.md). See [protocol](../docs/EXPERIMENT_PROTOCOL.md), [runtime profile](../docs/RUNTIME_PROFILE.md) and [transfer studies](event_transfer/README.md).
