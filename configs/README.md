# Configurations

`base.xml` defines shared options. `datasets/` describes data layouts and labels; `dehcd/` selects model variants. `config.xml` redirects to a runnable preset. `runtime/` contains optional device execution profiles.

The batch registry in `experiments/` maps IDs to configurations and reusable groups. It contains executable settings and reference relationships. It does not store results or prescribe a publication schedule. Regenerate presets and their JSON index with:

```bash
python tools/build_experiment_configs.py
```

`event_transfer/` provides source-training and scene-disjoint transfer settings. `local.example.json` is a portable path template; copy it to ignored `local/settings.json` for machine settings.

See [configuration fields](../docs/CONFIGURATION.md), [launcher usage](../docs/RUN_WORKFLOW.md) and [transfer tools](event_transfer/README.md).
