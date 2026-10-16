# Event-transfer interface

Source training and sequential target evaluation are implemented in `tools/run_event_transfer.py`. Target events do not participate in source checkpoint selection. Model-size overlays inherit `configs/event_transfer_earthquake.xml`.

```bash
python tools/run_event_transfer.py --help
python tools/run_event_transfer.py --data-root /path/to/events --output /path/to/runs --preflight-only
```

Data/output/device options are resolved through the command line and optional workstation settings. Each model size has its own run directory and target-event outputs. Full test evidence includes checkpoint identity, evaluated sample coverage and raw confusion counts.

For target-label fine-tuning, scratch initialization and scene-disjoint folds, see [transfer tools](../configs/event_transfer/README.md). Model and runtime settings remain explicit in the resolved configuration. Generated summaries describe requested runs and are not part of the source repository.
