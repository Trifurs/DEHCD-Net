# Transfer tools

`tools/run_event_transfer.py` trains configured source models and evaluates selected target events. `tools/run_transfer_study.py` supports scene-disjoint leave-one-event-out and target-label adaptation. Use `--help` to inspect paths, event selection, initialization and runtime options.

```bash
python -W ignore tools/run_event_transfer.py --data-root /path/to/event_data --output /path/to/source_runs
python -W ignore tools/run_transfer_study.py --protocols fewshot --sizes m --events noto-earthquake --source-root /path/to/source_runs --data-root /path/to/event_data --output /path/to/adaptation_runs
```

Fine-tuning requires the selected source checkpoint. `--source-seed` identifies its source run; `--seeds` selects independent adaptation repetitions. The zero-shot arm evaluates a fixed source checkpoint once. Trainable arms use separate optimizer/scheduler state and the same target partition and training settings.

`--protocols loeo` holds out an event; `--loeo-init` chooses source-weight initialization or scratch training. Target events are checked before importing weights. `--train-fraction`, `--val-fraction` and `--split-seed` control scene-level partitions. Training and validation annotations are counted separately. Group isolation, class coverage and input fingerprints are validated before training.

Shared options are in `study.xml`; model variants inherit the size configurations. CLI options can override budgets, learning rate and workers. `--target-exposure-note` records caller-declared prior inspection of target data without imposing a project-specific history on new users.

`--dry-run` writes and audits manifests/configurations without training; `--prepare-only` also prepares reusable inputs. Saved outputs contain partition audits, run settings, checkpoints and evaluation records under the requested output directory. Repeating a compatible command verifies existing state and continues incomplete work. Incompatible partitions or training settings require a separate output directory.
