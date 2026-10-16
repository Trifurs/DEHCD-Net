# Event generalization and few-shot adaptation

`tools/run_transfer_study.py` supports two scene-disjoint protocols, in addition to Turkey-only source training in `tools/run_event_transfer.py`:

| Protocol | Training and selection | Test |
| --- | --- | --- |
| `loeo` | three source events; source-side validation scenes | the held-out event |
| `fewshot` | separate target training and validation scenes | remaining target scenes |

## Few-shot comparison

```bash
python -W ignore tools/run_transfer_study.py --protocols fewshot --sizes m --events noto-earthquake --seeds 42 1051 2060 --source-root /path/to/event_results --source-seed 3402 --data-root /path/to/event_data --output /path/to/fewshot_results
```

The specified source checkpoint must exist. Three arms use identical held-out test scenes:

- `zero_shot`: fixed Turkey-only source weights, evaluated once.
- `finetune`: source model weights initialize a new optimizer, scheduler and epoch state.
- `scratch`: random initialization with the same architecture and target training recipe as fine-tuning.

Training and validation each use approximately 10% of original target scenes by default. Report the actual patch count, scene count and class supports for both label budgets. Scene proportions are not patch proportions. The fixed split seed is 20261015; grouping and stratification use reference-label coverage and pixel counts, without prediction or score feedback. Infeasible class coverage is rejected. Scene IDs alone do not establish a geographic buffer.

Defaults are 60 epochs, patience 15, head learning rate 0.0003 and encoder multiplier 0.25 for both trainable arms. Each epoch samples at least 512 occurrences with independent augmentation; this does not increase the annotation count. `--fewshot-epochs 40` selects a 40-epoch budget for both arms. Adaptation seeds reuse the declared source seed and are not independent source pretraining runs.

## Leave-one-event-out

```bash
python -W ignore tools/run_transfer_study.py --protocols loeo --sizes m --seeds 42 1051 2060 --data-root /path/to/event_data --source-root /path/to/event_results --output /path/to/loeo_results
```

Default folds hold out Turkey, Haiti, Morocco and Noto in turn. `--loeo-init auto` reuses Turkey-only weights when Turkey is a source event and preserves its training/validation scene boundary. The Turkey-held-out fold starts from scratch. `--loeo-init scratch` starts all folds from scratch. Held-out events and previously trained scenes are checked before importing weights.

Scratch folds use up to 120 epochs; initialized folds use up to 80. Patience is 30. Training samples up to 6,000 occurrences per epoch with equal event probability and within-event rare-class weighting. Selection uses the equal-event mean validation foreground mIoU.

## Evidence and continuation

Shared study settings are in `study.xml`; size settings remain in `dehcd_s.xml`, `dehcd_m.xml` and `dehcd_l.xml`. Training-only labels determine class weights. BF16/TF32, fused AdamW and shared normalized input caches are recorded. Different size recipes are disclosed and do not form a strictly hyperparameter-controlled size ablation.

Use `--dry-run` to generate manifests/configurations and audit partitions without training; `--prepare-only` additionally prepares caches. Output defaults to `runs/event_transfer_study`, or the sibling of a configured event-output directory. Each protocol/target contains `manifest.csv`, `split_audit.json`, resolved settings and method/seed results. Summaries are `summary/aggregate.json` and `summary/per_seed.csv`. Repeating a compatible command resumes training and verifies existing tests.

Changing the protocol, split, initialization or training recipe requires a separate output directory. All training and checkpoint selection finishes before final test evaluation. Source exposure and prior use of target events remain recorded in experiment evidence; previously inspected targets must not be presented as a fresh blind evaluation. Report all selected events, initialization, annotation budgets, adaptation seeds and sample SD. A fixed zero-shot checkpoint has one independent evaluation.
