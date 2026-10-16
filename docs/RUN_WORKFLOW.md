# Running tools

Launch from the repository root with the configured Python environment:

```bash
python -W ignore tools/run_all.py --help
python -W ignore tools/run_all.py --data-base /path/to/data --output /path/to/runs --dry-run
python -W ignore tools/run_all.py --data-base /path/to/data --output /path/to/runs --preflight-only
python -W ignore tools/run_all.py --data-base /path/to/data --output /path/to/runs
```

`--dry-run` writes a plan without loading data or training. `--audit-only` reads current definitions and existing run evidence without writing the result directory. `--prepare-only` seals data and prepares shared normalized inputs. The batch queue completes one requested seed before the next; xBD and CAU-Flood are placed after BRIGHT and Haiti within each seed. CLI seed order is preserved.

Use `--groups`, `--datasets` and `--seeds` to choose work. `tools/run_multiseed.py --config` accepts an isolated XML/JSON configuration. Presets are indexed by canonical IDs, so overlapping group selections do not duplicate a job.

## Progress and continuation

Console output includes overall and current progress, provisional ETA, best validation score, learning rate, early-stop progress and numerical-health counters. Warnings can be suppressed with `-W ignore`; errors retain their log paths. ETA adapts to measured timings and model types.

Repeating a compatible command resumes interrupted work from `last.pth` and verifies completed outputs. Continuation restores optimizer, scheduler, RNG and tracker state. Inference-only weight files cannot restore training state. Data, configuration and source checks reject incompatible reuse. Changes to weights, data processing or optimization require separate validation.

## Output layout

All generated files belong under the selected output directory:

```text
output/
  plan.json
  protocol.json
  summary/
  cache/
  runs/<dataset>/<group>/<configuration>/seed_<seed>/
    config_snapshot.json
    history.jsonl
    checkpoints/{best,last}.pth
    logs/
    test/
    artifacts/
```

The saved configuration identifies the checkpoint's model and input processing. `tools/test.py`, `tools/infer.py`, `tools/export_features.py` and `tools/benchmark.py` expose their checkpoint and output options through `--help`. Evaluation requires complete split coverage and finite predictions. Summary tools read the requested repetition set from the plan and leave incomplete inputs incomplete.

## Input reuse and profiling

Deterministic input preparation is cached before random augmentation. Cache identities include data and preprocessing settings; generated cache corruption is detected. Each sampled occurrence retains its own augmentation seed.

Parameter counting and forward profiling are separate tools. Benchmark inputs are randomly initialized device tensors; the report identifies this scope and excludes data loading, output writing and TTA. Compare timings only under matched device, shape, batch, precision and scan backend. Forward latency is not epoch training time.
