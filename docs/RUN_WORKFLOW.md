# Campaign progress and shared preparation

## Local entry point

```bash
python -W ignore tools/run_all.py
```

The local defaults select every catalog configuration on BRIGHT, Haiti, CAU-Flood
and xBD, with seeds 42, 1051 and 2060, `cuda:0` and the RTX 5090 runtime profile.
Inputs are under `~/桌面/myData/Hete_CD`; outputs are under
`~/桌面/myResult/DEHCD-Net`. The resolved protocol is sealed before training.
`--data-base` and `--output` override those locations. The Python environment must
already provide PyTorch and the installed selective-scan extension.

`--preflight-only` checks the entire plan and datasets. `--prepare-only` also
builds shared input caches and stops before training. `--dry-run` only writes a
plan. The normal command automatically reuses valid caches and resumes unfinished
runs. Completed runs are verified and skipped. For individual suites, manifests,
and advanced settings use `tools/run_multiseed.py` directly.

## Progress

An interactive terminal displays three progress bars: completed runs, the current
run's epochs, and train/validation/test batches. Batch loss and remaining time are
refreshed without transferring full prediction masks to the CPU. Epoch timing
includes validation and checkpoint snapshot overhead. The total ETA uses measured
same-experiment timings first, then measured model-family or dataset rates for
unseen configurations. It is explicitly an estimate and is recalibrated as runs
finish; it is not a promised completion deadline. Initialization is shown until
there is enough timing information.

`progress.txt` and `progress.json` at the output root can be read at any time.
They also work when stdout is redirected or the terminal is closed. Per-run logs
record detailed metrics and checkpoint events. Python warnings are suppressed in
the launcher and inherited subprocesses. Child console streams are stored in the
run's `logs/console.log`; failed tasks still show their error and log path in the
main console. Numerical guards, failure records and fail-fast behavior remain.

## Files

```text
DEHCD-Net/
  protocol.json
  progress.txt
  progress.json
  campaign/configs/<experiment>__seed_<seed>.json
  summary/
    jobs.csv
    per_seed.csv
    aggregate.json
  cache/<dataset>/<content-key>/
    manifest.json
    optical.bin
    sar.bin
    label.bin
    label_counts.npy             # training split only
  runs/<dataset>/<suite>/<experiment>/seed_<seed>/
    config_snapshot.json
    protocol.json
    progress.json
    history.jsonl
    training_summary.json
    checkpoints/{best,last}.pth
    logs/{console.log,run.log,...}
    test/
      result.json
      sample_metrics.csv
      sample_confusion_matrices.json
      per_event_metrics.json
      summary.csv
```

Both `best.pth` and `last.pth` retain model, optimizer, scheduler, scaler and RNG
states. Best uses the same validation metric and strict-improvement criterion;
last is updated every epoch. CPU copies isolate pending writes from subsequent
parameter updates. At most one write is outstanding; write failures are raised,
and a completion summary is published only after all writes finish. Atomic
replacement prevents partial checkpoint files. When best and last represent the
same epoch, a hard link avoids serializing identical bytes twice. Subsequent last
updates replace the file and do not change the older best state. Filesystems
without hard-link support use a copy. No completed checkpoints are removed.

## Shared preparation and equivalent computation

The campaign scans and seals each distinct data configuration once before model
training. It prepares normalized, aligned FP32 optical/SAR tensors and valid
integer labels before random cropping or augmentation, including Haiti's quality
mask policy. Labels that fit in uint8 are stored exactly and loaded as int64;
input floats are never quantized. Random crop choice, sampling with replacement,
augmentation, model initialization and epoch RNG schedules still run as configured.
Only the deterministic processing is cached. Metadata-return workflows can use
`--cache none` in the general runner.

All training label counts are stored, so configurations can select their original
statistics prefix and compute their own unchanged sampling weights without reading
labels again. Cache identities include source-file fingerprints, preprocessing
settings and preprocessing code. Cache bytes are hashed and verified on reuse;
changed source files fail the sealed protocol check, and damaged generated caches
are rebuilt. Source files are rechecked before a new cache is published. Cache
files are mapped read-only; each sampled item gets its own writable arrays before
augmentation. Jobs are grouped by dataset to reuse the OS page cache.

Confusion matrices use exact integer counts on the GPU and retain the same CPU
metric formulas. Input/output/gradient/parameter finite checks remain active;
grouping checks reduces host synchronization without permitting bad updates.
Per-image test metrics reuse their already-computed confusion matrices. Rolling
aggregation reuses validated metric rows; final reporting still verifies all
checkpoint identities. Model structure, epoch budgets, optimizer updates, losses,
physical and effective batches, precision, validation frequency and selection
criteria are unchanged. CUDA flow/grid_sample backward can still be nondeterministic,
so bitwise CUDA training reproducibility is not implied.
