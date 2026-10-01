# Campaign workflow

## Entry point

Run from the repository with the existing training Python environment:

```bash
python -W ignore tools/run_all.py --audit-only
python -W ignore tools/run_all.py --dry-run
python -W ignore tools/run_all.py --preflight-only
python -W ignore tools/run_all.py
```

`--audit-only` reads current definitions and results without modifying the result directory. `--dry-run` writes the target plan. `--preflight-only` audits data and existing evidence, registers compatible imports and stops before training. The default command completes outstanding work, and repeating the same command continues after interruption. `--prepare-only` prepares shared deterministic inputs without training.

The default collection is 80 unique configurations with seeds 42,1051,2060, hence 240 target tasks including existing compatible work. Dataset order is **BRIGHT, Haiti, xBD, CAU-Flood** so xBD and CAU-Flood remain last. Within each dataset the order is completed-result reuse, required reevaluation, compatible continuation and missing/new training. Current task state, not a stale configuration-directory scan, determines the remaining queue.

Defaults: data at `~/桌面/myData/Hete_CD/{BRIGHT1,Haiti1,xBD1,CAU1}`, results at `~/桌面/myResult/DEHCD-Net`, device cuda:0 and RTX5090 runtime profile. Use `--data-base`/`--output` for different locations. No environment upgrade or CUDA compiler rebuild is needed for ordinary use of the configured environment.

```bash
python -W ignore tools/run_all.py --groups main ablation sensitivity scaling --dry-run
python -W ignore tools/compare_results.py --campaign "$HOME/桌面/myResult/DEHCD-Net" --groups main ablation sensitivity scaling --output "$HOME/桌面/myResult/DEHCD-Net/summary/analysis"
```

Groups refer to canonical IDs. Their union is taken before seed expansion, so main/scaling/default points do not repeat training. The [generated catalog](EXPERIMENT_CATALOG.md) lists all groups and figure coverage. Group statistics report only verified per-seed test results, n, expected_n=3 and complete status. Missing runs remain incomplete, not zero-filled.

## Progress

Interactive output shows total runs, current epoch and train/validation/test batches, with provisional ETA. It includes the best validation foreground_miou and its epoch, current validation result, learning rate and no-improvement count. **Ordinary early stopping is disabled** for the formal fixed-budget protocol; no-improvement monitoring is not an active patience countdown. Foreground collapse/stall counters are shown separately as numerical/learning-health protection. These failures cannot count as completed experiments.

Total ETA uses measured timings, preferring the same experiment or model family, then dataset rates. It is provisional and recalibrates as different architectures finish; unmeasured model families can differ greatly. It is not a completion deadline or a guaranteed upper bound. Initialization is shown until timings are available.

Root progress.json/progress.txt and each run's progress.json/logs remain readable when stdout is redirected. Python warnings are suppressed in the launcher and subprocesses. Real errors and their log paths remain visible. No full prediction masks are transferred just to refresh console counters.

## Files and provenance

```text
DEHCD-Net/
  plan.json
  protocol.json
  reuse_manifest.json
  cleanup_manifest.json
  audit_report.json
  progress.json
  progress.txt
  campaign/configs/<canonical_id>__seed_<seed>.json
  summary/{jobs.csv,per_seed.csv,aggregate.json,...}
  cache/<dataset>/<content-key>/
  runs/<dataset>/<suite>/<canonical_id>/seed_<seed>/
    config_snapshot.json
    protocol.json
    history.jsonl
    progress.json
    training_summary.json
    checkpoints/{best,last}.pth
    logs/
    test/{result.json,sample_metrics.csv,sample_confusion_matrices.json,...}
    artifacts/
```

Provenance records preserve the original source/config hashes and connect imported evidence to the current canonical task. Documentation and display metadata do not invalidate scientifically compatible weights; training-affecting changes still require strict review. No unconditional force mode bypasses scientific compatibility. Historical stat verification is labeled stat_verified and cannot be promoted to proof of historical byte content by hashing today.

Best and last retain full model/optimizer/scheduler/scaler/RNG/epoch/best/protection state. Last updates atomically each epoch. Asynchronous writing uses immutable CPU snapshots and at most one pending write, whose failure propagates before completion is recorded. Identical best/last bytes may share a hard link; later atomic last replacement leaves the older best unchanged. Inference-only weights cannot provide full continuation. Active run files are not hot-edited, migrated or deleted.

Cleanup is restricted to confirmed project/results paths and recorded with reason, migration validation and actual status. Required heads/licenses, source evidence, unaudited checkpoints and data are retained.

## Shared preparation

Each distinct data version is sealed once. Deterministic alignment, normalization and label preparation produce lossless FP32 inputs and exact integer labels before random crops/augmentation. Haiti quality masks are preserved. Full label counts support unchanged weighting/sampling; every sampled occurrence keeps its own run/epoch/draw RNG seed. Read-only memmaps are copied before augmentation.

Cache keys include source fingerprints, preprocessing settings/code. Cache bytes are verified on reuse, corrupt generated caches rebuilt, and source identity rechecked before publication. Crop freedom/target applicability and actual sampler nonuniformity are audited from current data, without adding training groups. Integer GPU confusion accumulation and shared per-image counts avoid redundant transfers while retaining metric definitions.

## Checkpoint postprocessing

Use existing run snapshots, not unrelated templates, for evaluation and efficiency. Seed42 and the first16 lexical test IDs are fixed for displays; feature --sample-id may select only a subset of that pool. Feature response PNGs are mean absolute activations, scaled independently per image/module. Raw CHW arrays and scaling metadata are saved; colors across independently scaled modules are not comparable absolute magnitudes.

```bash
python -W ignore tools/export_features.py --run-dir "$HOME/桌面/myResult/DEHCD-Net/runs/bright/main/bright_dehcd_l/seed_42" --output "$HOME/桌面/myResult/DEHCD-Net/runs/bright/main/bright_dehcd_l/seed_42/artifacts/features" --device cuda:0
python -W ignore tools/benchmark.py --config "$HOME/桌面/myResult/DEHCD-Net/runs/bright/main/bright_dehcd_l/seed_42/config_snapshot.json" --checkpoint "$HOME/桌面/myResult/DEHCD-Net/runs/bright/main/bright_dehcd_l/seed_42/checkpoints/best.pth" --optical-channels 3 --sar-channels 1 --size 256 256 --batch-size 1 --device cuda:0 --warmup 20 --iterations 100 --output "$HOME/桌面/myResult/DEHCD-Net/runs/bright/main/bright_dehcd_l/seed_42/artifacts/benchmark.json"
```

These commands require that run's checkpoint to exist. Feature export never trains or creates accuracy results. HOG keeps optical/SAR streams; after DPM fusion the captured representation is shared, not two independent modality rows. Feature maps cannot prove causal synergy or physical separation. Parameter counting and matched-device profiling need not repeat for every seed. Benchmark reports checkpoint SHA, backend, execution precision, forward latency/throughput and peak allocated memory on synthetic device-resident inputs; it excludes disk I/O and TTA and is not epoch training time. Omit --checkpoint only for explicitly labeled random-initialization structural profiling; --weights-only selects restricted loading of plain weight files.

All full-test results still require complete split coverage and finite outputs. No feature/visualization subset or partial smoke result enters formal statistics.
