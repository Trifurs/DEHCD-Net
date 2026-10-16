# RTX 5090 training settings

Pass `--runtime-profile rtx5090` to `tools/run_multiseed.py`. The profile is stored
in `configs/runtime/rtx5090.json`; all selected configurations and their references
receive the same execution settings within each dataset. CLI worker/batch/epoch
options override the profile uniformly and are recorded in the resolved configs.
Use a new output directory whenever these settings change.

| Dataset | Input | Physical batch | Accumulation | Nominal effective batch | Epochs | Main learning rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BRIGHT | 256×256 | 4 | 2 | 8 | 100 | 0.001 |
| Haiti | 128×128 | 24 | 1 | 24 | 1000 | 0.0005 |
| CAU-Flood | 256×256 | 4 | 3 | 12 | 100 | 0.001 |
| xBD | 256×256 | 4 | 3 | 12 | 100 | 0.0008 |

No model-specific automatic batch reduction, resolution change,
pretraining or shorter epoch budget is applied. Early-stop patience is shared within each dataset: 30 on BRIGHT/xBD/CAU-Flood and 100 on Haiti. The final partial
accumulation window uses its actual number of microbatches. Accumulation preserves
the nominal batch budget but is not mathematically identical to a large physical
batch for BatchNorm and batch-dependent Dice/Lovasz losses. Every model in a
comparison uses the same physical batch and accumulation count.

Execution settings:

- FP32 tensors and losses, AMP disabled; TF32 enabled for supported matrix and
  convolution operations. TF32 reduces multiplication mantissa precision and is
  recorded explicitly; this is not a claim of strict IEEE FP32 arithmetic.
- Fused AdamW on CUDA and batched gradient operations; non-finite checks, norm
  clipping and foreground-collapse detection remain enabled.
- 8 CPU threads, 8 loader workers and prefetch factor 2. Persistent training
  workers use a seed attached to each sampled occurrence, derived from the run
  seed, epoch and draw position. Repeated sampled tiles receive fresh augmentation.
  Worker count and restart do not determine augmentation values. CUDA flow/scan
  kernels can still prevent bitwise reproducibility.
- Full validation every epoch; best checkpoint still selected by foreground mIoU.
- Best and last checkpoints are retained. `save_interval=0` disables additional
  periodic epoch archives. No existing checkpoint is removed.
- Progress metrics refreshed every 20 batches; all batches still contribute to
  epoch metrics. Validation image grids are disabled; scalar logs remain enabled.
- Normalized inputs before random augmentation and full per-image label counts are prepared once and shared across all compatible models/seeds. Integer GPU confusion counts avoid full-image CPU transfers. Full checkpoints use bounded asynchronous CPU snapshots and atomic replacement; identical best/last states can share an immutable file link. See [campaign workflow](RUN_WORKFLOW.md). Repeated FLOPs
  profiling is disabled during the multiseed test phase; use `tools/benchmark.py`
  separately with identical device, precision and shapes for all models.

The profile targets RTX 5090 32 GiB. Memory demand depends on architecture and concurrent GPU use; verify a bounded forward/backward pass on a new device before a full campaign. Numerical and resource checks do not establish convergence.

```bash
python tools/run_multiseed.py --suite all --seeds 42 1051 2060 \
  --runtime-profile rtx5090 \
  --data-root bright=/path/to/BRIGHT1 --data-root haiti=/path/to/Haiti1 \
  --data-root cau_flood=/path/to/CAU1 --data-root xbd=/path/to/xBD1 \
  --device cuda:0 --output runs/all_3seeds_rtx5090 --resume
```

The catalog contains 80 unique configurations (240 target tasks at three seeds). Compatible existing training counts toward this target; the reuse manifest identifies remaining work. Split overlap
checks remain mandatory. Disk requirements include full best/last model and
optimizer states for every run; disabling periodic archives does not make those
states small. Reserve sufficient storage before launching the whole catalog.
