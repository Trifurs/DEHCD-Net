# Runtime profiles

`--runtime-profile` selects execution settings from `configs/runtime/`. The default `none` uses the configuration directly. The `rtx5090` profile supplies shared microbatch, accumulation, worker, TF32 and optimizer settings for its target GPU.

Profile settings apply uniformly within each dataset. CLI worker/batch options override the selected profile and are saved in run-specific configurations. The tool does not silently lower batch size or change image resolution for individual models.

The supplied profile uses FP32 tensors/losses with TF32 enabled, fused CUDA optimizer operations, persistent workers and batched gradient checks. Early stopping and checkpoint selection remain configured explicitly. Accumulation uses the actual size of the final partial window; batch-dependent normalization and losses can differ from a single larger physical batch.

On another device, inspect the profile and validate a bounded forward/backward pass before a full run. BF16 requires a supporting GPU. ChangeMamba CUDA execution requires a matching compiled scan extension. See [extension build](../compare/UPSTREAM.md), [configuration](CONFIGURATION.md) and [running tools](RUN_WORKFLOW.md).
