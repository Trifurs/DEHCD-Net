# Forward complexity

`python -W ignore tools/profile_models.py` profiles configured models without training or reading image data. It writes structural profiles under the configured output directory's `model_profiles/`. The generated inputs and randomly initialized weights are used for structural accounting, not accuracy measurement.

```bash
python -W ignore tools/profile_models.py --device cuda:0
python -W ignore tools/profile_models.py --config configs/experiments/main/bright_dehcd_l.xml --size 256 256 --device cuda:0 --output /path/to/profiles
```

`--groups` and `--datasets` select preset configurations. Without `--size`, each configuration's patch size is used; the default batch size is one paired input. Channels come from the declared dataset/model contract. For custom layouts, pass `--optical-channels` and `--sar-channels` explicitly. Input shapes, configuration hash, implementation details and backend are saved with each profile.

`tools/test.py` uses the same counter and includes the profile in evaluation outputs. Its `--profile-size` overrides only the profiling input size. It does not resize the test dataset. `--no-profile` skips this optional step.

## Counting convention

The `scalar_forward_v1_fma2` convention counts theoretical scalar operations in one complete model forward. A multiply-add counts as two FLOPs. A scalar addition, multiplication, division, absolute value or transcendental function counts as one operation. This is an arithmetic convention, not a count of device instructions or clock cycles. It is independent of TF32/Tensor Core throughput.

Memory movement, allocation, reshaping, indexing, integer arithmetic, comparisons and selections count as zero floating arithmetic. This includes ReLU and max pooling. Their runtime and memory costs still matter for latency. Input loading/preprocessing outside the model, loss, backward, optimizer updates and TTA are outside the forward scope.

The counter observes actual ATen execution, including function calls that bypass `nn.Module.forward`. Repeated calls to shared modules are counted each time. It includes input/output adapters and every head executed by the model, even if only the primary logits are selected afterwards. Matrix attention, functional projections/convolutions, dynamic convolution, wavelet filters, HOG arithmetic and scatter accumulation are therefore included.

| Operation | Rule |
|---|---|
| Dense/grouped convolution | Twice the weight products at all output locations, plus output bias additions; padded zero products use the dense-window convention. |
| Transposed convolution | Products use input locations and the actual grouped weight shape, plus output bias additions. |
| Matrix products | Two operations per multiply-add; fused bias, alpha and beta arithmetic is additional. |
| Softmax | Stable shift, exponentiation, reduction and division; comparisons are excluded. |
| Fused attention | QK and AV products, scaling, softmax and floating mask additions; kernel-internal work is counted analytically once. |
| Normalization | Mean/variance when required, centering, epsilon, reciprocal square root and affine terms. Canonical group/layer normalization is `5E + 2G`, plus `E` for each affine multiplier/bias, where `E` is input elements and `G` is normalization groups. |
| Bilinear resize | Four shared weights per output pixel and seven blending operations per output value, plus separable source-coordinate/weight construction. |
| Bilinear grid sampling | Seven blending operations per output value plus fourteen coordinate/weight operations per grid point; zero/border padding uses comparison/selection without additional FLOPs. |
| Average pooling | Dense window accumulation and division; adaptive pooling uses actual bin extents. |
| Activations | Sigmoid: four; SiLU: five; exact GELU: five; tanh-approximate GELU: nine scalar operations per element. |

Fused operations use the documented canonical expressions. Unfused paths are counted as executed; different decompositions can have different scalar-operation totals. Compare models with the same counter convention, device/backend, batch size and explicitly recorded input dimensions. Internal adapter resizing is included and remains visible in implementation metadata.

## Selective state-space scan

The CUDA scan is opaque to ordinary ATen counters. A profiling-only wrapper adds its analytical cost and suppresses internal double counting. The same formula applies to the torch reference backend; temporary wrappers are restored after profiling, including on failure.

For `O = batch × channels × sequence_length` and state width `N`, the recurrence is:

```text
h[t] = exp(delta[t] * A) * h[t-1] + delta[t] * B[t] * u[t]
y[t] = sum(C[t] * h[t]) + D * u[t]
```

The recurrence and state reduction cost `O × (8N − 1)`. Optional skip `D*u` adds `2O`, delta bias adds `O`, and softplus adds `3O`. Directional projections, merges, gating and output layers are counted separately from their executed operators. CPU reference arithmetic and CUDA scan counting use this same convention.

## Output and coverage

- `profiles/<configuration>.json`: parameters, shapes, scalar `flops`/`gflops`, per-operator counts, leading module contributions and backend metadata.
- `summary.csv`: parameter counts, GFLOPs and coverage status.
- `index.json`: completion status and failed configurations.

`complexity.complete=true` means every encountered tensor operator and known opaque model kernel has a counting rule under this convention. Unsupported operators are listed explicitly. Any unsupported operation makes `flops` and `gflops` unavailable, and the standalone command exits unsuccessfully after writing diagnostics. Unknown operations are never silently assigned zero.

`matrix_macs`/`matrix_gmacs` describe only convolution and matrix-product multiply-accumulates. They are components of the total, not total FLOPs divided by two. The deprecated `macs`, `gmacs`, `flops_2x_macs` and `gflops_2x_macs` fields remain unset so consumers must explicitly choose the full `gflops` field.

Profiling restores model train/eval flags, RNG state, counting hooks and scan wrappers. It does not update weights or optimizer state. Use `tools/benchmark.py` separately for measured latency, throughput and allocated GPU memory; those quantities cannot be inferred from FLOPs alone.
