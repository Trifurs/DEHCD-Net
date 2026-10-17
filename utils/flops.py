"""Forward scalar-operation accounting, including functional and opaque kernels.

FMA=2; transcendental functions are one scalar operation. Counts describe the
executed inference path, not device instructions or latency. Unknown operators
make the total unavailable instead of being silently counted as zero.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from math import ceil, prod
import sys
import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode

CONVENTION = "scalar_forward_v1_fma2"
_ACTIVE = ContextVar("dehcd_flop_counter", default=None)
_LOCK = threading.RLock()

# Explicitly classified data movement, allocation, indexing and selection. These
# perform no counted floating arithmetic. An unknown name never joins this list
# just because its contribution seems small.
_ZERO = set("""
alias detach view view_as reshape _reshape_alias _unsafe_view flatten squeeze
unsqueeze permute transpose t expand expand_as contiguous clone copy_ _to_copy
to type_as lift_fresh lift_fresh_copy resolve_conj resolve_neg slice select
split split_with_sizes unbind chunk cat stack repeat repeat_interleave tile
flip roll index index_select gather
take take_along_dim _unsafe_index masked_select masked_fill masked_fill_ where
empty empty_like empty_strided new_empty new_empty_strided zeros zeros_like
new_zeros ones ones_like new_ones full full_like new_full fill_ zero_
constant_pad_nd reflection_pad1d reflection_pad2d reflection_pad3d
replication_pad1d replication_pad2d replication_pad3d unfold im2col
size sym_size stride sym_stride numel sym_numel is_contiguous storage_offset
_local_scalar_dense _assert_async _assert_scalar _assert_tensor_metadata
eq ne gt ge lt le logical_and logical_or logical_not bitwise_and bitwise_or
bitwise_not any all isnan isinf isfinite argmax argmin max min amax amin
maximum minimum fmax fmin clamp clamp_min clamp_max clamp_ clamp_min_ clamp_max_
relu relu_ hardtanh hardtanh_ threshold max_pool1d max_pool2d max_pool3d
max_pool2d_with_indices max_pool3d_with_indices adaptive_max_pool2d
_adaptive_max_pool2d upsample_nearest1d upsample_nearest2d upsample_nearest3d
_upsample_nearest_exact1d _upsample_nearest_exact2d _upsample_nearest_exact3d
embedding _pin_memory is_pinned record_stream _has_compatible_shallow_copy_type
""".split())
_UNARY = set("""
abs neg positive exp exp2 expm1 log log2 log10 log1p sqrt rsqrt reciprocal
sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh erf erfc erfinv
floor ceil round trunc frac sign sgn square pow atan2 hypot fmod
""".split())


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _first(value):
    return next(_tensors(value))


def _float_elements(value):
    return sum(t.numel() for t in _tensors(value) if t.is_floating_point())


def _arg(args, kwargs, index, key, default=None):
    return args[index] if len(args) > index else kwargs.get(key, default)


def selective_scan_flops(u, delta, A, B, C, D=None, delta_bias=None,
                         delta_softplus=False, *unused, **kwargs):
    """Count the real grouped SSM recurrence, including discretization/output.

    Per state/time: dt*A, exp, exp*state, dt*B*u, state add, state*C;
    summing N states takes N-1 adds. Optional D*u, delta bias and softplus
    are additional. CPU reference and CUDA kernels use this same convention.
    """
    if u.ndim != 3 or A.ndim != 2 or u.shape[1] != A.shape[0]:
        raise ValueError("Unsupported selective-scan input/state shapes")
    if B.ndim not in (3, 4) or C.ndim not in (3, 4):
        raise ValueError("Only real input-dependent B/C scans are supported")
    if any(t.is_complex() for t in (u, delta, A, B, C)):
        raise ValueError("Complex selective scans need a separate counting rule")
    outputs, states = u.numel(), A.shape[-1]
    count = outputs * (8 * states - 1)
    if D is not None:
        count += 2 * outputs
    if delta_bias is not None:
        count += outputs
    if delta_softplus:
        count += 3 * outputs  # exp + add + log, beta=1
    return int(count)


def _norm_flops(x, weight, bias, groups, moments):
    n = x.numel()
    # Mean, centering, variance and normalization; eps + rsqrt per group.
    base = 5 * n + 2 * groups if moments else 2 * n + 2 * groups
    return base + (n if weight is not None else 0) + (n if bias is not None else 0)


def _pool_area(input_size, output_size):
    return sum(ceil((i + 1) * input_size / output_size) - i * input_size // output_size
               for i in range(output_size))


def _rule(name, args, kwargs, output):
    """Return (scalar FLOPs, matrix/convolution MACs), or None if unsupported."""
    n = _float_elements(output)
    if name in _ZERO:
        return 0, 0
    if any(t.is_complex() for t in _tensors((args, output))):
        return None
    # Integer and boolean arithmetic is outside the FLOP convention.
    if not n and not any(t.is_floating_point() for t in _tensors(args)):
        return 0, 0
    if name in {"convolution", "_convolution", "conv1d", "conv2d", "conv3d"}:
        x, w = args[:2]
        y = _first(output)
        transposed = _arg(args, kwargs, 6, "transposed", False) if name in {"convolution", "_convolution"} else False
        spatial = x.shape[2:] if transposed else y.shape[2:]
        macs = x.shape[0] * prod(spatial) * w.numel()
        bias = _arg(args, kwargs, 2, "bias")
        return 2 * macs + (y.numel() if bias is not None else 0), macs
    if name in {"mm", "bmm", "matmul", "mv", "dot", "addmm", "baddbmm", "addmv"}:
        fused = name in {"addmm", "baddbmm", "addmv"}
        a, b = args[1:3] if fused else args[:2]
        macs = _first(output).numel() * (a.shape[-1] if a.ndim else 1)
        extra = 0
        if fused:
            alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 1)
            if alpha != 1:
                extra += n
            if beta != 0:
                extra += n * (1 + (beta != 1))
        return 2 * macs + extra, macs
    if name == "linear":
        x, weight = args[:2]
        macs = _first(output).numel() * weight.shape[-1]
        return 2 * macs + (n if _arg(args, kwargs, 2, "bias") is not None else 0), macs
    if name in {"index_put", "index_put_", "_unsafe_index_put"}:
        accumulated = _arg(args, kwargs, 3, "accumulate", False)
        return (_float_elements(_arg(args, kwargs, 2, "values")) if accumulated else 0), 0
    if name in {"scatter", "scatter_"}:
        reduction = kwargs.get("reduce")
        if reduction is None:
            return 0, 0
        if reduction in {"add", "multiply"}:
            return _float_elements(_arg(args, kwargs, 3, "src")), 0
        return None
    if name in {"add", "add_", "sub", "sub_", "rsub"}:
        return n * (1 + (kwargs.get("alpha", 1) != 1)), 0
    if name in {"mul", "mul_", "div", "div_", "true_divide", "floor_divide"}:
        return n * (1 + (kwargs.get("rounding_mode") is not None or name == "floor_divide")), 0
    if name == "addcmul" or name == "addcdiv":
        return n * (2 + (kwargs.get("value", 1) != 1)), 0
    if name == "remainder":
        return 4 * n, 0  # divide, floor, multiply, subtract
    if name in _UNARY:
        return n, 0
    if name == "sigmoid":
        return 4 * n, 0
    if name in {"silu", "silu_"}:
        return 5 * n, 0
    if name == "gelu":
        return (9 if kwargs.get("approximate", "none") == "tanh" else 5) * n, 0
    if name == "softplus":
        return (3 + 2 * (_arg(args, kwargs, 1, "beta", 1) != 1)) * n, 0
    if name == "logit":
        return 3 * n, 0
    if name in {"leaky_relu", "leaky_relu_", "prelu"}:
        return n, 0
    if name in {"_softmax", "softmax", "_log_softmax", "log_softmax"}:
        x = args[0]
        dim = int(_arg(args, kwargs, 1, "dim", -1))
        rows = x.numel() // max(x.shape[dim], 1)
        return 4 * x.numel() - (0 if "log_softmax" in name else rows), 0
    if name in {"sum", "mean", "nansum"}:
        input_n = _float_elements(args[0])
        return max(0, input_n - n) + (n if name == "mean" else 0), 0
    if name == "prod":
        return max(0, _float_elements(args[0]) - n), 0
    if name in {"cumsum", "cumprod"}:
        x = args[0]
        dim = int(_arg(args, kwargs, 1, "dim", -1))
        return max(0, n - n // max(x.shape[dim], 1)), 0
    if name in {"scatter_add", "scatter_add_", "index_add", "index_add_"}:
        return _float_elements(_arg(args, kwargs, 3, "src", kwargs.get("source"))), 0
    if name == "linalg_vector_norm":
        x = args[0]
        order = _arg(args, kwargs, 1, "ord", 2)
        if order == 2:
            return 2 * x.numel(), 0
        if order == 1:
            return 2 * x.numel() - n, 0
        if order in (0, float("inf"), -float("inf")):
            return (x.numel() if order else 0), 0
        return 3 * x.numel(), 0
    if name in {"native_layer_norm", "layer_norm"}:
        x, shape = args[:2]
        groups = x.numel() // prod(shape)
        return _norm_flops(x, _arg(args, kwargs, 2, "weight"), _arg(args, kwargs, 3, "bias"), groups, True), 0
    if name in {"native_group_norm", "group_norm"}:
        x = args[0]
        if name == "native_group_norm":
            weight, bias, groups = args[1], args[2], args[6]
        else:
            groups, weight, bias = args[1], _arg(args, kwargs, 2, "weight"), _arg(args, kwargs, 3, "bias")
        return _norm_flops(x, weight, bias, x.shape[0] * groups, True), 0
    if name in {"native_batch_norm", "cudnn_batch_norm", "miopen_batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training", "batch_norm"}:
        x, weight, bias = args[:3]
        training = False if name.endswith("no_training") else _arg(args, kwargs, 5, "training", False)
        return _norm_flops(x, weight, bias, x.shape[1], training), 0
    if name in {"_adaptive_avg_pool2d", "adaptive_avg_pool2d", "avg_pool2d"}:
        x, y = args[0], _first(output)
        if name == "avg_pool2d":
            kernel = list(args[1]); kernel = kernel * 2 if len(kernel) == 1 else kernel
            # Include padded zero terms: a dense window convention like convolution.
            return y.numel() * prod(kernel), 0
        area = _pool_area(x.shape[-2], y.shape[-2]) * _pool_area(x.shape[-1], y.shape[-1])
        return x.shape[0] * x.shape[1] * area, 0
    if name == "upsample_bilinear2d":
        h, w = _first(output).shape[-2:]
        # Four shared bilinear weights per pixel, seven blend operations/value;
        # source-coordinate affine map, floor, fraction and complement per axis.
        return 7 * n + 4 * h * w + 5 * (h + w), 0
    if name in {"grid_sampler", "grid_sampler_2d"}:
        grid = args[1]
        mode, padding = _arg(args, kwargs, 2, "interpolation_mode", 0), _arg(args, kwargs, 3, "padding_mode", 0)
        pixels = prod(grid.shape[:-1])
        if padding not in (0, 1):
            return None  # reflection needs an additional coordinate rule
        if mode == 0:
            return 7 * n + 14 * pixels, 0
        if mode == 1:
            return 6 * pixels, 0
        return None
    if name == "linspace":
        return (2 * n + 2 if n > 1 else 0), 0
    if name == "arange":
        return 2 * n, 0
    if name in {"native_dropout", "dropout"}:
        p = _arg(args, kwargs, 1, "p", 0)
        training = _arg(args, kwargs, 2, "train", True)
        return (2 * _float_elements(args[0]) if training and 0 < p < 1 else 0), 0
    if name.startswith("_scaled_dot_product_") or name == "scaled_dot_product_attention":
        q, k, v = args[:3]
        rows = prod(q.shape[:-1])
        scores = rows * k.shape[-2]
        macs = scores * (q.shape[-1] + v.shape[-1])
        # QK, score scaling, stable softmax, AV. Dropout/mask are backend args.
        extra = scores + 4 * scores - rows
        mask = kwargs.get("attn_mask", kwargs.get("attn_bias"))
        if mask is not None and mask.is_floating_point():
            extra += scores
        p = kwargs.get("dropout_p", 0)
        if p:
            extra += 2 * scores
        return 2 * macs + extra, macs
    return None


class FlopCounter(TorchDispatchMode):
    """Count executed ATen operations; publish a total only with full coverage."""

    def __init__(self):
        super().__init__()
        self.flops = 0
        self.matrix_macs = 0
        self.operators = defaultdict(lambda: {"calls": 0, "flops": 0, "matrix_macs": 0})
        self.unsupported = Counter()
        self.module_flops = Counter()
        self.stack = []
        self.suspended = 0

    def record(self, key, result):
        self.operators[key]["calls"] += 1
        if result is None:
            self.unsupported[key] += 1
            return
        flops, macs = map(int, result)
        self.flops += flops
        self.matrix_macs += macs
        self.operators[key]["flops"] += flops
        self.operators[key]["matrix_macs"] += macs
        self.module_flops[self.stack[-1] if self.stack else "<functional>"] += flops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        if not self.suspended:
            name = func._schema.name.split("::")[-1]
            # Namespace is significant: a custom op named "add" is not ATen add.
            bound = dict(kwargs)
            bound.update({schema.name: value for schema, value in zip(func._schema.arguments, args)})
            result = _rule(name, args, bound, output) if func._schema.name.startswith("aten::") else None
            self.record(str(func), result)
        return output

    @contextmanager
    def opaque_kernels(self):
        restored = []
        sentinel = object()
        token = _ACTIVE.set(self)

        def install(cls, name, formula):
            previous = cls.__dict__.get("apply", sentinel)
            original = cls.apply

            def apply(*args, **kwargs):
                counter = _ACTIVE.get()
                if counter is not self or self.suspended:
                    return original(*args, **kwargs)
                self.suspended += 1
                try:
                    result = original(*args, **kwargs)
                finally:
                    self.suspended -= 1
                self.record(name, (formula(*args, **kwargs), 0))
                return result

            setattr(cls, "apply", staticmethod(apply))
            restored.append((cls, previous))

        try:
            scan = sys.modules.get("compare.official.changemamba.scan_backend")
            if scan:
                for name in ("ReferenceScan", "AutoScan"):
                    install(getattr(scan, name), "custom::selective_scan", selective_scan_flops)
            vmamba = sys.modules.get("compare.official.changemamba.vmamba")
            if vmamba:
                for name in ("SelectiveScanMamba", "SelectiveScanCore", "SelectiveScanOflex"):
                    install(getattr(vmamba, name), "custom::selective_scan", selective_scan_flops)
            triton = sys.modules.get("compare.official.changemamba.csm_triton")
            if triton:
                for name in ("CrossScanTriton", "CrossScanTriton1b1"):
                    install(getattr(triton, name), "custom::cross_scan_copy", lambda x: 0)
                install(triton.CrossMergeTriton, "custom::cross_merge", lambda x: 3 * x.numel() // 4)
            yield
        finally:
            for cls, previous in reversed(restored):
                if previous is sentinel:
                    delattr(cls, "apply")
                else:
                    setattr(cls, "apply", previous)
            _ACTIVE.reset(token)

    def report(self):
        complete = not self.unsupported
        return {
            "convention": CONVENTION, "complete": complete,
            "flops": self.flops if complete else None,
            "gflops": self.flops / 1e9 if complete else None,
            "counted_flops": self.flops,
            "matrix_macs": self.matrix_macs,
            "matrix_gmacs": self.matrix_macs / 1e9,
            "matrix_macs_scope": "Convolutions and dense matrix products only; not total FLOPs/2",
            "macs": None, "gmacs": None, "flops_2x_macs": None, "gflops_2x_macs": None,
            "unsupported_operators": dict(sorted(self.unsupported.items())),
            "operators": dict(sorted(self.operators.items())),
            "top_modules_by_flops": [{"module": name, "flops": value, "gflops": value / 1e9}
                                     for name, value in self.module_flops.most_common(30)],
            "scope": "One full model forward, including executed input/output adapters and auxiliary heads; excludes data loading, loss, backward and TTA",
            "note": "Theoretical scalar operations under the declared convention, not hardware instructions. FMA=2. Transcendentals count as one operation; memory/indexing/comparisons/selections=0. Fused kernels use documented analytical rules. A total is available only when all encountered operators have a rule.",
        }


def profile_forward(model, optical, sar):
    """Profile an eval forward while restoring flags, RNG and temporary hooks."""
    from utils.model_outputs import extract_logits

    devices = sorted({x.device.index for x in (optical, sar) if x.is_cuda})
    flags = [(module, module.training) for module in model.modules()]
    hooks = []
    counter = FlopCounter()
    with _LOCK, torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            for name, module in model.named_modules():
                label = name or "<model>"
                hooks.append(module.register_forward_pre_hook(lambda mod, args, label=label: counter.stack.append(label)))
                def leave(mod, args, result, label=label):
                    if counter.stack and counter.stack[-1] == label:
                        counter.stack.pop()
                hooks.append(module.register_forward_hook(leave, always_call=True))
            with torch.no_grad(), counter.opaque_kernels(), counter:
                output = model(optical, sar)
            logits = extract_logits(output)
        finally:
            for hook in hooks:
                hook.remove()
            for module, training in flags:
                module.training = training
    return {
        "input_shape": {"optical": list(optical.shape), "sar": list(sar.shape), "logits": list(logits.shape)},
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_millions": sum(p.numel() for p in model.parameters()) / 1e6,
            "trainable_millions": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
            "parameter_memory_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2,
            "buffer_memory_mb": sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 ** 2,
        },
        "complexity": counter.report(),
    }
