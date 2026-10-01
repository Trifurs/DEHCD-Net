"""Explicit VMamba scan backends; the reference path evaluates the actual SSM.

The recurrence is h_t = exp(delta_t A) h_(t-1) + delta_t B_t u_t,
y_t = C_t h_t + D u_t, with softplus(delta + bias). Four spatial scan
directions, projections and merging remain in the official VMamba code.
"""
from functools import partial
import torch
import torch.nn.functional as F


def selective_scan_reference(u, delta, A, B, C, D=None, delta_bias=None,
                             delta_softplus=False, nrows=1, backnrows=1, oflex=True):
    dtype = u.dtype
    # Keep double for numerical gradient tests; real inference/training uses FP32.
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    u, delta, A, B, C = [t.to(work_dtype) for t in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.to(work_dtype)[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, channels, length = u.shape
    if B.ndim == 3:
        B = B.unsqueeze(1)
    if C.ndim == 3:
        C = C.unsqueeze(1)
    if B.ndim != 4 or C.ndim != 4 or channels % B.shape[1] or channels % C.shape[1]:
        raise ValueError("Expected grouped, input-dependent B/C with shape (batch, groups, state, length)")
    B = B.repeat_interleave(channels // B.shape[1], dim=1)
    C = C.repeat_interleave(channels // C.shape[1], dim=1)
    state = u.new_zeros(batch, channels, A.shape[-1])
    values = []
    for t in range(length):
        dt = delta[:, :, t, None]
        state = torch.exp(dt * A[None]) * state + dt * B[:, :, :, t] * u[:, :, t, None]
        values.append((state * C[:, :, :, t]).sum(-1))
    y = torch.stack(values, dim=-1)
    if D is not None:
        y = y + D.to(work_dtype)[None, :, None] * u
    return y if oflex else y.to(dtype)


class ReferenceScan:
    # Ordinary torch autograd, not an opaque custom Function without a backward.
    apply = staticmethod(selective_scan_reference)


class AutoScan:
    @staticmethod
    def apply(u, *args):
        if not u.is_cuda:
            return selective_scan_reference(u, *args)
        from . import vmamba
        if not hasattr(vmamba, "selective_scan_cuda_oflex"):
            raise RuntimeError("ChangeMamba CUDA requires selective_scan_cuda_oflex. Install "
                               "compare/kernels/selective_scan (see compare/UPSTREAM.md), or explicitly "
                               "set model.selective_scan_backend=torch for the slower reference SSM.")
        return vmamba.SelectiveScanOflex.apply(u, *args)


def configure_scan_backend(model, backend="auto"):
    from . import vmamba
    if backend not in {"auto", "torch", "cuda"}:
        raise ValueError("selective_scan_backend must be auto, torch or cuda")
    if backend == "cuda" and not hasattr(vmamba, "selective_scan_cuda_oflex"):
        raise RuntimeError("selective_scan_cuda_oflex is not installed; see compare/UPSTREAM.md")
    scan = {"auto": AutoScan, "torch": ReferenceScan, "cuda": vmamba.SelectiveScanOflex}[backend]
    for module in model.modules():
        if isinstance(module, vmamba.SS2D):
            # Published Tiny v3noz: same v3 forward and ordinary four-direction scan.
            module.forward_core = partial(module.forward_corev2, force_fp32=False, SelectiveScan=scan)
    model.selective_scan_backend = backend


def resolved_backend(backend, device):
    if backend == "torch" or (backend == "auto" and torch.device(device).type == "cpu"):
        return "torch_reference_ssm"
    return "selective_scan_cuda_oflex"
