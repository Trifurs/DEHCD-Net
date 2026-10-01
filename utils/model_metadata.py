"""Describe the actual implementation/backend, separately from task accuracy."""
from utils.checkpoint import unwrap_model


def model_metadata(model, device):
    model = unwrap_model(model)
    info = dict(getattr(model, "implementation_info", {}))
    info["core_precision_policy"] = getattr(model, "core_precision_policy", "outer_autocast")
    info["parameters"] = sum(p.numel() for p in model.parameters())
    info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    resize_to = getattr(model, "resize_to", None)
    info["adapter_spatial_transform"] = ({"type": "bilinear_resize", "core_input_size": [resize_to, resize_to],
                                          "output": "resize_logits_to_input", "align_corners": False}
                                         if resize_to else {"type": "no_adapter_resize"})
    if hasattr(model, "adapt_batchnorm"):
        info["batchnorm_replaced"] = bool(model.adapt_batchnorm)
    core = getattr(model, "model", model)
    backend = getattr(core, "selective_scan_backend", None)
    if backend is not None:
        from compare.official.changemamba.scan_backend import resolved_backend
        info["selective_scan_backend_requested"] = backend
        info["selective_scan_backend_resolved"] = resolved_backend(backend, device)
        if info["selective_scan_backend_resolved"] == "selective_scan_cuda_oflex":
            from compare.official.changemamba import vmamba
            from utils.protocol import file_digest
            extension = getattr(vmamba, "selective_scan_cuda_oflex", None)
            if extension is None:
                raise RuntimeError("The declared CUDA scan backend is unavailable")
            info["selective_scan_extension_sha256"] = file_digest(extension.__file__)
    return info
