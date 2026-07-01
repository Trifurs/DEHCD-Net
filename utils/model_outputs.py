from __future__ import annotations

from typing import Any

import torch


def extract_logits(output: Any) -> torch.Tensor:
    """Return the primary logits tensor from a tensor/list/dict model output."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "out", "prediction", "pred"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError(f"Model output dict does not contain logits keys: {list(output.keys())}")
    if isinstance(output, (list, tuple)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def extract_aux_logits(output: Any) -> list[torch.Tensor]:
    """Return auxiliary logits used for deep supervision when present."""

    if not isinstance(output, dict):
        return []
    aux = output.get("aux_logits", output.get("aux", []))
    if isinstance(aux, torch.Tensor):
        return [aux]
    if isinstance(aux, (list, tuple)):
        return [item for item in aux if isinstance(item, torch.Tensor)]
    return []


def extract_feature_pairs(output: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return paired modality features for optional alignment losses."""

    if not isinstance(output, dict):
        return []
    pairs = output.get("feature_pairs", [])
    if not isinstance(pairs, (list, tuple)):
        return []
    valid_pairs = []
    for item in pairs:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], torch.Tensor)
            and isinstance(item[1], torch.Tensor)
        ):
            valid_pairs.append((item[0], item[1]))
    return valid_pairs
