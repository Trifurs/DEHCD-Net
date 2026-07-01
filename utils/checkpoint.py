from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def load_model_state(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    unwrap_model(model).load_state_dict(clean_state_dict(state_dict))


def save_checkpoint(path: str | Path, model: torch.nn.Module, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["model"] = unwrap_model(model).state_dict()
    torch.save(payload, path)
