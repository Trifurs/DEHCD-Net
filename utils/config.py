from __future__ import annotations

import ast
import copy
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class ConfigNode(dict):
    """Small dict wrapper with dotted-key lookup for XML-driven experiments."""

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if not isinstance(key, str) or "." not in key:
            return super().get(key, default)
        cur: Any = self
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def require(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None:
            raise KeyError(f"Missing required config key: {key}")
        return value

    def as_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)

    def flatten(self, prefix: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for key, value in self.items():
            joined = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Mapping):
                flat.update(ConfigNode(value).flatten(joined))
            else:
                flat[joined] = value
        return flat


class XMLConfigParser:
    """Parse both project-style nested XML and legacy ``<startParam>`` XML files."""

    def __init__(self, config_path: str | os.PathLike[str]):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config_dir = self.config_path.parent

    def parse(self) -> ConfigNode:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        data = self._parse_file(self.config_path, visited=set())
        return ConfigNode(data)

    def _parse_file(self, path: Path, visited: set[Path]) -> Dict[str, Any]:
        path = path.expanduser().resolve()
        if path in visited:
            raise ValueError(f"Circular XML base_param include detected: {path}")
        visited.add(path)

        root = ET.parse(path).getroot()
        data = self._parse_element(root)

        if root.tag == "config_ref" and "path" in data:
            ref = self._resolve_include_path(path.parent, data["path"])
            return self._parse_file(ref, visited)

        base_path = data.pop("base_param", None)
        if base_path:
            base_file = self._resolve_include_path(path.parent, str(base_path))
            base_data = self._parse_file(base_file, visited)
            data = merge_dicts(base_data, data)

        return data

    def _parse_element(self, element: ET.Element) -> Any:
        children = list(element)
        if not children:
            return self._convert_value(element.text, element.attrib.get("type"))

        if all(child.tag == "param" for child in children):
            return self._parse_legacy_params(children)

        grouped: Dict[str, list[Any]] = {}
        for child in children:
            grouped.setdefault(child.tag, []).append(self._parse_element(child))

        parsed: Dict[str, Any] = {}
        for key, values in grouped.items():
            parsed[key] = values[0] if len(values) == 1 else values
        return parsed

    def _parse_legacy_params(self, params: Iterable[ET.Element]) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        for param in params:
            name = self._child_text(param, "name")
            if not name:
                continue
            value_type = self._child_text(param, "type")
            value = self._child_text(param, "value")
            parsed[name] = self._convert_value(value, value_type)
        return parsed

    @staticmethod
    def _child_text(element: ET.Element, tag: str) -> Optional[str]:
        child = element.find(tag)
        if child is None:
            return None
        return child.text

    @staticmethod
    def _resolve_include_path(base_dir: Path, include_path: str) -> Path:
        path = Path(include_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()

    @staticmethod
    def _convert_value(text: Optional[str], value_type: Optional[str]) -> Any:
        raw = "" if text is None else text.strip()
        type_name = (value_type or "").strip().lower()

        if raw.lower() in {"none", "null"}:
            return None
        if raw == "" and type_name not in {"str", "string", "path", "file", "folder"}:
            return None

        if type_name in {"int", "integer"}:
            return int(raw)
        if type_name in {"float", "double"}:
            return float(raw)
        if type_name in {"bool", "boolean"}:
            return raw.lower() in {"true", "1", "yes", "y", "on"}
        if type_name in {"list", "tuple", "dict"}:
            if raw == "":
                return []
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return [item.strip() for item in raw.split(",") if item.strip()]
        if type_name in {"path", "file", "folder"}:
            return raw

        return raw


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_plain_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value
