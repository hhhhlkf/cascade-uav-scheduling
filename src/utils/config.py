from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    defaults = data.get("defaults", [])
    if not defaults:
        return data

    merged: Dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, str):
            default_path = path.parent / f"{item}.yaml"
        elif isinstance(item, Mapping):
            name = next(iter(item.values()))
            default_path = path.parent / f"{name}.yaml"
        else:
            continue
        if default_path.exists():
            merged = deep_merge(merged, load_yaml_config(default_path))
    return deep_merge(merged, data)

