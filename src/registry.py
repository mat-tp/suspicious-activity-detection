from __future__ import annotations
from .common import *
from .paths import *
from .interfaces import *

_REGISTRIES: Dict[str, Dict[str, type]] = {
    "detector": {}, "tracker": {}, "preprocessor": {},
    "feature": {}, "model": {},
}


def register(category: str, name: str) -> Callable:
    if category not in _REGISTRIES:
        raise KeyError(f"Unknown category '{category}'. Valid: {list(_REGISTRIES.keys())}")
    def decorator(cls: type) -> type:
        if name in _REGISTRIES[category]:
            raise ValueError(f"'{name}' already registered under '{category}'")
        _REGISTRIES[category][name] = cls
        return cls
    return decorator


def build(category: str, name: str, **kwargs) -> Any:
    if category not in _REGISTRIES or name not in _REGISTRIES[category]:
        available_list = list(_REGISTRIES.get(category, {}).keys())
        raise KeyError(f"'{name}' not registered under '{category}'. Available: {available_list}")
    return _REGISTRIES[category][name](**kwargs)


def available(category: str) -> list:
    return sorted(_REGISTRIES.get(category, {}).keys())


def is_registered(category: str, name: str) -> bool:
    return name in _REGISTRIES.get(category, {})
