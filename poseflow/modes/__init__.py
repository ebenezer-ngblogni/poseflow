from .base import Mode, Annotation, Scene
from .crowd_safety import CrowdSafety
from .football import Football
from .fitness import Fitness
from .flow import Flow
from .action import LearnedAction

_REGISTRY = {m.key: m for m in (CrowdSafety, Football, Fitness, Flow, LearnedAction)}
MODES = {k: v.title for k, v in _REGISTRY.items()}


def get_mode(key, **kw):
    if key not in _REGISTRY:
        raise ValueError(f"unknown mode {key!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[key](**kw)


__all__ = ["Mode", "Annotation", "Scene", "MODES", "get_mode"]
