"""Algorithmes de crop CNI indépendants de l'interface utilisateur."""

from .contracts import METHOD_DESCRIPTIONS, METHOD_LABELS
from .registry import available_crop_methods, run_registered_crop_method

__all__ = [
    "METHOD_DESCRIPTIONS",
    "METHOD_LABELS",
    "available_crop_methods",
    "run_registered_crop_method",
]
