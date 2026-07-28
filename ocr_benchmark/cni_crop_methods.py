"""Façade rétrocompatible des méthodes de crop CNI.

Le reste de l'application continue d'importer ce module. Les implémentations,
le registre et les descriptions sont désormais séparés dans
``ocr_benchmark.cni_crop_algorithms``.
"""

from .cni_crop_algorithms.contracts import METHOD_DESCRIPTIONS, METHOD_LABELS
from .cni_crop_algorithms.implementations import normalise_crop_lab_source
from .cni_crop_algorithms.registry import run_registered_crop_method


def run_crop_method(*args, **kwargs):
    """Exécute la méthode demandée via le registre central."""
    return run_registered_crop_method(*args, **kwargs)


__all__ = [
    "METHOD_DESCRIPTIONS",
    "METHOD_LABELS",
    "normalise_crop_lab_source",
    "run_crop_method",
]
