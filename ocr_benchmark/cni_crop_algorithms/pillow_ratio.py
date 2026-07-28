"""Adaptateur de la recherche d'angle par ratio avec Pillow."""

from pathlib import Path
from typing import Any

from .implementations import _pillow_ratio_pipeline


def run(source_path: Path, output_dir: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    """Recherche l'angle minimisant l'écart au ratio physique d'une CNI."""
    return _pillow_ratio_pipeline(source_path, output_dir, parameters)
