"""Adaptateur de la méthode Smart Crop V4."""

from pathlib import Path
from typing import Any

from .implementations import _hybrid_v4_pipeline


def run(source_path: Path, output_dir: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    """Exécute Smart Crop V4 et retourne le résultat et ses diagnostics."""
    return _hybrid_v4_pipeline(source_path, output_dir, parameters)
