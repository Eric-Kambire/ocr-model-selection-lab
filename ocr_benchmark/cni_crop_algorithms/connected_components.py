"""Adaptateur de l'analyse par composants connectés."""

from pathlib import Path
from typing import Any

from .implementations import _connected_components_pipeline


def run(source_path: Path, output_dir: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    """Conserve les composantes dominantes puis valide la carte candidate."""
    return _connected_components_pipeline(source_path, output_dir, parameters)
