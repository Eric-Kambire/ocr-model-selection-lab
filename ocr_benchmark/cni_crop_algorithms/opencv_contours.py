"""Adaptateurs des méthodes de contours OpenCV."""

from pathlib import Path
from typing import Any

from .implementations import _canny_contours_pipeline, _min_area_rect_pipeline


def run_canny(source_path: Path, output_dir: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    """Exécute Canny, classe les quadrilatères et redresse le meilleur."""
    return _canny_contours_pipeline(source_path, output_dir, parameters)


def run_min_area_rect(
    source_path: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Exécute la référence minAreaRect sur le premier plan détecté."""
    return _min_area_rect_pipeline(source_path, output_dir, parameters)
