"""Registre unique des méthodes de crop.

Ajouter une méthode ne demande plus de modifier une longue chaîne ``if/elif`` :
il suffit d'enregistrer une fonction respectant le contrat ``CropMethod``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from . import connected_components, opencv_contours, pillow_ratio, smart_v4
from .contracts import METHOD_LABELS
from .diagnostics import GENERATE_INTERMEDIATE_STEPS

CropMethod = Callable[[Path, Path, dict[str, Any]], dict[str, Any]]

_METHODS: dict[str, CropMethod] = {
    "connected_components": connected_components.run,
    "canny_contours": opencv_contours.run_canny,
    "min_area_rect": opencv_contours.run_min_area_rect,
    "pillow_ratio": pillow_ratio.run,
    "hybrid_v4": smart_v4.run,
}


def available_crop_methods() -> tuple[str, ...]:
    """Retourne les identifiants stables proposés par l'interface."""
    return tuple(_METHODS)


def _json_safe(value: Any) -> Any:
    """Convertit récursivement les types OpenCV/NumPy en types JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def run_registered_crop_method(
    source_path: Path,
    output_dir: Path,
    *,
    method: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Valide, exécute et journalise une méthode enregistrée."""
    method = "hybrid_v4" if method == "hybrid_v3" else method
    implementation = _METHODS.get(method)
    if implementation is None:
        raise ValueError(f"Méthode inconnue : {method}")

    output_dir.mkdir(parents=True, exist_ok=True)
    values = dict(parameters or {})
    started = time.perf_counter()
    token = GENERATE_INTERMEDIATE_STEPS.set(bool(values.get("generate_steps", True)))
    try:
        result = _json_safe(implementation(Path(source_path), output_dir, values))
    finally:
        GENERATE_INTERMEDIATE_STEPS.reset(token)
    result["method"] = method
    result["method_label"] = METHOD_LABELS[method]
    result["parameters"] = values
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    report_path = output_dir / "crop_method_report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["report_path"] = str(report_path)
    return result
