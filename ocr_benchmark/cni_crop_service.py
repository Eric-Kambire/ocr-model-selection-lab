"""Point d'entrée unique des méthodes de recadrage utilisées par le benchmark CNI.

Le laboratoire de crop et le pipeline OCR doivent exécuter exactement le même
algorithme. Ce module adapte donc leurs résultats à un contrat stable pour le
runner, sans dépendre de Gradio.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any

from PIL import Image, ImageOps

from .cni_crop_methods import run_crop_method
from .cni_images import crop_cni_from_a4


SMART_CROP_V4 = "smart_crop_v4"
CONNECTED_COMPONENTS = "connected_components"
CANNY_CONTOURS = "canny_contours"
LEGACY_OPENCV = "legacy_opencv"
ORIGINAL_IMAGE = "original"
DEFAULT_SMART_CROP_MIN_SCORE = 0.55
DEFAULT_SMART_CROP_MARGIN = 0.012
SUPPORTED_CROP_METHODS = {
    SMART_CROP_V4,
    CONNECTED_COMPONENTS,
    CANNY_CONTOURS,
    LEGACY_OPENCV,
    ORIGINAL_IMAGE,
}


def crop_cni_for_benchmark(
    source_path: Path,
    output_dir: Path,
    *,
    output_path: Path | None = None,
    method: str = SMART_CROP_V4,
    minimum_score: float = DEFAULT_SMART_CROP_MIN_SCORE,
    margin_ratio: float = DEFAULT_SMART_CROP_MARGIN,
) -> dict[str, Any]:
    """Recadre une image normalisée et retourne le contrat attendu par le runner.

    Entrées :
    - ``source_path`` : PNG pleine résolution créé à partir du PDF ou de l'image.
    - ``output_dir`` : dossier d'artefacts propre à une face et à un run.
    - ``method`` : Smart Crop V4, composants connectés, Canny, ancien OpenCV
      ou aucun crop.

    Sortie :
    - ``image_path`` désigne toujours une image réellement lisible ;
    - ``source_sent_unchanged`` vaut vrai lorsque le détecteur est incertain ;
    - les scores, étapes et rapports restent disponibles pour le diagnostic.

    Smart Crop V4 peut réduire une copie uniquement pour détecter les coins.
    L'homographie finale est toujours calculée sur les pixels de ``source_path``.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_output = Path(output_path) if output_path is not None else None
    selected_method = method if method in SUPPORTED_CROP_METHODS else SMART_CROP_V4

    if selected_method == ORIGINAL_IMAGE:
        return _preserve_original(source_path, selected_method, "crop_disabled")

    if selected_method == LEGACY_OPENCV:
        legacy = crop_cni_from_a4(
            source_path,
            requested_output or output_dir / "final_legacy_opencv.png",
            debug_path=output_dir / "legacy_opencv_debug.png",
        )
        unchanged = bool(legacy.get("source_sent_unchanged")) or (
            str(legacy.get("crop_status") or "") != "crop_detected"
        )
        return {
            **legacy,
            "crop_method": selected_method,
            "source_sent_unchanged": unchanged,
            "score": legacy.get("score"),
            "stages": legacy.get("stages", []),
            "report_path": legacy.get("report_path"),
        }

    registered_method = {
        SMART_CROP_V4: "hybrid_v4",
        CONNECTED_COMPONENTS: "connected_components",
        CANNY_CONTOURS: "canny_contours",
    }[selected_method]
    parameters: dict[str, Any]
    if selected_method == SMART_CROP_V4:
        parameters = {
            "hybrid_min_score": _bounded(
                minimum_score, DEFAULT_SMART_CROP_MIN_SCORE, 0.0, 1.0
            ),
            "hybrid_margin": _bounded(
                margin_ratio, DEFAULT_SMART_CROP_MARGIN, 0.0, 0.08
            ),
        }
    elif selected_method == CONNECTED_COMPONENTS:
        parameters = {
            "component_mask_mode": "adaptive",
            "component_selection": "scored",
            "component_min_area_pct": 0.15,
            # Analyse le masque brut et une variante où les ponts fins ont été
            # cassés. Le meilleur candidat géométrique gagne.
            "component_break_bridges": True,
        }
    else:
        parameters = {
            "canny_low": 45,
            "canny_high": 135,
            "contour_kernel": 7,
            "contour_min_area_pct": 0.2,
            "contour_min_score": 0.64,
        }

    result = run_crop_method(
        source_path,
        output_dir,
        method=registered_method,
        parameters=parameters,
    )
    unchanged = bool(result.get("source_sent_unchanged"))
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    final_path = Path(str(result.get("final_path") or source_path))
    if requested_output is not None and not unchanged:
        requested_output.parent.mkdir(parents=True, exist_ok=True)
        copyfile(final_path, requested_output)
        final_path = requested_output
    # Conserver le nom historique ``smart_v4`` dans les résultats existants.
    # Les deux nouvelles méthodes utilisent leur identifiant public directement.
    status_method = (
        "smart_v4" if selected_method == SMART_CROP_V4 else selected_method
    )
    return {
        "image_path": str(final_path),
        "crop_status": (
            f"crop_fallback_original_{status_method}"
            if unchanged
            else f"crop_detected_{status_method}"
        ),
        "crop_method": selected_method,
        "source_sent_unchanged": unchanged,
        "score": summary.get("score"),
        "detector": summary.get("detector"),
        "crop_box": None,
        "coverage": None,
        "stages": result.get("stages", []),
        "report_path": result.get("report_path"),
        "summary": summary,
    }


def _preserve_original(source_path: Path, method: str, status: str) -> dict[str, Any]:
    """Retourne l'image entière sans la réencoder ni modifier ses dimensions."""
    with Image.open(source_path) as opened:
        width, height = ImageOps.exif_transpose(opened).size
    return {
        "image_path": str(source_path),
        "crop_status": status,
        "crop_method": method,
        "source_sent_unchanged": True,
        "score": None,
        "crop_box": None,
        "coverage": 1.0,
        "width": width,
        "height": height,
        "stages": [],
        "report_path": None,
    }


def _bounded(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    """Normalise les valeurs provenant du JSON de configuration."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if minimum <= number <= maximum else fallback
