"""Méthodes comparables de détection et de redressement d'une CNI.

Ce module ne dépend ni de Gradio ni d'Ollama. Il produit une suite d'artefacts
PNG et un rapport structuré que l'interface, les tests ou un futur worker
peuvent exploiter. L'image source n'est jamais écrasée.
"""

from __future__ import annotations

import json
import math
import time
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


CARD_RATIO = 85.60 / 53.98

METHOD_LABELS = {
    "connected_components": "Composants connectés",
    "canny_contours": "Canny + contours quadrilatères",
    "min_area_rect": "Rectangle orienté global (minAreaRect)",
    "pillow_ratio": "Pillow — recherche d'angle par ratio",
    "hybrid_v3": "Hybride V3 — multi-détecteurs",
}

METHOD_DESCRIPTIONS = {
    "connected_components": (
        "Isole les blocs de pixels connectés, affiche leur aire, remplit le "
        "composant retenu, puis valide sa forme avant le redressement."
    ),
    "canny_contours": (
        "Détecte les bords, ferme les petites coupures, classe plusieurs "
        "quadrilatères et redresse le meilleur candidat crédible."
    ),
    "min_area_rect": (
        "Calcule un rectangle autour de tous les pixels du masque. Cette méthode "
        "est utile comme référence, mais elle est volontairement sensible au bruit éloigné."
    ),
    "pillow_ratio": (
        "Tourne une copie réduite de la page à plusieurs angles et retient le "
        "rectangle englobant dont le ratio est le plus proche d'une CNI."
    ),
    "hybrid_v3": (
        "Combine contours, lignes Hough/LSD, texture et premier plan. Plusieurs "
        "quadrilatères sont classés par continuité des bords, ratio, angles, "
        "densité et distance au cadre avant la correction de perspective."
    ),
}


def normalise_crop_lab_source(
    source_path: Path,
    output_path: Path,
    *,
    dpi: int = 300,
    page_number: int = 1,
) -> dict[str, Any]:
    """Rend une page PDF ou normalise une image en PNG RGB.

    ``page_number`` commence à 1 pour correspondre à ce que voit l'utilisateur.
    Les métadonnées EXIF des images téléphone sont appliquées sans modifier le
    fichier fourni.
    """
    source_path = Path(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix.casefold()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF est requis pour ouvrir un PDF.") from exc
        if not 72 <= int(dpi) <= 600:
            raise ValueError("Le DPI doit être compris entre 72 et 600.")
        with fitz.open(source_path) as document:
            if document.page_count == 0:
                raise ValueError("Le PDF ne contient aucune page.")
            index = int(page_number) - 1
            if not 0 <= index < document.page_count:
                raise ValueError(
                    f"Page {page_number} absente : le PDF contient {document.page_count} page(s)."
                )
            page = document.load_page(index)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(float(dpi) / 72.0, float(dpi) / 72.0),
                alpha=False,
            )
            pixmap.save(str(output_path))
            page_count = document.page_count
        source_kind = "pdf"
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        with Image.open(source_path) as opened:
            ImageOps.exif_transpose(opened).convert("RGB").save(output_path, format="PNG")
        page_count = 1
        source_kind = suffix.lstrip(".")
    else:
        raise ValueError(f"Format non pris en charge : {suffix or 'sans extension'}")

    with Image.open(output_path) as image:
        width, height = image.size
    return {
        "image_path": str(output_path),
        "source_kind": source_kind,
        "source_name": source_path.name,
        "page_number": int(page_number),
        "page_count": page_count,
        "dpi": int(dpi),
        "width": width,
        "height": height,
    }


def run_crop_method(
    source_path: Path,
    output_dir: Path,
    *,
    method: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exécute une méthode et retourne toutes ses étapes visualisables."""
    if method not in METHOD_LABELS:
        raise ValueError(f"Méthode inconnue : {method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    values = dict(parameters or {})
    started = time.perf_counter()
    if method == "connected_components":
        result = _connected_components_pipeline(source_path, output_dir, values)
    elif method == "canny_contours":
        result = _canny_contours_pipeline(source_path, output_dir, values)
    elif method == "min_area_rect":
        result = _min_area_rect_pipeline(source_path, output_dir, values)
    elif method == "pillow_ratio":
        result = _pillow_ratio_pipeline(source_path, output_dir, values)
    else:
        result = _hybrid_v3_pipeline(source_path, output_dir, values)
    # OpenCV retourne certains nombres et tableaux dans des types NumPy. Le
    # rapport, Gradio et les futurs exports doivent recevoir uniquement des
    # structures Python sérialisables.
    result = _json_safe(result)
    result["method"] = method
    result["method_label"] = METHOD_LABELS[method]
    result["parameters"] = values
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    report_path = output_dir / "crop_method_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["report_path"] = str(report_path)
    return result


def _json_safe(value: Any) -> Any:
    """Convertit récursivement les valeurs OpenCV/NumPy pour JSON."""
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


def _hybrid_v3_pipeline(
    source_path: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Exécute le détecteur V3 et expose ses décisions étape par étape.

    La détection travaille éventuellement sur une copie réduite, mais les
    quatre coins sont remis à l'échelle avant le crop de l'image originale.
    Un score insuffisant active le fallback sûr vers l'image entière.
    """
    cv2, np = _opencv()
    from .cni_smart_crop_v3 import (
        DetectorConfig,
        detect_card,
        draw_debug,
        order_quad,
        resize_for_work,
        warp_card,
    )

    original = _read_bgr(source_path, cv2, np)
    config = DetectorConfig(
        expected_aspect_ratio=float(parameters.get("hybrid_ratio") or CARD_RATIO),
        min_area_ratio=float(parameters.get("hybrid_min_area") or 0.035),
        max_area_ratio=float(parameters.get("hybrid_max_area") or 0.92),
        edge_tolerance_ratio=float(parameters.get("hybrid_edge_tolerance") or 0.004),
        final_margin_ratio=float(parameters.get("hybrid_margin") or 0.012),
    )
    minimum_score = float(parameters.get("hybrid_min_score") or 0.55)
    stages: list[dict[str, Any]] = []
    _save_stage(
        stages,
        output_dir,
        original,
        name="Source normalisée",
        explanation=(
            "Image complète de référence. Elle reste intacte pendant toute la détection."
        ),
    )

    best, candidates, maps, scale = detect_card(original, config)
    working, _ = resize_for_work(original, config.max_working_side)
    diagnostic_maps = (
        ("Niveaux de gris", "gray", "Luminance utilisée par les détecteurs."),
        (
            "Gradient Sobel",
            "gradient",
            "Les variations fortes d'intensité révèlent les limites possibles.",
        ),
        (
            "Bords reconnectés",
            "connected_edges",
            "Canny et Sobel sont réunis puis les petites coupures sont fermées.",
        ),
        (
            "Masque premier plan",
            "foreground_mask",
            "Le fond est estimé sur le cadre ; les objets éloignés restent séparés.",
        ),
        (
            "Masque densité et texture",
            "density_mask",
            "Le texte, la photo et les motifs forment des régions localement denses.",
        ),
        (
            "Segments de lignes",
            "line_mask",
            "Hough et LSD proposent des côtés malgré un contour interrompu.",
        ),
    )
    for name, key, explanation in diagnostic_maps:
        image = maps.get(key)
        if image is not None:
            _save_stage(
                stages,
                output_dir,
                image,
                name=name,
                explanation=explanation,
            )

    debug = draw_debug(working, candidates, limit=12)
    _save_stage(
        stages,
        output_dir,
        debug,
        name="Candidats classés",
        explanation=(
            "Chaque quadrilatère vient d'un détecteur. Le vert porte le meilleur "
            "score global ; les autres restent visibles pour le diagnostic."
        ),
        metrics={
            "candidate_count": len(candidates),
            "top_candidates": [candidate.to_json() for candidate in candidates[:5]],
        },
    )

    selected_overlay = working.copy()
    accepted = best is not None and float(best.score) >= minimum_score
    if best is not None:
        colour = (0, 180, 0) if accepted else (0, 0, 220)
        points = np.round(order_quad(best.quad)).astype(np.int32)
        cv2.polylines(selected_overlay, [points], True, colour, 4, cv2.LINE_AA)
        cv2.putText(
            selected_overlay,
            f"score={best.score:.3f} seuil={minimum_score:.3f}",
            tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            colour,
            2,
            cv2.LINE_AA,
        )
    _save_stage(
        stages,
        output_dir,
        selected_overlay,
        name="Décision du détecteur",
        explanation=(
            "Vert : candidat accepté. Rouge : meilleur candidat visible mais "
            "refusé ; l'image entière sera alors transmise."
        ),
        metrics={
            "accepted": accepted,
            "minimum_score": minimum_score,
            "best_score": float(best.score) if best is not None else None,
            "best_source": best.source if best is not None else None,
            "best_metrics": best.metrics if best is not None else None,
        },
    )

    if not accepted or best is None:
        return _fallback_result(
            source_path,
            stages,
            reason=(
                "aucun candidat"
                if best is None
                else f"score {best.score:.3f} inférieur au minimum {minimum_score:.3f}"
            ),
            method_metrics={
                "candidate_count": len(candidates),
                "best": best.to_json() if best is not None else None,
                "working_scale": scale,
            },
        )

    original_quad = order_quad(best.quad) / scale
    crop = warp_card(
        original,
        original_quad,
        config.expected_aspect_ratio,
        margin_ratio=config.final_margin_ratio,
    )
    final_rotation = int(parameters.get("final_rotation") or 0)
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if final_rotation in rotations:
        crop = cv2.rotate(crop, rotations[final_rotation])
    final_path = output_dir / "final_hybrid_v3.png"
    if not cv2.imwrite(str(final_path), crop):
        raise OSError(f"Impossible d'écrire le crop final : {final_path}")
    _save_stage(
        stages,
        output_dir,
        crop,
        name="Crop redressé",
        explanation=(
            "Les quatre coins remis à l'échelle sur l'original sont projetés "
            "vers un rectangle par homographie."
        ),
        metrics={
            "score": float(best.score),
            "detector": best.source,
            "working_scale": scale,
            "quad_original": np.round(original_quad, 2).tolist(),
            "final_rotation": final_rotation,
        },
    )
    return {
        "status": "crop_detected",
        "final_path": str(final_path),
        "source_sent_unchanged": False,
        "stages": stages,
        "summary": {
            "score": float(best.score),
            "detector": best.source,
            "candidate_count": len(candidates),
            "metrics": best.metrics,
        },
    }


def _opencv() -> tuple[Any, Any]:
    """Importe OpenCV tardivement pour garder des erreurs de démarrage lisibles."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV et NumPy sont requis. Installez requirements.txt."
        ) from exc
    return cv2, np


def _read_bgr(source_path: Path, cv2: Any, np: Any) -> Any:
    with Image.open(source_path) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _save_stage(
    stages: list[dict[str, Any]],
    output_dir: Path,
    cv_image: Any,
    *,
    name: str,
    explanation: str,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Écrit un PNG sans perte et ajoute sa description au rapport."""
    cv2, _ = _opencv()
    index = len(stages) + 1
    path = output_dir / f"{index:02d}_{_safe_name(name)}.png"
    if not cv2.imwrite(str(path), cv_image):
        raise OSError(f"Impossible d'écrire l'étape : {path}")
    height, width = cv_image.shape[:2]
    stages.append(
        {
            "index": index - 1,
            "name": name,
            "image_path": str(path),
            "explanation": explanation,
            "metrics": {
                "width": int(width),
                "height": int(height),
                **dict(metrics or {}),
            },
        }
    )
    return str(path)


def _safe_name(value: str) -> str:
    """Produit un nom ASCII stable, y compris sous Windows et OpenCV."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "".join(
        character if character.isalnum() else "_" for character in ascii_value
    ).strip("_").lower()


def _gray_and_contrast(image: Any, cv2: Any) -> tuple[Any, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray, contrast


def _foreground_mask(
    contrast: Any,
    cv2: Any,
    *,
    mode: str,
    threshold: int,
    kernel_size: int,
) -> Any:
    """Construit le masque utilisé par les composants connectés."""
    size = max(3, int(kernel_size) | 1)
    if mode == "canny":
        raw = cv2.Canny(cv2.GaussianBlur(contrast, (5, 5), 0), 45, 135)
    elif mode == "otsu":
        _, raw = cv2.threshold(
            contrast,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
    else:
        block = max(15, size * 2 + 1) | 1
        raw = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            max(2, int((255 - int(threshold)) / 6)),
        )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size * 2 + 1, size))
    connected = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
    return cv2.morphologyEx(
        connected,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )


def _connected_components_pipeline(
    source_path: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    cv2, np = _opencv()
    original = _read_bgr(source_path, cv2, np)
    stages: list[dict[str, Any]] = []
    _save_stage(
        stages,
        output_dir,
        original,
        name="Source normalisée",
        explanation="Image de référence. Elle reste intacte pendant toute l'analyse.",
    )
    gray, contrast = _gray_and_contrast(original, cv2)
    _save_stage(
        stages,
        output_dir,
        contrast,
        name="Gris et contraste local",
        explanation="CLAHE renforce localement les bords malgré une ombre ou un scan inégal.",
    )
    min_dimension = min(original.shape[:2])
    kernel_size = int(parameters.get("component_kernel") or max(5, round(min_dimension * 0.012)))
    mask = _foreground_mask(
        contrast,
        cv2,
        mode=str(parameters.get("component_mask_mode") or "adaptive"),
        threshold=int(parameters.get("component_threshold") or 235),
        kernel_size=kernel_size,
    )
    _save_stage(
        stages,
        output_dir,
        mask,
        name="Masque des objets",
        explanation=(
            "Les pixels blancs sont les objets candidats. Les points éloignés restent visibles "
            "ici afin de vérifier qu'ils seront ensuite séparés de la carte."
        ),
        metrics={"kernel_size": kernel_size},
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = float(mask.shape[0] * mask.shape[1])
    minimum_area = image_area * float(parameters.get("component_min_area_pct") or 0.15) / 100.0
    components: list[dict[str, Any]] = []
    component_map = np.zeros((*mask.shape, 3), dtype=np.uint8)
    palette = (
        (64, 126, 201), (68, 170, 112), (203, 133, 61), (156, 94, 196),
        (185, 77, 93), (74, 166, 178), (112, 132, 149),
    )
    height, width = mask.shape
    for label_id in range(1, count):
        x, y, component_width, component_height, area = [
            int(value) for value in stats[label_id]
        ]
        if area < minimum_area:
            continue
        long_side = max(component_width, component_height)
        short_side = max(1, min(component_width, component_height))
        ratio = long_side / short_side
        coverage = area / image_area
        touches = sum(
            (
                x <= 1,
                y <= 1,
                x + component_width >= width - 1,
                y + component_height >= height - 1,
            )
        )
        ratio_fit = max(0.0, 1.0 - abs(ratio - CARD_RATIO) / 0.65)
        fill_ratio = area / max(1.0, component_width * component_height)
        score = 0.55 * ratio_fit + 0.30 * min(1.0, fill_ratio / 0.65) + 0.15 * min(1.0, coverage / 0.08)
        if touches >= 2:
            score -= 0.35
        components.append(
            {
                "label": label_id,
                "area": area,
                "box": [x, y, x + component_width, y + component_height],
                "ratio": ratio,
                "coverage": coverage,
                "fill_ratio": fill_ratio,
                "touches_borders": touches,
                "score": score,
            }
        )
        component_map[labels == label_id] = palette[(label_id - 1) % len(palette)]

    _save_stage(
        stages,
        output_dir,
        component_map,
        name="Composants séparés",
        explanation=(
            "Une couleur correspond à un objet connecté. La barre et les poussières "
            "doivent apparaître comme des objets distincts de la CNI."
        ),
        metrics={
            "components_total": max(0, count - 1),
            "components_after_area_filter": len(components),
            "minimum_area_px": round(minimum_area),
        },
    )
    selection_mode = str(parameters.get("component_selection") or "scored")
    if selection_mode == "largest":
        selected = max(components, key=lambda item: item["area"], default=None)
    else:
        selected = max(components, key=lambda item: item["score"], default=None)

    if selected is None:
        return _fallback_result(
            source_path,
            stages,
            reason="Aucun composant ne dépasse la surface minimale.",
            method_metrics={"components": components},
        )

    selected_mask = np.where(labels == selected["label"], 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_result(
            source_path,
            stages,
            reason="Le composant retenu n'a pas de contour extérieur.",
            method_metrics={"components": components, "selected": selected},
        )
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(selected_mask)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    _save_stage(
        stages,
        output_dir,
        filled,
        name="Composant retenu et trous bouchés",
        explanation=(
            "Seul le contour externe du composant retenu est rempli. Les lacunes "
            "internes de la carte n'agrandissent plus son rectangle."
        ),
        metrics=selected,
    )

    points = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")
    overlay = original.copy()
    cv2.polylines(overlay, [points.astype("int32")], True, (0, 190, 0), 4)
    for component in components:
        if component["label"] == selected["label"]:
            continue
        x1, y1, x2, y2 = component["box"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 210), 2)
    accepted, reason = _validate_candidate(selected)
    _save_stage(
        stages,
        output_dir,
        overlay,
        name="Rectangle du composant",
        explanation=(
            "Vert : composant retenu. Rouge : autres composants. La validation "
            f"géométrique conclut : {reason}"
        ),
        metrics={**selected, "accepted": accepted, "decision": reason},
    )
    if not accepted:
        return _fallback_result(
            source_path,
            stages,
            reason=reason,
            method_metrics={"components": components, "selected": selected},
        )
    final_path = _warp_candidate(
        original,
        points,
        output_dir / "final_connected_components.png",
        int(parameters.get("final_rotation") or 0),
        cv2,
        np,
    )
    final = cv2.imread(str(final_path))
    _save_stage(
        stages,
        output_dir,
        final,
        name="Crop redressé",
        explanation="Les quatre coins du rectangle sont projetés sur une image plane.",
        metrics={"final_rotation": int(parameters.get("final_rotation") or 0)},
    )
    return {
        "status": "crop_detected",
        "final_path": str(final_path),
        "source_sent_unchanged": False,
        "stages": stages,
        "summary": selected,
    }


def _canny_contours_pipeline(
    source_path: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    cv2, np = _opencv()
    original = _read_bgr(source_path, cv2, np)
    stages: list[dict[str, Any]] = []
    _save_stage(
        stages,
        output_dir,
        original,
        name="Source normalisée",
        explanation="Image complète avant toute détection.",
    )
    _, contrast = _gray_and_contrast(original, cv2)
    low = int(parameters.get("canny_low") or 45)
    high = max(low + 1, int(parameters.get("canny_high") or 135))
    edges = cv2.Canny(cv2.GaussianBlur(contrast, (5, 5), 0), low, high)
    _save_stage(
        stages,
        output_dir,
        edges,
        name="Bords Canny",
        explanation="Chaque pixel blanc représente une variation locale assez forte.",
        metrics={"canny_low": low, "canny_high": high},
    )
    kernel_size = max(3, int(parameters.get("contour_kernel") or 7) | 1)
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )
    _save_stage(
        stages,
        output_dir,
        closed,
        name="Bords reconnectés",
        explanation="La fermeture relie de petites interruptions sans remplir toute la page.",
        metrics={"kernel_size": kernel_size},
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(original.shape[0] * original.shape[1])
    minimum_area = image_area * float(parameters.get("contour_min_area_pct") or 0.2) / 100.0
    candidates: list[dict[str, Any]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:120]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        is_quad = len(approximation) == 4 and cv2.isContourConvex(approximation)
        rectangle = cv2.minAreaRect(contour)
        rect_width, rect_height = rectangle[1]
        if min(rect_width, rect_height) < 12:
            continue
        points = (
            approximation.reshape(4, 2).astype("float32")
            if is_quad
            else cv2.boxPoints(rectangle).astype("float32")
        )
        long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
        ratio = long_side / max(1.0, short_side)
        rect_area = max(1.0, rect_width * rect_height)
        rectangularity = min(1.0, contour_area / rect_area)
        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = min(1.0, contour_area / hull_area)
        coverage = rect_area / image_area
        ratio_fit = max(0.0, 1.0 - abs(ratio - CARD_RATIO) / 0.45)
        score = (
            0.48 * ratio_fit
            + 0.24 * rectangularity
            + 0.14 * solidity
            + (0.14 if is_quad else 0.0)
        )
        candidates.append(
            {
                "points": points,
                "area": rect_area,
                "ratio": ratio,
                "coverage": coverage,
                "rectangularity": rectangularity,
                "solidity": solidity,
                "is_quad": is_quad,
                "score": score,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    overlay = original.copy()
    minimum_score = float(parameters.get("contour_min_score") or 0.64)
    for position, candidate in enumerate(candidates[:20]):
        accepted, _ = _validate_candidate(candidate, minimum_score=minimum_score)
        color = (0, 165, 255) if accepted else (0, 0, 210)
        thickness = 4 if position == 0 else 2
        cv2.polylines(
            overlay,
            [candidate["points"].astype("int32")],
            True,
            color,
            thickness,
        )
    _save_stage(
        stages,
        output_dir,
        overlay,
        name="Quadrilatères candidats",
        explanation="Orange : candidat acceptable. Rouge : candidat rejeté par le score.",
        metrics={
            "candidates": len(candidates),
            "minimum_area_px": round(minimum_area),
            "minimum_score": minimum_score,
        },
    )
    selected = candidates[0] if candidates else None
    accepted, reason = _validate_candidate(selected, minimum_score=minimum_score)
    selected_overlay = original.copy()
    if selected is not None:
        cv2.polylines(
            selected_overlay,
            [selected["points"].astype("int32")],
            True,
            (0, 190, 0) if accepted else (0, 0, 220),
            5,
        )
    _save_stage(
        stages,
        output_dir,
        selected_overlay,
        name="Meilleur contour",
        explanation=f"Le meilleur score est contrôlé avant le crop : {reason}",
        metrics={**(selected or {}), "accepted": accepted, "decision": reason},
    )
    if not accepted or selected is None:
        return _fallback_result(
            source_path,
            stages,
            reason=reason,
            method_metrics={"candidate_count": len(candidates)},
        )
    final_path = _warp_candidate(
        original,
        selected["points"],
        output_dir / "final_canny_contours.png",
        int(parameters.get("final_rotation") or 0),
        cv2,
        np,
    )
    final = cv2.imread(str(final_path))
    _save_stage(
        stages,
        output_dir,
        final,
        name="Perspective corrigée",
        explanation="Le quadrilatère retenu est transformé en rectangle horizontal.",
        metrics={"final_rotation": int(parameters.get("final_rotation") or 0)},
    )
    return {
        "status": "crop_detected",
        "final_path": str(final_path),
        "source_sent_unchanged": False,
        "stages": stages,
        "summary": selected,
    }


def _min_area_rect_pipeline(
    source_path: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Méthode témoin : un seul rectangle englobe tous les pixels du masque."""
    cv2, np = _opencv()
    original = _read_bgr(source_path, cv2, np)
    stages: list[dict[str, Any]] = []
    _save_stage(
        stages,
        output_dir,
        original,
        name="Source normalisée",
        explanation="Cette méthode travaille sur toute l'image.",
    )
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    threshold = int(parameters.get("global_threshold") or 235)
    mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
    margin_pct = max(0.0, min(15.0, float(parameters.get("ignore_border_pct") or 0.0)))
    if margin_pct:
        margin_y = round(mask.shape[0] * margin_pct / 100.0)
        margin_x = round(mask.shape[1] * margin_pct / 100.0)
        mask[:margin_y, :] = 0
        mask[-margin_y:, :] = 0
        mask[:, :margin_x] = 0
        mask[:, -margin_x:] = 0
    _save_stage(
        stages,
        output_dir,
        mask,
        name="Masque global",
        explanation=(
            "Tous les pixels blancs participent au même calcul. Un seul point éloigné "
            "peut donc agrandir le rectangle : c'est la faiblesse observée."
        ),
        metrics={"threshold": threshold, "ignored_border_pct": margin_pct},
    )
    points = cv2.findNonZero(mask)
    if points is None:
        return _fallback_result(source_path, stages, reason="Le masque est vide.")
    rectangle = cv2.minAreaRect(points)
    box = cv2.boxPoints(rectangle).astype("float32")
    rect_width, rect_height = rectangle[1]
    ratio = max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
    coverage = rect_width * rect_height / max(1.0, original.shape[0] * original.shape[1])
    candidate = {
        "points": box,
        "area": rect_width * rect_height,
        "ratio": ratio,
        "coverage": coverage,
        "rectangularity": 1.0,
        "score": max(0.0, 1.0 - abs(ratio - CARD_RATIO) / 0.45),
    }
    accepted, reason = _validate_candidate(candidate, minimum_score=0.58)
    overlay = original.copy()
    cv2.polylines(
        overlay,
        [box.astype("int32")],
        True,
        (0, 190, 0) if accepted else (0, 0, 220),
        5,
    )
    _save_stage(
        stages,
        output_dir,
        overlay,
        name="Rectangle global",
        explanation=f"Le rectangle contient tous les pixels du masque. Décision : {reason}",
        metrics={**candidate, "accepted": accepted, "decision": reason},
    )
    if not accepted:
        return _fallback_result(source_path, stages, reason=reason, method_metrics=candidate)
    final_path = _warp_candidate(
        original,
        box,
        output_dir / "final_min_area_rect.png",
        int(parameters.get("final_rotation") or 0),
        cv2,
        np,
    )
    final = cv2.imread(str(final_path))
    _save_stage(
        stages,
        output_dir,
        final,
        name="Crop global redressé",
        explanation="Résultat produit uniquement si le rectangle global reste plausible.",
    )
    return {
        "status": "crop_detected",
        "final_path": str(final_path),
        "source_sent_unchanged": False,
        "stages": stages,
        "summary": candidate,
    }


def _pillow_ratio_pipeline(
    source_path: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Reproduit la recherche d'angle historique du Stepper avec ses itérations."""
    cv2, np = _opencv()
    with Image.open(source_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
    stages: list[dict[str, Any]] = []
    source_bgr = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
    _save_stage(
        stages,
        output_dir,
        source_bgr,
        name="Source normalisée",
        explanation="Pillow tourne une copie de cette image ; l'original reste intact.",
    )
    maximum = max(source.size)
    scale = min(1.0, 1000.0 / maximum)
    preview = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    threshold = int(parameters.get("pillow_threshold") or 235)
    coarse_step = max(3, int(parameters.get("pillow_coarse_step") or 9))
    fine_radius = max(1, int(parameters.get("pillow_fine_radius") or 3))
    candidates: list[dict[str, Any]] = []

    def evaluate(angle: int, phase: str, save_iteration: bool) -> None:
        rotated = preview.rotate(
            angle,
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor="white",
        )
        mask = ImageOps.grayscale(rotated).point(
            lambda pixel: 255 if pixel < threshold else 0
        )
        box = mask.getbbox()
        if box is None:
            return
        width = box[2] - box[0]
        height = box[3] - box[1]
        ratio = max(width, height) / max(1, min(width, height))
        coverage = width * height / max(1, preview.width * preview.height)
        score = abs(ratio - CARD_RATIO)
        item = {
            "angle": angle,
            "phase": phase,
            "box": list(box),
            "ratio": ratio,
            "coverage": coverage,
            "score": score,
        }
        candidates.append(item)
        if save_iteration:
            frame = rotated.copy()
            draw = ImageDraw.Draw(frame)
            draw.rectangle(box, outline="#1d73be", width=max(2, frame.width // 350))
            frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            _save_stage(
                stages,
                output_dir,
                frame_bgr,
                name=f"Rotation {phase} {angle:+d} degrés",
                explanation=(
                    "La page complète est tournée, puis le rectangle de tous les pixels "
                    f"sombres est recalculé. Score = |{ratio:.4f} - {CARD_RATIO:.4f}|."
                ),
                metrics=item,
            )

    coarse_angles = list(range(-90, 91, coarse_step))
    if 90 not in coarse_angles:
        coarse_angles.append(90)
    for angle in coarse_angles:
        evaluate(angle, "large", True)
    if not candidates:
        return _fallback_result(source_path, stages, reason="Aucun rectangle après rotation.")
    best_coarse = min(candidates, key=lambda item: item["score"])
    fine_angles = range(
        int(best_coarse["angle"]) - fine_radius,
        int(best_coarse["angle"]) + fine_radius + 1,
    )
    for angle in fine_angles:
        evaluate(angle, "affinage", True)
    selected = min(candidates, key=lambda item: item["score"])
    rotated_full = source.rotate(
        int(selected["angle"]),
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )
    full_mask = ImageOps.grayscale(rotated_full).point(
        lambda pixel: 255 if pixel < threshold else 0
    )
    box = full_mask.getbbox()
    ratio = float(selected["ratio"])
    coverage = (
        (box[2] - box[0]) * (box[3] - box[1]) / max(1, source.width * source.height)
        if box else 0.0
    )
    candidate = {
        "ratio": ratio,
        "coverage": coverage,
        "score": max(0.0, 1.0 - float(selected["score"]) / 0.45),
    }
    accepted, reason = _validate_candidate(candidate, minimum_score=0.58)
    if not accepted or box is None:
        return _fallback_result(
            source_path,
            stages,
            reason=reason,
            method_metrics={"selected": selected, "iterations": len(candidates)},
        )
    crop = rotated_full.crop(box)
    final_rotation = int(parameters.get("final_rotation") or 0)
    if final_rotation:
        crop = crop.rotate(-final_rotation, expand=True, fillcolor="white")
    final_path = output_dir / "final_pillow_ratio.png"
    crop.save(final_path, format="PNG")
    final_bgr = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    _save_stage(
        stages,
        output_dir,
        final_bgr,
        name="Meilleur angle et crop",
        explanation=(
            f"Angle retenu : {selected['angle']:+d}°. Cette méthode ne corrige "
            "pas une perspective en trapèze."
        ),
        metrics={**selected, "accepted": True, "decision": reason},
    )
    return {
        "status": "crop_detected",
        "final_path": str(final_path),
        "source_sent_unchanged": False,
        "stages": stages,
        "summary": {
            "selected": selected,
            "iterations": len(candidates),
        },
    }


def _validate_candidate(
    candidate: dict[str, Any] | None,
    *,
    minimum_score: float = 0.55,
) -> tuple[bool, str]:
    """Valide la géométrie de la carte, jamais le ratio de la page source."""
    if not candidate:
        return False, "aucun candidat"
    ratio = float(candidate.get("ratio") or 0.0)
    coverage = float(candidate.get("coverage") or 0.0)
    score = float(candidate.get("score") or 0.0)
    if not 1.15 <= ratio <= 2.20:
        return False, f"ratio {ratio:.3f} hors plage"
    if not 0.002 <= coverage <= 0.97:
        return False, f"couverture {coverage * 100:.2f} % hors plage"
    if int(candidate.get("touches_borders") or 0) >= 2:
        return False, "le composant touche plusieurs bords de l'image"
    if score < minimum_score:
        return False, f"score {score:.3f} inférieur au minimum {minimum_score:.3f}"
    return True, "candidat accepté"


def _warp_candidate(
    original: Any,
    points: Any,
    output_path: Path,
    final_rotation: int,
    cv2: Any,
    np: Any,
) -> Path:
    ordered = _order_points(points, np)
    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    target = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered, target)
    warped = cv2.warpPerspective(
        original,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if int(final_rotation) in rotations:
        warped = cv2.rotate(warped, rotations[int(final_rotation)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), warped):
        raise OSError(f"Impossible d'écrire le crop final : {output_path}")
    return output_path


def _order_points(points: Any, np: Any) -> Any:
    points = np.asarray(points, dtype="float32").reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype="float32",
    )


def _fallback_result(
    source_path: Path,
    stages: list[dict[str, Any]],
    *,
    reason: str,
    method_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retourne l'original si la méthode ne démontre pas un crop fiable."""
    return {
        "status": "fallback_original",
        "final_path": str(source_path),
        "source_sent_unchanged": True,
        "stages": stages,
        "summary": {
            "reason": reason,
            **dict(method_metrics or {}),
        },
    }
