"""Détection prudente d'une CNI dans une image de scan ou de téléphone.

Ce module est volontairement indépendant de Gradio, d'Ollama et des labels.
Son contrat est simple : il retourne une CNI redressée seulement lorsque la
confiance est suffisante. Dans tous les autres cas, il retourne exactement
l'image source normalisée, sans rotation ni recadrage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


# Le rapport largeur / hauteur d'une carte ISO ID-1 est 85.60 / 53.98.
# Il sert de préférence de classement, jamais d'hypothèse sur le format de la
# page source (A4, photo téléphone, image déjà recadrée, etc.).
CARD_RATIO = 85.60 / 53.98


def crop_cni_document(
    source_path: Path,
    output_path: Path,
    *,
    debug_path: Path | None = None,
) -> dict[str, Any]:
    """Détecte et redresse une CNI, ou transmet la source sans la modifier.

    Entrées:
        source_path: PNG/JPEG de travail déjà orienté via EXIF ou rendu depuis PDF.
        output_path: emplacement de la CNI redressée si elle est fiable.
        debug_path: aperçu optionnel des candidats et de la sélection.

    Sortie:
        ``image_path`` est *toujours* le fichier à envoyer au modèle. En repli,
        il pointe vers ``source_path`` afin de préserver l'image complète.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _return_original(source_path, "crop_unavailable_opencv")

    # Pillow respecte les métadonnées EXIF d'un téléphone. La conversion vers
    # BGR ne sert qu'à OpenCV ; le fichier source de travail reste intact.
    with Image.open(source_path) as opened:
        source_rgb = ImageOps.exif_transpose(opened).convert("RGB")
    original = cv2.cvtColor(np.asarray(source_rgb), cv2.COLOR_RGB2BGR)

    if original.size == 0:
        return _return_original(source_path, "crop_uncertain_empty_source")

    # Les bandes noires du scanner sont retirées uniquement pour la détection.
    # Elles ne modifient pas l'original et ne peuvent donc jamais être envoyées
    # comme un faux crop si aucune carte n'est reconnue.
    trim = _detect_dark_edge_bands(original, cv2, np)
    left, top, right, bottom = trim
    working = original[top:bottom, left:right]
    if working.size == 0:
        return _return_original(source_path, "crop_uncertain_invalid_edge_trim", trim=trim)

    preview, scale = _resize_for_detection(working, cv2)
    candidates = _find_card_candidates(preview, cv2, np)
    selected = next((candidate for candidate in candidates if candidate["accepted"]), None)

    if selected is None:
        _write_debug_preview(debug_path, original, trim, [], None, cv2, np)
        return _return_original(
            source_path,
            "crop_uncertain_no_card",
            trim=trim,
            candidates=len(candidates),
            debug_image_path=str(debug_path) if debug_path and debug_path.is_file() else None,
        )

    # Les points ont été détectés sur un aperçu réduit : on les remappe vers
    # l'image source pleine résolution avant toute transformation.
    points = selected["points"] / scale
    points[:, 0] += left
    points[:, 1] += top
    ordered = _order_corners(points, np)
    width, height = _target_dimensions(ordered, np)
    if width < 80 or height < 50:
        _write_debug_preview(debug_path, original, trim, candidates, selected, cv2, np)
        return _return_original(source_path, "crop_uncertain_too_small", trim=trim, score=selected["score"])

    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered.astype("float32"), target)
    warped = cv2.warpPerspective(
        original,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Une CNI doit être lue au format paysage. Cette rotation ne concerne que
    # la carte validée, jamais une page source entière.
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), warped):
        raise OSError(f"Impossible d'écrire le crop CNI : {output_path}")

    _write_debug_preview(debug_path, original, trim, candidates, selected, cv2, np, ordered)
    full_area = original.shape[0] * original.shape[1]
    return {
        "image_path": str(output_path),
        "crop_status": "crop_detected_perspective",
        "crop_box": _points_box(ordered),
        "coverage": round(float(selected["area"]) / max(1, full_area * scale * scale), 4),
        "score": round(float(selected["score"]), 4),
        "ratio": round(float(selected["ratio"]), 4),
        "rectangularity": round(float(selected["rectangularity"]), 4),
        "source_sent_unchanged": False,
        "edge_trim": list(trim),
        "debug_image_path": str(debug_path) if debug_path and debug_path.is_file() else None,
    }


def _return_original(source_path: Path, status: str, **details: Any) -> dict[str, Any]:
    """Retourne la source elle-même : aucun réencodage ni traitement caché."""
    return {
        "image_path": str(source_path),
        "crop_status": status,
        "crop_box": None,
        "coverage": None,
        "score": None,
        "source_sent_unchanged": True,
        **{key: value for key, value in details.items() if value is not None},
    }


def _detect_dark_edge_bands(image: Any, cv2: Any, np: Any) -> tuple[int, int, int, int]:
    """Trouve les bandes sombres continues au bord sans supposer une page A4."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray <= 38
    height, width = dark.shape
    max_rows, max_cols = max(1, int(height * 0.12)), max(1, int(width * 0.12))

    def count_from_start(values: Any, limit: int) -> int:
        count = 0
        for value in values[:limit]:
            # Une bande de scanner couvre presque toute la ligne/colonne.
            # Un texte ou le bord d'une carte ne satisfait normalement pas 88 %.
            if float(value) >= 0.88:
                count += 1
            else:
                break
        return count

    top = count_from_start(dark.mean(axis=1), max_rows)
    bottom = count_from_start(dark.mean(axis=1)[::-1], max_rows)
    left = count_from_start(dark.mean(axis=0), max_cols)
    right = count_from_start(dark.mean(axis=0)[::-1], max_cols)
    return left, top, max(left + 1, width - right), max(top + 1, height - bottom)


def _resize_for_detection(image: Any, cv2: Any) -> tuple[Any, float]:
    """Réduit uniquement l'aperçu d'analyse ; le crop final reste pleine qualité."""
    height, width = image.shape[:2]
    maximum = max(width, height)
    if maximum <= 1500:
        return image, 1.0
    scale = 1500.0 / maximum
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA), scale


def _find_card_candidates(image: Any, cv2: Any, np: Any) -> list[dict[str, Any]]:
    """Construit, score et trie les rectangles qui peuvent être une CNI."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < image_area * 0.002:
            continue
        rectangle = cv2.minAreaRect(contour)
        rect_width, rect_height = rectangle[1]
        if rect_width < 8 or rect_height < 8:
            continue
        long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
        ratio = long_side / short_side
        rect_area = max(1.0, rect_width * rect_height)
        coverage = rect_area / image_area
        rectangularity = min(1.0, contour_area / rect_area)
        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = min(1.0, contour_area / hull_area)
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        is_quad = len(approximation) == 4 and cv2.isContourConvex(approximation)
        points = approximation.reshape(4, 2).astype("float32") if is_quad else cv2.boxPoints(rectangle).astype("float32")
        box = tuple(_points_box(points))
        if box in seen:
            continue
        seen.add(box)

        # Le ratio est une composante majeure, sans être une condition isolée.
        # Cela protège contre un A4 ou une ombre rectangulaire qui englobe la page.
        ratio_fit = max(0.0, 1.0 - abs(ratio - CARD_RATIO) / 0.34)
        score = 0.50 * ratio_fit + 0.25 * rectangularity + 0.15 * solidity + (0.10 if is_quad else 0.0)
        if coverage > 0.80 and abs(ratio - CARD_RATIO) > 0.18:
            score -= 0.25
        accepted = bool(
            1.15 <= ratio <= 2.20
            and 0.002 <= coverage <= 0.96
            and rectangularity >= 0.42
            and score >= 0.66
        )
        candidates.append({
            "points": points,
            "area": rect_area,
            "ratio": ratio,
            "rectangularity": rectangularity,
            "solidity": solidity,
            "score": score,
            "accepted": accepted,
        })
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def _order_corners(points: Any, np: Any) -> Any:
    """Ordonne quatre coins : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    by_x = points[np.argsort(points[:, 0])]
    left, right = by_x[:2], by_x[2:]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return np.asarray([left[0], right[0], right[1], left[1]], dtype="float32")


def _target_dimensions(points: Any, np: Any) -> tuple[int, int]:
    """Mesure les côtés opposés afin que la perspective soit corrigée sans étirement."""
    width = int(round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3]))))
    height = int(round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))))
    return width, height


def _points_box(points: Any) -> list[int]:
    """Convertit quatre points en boîte englobante sérialisable."""
    return [int(points[:, 0].min()), int(points[:, 1].min()), int(points[:, 0].max()), int(points[:, 1].max())]


def _write_debug_preview(
    debug_path: Path | None,
    original: Any,
    trim: tuple[int, int, int, int],
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    cv2: Any,
    np: Any,
    selected_points: Any | None = None,
) -> None:
    """Écrit un aperçu léger : rouge=écarté, orange=retenu, vert=sélectionné."""
    if debug_path is None:
        return
    preview = original.copy()
    left, top, right, bottom = trim
    cv2.rectangle(preview, (left, top), (right - 1, bottom - 1), (180, 180, 0), 2)
    for candidate in candidates[:12]:
        color = (0, 0, 220) if not candidate["accepted"] else (0, 165, 255)
        cv2.polylines(preview, [candidate["points"].astype("int32")], True, color, 1)
    if selected_points is not None:
        cv2.polylines(preview, [selected_points.astype("int32")], True, (0, 180, 0), 3)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), preview)
