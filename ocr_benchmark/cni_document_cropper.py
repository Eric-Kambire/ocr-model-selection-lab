"""Détection prudente d'une CNI dans une image de scan ou de téléphone.

Le module retourne une carte redressée seulement lorsqu'elle est suffisamment
crédible. Sinon il renvoie l'image source complète, sans rotation ni crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

CARD_RATIO = 85.60 / 53.98


def crop_cni_document(source_path: Path, output_path: Path, *, debug_path: Path | None = None) -> dict[str, Any]:
    """Détecte/redresse une CNI ; en cas de doute, transmet la source intacte."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _return_original(source_path, "crop_unavailable_opencv")
    with Image.open(source_path) as opened:
        source_rgb = ImageOps.exif_transpose(opened).convert("RGB")
    original = cv2.cvtColor(np.asarray(source_rgb), cv2.COLOR_RGB2BGR)
    if original.size == 0:
        return _return_original(source_path, "crop_uncertain_empty_source")
    trim = _detect_dark_edge_bands(original, cv2)
    left, top, right, bottom = trim
    working = original[top:bottom, left:right]
    if working.size == 0:
        return _return_original(source_path, "crop_uncertain_invalid_edge_trim", trim=trim)
    preview, scale = _resize_for_detection(working, cv2)
    candidates = _find_card_candidates(preview, cv2)
    selected = next((candidate for candidate in candidates if candidate["accepted"]), None)
    if selected is None:
        _write_debug_preview(debug_path, original, trim, [], None, cv2)
        return _return_original(source_path, "crop_uncertain_no_card", trim=trim, candidates=len(candidates))
    points = selected["points"] / scale
    points[:, 0] += left
    points[:, 1] += top
    ordered = _order_corners(points, np)
    width, height = _target_dimensions(ordered, np)
    if width < 80 or height < 50:
        return _return_original(source_path, "crop_uncertain_too_small", trim=trim, score=selected["score"])
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    warped = cv2.warpPerspective(original, cv2.getPerspectiveTransform(ordered.astype("float32"), target), (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), warped):
        raise OSError(f"Impossible d'écrire le crop CNI : {output_path}")
    _write_debug_preview(debug_path, original, trim, [], selected, cv2, ordered)
    return {
        "image_path": str(output_path), "crop_status": "crop_detected_perspective",
        "crop_box": _points_box(ordered),
        "coverage": round(float(selected["area"]) / max(1, original.shape[0] * original.shape[1] * scale * scale), 4),
        "score": round(float(selected["score"]), 4), "ratio": round(float(selected["ratio"]), 4),
        "rectangularity": round(float(selected["rectangularity"]), 4),
        "source_sent_unchanged": False, "edge_trim": list(trim),
        "debug_image_path": str(debug_path) if debug_path and debug_path.is_file() else None,
    }


def _return_original(source_path: Path, status: str, **details: Any) -> dict[str, Any]:
    return {"image_path": str(source_path), "crop_status": status, "crop_box": None, "coverage": None, "score": None, "source_sent_unchanged": True, **{key: value for key, value in details.items() if value is not None}}


def _detect_dark_edge_bands(image: Any, cv2: Any) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray <= 38
    height, width = dark.shape
    def count(values: Any, limit: int) -> int:
        total = 0
        for value in values[:limit]:
            if float(value) >= 0.88:
                total += 1
            else:
                break
        return total
    max_rows, max_cols = max(1, int(height * .12)), max(1, int(width * .12))
    top, bottom = count(dark.mean(axis=1), max_rows), count(dark.mean(axis=1)[::-1], max_rows)
    left, right = count(dark.mean(axis=0), max_cols), count(dark.mean(axis=0)[::-1], max_cols)
    return left, top, max(left + 1, width - right), max(top + 1, height - bottom)


def _resize_for_detection(image: Any, cv2: Any) -> tuple[Any, float]:
    height, width = image.shape[:2]
    maximum = max(width, height)
    if maximum <= 1500:
        return image, 1.0
    scale = 1500.0 / maximum
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA), scale


def _find_card_candidates(image: Any, cv2: Any) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(clahe, (5, 5), 0), 45, 135)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    candidates, seen = [], set()
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < image_area * .002:
            continue
        rectangle = cv2.minAreaRect(contour)
        rect_width, rect_height = rectangle[1]
        if rect_width < 8 or rect_height < 8:
            continue
        long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
        ratio, rect_area = long_side / short_side, max(1.0, rect_width * rect_height)
        coverage, rectangularity = rect_area / image_area, min(1.0, contour_area / rect_area)
        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = min(1.0, contour_area / hull_area)
        approximation = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        is_quad = len(approximation) == 4 and cv2.isContourConvex(approximation)
        points = approximation.reshape(4, 2).astype("float32") if is_quad else cv2.boxPoints(rectangle).astype("float32")
        box = tuple(_points_box(points))
        if box in seen:
            continue
        seen.add(box)
        ratio_fit = max(0.0, 1.0 - abs(ratio - CARD_RATIO) / .34)
        score = .50 * ratio_fit + .25 * rectangularity + .15 * solidity + (.10 if is_quad else 0.0)
        if coverage > .80 and abs(ratio - CARD_RATIO) > .18:
            score -= .25
        candidates.append({"points": points, "area": rect_area, "ratio": ratio, "rectangularity": rectangularity, "score": score, "accepted": bool(1.15 <= ratio <= 2.20 and .002 <= coverage <= .96 and rectangularity >= .42 and score >= .66)})
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def _order_corners(points: Any, np: Any) -> Any:
    by_x = points[np.argsort(points[:, 0])]
    left, right = by_x[:2][np.argsort(by_x[:2, 1])], by_x[2:][np.argsort(by_x[2:, 1])]
    return np.asarray([left[0], right[0], right[1], left[1]], dtype="float32")


def _target_dimensions(points: Any, np: Any) -> tuple[int, int]:
    return int(round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))), int(round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))))


def _points_box(points: Any) -> list[int]:
    return [int(points[:, 0].min()), int(points[:, 1].min()), int(points[:, 0].max()), int(points[:, 1].max())]


def _write_debug_preview(debug_path: Path | None, original: Any, trim: tuple[int, int, int, int], candidates: list[dict[str, Any]], selected: dict[str, Any] | None, cv2: Any, selected_points: Any | None = None) -> None:
    if debug_path is None:
        return
    preview = original.copy()
    left, top, right, bottom = trim
    cv2.rectangle(preview, (left, top), (right - 1, bottom - 1), (180, 180, 0), 2)
    if selected_points is not None:
        cv2.polylines(preview, [selected_points.astype("int32")], True, (0, 180, 0), 3)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), preview)
