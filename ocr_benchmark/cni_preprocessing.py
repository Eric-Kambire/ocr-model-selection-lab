"""Préparation traçable des sources CNI PDF, JPEG et PNG.

Les originaux ne sont jamais modifiés : tous les artefacts sont écrits dans le
répertoire du run afin de pouvoir auditer exactement l'entrée du modèle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .cni_images import render_single_page_pdf

SUPPORTED_CNI_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def prepare_cni_source(source_path: Path, output_path: Path, dpi: int = 300) -> dict[str, Any]:
    """Crée un PNG de travail depuis un PDF mono-page, JPEG ou PNG."""
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        rendered = render_single_page_pdf(source_path, output_path, dpi=dpi)
        return {"source_type": "pdf", "source_path": str(source_path), **rendered}
    if suffix not in SUPPORTED_CNI_IMAGE_SUFFIXES:
        raise ValueError(f"Format CNI non pris en charge : {source_path.suffix}")

    # EXIF peut indiquer qu'un JPEG doit être tourné. L'orientation est appliquée
    # une seule fois, puis la pipeline utilise toujours un PNG RGB homogène.
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        width, height = image.size
    return {
        "source_type": suffix.lstrip("."),
        "source_path": str(source_path),
        "image_path": str(output_path),
        "width": width,
        "height": height,
        "dpi": None,
        "exif_orientation_applied": True,
    }


def preprocess_cni_image(
    source_path: Path,
    output_path: Path,
    *,
    deskew: bool = False,
    perspective: bool = False,
    contrast: bool = False,
    denoise: bool = False,
) -> dict[str, Any]:
    """Applique les options OpenCV activées et journalise le résultat.

    OpenCV est importé uniquement au moment où une option est demandée : PDF,
    JPEG et PNG fonctionnent donc normalement sans cette dépendance optionnelle.
    """
    enabled = {"deskew": deskew, "perspective": perspective, "contrast": contrast, "denoise": denoise}
    if not any(enabled.values()):
        return {"status": "disabled", "image_path": str(source_path), "operations": []}
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV est requis pour les options de prétraitement sélectionnées. "
            "Installez opencv-python-headless."
        ) from exc

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV ne peut pas lire l'image : {source_path}")
    operations: list[dict[str, Any]] = []

    if contrast:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        image = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
        operations.append({"name": "clahe_contrast", "clip_limit": 2.0, "grid": [8, 8]})

    if denoise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        operations.append({"name": "denoise", "strength": 5})

    if deskew:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        points = cv2.findNonZero(mask)
        if points is None:
            operations.append({"name": "deskew", "status": "skipped_no_foreground"})
        else:
            angle = float(cv2.minAreaRect(points)[-1])
            angle = -(90.0 + angle) if angle < -45.0 else -angle
            if abs(angle) > 15.0:
                operations.append({"name": "deskew", "angle_degrees": round(angle, 3), "status": "skipped_out_of_range"})
            else:
                height, width = image.shape[:2]
                center = (width / 2.0, height / 2.0)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
                target_width = int(height * sine + width * cosine)
                target_height = int(height * cosine + width * sine)
                matrix[0, 2] += target_width / 2.0 - center[0]
                matrix[1, 2] += target_height / 2.0 - center[1]
                image = cv2.warpAffine(image, matrix, (target_width, target_height), borderValue=(255, 255, 255))
                operations.append({"name": "deskew", "angle_degrees": round(angle, 3), "status": "applied"})

    if perspective:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quadrilateral = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:25]:
            perimeter = cv2.arcLength(contour, True)
            candidate = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(candidate) == 4 and cv2.contourArea(candidate) > image.shape[0] * image.shape[1] * 0.02:
                quadrilateral = candidate.reshape(4, 2).astype("float32")
                break
        if quadrilateral is None:
            operations.append({"name": "perspective", "status": "skipped_no_quadrilateral"})
        else:
            ordered = _order_corners(quadrilateral, np)
            width = int(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])))
            height = int(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))
            if width > 0 and height > 0:
                target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                matrix = cv2.getPerspectiveTransform(ordered, target)
                image = cv2.warpPerspective(image, matrix, (width, height), borderValue=(255, 255, 255))
                operations.append({"name": "perspective", "status": "applied", "target_size_px": [width, height]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Impossible d'écrire l'image prétraitée : {output_path}")
    return {"status": "applied", "image_path": str(output_path), "operations": operations}


def _order_corners(points: Any, np: Any) -> Any:
    """Ordonne les coins : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered
