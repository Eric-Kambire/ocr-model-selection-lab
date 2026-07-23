"""Normalisation et prétraitement optionnel des sources CNI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .cni_images import render_single_page_pdf


def prepare_cni_source(source_path: Path, output_path: Path, dpi: int) -> dict[str, Any]:
    """Copie une image ou rend un PDF en PNG de travail sans toucher à la source."""
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        rendered = render_single_page_pdf(source_path, output_path, dpi=dpi)
        return {"source_type": "pdf", "source_path": str(source_path), **rendered}
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Format CNI non supporté : {source_path.suffix}")
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        width, height = image.size
    return {"source_type": suffix.lstrip("."), "source_path": str(source_path), "image_path": str(output_path), "width": width, "height": height, "dpi": None}


def preprocess_cni_image(source_path: Path, output_path: Path, options: dict[str, bool]) -> dict[str, Any]:
    """Applique les traitements activés et trace chaque décision dans le run.

    Les rotations Pillow et OpenCV sont exclusives : Pillow recherche l'angle
    dont le contenu ressemble le plus au ratio d'une carte ; OpenCV part du
    rectangle orienté ``minAreaRect``. La perspective est ensuite corrigée, si
    un quadrilatère crédible est détecté. En cas d'échec, l'image précédente est
    conservée au lieu de produire une image vide.
    """
    enabled = {name: bool(options.get(name, False)) for name in (
        "rotation_pillow", "rotation_opencv", "perspective", "deskew", "contrast", "denoise",
    )}
    if enabled["rotation_pillow"] and enabled["rotation_opencv"]:
        raise ValueError("Choisissez une seule méthode de rotation : Pillow ou OpenCV.")
    if not any(enabled.values()):
        return {"status": "disabled", "image_path": str(source_path), "operations": []}
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV est requis pour le prétraitement sélectionné.") from exc
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV ne peut pas lire {source_path}")
    operations: list[str] = []
    metadata: dict[str, Any] = {}
    if enabled["contrast"]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        image = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
        operations.append("contrast_clahe")
    if enabled["denoise"]:
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        operations.append("denoise")
    if enabled["rotation_pillow"]:
        image, rotation = _rotate_with_pillow_search(image)
        operations.append(f"rotation_pillow:{rotation['angle_degrees']:.2f}")
        metadata["rotation"] = rotation
    elif enabled["rotation_opencv"]:
        image, rotation = _rotate_with_opencv(image, cv2)
        operations.append(f"rotation_opencv:{rotation['angle_degrees']:.2f}")
        metadata["rotation"] = rotation
    elif enabled["deskew"]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        points = cv2.findNonZero(mask)
        if points is not None:
            angle = float(cv2.minAreaRect(points)[-1])
            angle = -(90 + angle) if angle < -45 else -angle
            if abs(angle) <= 15:
                height, width = image.shape[:2]
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                image = cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))
                operations.append(f"deskew:{angle:.2f}")
                metadata["rotation"] = {"method": "opencv_deskew_legacy", "angle_degrees": round(angle, 3)}
    if enabled["perspective"]:
        image, perspective = _correct_perspective(image, cv2)
        metadata["perspective"] = perspective
        operations.append(f"perspective:{perspective['status']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Écriture impossible : {output_path}")
    return {"status": "applied", "image_path": str(output_path), "operations": operations, **metadata}


def _content_ratio(image: Image.Image) -> float | None:
    """Retourne le ratio du rectangle de pixels non presque blancs."""
    mask = ImageOps.grayscale(image).point(lambda pixel: 255 if pixel < 242 else 0)
    box = mask.getbbox()
    if box is None:
        return None
    width, height = box[2] - box[0], box[3] - box[1]
    return width / height if height else None


def _rotate_with_pillow_search(image: Any) -> tuple[Any, dict[str, Any]]:
    """Recherche puis affine un angle Pillow sans parcourir l'image haute résolution."""
    source = Image.fromarray(image[:, :, ::-1])
    preview = source.copy()
    preview.thumbnail((700, 700))
    target_ratio, best_angle, best_score = 1.586, 0.0, float("inf")
    for angle in range(-90, 91, 6):
        ratio = _content_ratio(preview.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor="white"))
        if ratio is not None and abs(ratio - target_ratio) < best_score:
            best_angle, best_score = float(angle), abs(ratio - target_ratio)
    coarse_best = best_angle
    for angle in range(int(best_angle) - 5, int(best_angle) + 6):
        ratio = _content_ratio(preview.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor="white"))
        if ratio is not None and abs(ratio - target_ratio) < best_score:
            best_angle, best_score = float(angle), abs(ratio - target_ratio)
    rotated = source.rotate(best_angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor="white")
    import numpy as np
    return np.asarray(rotated)[:, :, ::-1].copy(), {
        "method": "pillow_ratio_search", "angle_degrees": best_angle,
        "target_ratio": target_ratio, "coarse_step_degrees": 6,
        "score": round(best_score, 5), "coarse_best_angle_degrees": coarse_best,
    }


def _rotate_with_opencv(image: Any, cv2: Any) -> tuple[Any, dict[str, Any]]:
    """Déduit une rotation à partir du rectangle orienté OpenCV, sans couper le canevas."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(mask)
    if points is None:
        return image, {"method": "opencv_min_area_rect", "angle_degrees": 0.0, "status": "no_foreground"}
    raw_angle = float(cv2.minAreaRect(points)[-1])
    angle = -(90 + raw_angle) if raw_angle < -45 else -raw_angle
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    target_width, target_height = int(height * sine + width * cosine), int(height * cosine + width * sine)
    matrix[0, 2] += target_width / 2.0 - width / 2.0
    matrix[1, 2] += target_height / 2.0 - height / 2.0
    rotated = cv2.warpAffine(image, matrix, (target_width, target_height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return rotated, {"method": "opencv_min_area_rect", "angle_degrees": round(angle, 3), "raw_angle_degrees": raw_angle}


def _correct_perspective(image: Any, cv2: Any) -> tuple[Any, dict[str, Any]]:
    """Applique une homographie seulement lorsqu'un grand quadrilatère est trouvé."""
    import numpy as np
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = image.shape[0] * image.shape[1] * 0.02
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:25]:
        candidate = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(candidate) != 4 or cv2.contourArea(candidate) <= minimum_area:
            continue
        points = candidate.reshape(4, 2).astype("float32")
        ordered = np.zeros((4, 2), dtype="float32")
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).reshape(-1)
        ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
        ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
        width = int(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])))
        height = int(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))
        if width <= 0 or height <= 0:
            continue
        target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        corrected = cv2.warpPerspective(image, cv2.getPerspectiveTransform(ordered, target), (width, height), borderValue=(255, 255, 255))
        return corrected, {"status": "applied", "corners": ordered.astype(int).tolist(), "target_size_px": [width, height]}
    return image, {"status": "skipped_no_quadrilateral"}
