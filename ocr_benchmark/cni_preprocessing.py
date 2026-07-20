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
    """Applique seulement les traitements cochés et en conserve la trace."""
    enabled = {name: bool(options.get(name, False)) for name in ("deskew", "contrast", "denoise")}
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
    if enabled["contrast"]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        image = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
        operations.append("contrast_clahe")
    if enabled["denoise"]:
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        operations.append("denoise")
    if enabled["deskew"]:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Écriture impossible : {output_path}")
    return {"status": "applied", "image_path": str(output_path), "operations": operations}
