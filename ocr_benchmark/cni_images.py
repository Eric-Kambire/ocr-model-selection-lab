"""PDF rendering and image preparation for Moroccan CNI benchmark inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def render_single_page_pdf(pdf_path: Path, output_path: Path, dpi: int = 300) -> dict[str, Any]:
    """Render exactly one PDF page to PNG with an explicit page-count check."""
    if dpi < 72 or dpi > 600:
        raise ValueError("CNI render DPI must be between 72 and 600.")
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render CNI PDFs. Install requirements.txt.") from exc
    with fitz.open(pdf_path) as document:
        if document.page_count != 1:
            raise ValueError(f"Expected exactly one PDF page, found {document.page_count}: {pdf_path.name}")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(output_path))
    with Image.open(output_path) as image:
        width, height = image.size
    return {"image_path": str(output_path), "width": width, "height": height, "dpi": dpi}


def crop_cni_from_a4(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Crop a non-white CNI from a page; save full page with visible fallback."""
    with Image.open(source_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    bbox = ImageOps.grayscale(original).point(lambda pixel: 255 if pixel < 242 else 0).getbbox()
    if bbox is None:
        return _copy_full_page(original, output_path, "crop_not_detected")
    left, top, right, bottom = bbox
    padding = max(12, int(max(original.size) * 0.015))
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(original.width, right + padding), min(original.height, bottom + padding)
    width, height = right - left, bottom - top
    ratio = width / height if height else 0
    coverage = (width * height) / (original.width * original.height)
    if not 1.20 <= ratio <= 2.05 or coverage > 0.65 or coverage < 0.02:
        return _copy_full_page(original, output_path, "crop_fallback_full_page")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original.crop((left, top, right, bottom)).save(output_path, format="PNG")
    return {"image_path": str(output_path), "crop_status": "crop_detected", "crop_box": [left, top, right, bottom], "coverage": round(coverage, 4)}


def build_vertical_cni_composite(recto_path: Path, verso_path: Path, output_path: Path) -> str:
    """Create one image with recto on top and verso underneath."""
    with Image.open(recto_path) as source:
        recto = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(verso_path) as source:
        verso = ImageOps.exif_transpose(source).convert("RGB")
    target_width = max(recto.width, verso.width)
    recto, verso = _resize_to_width(recto, target_width), _resize_to_width(verso, target_width)
    separator = 36
    canvas = Image.new("RGB", (target_width, recto.height + separator + verso.height), "white")
    canvas.paste(recto, (0, 0)); canvas.paste(verso, (0, recto.height + separator))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, recto.height + separator // 2, target_width, recto.height + separator // 2), fill="black", width=2)
    draw.text((8, 8), "RECTO", fill="black")
    draw.text((8, recto.height + separator + 8), "VERSO", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return str(output_path)


def _copy_full_page(image: Image.Image, output_path: Path, status: str) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return {"image_path": str(output_path), "crop_status": status, "crop_box": None, "coverage": None}


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    return image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
