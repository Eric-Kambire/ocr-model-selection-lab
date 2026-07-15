"""Image-only preparation helpers for Moroccan CNI benchmark inputs.

This module deliberately knows nothing about labels, prompts or model calls. It
turns the one-page PDF input contract into reproducible PNG artefacts that can
be inspected when a model produces an unexpected result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def render_single_page_pdf(pdf_path: Path, output_path: Path, dpi: int = 300) -> dict[str, Any]:
    """Render one source PDF page to a PNG kept with the benchmark artefacts.

    A CNI face is expected to be one PDF page. Rejecting a multi-page PDF here
    is intentional: otherwise the runner could silently benchmark an arbitrary
    page and make a result impossible to reproduce.

    Returns the saved path and final raster dimensions for ``preparation.json``.
    """
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
    """Try to crop an ID-card-shaped non-white region from an A4 scan.

    The scanner may put the card in any corner of a white page. This is only a
    conservative heuristic, never a destructive transform: an uncertain crop
    writes the full rendered page and records a ``crop_*`` status instead.
    """
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
    # The ID-1 card ratio is about 1.586. The wider interval accepts scanner
    # shadows/perspective while rejecting almost the entire A4 page.
    if not 1.20 <= ratio <= 2.05 or coverage > 0.65 or coverage < 0.02:
        return _copy_full_page(original, output_path, "crop_fallback_full_page")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original.crop((left, top, right, bottom)).save(output_path, format="PNG")
    return {"image_path": str(output_path), "crop_status": "crop_detected", "crop_box": [left, top, right, bottom], "coverage": round(coverage, 4)}


def build_vertical_cni_composite(recto_path: Path, verso_path: Path, output_path: Path) -> str:
    """Create a labelled recto-over-verso image for the combined VLM strategy.

    Both faces are resized to one width before stacking. The labels and divider
    reduce the risk that a VLM mistakes the second face for a continuation of
    the first one. The separate-call strategy still creates this artefact so a
    run stays visually diagnosable.
    """
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
    """Persist the safe fallback and expose why the crop was not trusted."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return {"image_path": str(output_path), "crop_status": status, "crop_box": None, "coverage": None}


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    """Resize proportionally; never distort a document merely to stack it."""
    if image.width == width:
        return image
    return image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
