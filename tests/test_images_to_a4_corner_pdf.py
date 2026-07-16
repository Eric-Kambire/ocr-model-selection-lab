from pathlib import Path

import fitz
import pytest
from PIL import Image

from scripts.images_to_a4_corner_pdf import (
    A4_HEIGHT_POINTS,
    A4_WIDTH_POINTS,
    a4_corner_rect,
    create_a4_pdf,
    parse_selection,
)


def test_parse_selection_accepts_all_and_ranges():
    assert parse_selection("all", 4) == [0, 1, 2, 3]
    assert parse_selection("1, 3-4, 1", 4) == [0, 2, 3]


def test_parse_selection_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="positions vont de 1 à 3"):
        parse_selection("4", 3)


def test_a4_corner_rect_places_image_at_requested_corner():
    rect = a4_corner_rect(1000, 500, corner="top-right", margin_mm=12, max_width_mm=120, max_height_mm=90)
    assert rect.x1 == pytest.approx(A4_WIDTH_POINTS - 12 * 72 / 25.4)
    assert rect.y0 == pytest.approx(12 * 72 / 25.4)


def test_create_a4_pdf_creates_a4_page(tmp_path: Path):
    image_path = tmp_path / "cin_recto.png"
    output_path = tmp_path / "cin_recto.pdf"
    Image.new("RGB", (640, 320), "white").save(image_path)

    create_a4_pdf(image_path, output_path, corner="bottom-left")

    document = fitz.open(output_path)
    try:
        assert document.page_count == 1
        page = document[0]
        assert page.rect.width == pytest.approx(A4_WIDTH_POINTS)
        assert page.rect.height == pytest.approx(A4_HEIGHT_POINTS)
        assert len(page.get_images()) == 1
    finally:
        document.close()
