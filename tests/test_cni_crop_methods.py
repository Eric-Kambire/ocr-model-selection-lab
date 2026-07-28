"""Tests du moteur multi-méthodes, sans démarrer Gradio."""

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_benchmark.cni_crop_methods import (
    normalise_crop_lab_source,
    run_crop_method,
)


def _noisy_card_scan(path: Path) -> None:
    page = Image.new("RGB", (1000, 1400), (235, 235, 235))
    draw = ImageDraw.Draw(page)
    draw.rectangle((80, 360, 680, 740), fill=(165, 205, 190), outline=(25, 65, 65), width=8)
    draw.rectangle((130, 420, 360, 455), fill=(30, 70, 70))
    draw.rectangle((130, 500, 560, 530), fill=(30, 70, 70))
    # Bruits externes : barre lointaine et petits points.
    draw.rectangle((0, 18, 999, 27), fill=(15, 15, 15))
    for x, y in ((850, 180), (920, 620), (760, 980), (210, 1100)):
        draw.ellipse((x, y, x + 7, y + 7), fill=(20, 20, 20))
    page.save(path)


def test_connected_components_separates_distant_noise(tmp_path: Path):
    source = tmp_path / "scan.png"
    _noisy_card_scan(source)

    result = run_crop_method(
        source,
        tmp_path / "components",
        method="connected_components",
        parameters={
            "component_mask_mode": "canny",
            "component_kernel": 11,
            "component_min_area_pct": 0.05,
            "component_selection": "scored",
        },
    )

    assert result["status"] == "crop_detected"
    assert result["source_sent_unchanged"] is False
    assert Path(result["final_path"]).is_file()
    assert any(stage["name"] == "Composants séparés" for stage in result["stages"])


def test_global_rectangle_exposes_noise_without_forcing_bad_crop(tmp_path: Path):
    source = tmp_path / "scan.png"
    _noisy_card_scan(source)

    result = run_crop_method(
        source,
        tmp_path / "global",
        method="min_area_rect",
        parameters={"global_threshold": 235},
    )

    assert any(stage["name"] == "Rectangle global" for stage in result["stages"])
    assert result["status"] in {"crop_detected", "fallback_original"}


def test_image_source_is_normalised_without_changing_dimensions(tmp_path: Path):
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (640, 480), "white").save(source)
    output = tmp_path / "normalised.png"

    metadata = normalise_crop_lab_source(source, output, dpi=300, page_number=1)

    assert metadata["source_kind"] == "jpg"
    assert (metadata["width"], metadata["height"]) == (640, 480)
    assert output.is_file()


def test_hybrid_v3_exposes_candidates_and_keeps_a_safe_fallback(tmp_path: Path):
    """Le moteur V3 doit toujours produire des étapes et un fichier exploitable."""
    source = tmp_path / "scan.png"
    _noisy_card_scan(source)

    result = run_crop_method(
        source,
        tmp_path / "hybrid",
        method="hybrid_v3",
        parameters={"hybrid_min_score": 0.40},
    )

    assert result["status"] in {"crop_detected", "fallback_original"}
    assert Path(result["final_path"]).is_file()
    assert any(stage["name"] == "Candidats classés" for stage in result["stages"])
    assert any(stage["name"] == "Décision du détecteur" for stage in result["stages"])


def test_hybrid_v3_returns_original_when_confidence_is_insufficient(tmp_path: Path):
    """Une page vide ne doit jamais produire un crop artificiel."""
    source = tmp_path / "empty.png"
    Image.new("RGB", (900, 1200), "white").save(source)

    result = run_crop_method(
        source,
        tmp_path / "empty_hybrid",
        method="hybrid_v3",
        parameters={"hybrid_min_score": 0.90},
    )

    assert result["status"] == "fallback_original"
    assert result["source_sent_unchanged"] is True
    assert Path(result["final_path"]) == source
