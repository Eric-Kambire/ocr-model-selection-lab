"""Tests du moteur multi-méthodes, sans démarrer Gradio."""

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_benchmark.cni_crop_methods import (
    normalise_crop_lab_source,
    run_crop_method,
)
from ocr_benchmark.cni_smart_crop import (
    _normalise_hough_segments,
    detect_dark_frame_bands,
    foreground_leakage_penalty,
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
    assert any(stage["name"] == "Ponts fins supprimés" for stage in result["stages"])


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


def test_hybrid_v4_exposes_candidates_and_keeps_a_safe_fallback(tmp_path: Path):
    """Le moteur V4 doit toujours produire des étapes et un fichier exploitable."""
    source = tmp_path / "scan.png"
    _noisy_card_scan(source)

    result = run_crop_method(
        source,
        tmp_path / "hybrid",
        method="hybrid_v4",
        parameters={"hybrid_min_score": 0.40},
    )

    assert result["status"] in {"crop_detected", "fallback_original"}
    assert Path(result["final_path"]).is_file()
    assert any(stage["name"] == "Candidats classés" for stage in result["stages"])
    assert any(stage["name"] == "Décision du détecteur" for stage in result["stages"])


def test_hybrid_v4_returns_original_when_confidence_is_insufficient(tmp_path: Path):
    """Une page vide ne doit jamais produire un crop artificiel."""
    source = tmp_path / "empty.png"
    Image.new("RGB", (900, 1200), "white").save(source)

    result = run_crop_method(
        source,
        tmp_path / "empty_hybrid",
        method="hybrid_v4",
        parameters={"hybrid_min_score": 0.90},
    )

    assert result["status"] == "fallback_original"
    assert result["source_sent_unchanged"] is True
    assert Path(result["final_path"]) == source


def test_intermediate_stage_generation_can_be_disabled(tmp_path: Path):
    """La production peut garder source/sortie sans écrire tous les diagnostics."""
    source = tmp_path / "scan.png"
    _noisy_card_scan(source)

    result = run_crop_method(
        source,
        tmp_path / "compact",
        method="hybrid_v4",
        parameters={"hybrid_min_score": 0.40, "generate_steps": False},
    )

    assert Path(result["final_path"]).exists()
    assert len(result["stages"]) <= 3


def test_v4_detects_only_continuous_dark_frame_bands():
    """Une bordure continue est neutralisée, pas un simple objet sombre isolé."""
    import numpy as np

    image = np.full((500, 700, 3), 235, dtype=np.uint8)
    image[:12, :] = 10
    image[:, :9] = 10
    image[80:120, 620:660] = 5

    top, bottom, left, right, mask = detect_dark_frame_bands(image)

    assert (top, bottom, left, right) == (12, 0, 9, 0)
    assert int(mask[5, 300]) == 255
    assert int(mask[100, 640]) == 0


def test_v4_penalises_a_candidate_that_cuts_nearby_content():
    """Un bord interne doit perdre plus de points que le contour complet."""
    import numpy as np

    foreground = np.zeros((500, 700), dtype=np.uint8)
    foreground[140:360, 170:530] = 255
    full = np.array([[170, 140], [530, 140], [530, 360], [170, 360]], dtype=np.float32)
    truncated = np.array([[170, 140], [420, 140], [420, 360], [170, 360]], dtype=np.float32)

    full_penalty, _ = foreground_leakage_penalty(full, foreground)
    truncated_penalty, _ = foreground_leakage_penalty(truncated, foreground)

    assert truncated_penalty > full_penalty


def test_hough_segments_accepts_macos_and_standard_opencv_shapes():
    """Les formes ``N×4`` et ``N×1×4`` doivent donner les mêmes segments."""
    import numpy as np

    compact = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int32)
    standard = compact.reshape(-1, 1, 4)

    compact_result = _normalise_hough_segments(compact)
    standard_result = _normalise_hough_segments(standard)

    assert compact_result.shape == (2, 4)
    assert standard_result.shape == (2, 4)
    assert np.array_equal(compact_result, standard_result)
