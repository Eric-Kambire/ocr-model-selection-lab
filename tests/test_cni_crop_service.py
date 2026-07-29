"""Tests du point d'entrée de crop réellement utilisé par le benchmark CNI."""

from pathlib import Path

from PIL import Image

from ocr_benchmark import cni_crop_service


def test_original_method_preserves_the_full_resolution_file(tmp_path: Path):
    """Le mode sans crop transmet la source normalisée sans rééchantillonnage."""
    source = tmp_path / "source.png"
    Image.new("RGB", (2400, 1600), "white").save(source)

    result = cni_crop_service.crop_cni_for_benchmark(
        source,
        tmp_path / "crop",
        method="original",
    )

    assert result["image_path"] == str(source)
    assert result["source_sent_unchanged"] is True
    assert (result["width"], result["height"]) == (2400, 1600)


def test_smart_v4_parameters_and_result_are_adapted_for_the_runner(
    tmp_path: Path, monkeypatch
):
    """Le service transmet les réglages V4 et conserve son rapport de diagnostic."""
    source = tmp_path / "source.png"
    final = tmp_path / "final.png"
    report = tmp_path / "report.json"
    Image.new("RGB", (2000, 3000), "white").save(source)
    Image.new("RGB", (1600, 1000), "white").save(final)
    captured = {}

    def fake_run_crop_method(source_path, output_dir, *, method, parameters):
        captured.update(
            source_path=source_path,
            output_dir=output_dir,
            method=method,
            parameters=parameters,
        )
        return {
            "status": "crop_detected",
            "final_path": str(final),
            "source_sent_unchanged": False,
            "stages": [{"name": "Crop redressé"}],
            "summary": {"score": 0.81, "detector": "contour"},
            "report_path": str(report),
        }

    monkeypatch.setattr(cni_crop_service, "run_crop_method", fake_run_crop_method)
    result = cni_crop_service.crop_cni_for_benchmark(
        source,
        tmp_path / "artefacts",
        minimum_score=0.61,
        margin_ratio=0.018,
    )

    assert captured["method"] == "hybrid_v4"
    assert captured["parameters"] == {
        "hybrid_min_score": 0.61,
        "hybrid_margin": 0.018,
    }
    assert result["image_path"] == str(final)
    assert result["crop_status"] == "crop_detected_smart_v4"
    assert result["score"] == 0.81
    assert result["report_path"] == str(report)


def test_benchmark_dispatches_connected_components_and_canny(
    tmp_path: Path, monkeypatch
):
    """Les méthodes du laboratoire sont réellement appelées par le benchmark."""
    source = tmp_path / "source.png"
    final = tmp_path / "final.png"
    Image.new("RGB", (1200, 1800), "white").save(source)
    Image.new("RGB", (856, 540), "white").save(final)
    calls: list[tuple[str, dict]] = []

    def fake_run_crop_method(source_path, output_dir, *, method, parameters):
        calls.append((method, parameters))
        return {
            "status": "crop_detected",
            "final_path": str(final),
            "source_sent_unchanged": False,
            "stages": [],
            "summary": {"score": 0.75},
            "report_path": None,
        }

    monkeypatch.setattr(cni_crop_service, "run_crop_method", fake_run_crop_method)

    connected = cni_crop_service.crop_cni_for_benchmark(
        source,
        tmp_path / "connected",
        method="connected_components",
    )
    canny = cni_crop_service.crop_cni_for_benchmark(
        source,
        tmp_path / "canny",
        method="canny_contours",
    )

    assert calls[0][0] == "connected_components"
    assert calls[0][1]["component_break_bridges"] is True
    assert calls[1][0] == "canny_contours"
    assert connected["crop_status"] == "crop_detected_connected_components"
    assert canny["crop_status"] == "crop_detected_canny_contours"
