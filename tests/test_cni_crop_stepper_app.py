"""Tests du laboratoire visuel autonome de préparation des scans CNI."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.cni_crop_stepper_app import _mm_to_px, build_pipeline


def _write_card(path: Path) -> None:
    """Crée une carte de test rectangulaire avec du contenu sombre visible."""
    image = Image.new("RGB", (800, 500), "#d8ebf0")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 770, 470), outline="#123a6f", width=10)
    draw.text((80, 130), "CNI TEST BM42518", fill="#101820")
    image.save(path)


def test_simulation_a4_creates_pdf_metrics_and_reusable_log(tmp_path: Path):
    """Le chemin image→A4 doit conserver une trace complète des paramètres."""
    card = tmp_path / "carte.png"
    _write_card(card)

    result = build_pipeline(str(card), "simulate_a4", 150, 242, False, 7.5, 120)
    state = result[0]

    assert len(state["paths"]) == 6
    assert len(state["metrics"]) == 6
    assert Path(state["prepared_pdf"]).is_file()
    assert Path(state["report_path"]).is_file()
    assert state["source_preparation"]["mode"] == "simulate_a4"
    assert state["source_preparation"]["simulation_angle_degrees"] == 7.5
    assert state["metrics"][0]["width_px"] == _mm_to_px(210, 150)
    assert state["metrics"][0]["height_px"] == _mm_to_px(297, 150)

    report = json.loads(Path(state["report_path"]).read_text(encoding="utf-8"))
    assert report["dpi"] == 150
    assert report["metrics"][5]["step"] == "6. Crop final"

