from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from ocr_benchmark.cni import (
    build_cni_global_json,
    build_cni_prompt,
    crop_cni_from_a4,
    import_cni_zip,
    materialize_cni_labels,
    parse_cni_json_response,
    scan_cni_clients,
)
from ocr_benchmark.cni_runner import iter_cni_benchmark


def _write_pdf(path: Path) -> None:
    import fitz

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.draw_rect((30, 30, 330, 220), color=(0.1, 0.4, 0.7), fill=(0.8, 0.9, 1.0))
    page.insert_text((60, 80), "CNI TEST")
    document.save(path)
    document.close()


def test_scan_uses_folder_identifier_and_materializes_external_label(tmp_path: Path):
    clients_root = tmp_path / "clients"
    labels_root = tmp_path / "labels"
    client = clients_root / "folder-client-42"
    client.mkdir(parents=True)
    labels_root.mkdir()
    _write_pdf(client / "document-other-id_CIN_Recto.pdf")
    _write_pdf(client / "document-other-id_CIN_Verso.pdf")
    (labels_root / "folder-client-42.jsonb").write_text('{"nom":"TEST"}', encoding="utf-8")

    records = scan_cni_clients(clients_root, labels_root)
    assert records[0]["folder_client_id"] == "folder-client-42"
    assert records[0]["recto_document_id"] == "document-other-id"
    assert records[0]["status"] == "ready"

    updated = materialize_cni_labels(records)
    label = client / "folder-client-42.json"
    assert updated[0]["label_status"] == "label_materialized"
    assert json.loads(label.read_text(encoding="utf-8")) == {"nom": "TEST"}


def test_crop_detects_card_area_without_using_a4_as_the_result(tmp_path: Path):
    page = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((20, 20, 620, 395), fill=(160, 200, 180))
    source = tmp_path / "page.png"
    target = tmp_path / "crop.png"
    page.save(source)

    result = crop_cni_from_a4(source, target)
    assert result["crop_status"] == "crop_detected"
    with Image.open(target) as cropped:
        assert 1.2 <= cropped.width / cropped.height <= 2.05


def test_side_json_parser_and_global_preserve_both_cin_values():
    recto, error = parse_cni_json_response(
        '{"cin":"BM42518","nom":"ZAAD","prenom":"CHAIMAA","date_naissance":"2001-01-20","ville_naissance":"CASA","date_validite":"2029-03-21"}',
        "recto",
    )
    assert error is None
    global_json = build_cni_global_json(
        {"folder_client_id": "client-1", "label_status": "label_not_found"},
        recto,
        {"cin": "BM42518", "date_validite": "2029-03-21", "adresse": "CASA"},
    )
    assert global_json["cin_recto"] == "BM42518"
    assert global_json["cin_verso"] == "BM42518"
    assert global_json["cin_fusionne"] == "BM42518"
    assert global_json["cin_coherent"] is True


def test_cni_prompt_covers_old_new_layout_and_operator_instructions():
    prompt = build_cni_prompt("recto", instructions="Prioritize a sharp reading of the CIN identifier.")
    assert "old or new layout" in prompt
    assert "Prioritize a sharp reading" in prompt
    assert '"cin": null' in prompt


def test_zip_import_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("../outside.txt", "no")
    try:
        import_cni_zip(archive, tmp_path / "imports")
    except ValueError as exc:
        assert "Unsafe ZIP path" in str(exc)
    else:
        raise AssertionError("Unsafe ZIP path was accepted")


class _FakeModel:
    model_name = "FakeVision"

    def perform_ocr(self, image_path: str, *, prompt: str | None = None) -> dict:
        if prompt and "RECTO" in prompt:
            content = '{"cin":"AA1","nom":"NOM","prenom":"PRENOM","date_naissance":"2000-01-01","ville_naissance":"CASA","date_validite":"2030-01-01"}'
        else:
            content = '{"cin":"AA1","date_validite":"2030-01-01","adresse":"CASA"}'
        return {"text": content, "raw_response": content, "latency": 0.01, "status": "success", "device": "test"}

    def close(self) -> None:
        return None


class _FakeRegistry:
    def create(self, *args, **kwargs):
        return _FakeModel()


def test_separate_runner_creates_recto_verso_and_global_outputs(tmp_path: Path):
    client = tmp_path / "clients" / "folder-client"
    client.mkdir(parents=True)
    _write_pdf(client / "source_CIN_Recto.pdf")
    _write_pdf(client / "source_CIN_Verso.pdf")
    records = scan_cni_clients(tmp_path / "clients")

    events = list(iter_cni_benchmark(_FakeRegistry(), ["fake:vision"], records, tmp_path / "runs", timeout_seconds=5))
    result = events[-1]["result"]
    assert result["status"] == "success"
    assert Path(result["recto_json_path"]).is_file()
    assert Path(result["verso_json_path"]).is_file()
    assert Path(result["global_json_path"]).is_file()


class _RecordingModel:
    """Faux modèle qui mémorise les images réellement reçues par le runner."""

    model_name = "RecordingVision"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def perform_ocr(self, image_path: str, *, prompt: str | None = None) -> dict:
        self.calls.append((image_path, prompt or ""))
        if "RECTO at the top" in (prompt or ""):
            content = (
                '{"recto":{"cin":"AA1","nom":"NOM","prenom":"PRENOM",'
                '"date_naissance":"2000-01-01","ville_naissance":"CASA","date_validite":"2030-01-01"},'
                '"verso":{"cin":"AA1","date_validite":"2030-01-01","adresse":"CASA"}}'
            )
        elif "RECTO" in (prompt or ""):
            content = '{"cin":"AA1","nom":"NOM","prenom":"PRENOM","date_naissance":"2000-01-01","ville_naissance":"CASA","date_validite":"2030-01-01"}'
        else:
            content = '{"cin":"AA1","date_validite":"2030-01-01","adresse":"CASA"}'
        return {"text": content, "raw_response": content, "latency": 0.01, "status": "success", "device": "test"}


class _RecordingRegistry:
    def __init__(self) -> None:
        self.model = _RecordingModel()

    def create(self, *args, **kwargs):
        return self.model


def test_cni_strategies_send_expected_images_and_keep_pair_progress(tmp_path: Path):
    """Deux appels reçoivent recto/verso ; le collage n'est envoyé qu'une fois."""
    client = tmp_path / "clients" / "folder-client"
    client.mkdir(parents=True)
    _write_pdf(client / "source_CIN_Recto.pdf")
    _write_pdf(client / "source_CIN_Verso.pdf")
    records = scan_cni_clients(tmp_path / "clients")

    separate = _RecordingRegistry()
    separate_events = list(iter_cni_benchmark(separate, ["fake:vision"], records, tmp_path / "runs-separate", strategy="separate_calls"))
    assert len(separate.model.calls) == 2
    assert separate.model.calls[0][0].endswith("crop_recto.png")
    assert separate.model.calls[1][0].endswith("crop_verso.png")
    assert separate_events[-1]["completed"] == 1
    assert separate_events[-1]["total"] == 1  # Une paire client/modèle, malgré deux appels.

    combined = _RecordingRegistry()
    combined_events = list(iter_cni_benchmark(combined, ["fake:vision"], records, tmp_path / "runs-combined", strategy="combined_vertical"))
    assert len(combined.model.calls) == 1
    assert combined.model.calls[0][0].endswith("recto_verso_composite.png")
    assert combined_events[-1]["completed"] == 1
    assert combined_events[-1]["total"] == 1
