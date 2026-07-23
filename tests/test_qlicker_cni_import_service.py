"""Tests sans réseau du lot CNI QlickEER multi-clients."""

from pathlib import Path

from ocr_benchmark.application.qlicker_cni_import_service import (
    build_qlicker_cni_routes,
    iter_prepare_qlicker_cni_clients,
)


def _rows(*pairs):
    """Construit les lignes du tableau Gradio de paramètres."""
    return [[name, value, True] for name, value in pairs]


def test_batch_materializes_a_cni_pair_and_normalized_label(tmp_path, monkeypatch):
    """Un client API devient une paire locale compatible avec le scanner CNI."""
    calls = []

    def fake_get(_base, endpoint, params, **_options):
        calls.append((endpoint, list(params)))
        if endpoint == "documents":
            return {"response": {"status_code": 200, "body": {"response_data": {"documents_list": [
                "Qlickeer_A0000000_CIN_recto.pdf", "Qlickeer_A0000000_CIN_verso.pdf",
            ]}}}}
        if endpoint == "customer":
            return {"response": {"status_code": 200, "body": {"response_data": {"customer": {
                "id": "A0000000",
                "customer_data": {"cin_id": "A0000000", "first_name": "PRENOM", "last_name": "NOM"},
            }}}}}
        raise AssertionError(f"endpoint inattendu: {endpoint}")

    def fake_download(_base, endpoint, params, stem: Path, **_options):
        assert endpoint == "file"
        path = stem.with_suffix(".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-test")
        return {"path": str(path), "bytes": 9, "content_type": "application/pdf"}

    monkeypatch.setattr("ocr_benchmark.application.qlicker_cni_import_service.execute_qlicker_get", fake_get)
    monkeypatch.setattr("ocr_benchmark.application.qlicker_cni_import_service.download_qlicker_file", fake_download)

    routes = build_qlicker_cni_routes(
        "customer", _rows(("customerID", "placeholder")),
        "documents", _rows(("customerID", "placeholder"), ("filter", "")),
        "file", _rows(("customerID", "placeholder"), ("page", "9"), ("file", "placeholder"), ("other", "kept")),
    )
    events = list(iter_prepare_qlicker_cni_clients(
        [{"id": "A0000000", "last_name": "NOM", "first_name": "PRENOM"}],
        tmp_path,
        base_url="https://qlicker.internal",
        routes=routes,
        timeout_seconds=30,
        proxy_url=None,
        use_system_proxy=False,
        verify_ssl=True,
    ))

    assert [event["status"] for event in events] == [
        "discovered", "documents_detected", "downloaded", "label_normalized", "ready",
    ]
    client_dir = tmp_path / "A0000000"
    assert (client_dir / "A0000000_CIN_Recto.pdf").is_file()
    assert (client_dir / "A0000000_CIN_Verso.pdf").is_file()
    assert '"cin": "A0000000"' in (client_dir / "A0000000.json").read_text(encoding="utf-8")
    assert ("documents", [("customerID", "A0000000"), ("filter", "")]) in calls
    assert ("customer", [("customerID", "A0000000")]) in calls


def test_batch_keeps_documents_when_customer_label_is_unavailable(tmp_path, monkeypatch):
    """Une erreur de label ne supprime pas les deux documents déjà téléchargés."""
    def fake_get(_base, endpoint, _params, **_options):
        if endpoint == "documents":
            return {"response": {"status_code": 200, "body": {"response_data": {"documents_list": [
                "CIN_recto.pdf", "CIN_verso.pdf",
            ]}}}}
        return {"response": {"status_code": 500, "body": {}}}

    def fake_download(_base, _endpoint, _params, stem: Path, **_options):
        path = stem.with_suffix(".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
        return {"path": str(path), "bytes": 4, "content_type": "image/jpeg"}

    monkeypatch.setattr("ocr_benchmark.application.qlicker_cni_import_service.execute_qlicker_get", fake_get)
    monkeypatch.setattr("ocr_benchmark.application.qlicker_cni_import_service.download_qlicker_file", fake_download)
    routes = build_qlicker_cni_routes("customer", [], "documents", [], "file", [])

    events = list(iter_prepare_qlicker_cni_clients(
        [{"id": "B0000000"}], tmp_path, base_url="https://qlicker.internal", routes=routes,
        timeout_seconds=30, proxy_url=None, use_system_proxy=False, verify_ssl=True,
    ))

    assert events[-1]["status"] == "ready_without_label"
    assert (tmp_path / "B0000000" / "B0000000_CIN_Recto.jpg").is_file()
    assert not (tmp_path / "B0000000" / "B0000000.json").exists()
