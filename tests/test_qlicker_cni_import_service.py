"""Tests sans réseau du lot CNI QlickEER multi-clients."""

from pathlib import Path
import json

from ocr_benchmark.application.qlicker_cni_import_service import (
    build_qlicker_cni_routes,
    find_completed_qlicker_batch,
    iter_prepare_qlicker_cni_clients,
    qlicker_preparation_fingerprint,
    write_qlicker_preparation_manifest,
)


def _rows(*pairs):
    """Construit les lignes du tableau Gradio de paramètres."""
    return [[name, value, True] for name, value in pairs]


def test_batch_materializes_a_cni_pair_and_normalized_label(tmp_path, monkeypatch):
    """Un client API devient une paire locale compatible avec le scanner CNI."""
    calls = []
    downloaded_params = []

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
        downloaded_params.append(list(params))
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
        "discovered", "loading_documents", "documents_detected",
        "downloading_recto", "downloading_verso", "downloaded",
        "loading_label", "label_normalized", "ready",
    ]
    client_dir = tmp_path / "A0000000"
    assert (client_dir / "A0000000_CIN_Recto.pdf").is_file()
    assert (client_dir / "A0000000_CIN_Verso.pdf").is_file()
    assert '"cin": "A0000000"' in (client_dir / "A0000000.json").read_text(encoding="utf-8")
    manifest = json.loads((client_dir / ".qlickeer_documents.json").read_text(encoding="utf-8"))
    assert manifest["downloads"]["recto"]["filename"] == "A0000000_CIN_Recto.pdf"
    assert manifest["downloads"]["verso"]["bytes"] == 9
    assert ("documents", [("customerID", "A0000000"), ("filter", "")]) in calls
    assert ("customer", [("customerID", "A0000000")]) in calls
    # La valeur ``page`` est celle testée et configurée par l'opérateur. Elle
    # ne doit pas être remplacée silencieusement par un index arbitraire.
    assert downloaded_params == [
        [("customerID", "A0000000"), ("page", "9"), ("file", "Qlickeer_A0000000_CIN_recto.pdf"), ("other", "kept")],
        [("customerID", "A0000000"), ("page", "9"), ("file", "Qlickeer_A0000000_CIN_verso.pdf"), ("other", "kept")],
    ]


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


def test_batch_accepts_the_enriched_candidate_sent_by_gradio(tmp_path, monkeypatch):
    """La candidature UI contient le client sous ``customer`` et garde son ID."""
    requested_customer_ids = []

    def fake_get(_base, endpoint, params, **_options):
        requested_customer_ids.append((endpoint, dict(params).get("customerID")))
        if endpoint == "documents":
            return {"response": {"status_code": 200, "body": {"response_data": {"documents_list": [
                "CIN_recto.pdf", "CIN_verso.pdf",
            ]}}}}
        return {"response": {"status_code": 200, "body": {"response_data": {"customer": {
            "id": "C0000000", "customer_data": {"cin_id": "C0000000"},
        }}}}}

    def fake_download(_base, _endpoint, _params, stem: Path, **_options):
        path = stem.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return {"path": str(path), "bytes": 3, "content_type": "image/png"}

    monkeypatch.setattr("ocr_benchmark.application.qlicker_cni_import_service.execute_qlicker_get", fake_get)
    monkeypatch.setattr("ocr_benchmark.application.qlicker_cni_import_service.download_qlicker_file", fake_download)
    routes = build_qlicker_cni_routes("customer", [], "documents", [], "file", [])

    events = list(iter_prepare_qlicker_cni_clients(
        [{"client_id": "C0000000", "customer": {"id": "C0000000"}, "status": "discovered"}],
        tmp_path, base_url="https://qlicker.internal", routes=routes,
        timeout_seconds=30, proxy_url=None, use_system_proxy=False, verify_ssl=True,
    ))

    assert events[0]["client_id"] == "C0000000"
    assert events[-1]["status"] == "ready"
    assert (tmp_path / "C0000000" / "C0000000_CIN_Recto.png").is_file()
    assert ("documents", "C0000000") in requested_customer_ids
    assert ("customer", "C0000000") in requested_customer_ids


def test_completed_batch_is_found_from_the_same_preparation_fingerprint(tmp_path):
    """Un refresh peut réutiliser un lot fini sans rappeler QlickEER."""
    routes = build_qlicker_cni_routes("customer", [], "documents", [], "file", [])
    customers = [{"client_id": "D0000000", "customer": {"id": "D0000000"}}]
    fingerprint = qlicker_preparation_fingerprint(
        customers,
        base_url="https://qlicker.internal",
        routes=routes,
        recto_suffix="_CIN_Recto",
        verso_suffix="_CIN_Verso",
    )
    batch = tmp_path / "batch-existing"
    batch.mkdir()
    write_qlicker_preparation_manifest(
        batch, fingerprint=fingerprint, status="completed", selected_count=1, ready_count=1,
    )

    assert find_completed_qlicker_batch(tmp_path, fingerprint) == batch
    assert find_completed_qlicker_batch(tmp_path, "other") is None


def test_partially_failed_batch_is_not_reused(tmp_path):
    """Un lot sans paire prête doit être rejoué, pas figé par idempotence."""
    routes = build_qlicker_cni_routes("customer", [], "documents", [], "file", [])
    customers = [{"client_id": "E0000000", "customer": {"id": "E0000000"}}]
    fingerprint = qlicker_preparation_fingerprint(
        customers,
        base_url="https://qlicker.internal",
        routes=routes,
        recto_suffix="_CIN_Recto",
        verso_suffix="_CIN_Verso",
    )
    batch = tmp_path / "batch-failed"
    batch.mkdir()
    write_qlicker_preparation_manifest(
        batch, fingerprint=fingerprint, status="completed", selected_count=1, ready_count=0,
    )

    assert find_completed_qlicker_batch(tmp_path, fingerprint) is None


def test_qlicker_fingerprint_ignores_suffix_case():
    """Changer seulement R/r ne crée pas artificiellement un nouveau lot."""
    routes = build_qlicker_cni_routes("customer", [], "documents", [], "file", [])
    customers = [{"client_id": "F0000000", "customer": {"id": "F0000000"}}]
    upper = qlicker_preparation_fingerprint(
        customers, base_url="https://qlicker.internal", routes=routes,
        recto_suffix="_CIN_Recto", verso_suffix="_CIN_Verso",
    )
    lower = qlicker_preparation_fingerprint(
        customers, base_url="https://qlicker.internal", routes=routes,
        recto_suffix="_CIN_recto", verso_suffix="_CIN_verso",
    )

    assert upper == lower
