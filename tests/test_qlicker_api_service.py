"""Tests unitaires du constructeur de requêtes Qlicker sans réseau réel."""

import pytest

from ocr_benchmark.application.qlicker_api_service import (
    _qlicker_file_extension,
    build_qlicker_url,
    download_qlicker_file,
    editable_rows_to_query_pairs,
    merge_query_params,
    parse_qlicker_url,
    parse_extra_query_params,
    system_proxy_mapping,
    extract_customer_cni_label,
    find_cni_documents,
)


def test_extra_params_keep_empty_string_and_null():
    """Le formulaire doit distinguer une valeur vide d'un paramètre omis."""
    assert parse_extra_query_params('{"sort": null, "filter": ""}') == {
        "sort": None,
        "filter": "",
    }


def test_guided_params_override_extra_json():
    """Les cinq paramètres connus restent contrôlés par leurs champs dédiés."""
    assert merge_query_params({"page": 2, "step": None}, {"page": 99, "other": "x"}) == {
        "page": 2,
        "other": "x",
    }


def test_url_uses_base_and_endpoint_segment():
    """La Base URL commune et la fonction HTTP sont assemblées sans double slash."""
    assert build_qlicker_url("http://qlicker.internal/api/", "/GetCustomers") == "http://qlicker.internal/api/GetCustomers"


def test_invalid_extra_json_is_explicit():
    """Un JSON mal formé doit être corrigé avant qu'une requête parte."""
    with pytest.raises(ValueError, match="JSON invalide"):
        parse_extra_query_params("{invalid}")


def test_url_parser_preserves_blank_and_duplicate_parameters_for_editing():
    """Une URL Postman devient une table Gradio modifiable sans perte."""
    base_url, endpoint, rows = parse_qlicker_url(
        "https://qlicker.internal/api/get_signed_documents_list?customerID=42&filter=&tag=a&tag=b"
    )

    assert base_url == "https://qlicker.internal"
    assert endpoint == "api/get_signed_documents_list"
    assert rows == [
        ["customerID", "42", True],
        ["filter", "", True],
        ["tag", "a", True],
        ["tag", "b", True],
    ]
    assert editable_rows_to_query_pairs(rows + [["disabled", "x", False]]) == [
        ("customerID", "42"), ("filter", ""), ("tag", "a"), ("tag", "b"),
    ]


def test_system_proxy_keeps_only_http_schemes(monkeypatch):
    """Le mode proxy système ne transmet pas de réglage non HTTP à requests."""
    monkeypatch.setattr(
        "ocr_benchmark.application.qlicker_api_service.getproxies",
        lambda: {"http": "http://proxy:8080", "https": "http://proxy:8080", "ftp": "ftp://ignore"},
    )

    assert system_proxy_mapping() == {
        "http": "http://proxy:8080",
        "https": "http://proxy:8080",
    }


def test_customer_data_becomes_cni_label_with_confidence():
    """Les champs métier QlickEER deviennent le contrat CNI attendu par l'évaluateur."""
    payload = {"response": {"body": {"response_data": {"customer": {
        "id": "customer-1",
        "customer_data": {
            "cin_id": "A0000000", "cin_id_confidence": 100,
            "first_name": "PRENOM", "first_name_confidence": 95,
            "last_name": "NOM", "birth_date": "01/01/1990",
            "birth_place": "RABAT", "validity_date": "01/01/2030",
            "address": "ADRESSE",
        },
    }}}}}

    label = extract_customer_cni_label(payload)

    assert label["cin"] == "A0000000"
    assert label["date_naissance"] == "1990-01-01"
    assert label["date_validite"] == "2030-01-01"
    assert label["field_confidence"]["cin"] == 100


def test_document_list_detects_cni_pair_only():
    """Les conventions et autres pièces ne sont jamais prises pour une CNI."""
    assert find_cni_documents([
        "Qlickeer_A0000000_Convention_compte.pdf",
        "Qlickeer_A0000000_CIN_recto.pdf",
        "Qlickeer_A0000000_CIN_verso.pdf",
    ]) == {
        "recto": "Qlickeer_A0000000_CIN_recto.pdf",
        "verso": "Qlickeer_A0000000_CIN_verso.pdf",
    }


def test_file_download_extension_accepts_generic_mime_filename_or_signature():
    """view_file peut annoncer octet-stream tout en envoyant un vrai PDF/image."""
    assert _qlicker_file_extension(
        "application/octet-stream", 'attachment; filename="CIN_recto.pdf"', [], b"",
    ) == ".pdf"
    assert _qlicker_file_extension(
        "application/octet-stream", "", [("file", "CIN_verso.jpeg")], b"",
    ) == ".jpg"
    assert _qlicker_file_extension("", "", [], b"\x89PNG\r\n\x1a\nrest") == ".png"


def test_binary_signature_overrides_misleading_pdf_name_and_mime():
    """Une réponse JPEG appelée ``.pdf`` doit rester une image exploitable."""
    jpeg_prefix = b"\xff\xd8\xff\xe0JFIF"
    assert _qlicker_file_extension(
        "application/pdf", 'attachment; filename="CIN_recto.pdf"', [("file", "CIN_recto.pdf")], jpeg_prefix,
    ) == ".jpg"


def test_empty_binary_download_does_not_create_a_final_pdf(tmp_path, monkeypatch):
    """Une réponse PDF annoncée mais vide reste visible comme erreur, sans faux artefact."""
    class EmptyResponse:
        url = "https://qlicker.internal/view_file"
        headers = {"content-type": "application/pdf", "content-length": "0"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 64 * 1024
            return iter(())

        def close(self):
            return None

    class EmptySession:
        trust_env = True

        def get(self, *args, **kwargs):
            return EmptyResponse()

    monkeypatch.setattr("ocr_benchmark.application.qlicker_api_service.requests.Session", EmptySession)

    with pytest.raises(ValueError, match="0 octet"):
        download_qlicker_file(
            "https://qlicker.internal", "view_file", [("file", "cin.pdf")], tmp_path / "cin",
        )
    assert not (tmp_path / "cin.pdf").exists()
