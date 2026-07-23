"""Tests unitaires du constructeur de requêtes Qlicker sans réseau réel."""

import pytest

from ocr_benchmark.application.qlicker_api_service import (
    build_qlicker_url,
    editable_rows_to_query_pairs,
    merge_query_params,
    parse_qlicker_url,
    parse_extra_query_params,
    system_proxy_mapping,
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
