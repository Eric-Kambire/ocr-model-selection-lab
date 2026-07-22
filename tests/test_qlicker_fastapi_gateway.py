"""Tests locaux de la liste blanche FastAPI, sans appel réseau Qlicker."""

import pytest
from fastapi import HTTPException

from scripts.qlicker_fastapi_gateway import parse_url_for_gateway, validate_internal_endpoint


def test_gateway_accepts_only_configured_host(monkeypatch):
    monkeypatch.setenv("QLICKER_ALLOWED_HOSTS", "qlicker.intra.local,10.10.5.8")

    assert validate_internal_endpoint("https://qlicker.intra.local/api/list") == (
        "https://qlicker.intra.local/api/list", "qlicker.intra.local", 443
    )


def test_gateway_rejects_host_outside_allowlist(monkeypatch):
    monkeypatch.setenv("QLICKER_ALLOWED_HOSTS", "qlicker.intra.local")

    with pytest.raises(HTTPException) as error:
        validate_internal_endpoint("https://example.com/")

    assert error.value.status_code == 403


def test_parser_keeps_blank_and_duplicate_parameters():
    endpoint, parameters = parse_url_for_gateway("https://qlicker.intra.local/api?page=1&filter=&tag=a&tag=b")

    assert endpoint == "https://qlicker.intra.local/api"
    assert parameters == [
        {"name": "page", "value": "1"},
        {"name": "filter", "value": ""},
        {"name": "tag", "value": "a"},
        {"name": "tag", "value": "b"},
    ]
