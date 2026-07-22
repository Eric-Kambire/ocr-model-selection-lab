"""Tests locaux de la liste blanche FastAPI, sans appel réseau Qlicker."""

import pytest
from fastapi import HTTPException

from scripts.qlicker_fastapi_gateway import validate_internal_endpoint


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
