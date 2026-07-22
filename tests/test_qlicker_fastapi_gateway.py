"""Tests locaux de la liste blanche FastAPI, sans appel réseau Qlicker."""

import pytest
from fastapi import HTTPException

from scripts.qlicker_fastapi_gateway import ssl_verification_enabled, validate_internal_endpoint


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


def test_ssl_verification_is_enabled_by_default_and_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.delenv("QLICKER_VERIFY_SSL", raising=False)
    assert ssl_verification_enabled() is True

    monkeypatch.setenv("QLICKER_VERIFY_SSL", "false")
    assert ssl_verification_enabled() is False
