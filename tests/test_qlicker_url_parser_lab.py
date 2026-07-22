"""Tests sans réseau du parseur URL inspiré de Postman."""

from scripts import qlicker_url_parser_lab as parser_lab
from scripts.qlicker_url_parser_lab import parse_url_to_rows, rows_to_query_pairs


def test_parse_url_keeps_blank_and_duplicate_query_parameters():
    endpoint, rows, message = parse_url_to_rows(
        "http://qlicker.local/api/GetCustomers?page=1&filter=&tag=a&tag=b"
    )

    assert endpoint == "http://qlicker.local/api/GetCustomers"
    assert rows == [["page", "1", True], ["filter", "", True], ["tag", "a", True], ["tag", "b", True]]
    assert "4 paramètre" in message


def test_rows_to_query_pairs_omits_disabled_row():
    assert rows_to_query_pairs([["page", "1", True], ["filter", "", True], ["ignored", "x", False]]) == [
        ("page", "1"),
        ("filter", ""),
    ]


def test_get_uses_distinct_connect_and_read_timeouts(monkeypatch):
    """Le champ connexion et le champ réponse doivent arriver séparément à requests."""
    observed = {}

    class FakeResponse:
        url = "http://qlicker.local/api/GetCustomers?page=1"
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b"{}"

        def json(self):
            return {"ok": True}

    class FakeSession:
        trust_env = True

        def __enter__(self):
            observed["session"] = self
            return self

        def __exit__(self, *_args):
            return False

        def get(self, _url, *, params, timeout, proxies):
            observed["params"] = params
            observed["timeout"] = timeout
            observed["proxies"] = proxies
            return FakeResponse()

    monkeypatch.setattr(parser_lab.requests, "Session", FakeSession)
    _preview, status, body = parser_lab.execute_get(
        "http://qlicker.local/api/GetCustomers",
        [["page", "1", True]],
        123,
        456,
        False,
        "",
    )

    assert observed["timeout"] == (123.0, 456.0)
    assert observed["proxies"] is None
    assert observed["session"].trust_env is False
    assert "HTTP : `200`" in status
    assert '"ok": true' in body


def test_explicit_proxy_masks_password_and_dominates_environment_proxy(monkeypatch):
    """Le proxy explicite est transmis à Requests, sans exposer le secret."""
    observed = {}

    class FakeResponse:
        url = "https://qlicker.local/api/GetCustomers"
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b"{}"

        def json(self):
            return {"ok": True}

    class FakeSession:
        trust_env = True

        def __enter__(self):
            observed["session"] = self
            return self

        def __exit__(self, *_args):
            return False

        def get(self, _url, *, params, timeout, proxies):
            observed["proxies"] = proxies
            return FakeResponse()

    monkeypatch.setattr(parser_lab.requests, "Session", FakeSession)
    preview, _status, _body = parser_lab.execute_get(
        "https://qlicker.local/api/GetCustomers", [], 10, 10, True, "http://alice:secret@proxy.local:8080"
    )

    assert observed["proxies"] == {
        "http": "http://alice:secret@proxy.local:8080",
        "https": "http://alice:secret@proxy.local:8080",
    }
    assert "secret" not in preview
    assert "alice:***" in preview
