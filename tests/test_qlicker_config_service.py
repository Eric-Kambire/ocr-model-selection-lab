"""Tests de la persistance locale QlickEER, sans interface ni réseau."""

from ocr_benchmark.application.qlicker_config_service import (
    load_qlicker_config,
    qlicker_config_from_ui,
    reset_qlicker_config,
    save_qlicker_config,
)


def test_qlicker_config_round_trip_preserves_editable_routes(tmp_path):
    """Les endpoints et paramètres restent disponibles après un redémarrage."""
    path = tmp_path / "qlickeer_api.local.json"
    value = qlicker_config_from_ui(
        base_url="https://internal.example",
        timeout_seconds=300,
        use_system_proxy=True,
        verify_ssl=False,
        proxy_url="http://ncproxy:8080",
        import_root="D:/imports",
        routes={
            "list": {
                "raw_url": "https://internal.example/get_customer?page=1",
                "endpoint": "/get_customer",
                "params": [["page", "1", True], ["ignored", "x", False]],
            },
        },
    )

    save_qlicker_config(path, value, import_root="D:/default-imports")
    loaded = load_qlicker_config(path, import_root="D:/default-imports")

    assert loaded["base_url"] == "https://internal.example"
    assert loaded["timeout_seconds"] == 300
    assert loaded["routes"]["list"]["endpoint"] == "/get_customer"
    assert loaded["routes"]["list"]["params"] == [["page", "1", True], ["ignored", "x", False]]
    assert loaded["routes"]["view"]["params"] == []


def test_qlicker_config_reset_deletes_local_file(tmp_path):
    """Réinitialiser n'efface ni les imports ni les résultats, seulement la config."""
    path = tmp_path / "qlickeer_api.local.json"
    save_qlicker_config(path, {"base_url": "https://internal"}, import_root="D:/imports")

    defaults = reset_qlicker_config(path, import_root="D:/imports")

    assert not path.exists()
    assert defaults["base_url"] == ""
    assert defaults["import_root"] == "D:/imports"
