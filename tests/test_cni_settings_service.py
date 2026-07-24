"""Tests de persistance locale des réglages CNI Gradio."""

from ocr_benchmark.application.cni_settings_service import (
    cni_settings_from_ui,
    default_cni_settings,
    load_cni_settings,
    save_cni_settings,
)


def _defaults():
    return default_cni_settings(cpu_threads=4, system_prompt="system", prompt_instructions="user")


def test_cni_settings_round_trip_keeps_preprocessing_and_execution(tmp_path):
    """Les choix CNI restent identiques après un redémarrage de l'interface."""
    defaults = _defaults()
    value = cni_settings_from_ui(
        models=["ollama:test"], strategy="combined_vertical", dpi=350,
        timeout_seconds=600, cpu_threads=2, unload_after_task=False,
        continue_without_label=True, recto_suffix="_CIN_recto",
        verso_suffix="_CIN_verso", rotation_method="opencv",
        perspective_correction=True, preprocessing=["contrast", "denoise"],
        system_prompt="system personnalisé", prompt_instructions="user personnalisé",
    )
    path = tmp_path / "cni_settings.local.json"

    save_cni_settings(path, value, defaults=defaults)
    loaded = load_cni_settings(path, defaults=defaults)

    assert loaded["strategy"] == "combined_vertical"
    assert loaded["dpi"] == 350
    assert loaded["rotation_method"] == "opencv"
    assert loaded["perspective_correction"] is True
    assert loaded["preprocessing"] == ["contrast", "denoise"]
    assert loaded["models"] == ["ollama:test"]


def test_invalid_saved_settings_fall_back_to_safe_defaults(tmp_path):
    """Une configuration locale invalide ne casse jamais le démarrage."""
    defaults = _defaults()
    path = tmp_path / "cni_settings.local.json"
    path.write_text('{"schema_version": 1, "rotation_method": "invalid"}', encoding="utf-8")

    loaded = load_cni_settings(path, defaults=defaults)

    assert loaded["rotation_method"] == "none"
    assert loaded["strategy"] == "separate_calls"
