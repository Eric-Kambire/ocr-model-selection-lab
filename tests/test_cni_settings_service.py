"""Tests de persistance locale des réglages CNI Gradio."""

import json

from ocr_benchmark.application.cni_settings_service import (
    LEGACY_DEFAULT_SYSTEM_PROMPT,
    LEGACY_DEFAULT_USER_INSTRUCTIONS,
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
        ollama_ignore_environment_proxy=True,
        continue_without_label=True, recto_suffix="_CIN_recto",
        verso_suffix="_CIN_verso", crop_method="connected_components",
        smart_crop_min_score=0.62, smart_crop_margin=0.018,
        rotation_method="opencv",
        perspective_correction=True, preprocessing=["contrast", "denoise"],
        output_format_mode="json",
        model_output_modes={"ollama:lightonocr": "prompt", "": "json", "bad": "xml"},
        system_prompt="system personnalisé", prompt_instructions="user personnalisé",
        prompt_scope_mode="full_rules",
        prompt_delivery_mode="image_only",
        prompt_context_budget=16384,
    )
    path = tmp_path / "cni_settings.local.json"

    save_cni_settings(path, value, defaults=defaults)
    loaded = load_cni_settings(path, defaults=defaults)

    assert loaded["strategy"] == "combined_vertical"
    assert loaded["dpi"] == 350
    assert loaded["crop_method"] == "connected_components"
    assert loaded["smart_crop_min_score"] == 0.62
    assert loaded["smart_crop_margin"] == 0.018
    assert loaded["rotation_method"] == "opencv"
    assert loaded["perspective_correction"] is True
    assert loaded["preprocessing"] == ["contrast", "denoise"]
    assert loaded["output_format_mode"] == "json"
    assert loaded["model_output_modes"] == {"ollama:lightonocr": "prompt"}
    assert loaded["models"] == ["ollama:test"]
    assert loaded["ollama_ignore_environment_proxy"] is True
    assert loaded["prompt_scope_mode"] == "full_rules"
    assert loaded["prompt_delivery_mode"] == "image_only"
    assert loaded["prompt_context_budget"] == 16384


def test_invalid_saved_settings_fall_back_to_safe_defaults(tmp_path):
    """Une configuration locale invalide ne casse jamais le démarrage."""
    defaults = _defaults()
    path = tmp_path / "cni_settings.local.json"
    path.write_text('{"schema_version": 1, "rotation_method": "invalid"}', encoding="utf-8")

    loaded = load_cni_settings(path, defaults=defaults)

    assert loaded["rotation_method"] == "none"
    assert loaded["strategy"] == "separate_calls"
    assert loaded["crop_method"] == "smart_crop_v4"
    assert loaded["output_format_mode"] == "schema"
    assert loaded["prompt_scope_mode"] == "side_specific"
    assert loaded["prompt_delivery_mode"] == "application_prompt"
    assert loaded["ollama_ignore_environment_proxy"] is True


def test_invalid_crop_settings_fall_back_to_safe_v4_defaults(tmp_path):
    """Un réglage de crop invalide ne désactive jamais le repli sûr."""
    defaults = _defaults()
    path = tmp_path / "cni_settings.local.json"
    path.write_text(
        '{"schema_version": 1, "crop_method": "inconnue", '
        '"smart_crop_min_score": 8, "smart_crop_margin": -1}',
        encoding="utf-8",
    )

    loaded = load_cni_settings(path, defaults=defaults)

    assert loaded["crop_method"] == "smart_crop_v4"
    assert loaded["smart_crop_min_score"] == 0.55
    assert loaded["smart_crop_margin"] == 0.012


def test_legacy_default_prompts_are_migrated_without_overwriting_custom_text(
    tmp_path,
):
    """Une mise à jour remplace les anciens défauts, jamais un prompt métier."""
    defaults = default_cni_settings(
        cpu_threads=4,
        system_prompt="nouveau système",
        prompt_instructions="",
    )
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system_prompt": LEGACY_DEFAULT_SYSTEM_PROMPT,
                "prompt_instructions": LEGACY_DEFAULT_USER_INSTRUCTIONS,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    custom_path = tmp_path / "custom.json"
    custom_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system_prompt": "système personnalisé",
                "prompt_instructions": "règle personnalisée",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    migrated = load_cni_settings(legacy_path, defaults=defaults)
    custom = load_cni_settings(custom_path, defaults=defaults)

    assert migrated["system_prompt"] == "nouveau système"
    assert migrated["prompt_instructions"] == ""
    assert custom["system_prompt"] == "système personnalisé"
    assert custom["prompt_instructions"] == "règle personnalisée"
