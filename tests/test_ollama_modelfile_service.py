"""Tests du service de création de modèle Ollama dérivé."""

from __future__ import annotations

import subprocess

import pytest

from ocr_benchmark.application.ollama_modelfile_service import (
    bind_selected_base_model,
    build_modelfile_template,
    create_ollama_model,
)


def test_template_supports_a_long_multiline_system_prompt():
    prompt = "Première ligne\n" + ("Règle métier détaillée.\n" * 300)

    value = build_modelfile_template("qwen3-vl:4b", system_prompt=prompt)

    assert value.startswith("FROM qwen3-vl:4b\n")
    assert 'SYSTEM """\nPremière ligne' in value
    assert value.endswith('"""\n')
    assert "PARAMETER num_ctx 8192" in value


def test_selected_model_replaces_the_from_instruction():
    value = bind_selected_base_model(
        "FROM ancien:latest\nPARAMETER temperature 0\n",
        "nouveau-vlm:4b",
    )

    assert value.startswith("FROM nouveau-vlm:4b\n")
    assert "ancien:latest" not in value


def test_create_uses_an_argument_list_and_never_a_shell(monkeypatch):
    observed = {}

    def fake_run(command, **options):
        observed["command"] = command
        observed["options"] = options
        return subprocess.CompletedProcess(command, 0, "success", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = create_ollama_model(
        base_model="vision:4b",
        new_model_name="cni-ocr:test",
        modelfile="FROM autre\nSYSTEM test\n",
        timeout_seconds=123,
    )

    assert result.success is True
    assert observed["command"][:3] == ["ollama", "create", "cni-ocr:test"]
    assert observed["options"]["timeout"] == 123
    assert "shell" not in observed["options"]
    assert result.modelfile.startswith("FROM vision:4b\n")


@pytest.mark.parametrize("name", ["", "-danger", "nom avec espace", "x;whoami"])
def test_invalid_model_names_are_rejected(name):
    with pytest.raises(ValueError):
        create_ollama_model(
            base_model="vision:4b",
            new_model_name=name,
            modelfile="FROM vision:4b\n",
        )
