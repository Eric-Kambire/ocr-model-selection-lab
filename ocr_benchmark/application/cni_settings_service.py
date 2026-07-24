"""Persistance locale et automatique des réglages d'exécution CNI.

Ce module ne stocke ni document ni résultat : seulement les choix opérateur
utiles pour reprendre le même protocole après un rechargement de Gradio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def default_cni_settings(*, cpu_threads: int, system_prompt: str, prompt_instructions: str) -> dict[str, Any]:
    """Retourne une configuration CNI sûre et sérialisable par défaut."""
    return {
        "schema_version": 1,
        "models": [],
        "strategy": "separate_calls",
        "dpi": 300,
        "timeout_seconds": 300,
        "cpu_threads": max(1, int(cpu_threads)),
        "unload_after_task": True,
        "continue_without_label": False,
        "recto_suffix": "_CIN_Recto",
        "verso_suffix": "_CIN_Verso",
        "rotation_method": "none",
        "perspective_correction": False,
        "preprocessing": [],
        "system_prompt": system_prompt,
        "prompt_instructions": prompt_instructions,
    }


def load_cni_settings(path: Path, *, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Charge les réglages locaux, sans bloquer l'interface si le JSON est invalide."""
    if not path.is_file():
        return dict(defaults)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(defaults)
    return _normalise(value, defaults)


def save_cni_settings(path: Path, value: Mapping[str, Any], *, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Enregistre atomiquement les réglages après toute modification UI."""
    normalised = _normalise(value, defaults)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(normalised, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return normalised


def cni_settings_from_ui(
    *,
    models: Any,
    strategy: Any,
    dpi: Any,
    timeout_seconds: Any,
    cpu_threads: Any,
    unload_after_task: Any,
    continue_without_label: Any,
    recto_suffix: Any,
    verso_suffix: Any,
    rotation_method: Any,
    perspective_correction: Any,
    preprocessing: Any,
    system_prompt: Any,
    prompt_instructions: Any,
) -> dict[str, Any]:
    """Convertit les composants Gradio en données JSON simples."""
    return {
        "schema_version": 1,
        "models": [str(model) for model in (models or []) if str(model).strip()],
        "strategy": str(strategy or ""),
        "dpi": _positive_integer(dpi, 300),
        "timeout_seconds": _positive_integer(timeout_seconds, 300),
        "cpu_threads": _positive_integer(cpu_threads, 1),
        "unload_after_task": bool(unload_after_task),
        "continue_without_label": bool(continue_without_label),
        "recto_suffix": str(recto_suffix or "").strip(),
        "verso_suffix": str(verso_suffix or "").strip(),
        "rotation_method": str(rotation_method or ""),
        "perspective_correction": bool(perspective_correction),
        "preprocessing": [str(name) for name in (preprocessing or []) if str(name).strip()],
        "system_prompt": str(system_prompt or "").strip(),
        "prompt_instructions": str(prompt_instructions or "").strip(),
    }


def _normalise(value: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Conserve uniquement les types et options reconnus par le runner CNI."""
    raw = value if isinstance(value, Mapping) and value.get("schema_version") == 1 else {}
    fallback = dict(defaults)
    result = cni_settings_from_ui(
        models=raw.get("models", fallback["models"]),
        strategy=raw.get("strategy", fallback["strategy"]),
        dpi=raw.get("dpi", fallback["dpi"]),
        timeout_seconds=raw.get("timeout_seconds", fallback["timeout_seconds"]),
        cpu_threads=raw.get("cpu_threads", fallback["cpu_threads"]),
        unload_after_task=raw.get("unload_after_task", fallback["unload_after_task"]),
        continue_without_label=raw.get("continue_without_label", fallback["continue_without_label"]),
        recto_suffix=raw.get("recto_suffix", fallback["recto_suffix"]),
        verso_suffix=raw.get("verso_suffix", fallback["verso_suffix"]),
        rotation_method=raw.get("rotation_method", fallback["rotation_method"]),
        perspective_correction=raw.get("perspective_correction", fallback["perspective_correction"]),
        preprocessing=raw.get("preprocessing", fallback["preprocessing"]),
        system_prompt=raw.get("system_prompt", fallback["system_prompt"]),
        prompt_instructions=raw.get("prompt_instructions", fallback["prompt_instructions"]),
    )
    result["strategy"] = result["strategy"] if result["strategy"] in {"separate_calls", "combined_vertical"} else fallback["strategy"]
    result["rotation_method"] = result["rotation_method"] if result["rotation_method"] in {"none", "pillow", "opencv"} else fallback["rotation_method"]
    result["preprocessing"] = [name for name in result["preprocessing"] if name in {"contrast", "denoise"}]
    result["recto_suffix"] = result["recto_suffix"] or fallback["recto_suffix"]
    result["verso_suffix"] = result["verso_suffix"] or fallback["verso_suffix"]
    result["system_prompt"] = result["system_prompt"] or fallback["system_prompt"]
    result["prompt_instructions"] = result["prompt_instructions"] or fallback["prompt_instructions"]
    return result


def _positive_integer(value: Any, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback
