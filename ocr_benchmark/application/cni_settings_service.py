"""Persistance locale et automatique des réglages d'exécution CNI.

Ce module ne stocke ni document ni résultat : seulement les choix opérateur
utiles pour reprendre le même protocole après un rechargement de Gradio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..cni_crop_service import (
    DEFAULT_SMART_CROP_MARGIN,
    DEFAULT_SMART_CROP_MIN_SCORE,
    SMART_CROP_V4,
    SUPPORTED_CROP_METHODS,
)


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
        "crop_method": SMART_CROP_V4,
        "smart_crop_min_score": DEFAULT_SMART_CROP_MIN_SCORE,
        "smart_crop_margin": DEFAULT_SMART_CROP_MARGIN,
        "rotation_method": "none",
        "perspective_correction": False,
        "preprocessing": [],
        "output_format_mode": "schema",
        "model_output_modes": {},
        "system_prompt": system_prompt,
        "prompt_instructions": prompt_instructions,
        # Ce seuil sert uniquement à surveiller la taille du prompt dans l'UI.
        # Il ne modifie pas le contexte réellement alloué par Ollama.
        "prompt_context_budget": 8192,
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
    crop_method: Any,
    smart_crop_min_score: Any,
    smart_crop_margin: Any,
    rotation_method: Any,
    perspective_correction: Any,
    preprocessing: Any,
    output_format_mode: Any,
    model_output_modes: Any,
    system_prompt: Any,
    prompt_instructions: Any,
    prompt_context_budget: Any,
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
        "crop_method": str(crop_method or ""),
        "smart_crop_min_score": _bounded_float(
            smart_crop_min_score, DEFAULT_SMART_CROP_MIN_SCORE, 0.0, 1.0
        ),
        "smart_crop_margin": _bounded_float(
            smart_crop_margin, DEFAULT_SMART_CROP_MARGIN, 0.0, 0.08
        ),
        "rotation_method": str(rotation_method or ""),
        "perspective_correction": bool(perspective_correction),
        "preprocessing": [str(name) for name in (preprocessing or []) if str(name).strip()],
        "output_format_mode": str(output_format_mode or ""),
        "model_output_modes": _model_output_modes(model_output_modes),
        "system_prompt": str(system_prompt or "").strip(),
        "prompt_instructions": str(prompt_instructions or "").strip(),
        "prompt_context_budget": _positive_integer(prompt_context_budget, 8192),
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
        crop_method=raw.get("crop_method", fallback["crop_method"]),
        smart_crop_min_score=raw.get("smart_crop_min_score", fallback["smart_crop_min_score"]),
        smart_crop_margin=raw.get("smart_crop_margin", fallback["smart_crop_margin"]),
        rotation_method=raw.get("rotation_method", fallback["rotation_method"]),
        perspective_correction=raw.get("perspective_correction", fallback["perspective_correction"]),
        preprocessing=raw.get("preprocessing", fallback["preprocessing"]),
        output_format_mode=raw.get("output_format_mode", fallback["output_format_mode"]),
        model_output_modes=raw.get("model_output_modes", fallback["model_output_modes"]),
        system_prompt=raw.get("system_prompt", fallback["system_prompt"]),
        prompt_instructions=raw.get("prompt_instructions", fallback["prompt_instructions"]),
        prompt_context_budget=raw.get(
            "prompt_context_budget", fallback["prompt_context_budget"]
        ),
    )
    result["strategy"] = result["strategy"] if result["strategy"] in {"separate_calls", "combined_vertical"} else fallback["strategy"]
    result["crop_method"] = (
        result["crop_method"]
        if result["crop_method"] in SUPPORTED_CROP_METHODS
        else fallback["crop_method"]
    )
    result["rotation_method"] = result["rotation_method"] if result["rotation_method"] in {"none", "pillow", "opencv"} else fallback["rotation_method"]
    result["preprocessing"] = [name for name in result["preprocessing"] if name in {"contrast", "denoise"}]
    result["output_format_mode"] = (
        result["output_format_mode"]
        if result["output_format_mode"] in {"prompt", "json", "schema"}
        else fallback["output_format_mode"]
    )
    result["recto_suffix"] = result["recto_suffix"] or fallback["recto_suffix"]
    result["verso_suffix"] = result["verso_suffix"] or fallback["verso_suffix"]
    result["system_prompt"] = result["system_prompt"] or fallback["system_prompt"]
    result["prompt_instructions"] = result["prompt_instructions"] or fallback["prompt_instructions"]
    return result


def _model_output_modes(value: Any) -> dict[str, str]:
    """Conserve uniquement les exceptions explicites et reconnues."""
    if not isinstance(value, Mapping):
        return {}
    valid_modes = {"prompt", "json", "schema"}
    return {
        str(model).strip(): str(mode)
        for model, mode in value.items()
        if str(model).strip() and str(mode) in valid_modes
    }


def _positive_integer(value: Any, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback


def _bounded_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    """Convertit une valeur UI en nombre fini borné, sinon conserve le défaut."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not minimum <= number <= maximum:
        return fallback
    return number
