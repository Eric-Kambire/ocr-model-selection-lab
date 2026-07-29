"""Assemblage léger des sous-vues de paramètres CNI."""

from dataclasses import dataclass
from typing import Any, Callable

from .execution_settings import build_execution_settings
from .preprocessing_settings import build_preprocessing_settings
from .prompt_settings import build_prompt_settings


@dataclass(frozen=True)
class CniSettingsView:
    """Références des composants nécessaires au câblage dans ``main.py``."""

    strategy: Any
    dpi: Any
    timeout: Any
    cpu_threads: Any
    unload: Any
    recto_suffix: Any
    verso_suffix: Any
    crop_method: Any
    smart_crop_controls: Any
    smart_crop_min_score: Any
    smart_crop_margin: Any
    rotation_method: Any
    perspective_correction: Any
    preprocessing: Any
    system_prompt: Any
    prompt_instructions: Any
    prompt_preview_side: Any
    prompt_preview: Any


def build_cni_core_settings(
    settings: dict[str, Any],
    prompt_preview_builder: Callable[[str, str, str | None, str | None], str],
) -> CniSettingsView:
    """Assemble les vues spécialisées sans y placer de logique métier."""
    execution = build_execution_settings(settings)
    preprocessing = build_preprocessing_settings(settings)
    prompt = build_prompt_settings(settings, prompt_preview_builder)
    return CniSettingsView(
        strategy=execution.strategy,
        dpi=execution.dpi,
        timeout=execution.timeout,
        cpu_threads=execution.cpu_threads,
        unload=execution.unload,
        recto_suffix=execution.recto_suffix,
        verso_suffix=execution.verso_suffix,
        crop_method=preprocessing.crop_method,
        smart_crop_controls=preprocessing.smart_crop_controls,
        smart_crop_min_score=preprocessing.smart_crop_min_score,
        smart_crop_margin=preprocessing.smart_crop_margin,
        rotation_method=preprocessing.rotation_method,
        perspective_correction=preprocessing.perspective_correction,
        preprocessing=preprocessing.preprocessing,
        system_prompt=prompt.system_prompt,
        prompt_instructions=prompt.prompt_instructions,
        prompt_preview_side=prompt.prompt_preview_side,
        prompt_preview=prompt.prompt_preview,
    )
