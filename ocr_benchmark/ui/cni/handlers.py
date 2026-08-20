"""Handlers Gradio légers de l'espace CNI.

Les cas d'usage lourds restent dans ``ocr_benchmark.application``. Ces
fonctions traduisent seulement les valeurs des composants vers ces services.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import gradio as gr

from ...application.cni_settings_service import (
    cni_settings_from_ui,
    save_cni_settings,
)


LOGGER = logging.getLogger(__name__)


def create_persist_settings_handler(
    config_path: Path,
    defaults: Mapping[str, Any],
) -> Callable[..., None]:
    """Crée le handler d'auto-sauvegarde avec ses dépendances explicites."""

    def persist(
        models, pipeline_mode, llm_model, strategy, dpi, timeout, threads, unload,
        ollama_ignore_environment_proxy, continue_without_label,
        recto_suffix, verso_suffix, crop_method, smart_crop_min_score,
        smart_crop_margin, rotation_method, perspective_correction, preprocessing,
        output_format_mode, model_output_modes, system_prompt, prompt_instructions,
        prompt_scope_mode, prompt_delivery_mode, ollama_thinking_mode,
        prompt_context_budget, vlm_transcription_instructions,
        llm_system_prompt,
    ) -> None:
        value = cni_settings_from_ui(
            models=models,
            pipeline_mode=pipeline_mode,
            llm_model=llm_model,
            strategy=strategy,
            dpi=dpi,
            timeout_seconds=timeout,
            cpu_threads=threads,
            unload_after_task=unload,
            ollama_ignore_environment_proxy=ollama_ignore_environment_proxy,
            continue_without_label=continue_without_label,
            recto_suffix=recto_suffix,
            verso_suffix=verso_suffix,
            crop_method=crop_method,
            smart_crop_min_score=smart_crop_min_score,
            smart_crop_margin=smart_crop_margin,
            rotation_method=rotation_method,
            perspective_correction=perspective_correction,
            preprocessing=preprocessing,
            output_format_mode=output_format_mode,
            model_output_modes=model_output_modes,
            system_prompt=system_prompt,
            prompt_instructions=prompt_instructions,
            prompt_scope_mode=prompt_scope_mode,
            prompt_delivery_mode=prompt_delivery_mode,
            ollama_thinking_mode=ollama_thinking_mode,
            prompt_context_budget=prompt_context_budget,
            vlm_transcription_instructions=vlm_transcription_instructions,
            llm_system_prompt=llm_system_prompt,
        )
        try:
            save_cni_settings(config_path, value, defaults=defaults)
            LOGGER.info(
                "CNI settings saved | pipeline=%s | strategy=%s | output_format=%s | "
                "ollama_trust_environment=%s | ollama_thinking=%s | "
                "model_overrides=%d",
                value["pipeline_mode"],
                value["strategy"],
                value["output_format_mode"],
                not value["ollama_ignore_environment_proxy"],
                value["ollama_thinking_mode"],
                len(value["model_output_modes"]),
            )
        except OSError:
            LOGGER.exception("CNI settings auto-save failed")

    return persist


def request_cancel(
    client_records: list[dict[str, Any]] | None,
    alert_renderer: Callable[[str, str], str],
) -> tuple[Any, Any, str, str]:
    """Annule un run tout en conservant la sélection déjà préparée."""
    ready = sum(
        record.get("status") == "ready" for record in (client_records or [])
    )
    message = (
        f"Annulation demandée. La sélection est conservée ({ready} paire(s) prête(s)). "
        "Cliquez sur Relancer pour reprendre sans préparer les documents à nouveau."
    )
    return (
        gr.update(visible=True, value="Relancer"),
        gr.update(visible=False),
        alert_renderer("warning", message),
        "Annulation demandée ; sélection conservée.",
    )
