"""Composants Gradio relatifs à l'exécution du benchmark CNI."""

import os
from dataclasses import dataclass
from typing import Any

import gradio as gr


@dataclass(frozen=True)
class ExecutionSettings:
    strategy: Any
    dpi: Any
    timeout: Any
    cpu_threads: Any
    unload: Any
    ollama_ignore_environment_proxy: Any
    recto_suffix: Any
    verso_suffix: Any


def build_execution_settings(settings: dict[str, Any]) -> ExecutionSettings:
    """Construit l'onglet qui contrôle la stratégie et les ressources."""
    cpu_maximum = max(1, os.cpu_count() or 1)
    with gr.Tab("Exécution"):
        with gr.Row():
            strategy = gr.Radio(
                [
                    ("Deux appels : recto puis verso — recommandé", "separate_calls"),
                    ("Une image : recto en haut, verso en bas", "combined_vertical"),
                ],
                value=settings["strategy"],
                label="Stratégie d'envoi au modèle",
            )
            dpi = gr.Slider(
                150, 450, value=settings["dpi"], step=25,
                label="Résolution PDF (DPI)",
            )
            timeout = gr.Number(
                value=settings["timeout_seconds"], minimum=1, maximum=7200,
                precision=0, label="Temps maximum par appel (s)",
            )
        with gr.Row():
            cpu_threads = gr.Number(
                value=min(settings["cpu_threads"], cpu_maximum),
                minimum=1, maximum=cpu_maximum, precision=0,
                label="Threads CPU Ollama",
            )
            unload = gr.Checkbox(
                value=settings["unload_after_task"],
                label="Décharger le modèle après son lot de documents",
                info="Le modèle reste chargé entre recto et verso, puis est libéré avant le suivant.",
            )
            ollama_ignore_environment_proxy = gr.Checkbox(
                value=settings["ollama_ignore_environment_proxy"],
                label="Ignorer le proxy système pour Ollama",
                info=(
                    "Active trust_env=False. Recommandé pour un Ollama local "
                    "sur 127.0.0.1/localhost afin d'éviter un timeout de proxy. "
                    "Décochez seulement si votre serveur Ollama distant exige le proxy."
                ),
            )
        with gr.Row():
            recto_suffix = gr.Textbox(
                value=settings["recto_suffix"], label="Suffixe recto",
                info="Ex. _CIN_Recto. L'extension est ignorée.",
            )
            verso_suffix = gr.Textbox(
                value=settings["verso_suffix"], label="Suffixe verso",
                info="Ex. _CIN_Verso. L'extension est ignorée.",
            )
    return ExecutionSettings(
        strategy,
        dpi,
        timeout,
        cpu_threads,
        unload,
        ollama_ignore_environment_proxy,
        recto_suffix,
        verso_suffix,
    )
