"""Composants Gradio du prompt et du contrat de sortie CNI."""

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr


@dataclass(frozen=True)
class PromptSettings:
    output_format_mode: Any
    system_prompt: Any
    prompt_instructions: Any
    prompt_preview: Any
    refresh_prompt: Any


def build_prompt_settings(
    settings: dict[str, Any],
    prompt_preview_builder: Callable[[str, str | None, str | None], str],
) -> PromptSettings:
    """Construit les contrôles qui séparent le prompt du format fournisseur."""
    with gr.Tab("Prompt et sortie"):
        output_format_mode = gr.Radio(
            [
                ("Schéma JSON strict · recommandé", "schema"),
                ("Objet JSON libre", "json"),
                ("Prompt uniquement · Markdown possible", "prompt"),
            ],
            value=settings["output_format_mode"],
            label="Contrat de sortie du modèle",
            info=(
                "Le schéma est transmis à Ollama pour forcer les clés configurées, "
                "notamment avec LightOnOCR. La réponse brute reste conservée."
            ),
        )
        system_prompt = gr.Textbox(
            value=settings["system_prompt"], label="Prompt système", lines=5,
            info="Règle prioritaire commune aux deux faces.",
        )
        prompt_instructions = gr.Textbox(
            value=settings["prompt_instructions"],
            label="Consignes utilisateur complémentaires", lines=4,
            info="Ces consignes ne doivent pas modifier les clés du schéma.",
        )
        prompt_preview = gr.Code(
            value=prompt_preview_builder(
                settings["strategy"],
                settings["system_prompt"],
                settings["prompt_instructions"],
            ),
            label="Prompts réellement envoyés",
            lines=18,
            interactive=False,
        )
        refresh_prompt = gr.Button("Actualiser l’aperçu du prompt")
    return PromptSettings(
        output_format_mode,
        system_prompt,
        prompt_instructions,
        prompt_preview,
        refresh_prompt,
    )
