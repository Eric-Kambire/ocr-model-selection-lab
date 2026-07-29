"""Composants Gradio du prompt et du contrat de sortie CNI."""

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr


@dataclass(frozen=True)
class PromptSettings:
    system_prompt: Any
    prompt_instructions: Any
    prompt_preview: Any
    refresh_prompt: Any


def build_prompt_settings(
    settings: dict[str, Any],
    prompt_preview_builder: Callable[[str, str | None, str | None], str],
) -> PromptSettings:
    """Construit l'éditeur de prompt.

    Le contrat de sortie est volontairement placé à côté de la sélection des
    modèles dans la vue « Préparer ». L'opérateur choisit ainsi le comportement
    JSON en même temps que les modèles auxquels il sera appliqué.
    """
    with gr.Tab("Prompt et sortie"):
        gr.Markdown(
            "Le **mode de sortie JSON** se choisit dans `1. Préparer`, juste "
            "sous les modèles. Cette page ne contient que les consignes métier "
            "et l'aperçu exact des prompts."
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
            lines=14,
            interactive=False,
        )
        refresh_prompt = gr.Button("Recalculer l’aperçu", size="sm")
    return PromptSettings(
        system_prompt,
        prompt_instructions,
        prompt_preview,
        refresh_prompt,
    )
