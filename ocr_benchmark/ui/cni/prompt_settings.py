"""Composants Gradio du prompt et du contrat de sortie CNI."""

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr


@dataclass(frozen=True)
class PromptSettings:
    system_prompt: Any
    prompt_instructions: Any
    prompt_preview_side: Any
    prompt_context_budget: Any
    prompt_token_indicator: Any
    prompt_preview: Any


def build_prompt_settings(
    settings: dict[str, Any],
    prompt_preview_builder: Callable[[str, str, str | None, str | None], str],
    token_indicator_builder: Callable[
        [str, str, str | None, str | None, int | float | None], str
    ],
) -> PromptSettings:
    """Construit l'éditeur de prompt.

    Le contrat de sortie est volontairement placé à côté de la sélection des
    modèles dans la vue « Préparer ». L'opérateur choisit ainsi le comportement
    JSON en même temps que les modèles auxquels il sera appliqué.
    """
    with gr.Tab("Prompt et sortie"):
        gr.Markdown(
            "Construisez les consignes en deux niveaux, puis contrôlez le message "
            "exact envoyé pour chaque face. Le format JSON du fournisseur reste "
            "accessible avec ⚙ à côté des modèles dans `1. Préparer`."
        )
        with gr.Tabs(elem_id="cni-prompt-tabs"):
            with gr.Tab("1. Prompt système"):
                gr.Markdown(
                    "Instruction prioritaire commune au recto, au verso et au mode combiné. "
                    "Elle fixe le rôle du modèle et les règles qu’il ne doit jamais contourner."
                )
                system_prompt = gr.Textbox(
                    value=settings["system_prompt"],
                    label="Prompt système",
                    lines=7,
                    info="Une modification s’applique à tous les modèles du prochain run.",
                )
            with gr.Tab("2. Consignes utilisateur"):
                gr.Markdown(
                    "Ajoutez ici les règles métier propres aux CNI. Le code complète ensuite "
                    "automatiquement ces consignes avec les champs attendus de la face."
                )
                prompt_instructions = gr.Textbox(
                    value=settings["prompt_instructions"],
                    label="Consignes utilisateur complémentaires",
                    lines=7,
                    info="N’ajoutez pas de nouvelles clés JSON ici ; modifiez la configuration des champs.",
                )
            with gr.Tab("3. Prompt final envoyé"):
                gr.Markdown(
                    "Cet aperçu est recalculé automatiquement. Le sélecteur sert uniquement "
                    "à inspecter le message : il ne change ni la stratégie ni la face envoyée. "
                    "En mode séparé, le scan associe les fichiers aux rôles avec les suffixes "
                    "configurés, puis le runner impose **Recto → prompt recto** et "
                    "**Verso → prompt verso**. Le prompt système reste commun aux deux appels."
                )
                initial_side = (
                    "combined"
                    if settings["strategy"] == "combined_vertical"
                    else "recto"
                )
                with gr.Row():
                    prompt_preview_side = gr.Radio(
                        [
                            ("Recto", "recto"),
                            ("Verso", "verso"),
                            ("Image combinée", "combined"),
                        ],
                        value=initial_side,
                        label="Message à inspecter",
                    )
                    prompt_context_budget = gr.Number(
                        value=settings.get("prompt_context_budget", 8192),
                        minimum=256,
                        precision=0,
                        label="Budget de contexte surveillé",
                        info=(
                            "Seuil d’alerte en tokens. Il ne modifie pas le paramètre "
                            "num_ctx d’Ollama."
                        ),
                    )
                prompt_token_indicator = gr.HTML(
                    token_indicator_builder(
                        settings["strategy"],
                        initial_side,
                        settings["system_prompt"],
                        settings["prompt_instructions"],
                        settings.get("prompt_context_budget", 8192),
                    )
                )
                prompt_preview = gr.Code(
                    value=prompt_preview_builder(
                        settings["strategy"],
                        initial_side,
                        settings["system_prompt"],
                        settings["prompt_instructions"],
                    ),
                    label="Prompt réellement envoyé pour cette entrée",
                    lines=18,
                    interactive=False,
                )
    return PromptSettings(
        system_prompt,
        prompt_instructions,
        prompt_preview_side,
        prompt_context_budget,
        prompt_token_indicator,
        prompt_preview,
    )
