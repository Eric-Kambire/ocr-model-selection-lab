"""Composants Gradio du prompt et du contrat de sortie CNI."""

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr


def prompt_section_token_badge(text: str | None) -> str:
    """Retourne un compteur compact pour une section de texte du prompt.

    L'estimation utilise la taille UTF-8 car les modèles sélectionnés peuvent
    employer des tokenizers différents. La mesure exacte reste disponible
    après l'appel Ollama dans ``prompt_eval_count``.
    """
    estimated = max(0, (len((text or "").encode("utf-8")) + 3) // 4)
    return (
        "<div style='display:flex;justify-content:flex-end;margin:2px 0 6px'>"
        "<span style='display:inline-flex;align-items:center;gap:5px;"
        "padding:4px 9px;border:1px solid var(--border-color-primary);"
        "border-radius:999px;background:var(--block-background-fill);"
        "color:var(--body-text-color-subdued);font-size:12px'>"
        f"<strong>≈ {estimated:,}</strong> tokens texte"
        "</span></div>"
    )


@dataclass(frozen=True)
class PromptSettings:
    system_prompt: Any
    system_token_indicator: Any
    prompt_instructions: Any
    instructions_token_indicator: Any
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
            "Définissez le rôle, ajoutez les règles métier, puis vérifiez le message "
            "assemblé pour chaque face. Le réglage du format JSON reste accessible "
            "avec ⚙ à côté des modèles dans `1. Préparer`."
        )
        with gr.Tabs(elem_id="cni-prompt-tabs"):
            with gr.Tab("① Système"):
                gr.Markdown(
                    "Instruction prioritaire commune au recto, au verso et au mode combiné. "
                    "Elle fixe le rôle du modèle et les règles qu’il ne doit jamais contourner."
                )
                system_token_indicator = gr.HTML(
                    prompt_section_token_badge(settings["system_prompt"])
                )
                system_prompt = gr.Textbox(
                    value=settings["system_prompt"],
                    label="Instruction prioritaire",
                    lines=7,
                    info="Une modification s’applique à tous les modèles du prochain run.",
                )
            with gr.Tab("② Utilisateur"):
                gr.Markdown(
                    "Ajoutez ici les règles métier propres aux CNI. Le code complète ensuite "
                    "automatiquement ces consignes avec les champs attendus de la face."
                )
                instructions_token_indicator = gr.HTML(
                    prompt_section_token_badge(settings["prompt_instructions"])
                )
                prompt_instructions = gr.Textbox(
                    value=settings["prompt_instructions"],
                    label="Règles métier supplémentaires",
                    lines=7,
                    info="N’ajoutez pas de nouvelles clés JSON ici ; modifiez la configuration des champs.",
                )
            with gr.Tab("③ Aperçu final"):
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
                    label="Message complet inspecté",
                    lines=18,
                    interactive=False,
                )
    return PromptSettings(
        system_prompt,
        system_token_indicator,
        prompt_instructions,
        instructions_token_indicator,
        prompt_preview_side,
        prompt_context_budget,
        prompt_token_indicator,
        prompt_preview,
    )
