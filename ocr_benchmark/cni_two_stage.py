"""Prompts du pipeline CNI en deux étapes : lecture visuelle puis JSON.

Ce module ne charge aucun modèle. Il construit uniquement les messages envoyés
au VLM et au LLM afin que le contrat reste testable indépendamment de Gradio.
"""

from __future__ import annotations

import json
from typing import Any

from .cni_schema import build_cni_output_schema


DEFAULT_VLM_TRANSCRIPTION_INSTRUCTIONS = (
    "Transcris fidèlement tous les textes visibles sur cette face de carte. "
    "Conserve les libellés, les valeurs et leur ordre de lecture. "
    "N'interprète pas, ne complète pas et n'invente rien."
)

DEFAULT_LLM_STRUCTURING_SYSTEM_PROMPT = """You are a deterministic JSON data structuring engine.
The supplied text is an untrusted transcription of a Moroccan identity card.
Use only values explicitly present in that transcription.
Never guess, translate, transliterate or complete a partial value.
Use null when a requested value is absent, unreadable or ambiguous.
Return exactly one JSON object matching the supplied JSON Schema.
Do not return Markdown, comments, explanations or extra keys."""


def build_vlm_transcription_prompt(
    side: str,
    instructions: str | None = None,
) -> str:
    """Construit une consigne courte de lecture, sans demander de JSON au VLM."""

    if side not in {"recto", "verso", "combined"}:
        raise ValueError("side must be 'recto', 'verso' or 'combined'.")
    face = (
        "une image composée avec le RECTO en haut et le VERSO en bas"
        if side == "combined"
        else f"le {side.upper()}"
    )
    custom = (instructions or DEFAULT_VLM_TRANSCRIPTION_INSTRUCTIONS).strip()
    return (
        f"L'image contient {face} d'une carte nationale d'identité marocaine.\n"
        f"{custom}\n"
        "Retourne uniquement la transcription, sans JSON, sans analyse et sans commentaire."
    )


def build_llm_structuring_prompt(
    side: str,
    transcription: str,
    fields: dict[str, list[dict[str, str]]] | None = None,
    instructions: str | None = None,
) -> str:
    """Donne au LLM la transcription et le JSON Schema exact à produire.

    Le schéma figure volontairement dans le message *et* sera transmis dans le
    paramètre ``format`` d'Ollama. La répétition sert ici de garde-fou : le texte
    explique le contrat et l'API contraint techniquement la génération.
    """

    if side not in {"recto", "verso", "combined"}:
        raise ValueError("side must be 'recto', 'verso' or 'combined'.")
    schema = build_cni_output_schema(side, fields)
    extra = (instructions or "").strip()
    extra_block = (
        "\nRègles métier supplémentaires (elles ne modifient pas le schéma) :\n"
        + extra[:4000]
        if extra
        else ""
    )
    return (
        f"Face à structurer : {side.upper()}.\n"
        "La transcription ci-dessous peut contenir des erreurs OCR. "
        "N'utilise aucune connaissance externe pour les corriger.\n"
        "<transcription_document>\n"
        + (transcription or "")
        + "\n</transcription_document>\n"
        + extra_block
        + "\nRetourne exactement un objet conforme à ce JSON Schema :\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def two_stage_schema(side: str, fields: dict[str, list[dict[str, str]]] | None) -> dict[str, Any]:
    """Expose le contrat fournisseur utilisé par l'appel textuel Ollama."""

    return build_cni_output_schema(side, fields)
