"""Inspection des capacités déclarées par un modèle Ollama.

Un modèle textuel accepte parfois une requête contenant ``images`` sans
signaler d'erreur : il ignore alors l'image et répond uniquement au prompt.
Pour un benchmark OCR, cette réponse est un faux positif. Ce module centralise
donc la vérification *avant* l'inférence, sans dépendre de Gradio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaVisionCapability:
    """Résultat normalisé de l'inspection ``ollama.show``."""

    supported: bool
    capabilities: tuple[str, ...]
    reason: str


def inspect_ollama_vision_capability(
    client: Any,
    model_name: str,
) -> OllamaVisionCapability:
    """Vérifie qu'Ollama déclare explicitement la capacité ``vision``.

    La réponse du SDK peut être un dictionnaire ou un objet Pydantic selon sa
    version. Une ancienne version d'Ollama peut ne pas exposer
    ``capabilities`` ; dans ce cas, quelques métadonnées multimodales connues
    servent de repli. Sans preuve, la vérification échoue de manière fermée :
    une réponse textuelle ne doit jamais compter comme analyse d'image.
    """

    try:
        response = client.show(model=model_name)
    except Exception as exc:
        return OllamaVisionCapability(
            supported=False,
            capabilities=(),
            reason=(
                "Impossible de vérifier les capacités du modèle avec "
                f"ollama.show : {type(exc).__name__}: {exc}"
            ),
        )

    raw_capabilities = _read_value(response, "capabilities") or []
    capabilities = tuple(
        sorted(
            {
                str(value).strip().lower()
                for value in raw_capabilities
                if str(value).strip()
            }
        )
    )
    if "vision" in capabilities:
        return OllamaVisionCapability(
            supported=True,
            capabilities=capabilities,
            reason="Ollama déclare explicitement la capacité vision.",
        )
    if capabilities:
        return OllamaVisionCapability(
            supported=False,
            capabilities=capabilities,
            reason=(
                "Ollama ne déclare pas la capacité vision "
                f"(capacités : {', '.join(capabilities)})."
            ),
        )

    # Compatibilité défensive avec d'anciennes réponses ``show`` : les modèles
    # multimodaux exposent généralement des clés ``vision`` ou ``mm``.
    model_info = _read_value(response, "model_info") or {}
    keys = (
        model_info.keys()
        if isinstance(model_info, dict)
        else getattr(model_info, "model_fields_set", ())
    )
    normalized_keys = {str(key).lower() for key in keys}
    if any(".vision." in key or ".mm." in key for key in normalized_keys):
        return OllamaVisionCapability(
            supported=True,
            capabilities=("vision",),
            reason="Capacité vision détectée dans les métadonnées multimodales.",
        )

    return OllamaVisionCapability(
        supported=False,
        capabilities=(),
        reason=(
            "La réponse ollama.show ne contient aucune preuve de capacité "
            "vision. Mettez Ollama à jour ou choisissez un VLM."
        ),
    )


def _read_value(value: Any, key: str) -> Any:
    """Lit uniformément une clé depuis un dictionnaire ou un objet SDK."""

    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
