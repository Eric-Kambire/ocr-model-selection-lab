"""Contrat d'extraction CNI : champs, prompts, parsing JSON et fusion.

Ce module ne lit aucun fichier métier et n'appelle aucun modèle. Il garantit
que l'interface, le prompt et les artefacts utilisent le même contrat JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Ces tuples documentent le premier contrat supporté. La configuration active
# vient de ``config/cni_fields.json`` afin d'ajouter un champ sans dupliquer le
# texte du prompt dans plusieurs fichiers.
RECTO_FIELDS = (
    "cin", "nom", "prenom", "date_naissance", "ville_naissance", "date_validite",
)
VERSO_FIELDS = ("cin", "date_validite", "adresse")

DEFAULT_CNI_FIELD_CONFIG = {
    "recto": [
        {"key": "cin", "type": "text"}, {"key": "nom", "type": "text"},
        {"key": "prenom", "type": "text"}, {"key": "date_naissance", "type": "date"},
        {"key": "ville_naissance", "type": "text"}, {"key": "date_validite", "type": "date"},
    ],
    "verso": [
        {"key": "cin", "type": "text"}, {"key": "date_validite", "type": "date"},
        {"key": "adresse", "type": "text"},
    ],
}


def load_cni_field_config(config_path: Path | None = None) -> dict[str, list[dict[str, str]]]:
    """Charge et valide le contrat de champs CNI modifiable."""
    if config_path is None or not config_path.is_file():
        # Copie profonde : un appel peut adapter sa configuration en mémoire
        # sans modifier le défaut utilisé par les exécutions suivantes.
        return json.loads(json.dumps(DEFAULT_CNI_FIELD_CONFIG))
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid CNI field configuration: {config_path}: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(value.get(side), list) for side in ("recto", "verso")):
        raise ValueError("CNI field configuration must contain recto and verso arrays.")
    for side in ("recto", "verso"):
        for item in value[side]:
            if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                raise ValueError(f"Invalid field declaration for {side}.")
    return value


def _prompt_user_instructions(instructions: str | None) -> str:
    """Ajoute les consignes utilisateur sans autoriser un changement de schéma."""
    cleaned = (instructions or "").strip()
    if not cleaned:
        return ""
    return "\nAdditional user instructions (do not change the JSON schema):\n" + cleaned[:4000]


def build_cni_prompt(side: str, fields: dict[str, list[dict[str, str]]] | None = None, instructions: str | None = None) -> str:
    """Construit le message utilisateur JSON strict pour une seule face de CNI.

    Le message système est injecté séparément par l'adaptateur Ollama. Ici, on
    garde une demande courte, orientée extraction latine, pour limiter les
    digressions du modèle pendant la première phase du projet.
    """
    if side not in {"recto", "verso"}:
        raise ValueError("side must be 'recto' or 'verso'.")
    config = fields or load_cni_field_config()
    schema = {str(item["key"]): None for item in config[side]}
    side_focus = (
        "Read the identity number, surname, given name, birth date, birth city and validity date."
        if side == "recto"
        else "Read the identity number, validity date and full address. Do not use parent names or civil-status data."
    )
    return (
        f"Extract these fields from the {side.upper()} side of a Moroccan CNI (old or new layout).\n"
        "Read only values visibly printed in Latin characters. Do not translate, transliterate, infer, or add fields.\n"
        f"{side_focus}\n"
        "Use null when unreadable. Preserve spelling, punctuation and accents. Format a clearly readable date as YYYY-MM-DD.\n"
        "Return ONLY one valid JSON object: no Markdown, prose, comments or code fence.\n"
        "Required JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
        + _prompt_user_instructions(instructions)
    )


def build_combined_cni_prompt(fields: dict[str, list[dict[str, str]]] | None = None, instructions: str | None = None) -> str:
    """Construit le message utilisateur pour le composite recto-dessus-verso."""
    config = fields or load_cni_field_config()
    schema = {
        "recto": {str(item["key"]): None for item in config["recto"]},
        "verso": {str(item["key"]): None for item in config["verso"]},
    }
    return (
        "The image contains two sides of the same Moroccan national identity card, old or new layout: RECTO at the top and VERSO at the bottom.\n"
        "Read only values visibly printed in Latin characters. Do not translate, transliterate, infer, duplicate across sides or add fields.\n"
        "For ambiguous or unreadable values use null. Preserve spelling, punctuation and accents; format a clearly readable date as YYYY-MM-DD.\n"
        "Return ONLY one valid JSON object with recto and verso: no Markdown, prose, comments or code fence.\n"
        "Required schema:\n" + json.dumps(schema, ensure_ascii=False)
        + _prompt_user_instructions(instructions)
    )


def parse_cni_json_response(raw_text: str, side: str, fields: dict[str, list[dict[str, str]]] | None = None) -> tuple[dict[str, str | None], str | None]:
    """Parse une réponse modèle selon les seuls champs configurés de la face."""
    if side not in {"recto", "verso"}:
        raise ValueError("side must be 'recto' or 'verso'.")
    config = fields or load_cni_field_config()
    keys = [str(item["key"]) for item in config[side]]
    empty = {key: None for key in keys}
    # Les clés non prévues sont ignorées ; les clés prévues mais manquantes
    # deviennent ``None`` afin que tous les artefacts aient la même structure.
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        return empty, "model_response_not_json"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return empty, "model_response_invalid_json"
    if not isinstance(parsed, dict):
        return empty, "model_response_not_object"
    return {key: _string_or_none(parsed.get(key)) for key in keys}, None


def parse_combined_cni_json_response(raw_text: str, fields: dict[str, list[dict[str, str]]] | None = None) -> tuple[dict[str, str | None], dict[str, str | None], str | None]:
    """Parse une réponse combinée contenant les objets ``recto`` et ``verso``."""
    config = fields or load_cni_field_config()
    empty_recto = {str(item["key"]): None for item in config["recto"]}
    empty_verso = {str(item["key"]): None for item in config["verso"]}
    candidate = _extract_json_object(raw_text)
    # Un objet interne absent est traité comme une face vide : le benchmark
    # conserve ainsi les trois JSON de diagnostic au lieu de s'interrompre.
    if candidate is None:
        return empty_recto, empty_verso, "model_response_not_json"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return empty_recto, empty_verso, "model_response_invalid_json"
    if not isinstance(parsed, dict):
        return empty_recto, empty_verso, "model_response_not_object"
    recto = parsed.get("recto") if isinstance(parsed.get("recto"), dict) else {}
    verso = parsed.get("verso") if isinstance(parsed.get("verso"), dict) else {}
    return (
        {key: _string_or_none(recto.get(key)) for key in empty_recto},
        {key: _string_or_none(verso.get(key)) for key in empty_verso},
        None,
    )


def build_cni_global_json(client: dict[str, Any], recto: dict[str, str | None], verso: dict[str, str | None]) -> dict[str, Any]:
    """Fusionne les deux faces sans encore comparer au label externe."""
    # Conserver les deux lectures est essentiel : une CIN incohérente doit être
    # visible dans le résultat, jamais arbitrairement remplacée par une face.
    cin_value, cin_coherent = _merge_cross_side_value(recto.get("cin"), verso.get("cin"))
    valid_value, valid_coherent = _merge_cross_side_value(recto.get("date_validite"), verso.get("date_validite"))
    return {
        "folder_client_id": client["folder_client_id"], "recto_document_id": client.get("recto_document_id"),
        "verso_document_id": client.get("verso_document_id"), "label_status": client.get("label_status"),
        "recto": recto, "verso": verso, "cin_recto": recto.get("cin"), "cin_verso": verso.get("cin"),
        "cin_fusionne": cin_value, "cin_coherent": cin_coherent, "nom": recto.get("nom"),
        "prenom": recto.get("prenom"), "date_naissance": recto.get("date_naissance"),
        "ville_naissance": recto.get("ville_naissance"), "date_validite_recto": recto.get("date_validite"),
        "date_validite_verso": verso.get("date_validite"), "date_validite_fusionnee": valid_value,
        "date_validite_coherente": valid_coherent, "adresse": verso.get("adresse"),
    }


def _extract_json_object(text: str) -> str | None:
    """Accepte une clôture Markdown, sans tenter de réparer un JSON invalide."""
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:]).strip()
    first, last = value.find("{"), value.rfind("}")
    return value[first:last + 1] if first >= 0 and last > first else None


def _string_or_none(value: Any) -> str | None:
    """Conserve les scalaires, normalise les espaces et rejette les objets JSON."""
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        value = str(value).strip()
        return value or None
    return None


def _merge_cross_side_value(recto_value: str | None, verso_value: str | None) -> tuple[str | None, bool | None]:
    """Retourne une valeur fusionnée sûre et le drapeau de cohérence."""
    if recto_value and verso_value:
        # La comparaison tolère espaces et ponctuation, sans modifier la valeur
        # originale affichée à l'utilisateur dans le JSON final.
        if re.sub(r"[^A-Z0-9]", "", recto_value.upper()) == re.sub(r"[^A-Z0-9]", "", verso_value.upper()):
            return recto_value, True
        return None, False
    return recto_value or verso_value, None
