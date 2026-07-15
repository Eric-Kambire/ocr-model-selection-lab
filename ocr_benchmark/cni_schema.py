"""CNI extraction contract: editable fields, prompts, parsing and merging.

This module contains no file-system or image work.  It defines the stable JSON
contract shared by the interface, the model prompt and the persisted results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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
    """Load the editable CNI field contract.

    Args:
        config_path: JSON configuration path.  Missing paths use the default.

    Returns:
        A mapping with validated ``recto`` and ``verso`` field declarations.

    Raises:
        ValueError: If an existing configuration is malformed.
    """
    if config_path is None or not config_path.is_file():
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


def build_cni_prompt(side: str, fields: dict[str, list[dict[str, str]]] | None = None) -> str:
    """Build a strict JSON-only prompt for one CNI side.

    Values are read from Latin/French text. Arabic can identify a label but is
    never translated, transliterated or guessed.
    """
    if side not in {"recto", "verso"}:
        raise ValueError("side must be 'recto' or 'verso'.")
    config = fields or load_cni_field_config()
    schema = {str(item["key"]): None for item in config[side]}
    return (
        f"You extract structured data from the {side.upper()} side of a Moroccan national identity card (CNI).\n"
        "Read only the Latin/French value printed on the card. Arabic text may help identify a label, "
        "but do not translate, transliterate, guess, or add fields.\n"
        "Return ONLY one valid JSON object. Do not use Markdown, code fences, comments, or prose.\n"
        "For an unreadable or absent value, use null. Preserve spelling. Format a clearly readable date as YYYY-MM-DD.\n"
        "Required JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
    )


def build_combined_cni_prompt(fields: dict[str, list[dict[str, str]]] | None = None) -> str:
    """Build the strict JSON-only prompt for recto-above-verso composite input."""
    config = fields or load_cni_field_config()
    schema = {
        "recto": {str(item["key"]): None for item in config["recto"]},
        "verso": {str(item["key"]): None for item in config["verso"]},
    }
    return (
        "The image contains two sides of the same Moroccan national identity card: RECTO at the top and VERSO at the bottom.\n"
        "Extract only the Latin/French values visible on each side. Do not translate Arabic text, infer missing values, "
        "or add fields. Return ONLY one valid JSON object, with null for unreadable values.\n"
        "Format a clearly readable date as YYYY-MM-DD. Required schema:\n" + json.dumps(schema, ensure_ascii=False)
    )


def parse_cni_json_response(raw_text: str, side: str, fields: dict[str, list[dict[str, str]]] | None = None) -> tuple[dict[str, str | None], str | None]:
    """Parse one model response into exactly the configured side fields."""
    if side not in {"recto", "verso"}:
        raise ValueError("side must be 'recto' or 'verso'.")
    config = fields or load_cni_field_config()
    keys = [str(item["key"]) for item in config[side]]
    empty = {key: None for key in keys}
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
    """Parse a combined response with its ``recto`` and ``verso`` objects."""
    config = fields or load_cni_field_config()
    empty_recto = {str(item["key"]): None for item in config["recto"]}
    empty_verso = {str(item["key"]): None for item in config["verso"]}
    candidate = _extract_json_object(raw_text)
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
    """Merge side results without making a label-comparison decision."""
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
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:]).strip()
    first, last = value.find("{"), value.rfind("}")
    return value[first:last + 1] if first >= 0 and last > first else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        value = str(value).strip()
        return value or None
    return None


def _merge_cross_side_value(recto_value: str | None, verso_value: str | None) -> tuple[str | None, bool | None]:
    if recto_value and verso_value:
        if re.sub(r"[^A-Z0-9]", "", recto_value.upper()) == re.sub(r"[^A-Z0-9]", "", verso_value.upper()):
            return recto_value, True
        return None, False
    return recto_value or verso_value, None
