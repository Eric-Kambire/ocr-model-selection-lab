"""Comparaison déterministe entre une extraction CNI et son label de référence.

Ce module est volontairement indépendant de Gradio et d'Ollama : le même
calcul de score peut servir à l'interface, aux exports et à un futur worker.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any, Mapping


CNI_COMPARISON_FIELDS = (
    "cin",
    "prenom",
    "nom",
    "date_naissance",
    "ville_naissance",
    "date_validite",
    "adresse",
)


def compare_cni_extraction(
    label: Mapping[str, Any] | None,
    extraction: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare les champs canoniques et retourne les détails et le score.

    Une valeur de référence manquante est exclue du dénominateur. Une valeur
    attendue mais non extraite est comptée comme erreur. Si l'extraction globale
    est indisponible, aucun score n'est produit : un échec technique ne doit pas
    être déguisé en 0 % de qualité OCR.
    """
    label_is_available = isinstance(label, Mapping) and "status" not in label
    extraction_is_available = isinstance(extraction, Mapping) and "status" not in extraction
    confidence = _label_confidence(label) if label_is_available else {}
    rows: list[dict[str, Any]] = []

    for field in CNI_COMPARISON_FIELDS:
        expected = _label_value(label, field) if label_is_available else None
        actual = _extracted_value(extraction, field) if extraction_is_available else None
        if not label_is_available or _is_empty(expected):
            state = "reference_missing"
        elif not extraction_is_available:
            state = "extraction_unavailable"
        elif _is_empty(actual):
            state = "extracted_missing"
        elif _normalise(field, expected) == _normalise(field, actual):
            state = "correct"
        else:
            state = "different"
        rows.append(
            {
                "field": field,
                "expected": expected,
                "actual": actual,
                "reference_confidence": confidence.get(field),
                "state": state,
            }
        )

    if not label_is_available:
        score, score_status = None, "reference_unavailable"
    elif not extraction_is_available:
        score, score_status = None, "extraction_unavailable"
    else:
        comparable = [row for row in rows if row["state"] != "reference_missing"]
        if not comparable:
            score, score_status = None, "no_comparable_reference_field"
        else:
            score = sum(row["state"] == "correct" for row in comparable) / len(comparable)
            score_status = "scored"

    return {
        "rows": rows,
        "accuracy": score,
        "score_status": score_status,
        "correct_fields": sum(row["state"] == "correct" for row in rows),
        "comparable_fields": sum(row["state"] != "reference_missing" for row in rows),
    }


def field_state_map(comparison: Mapping[str, Any] | None) -> dict[str, str]:
    """Expose un état par clé pour les filtres et les tableaux compacts."""
    rows = comparison.get("rows", []) if isinstance(comparison, Mapping) else []
    return {
        str(row.get("field")): str(row.get("state"))
        for row in rows
        if isinstance(row, Mapping) and row.get("field")
    }


def _label_value(label: Mapping[str, Any] | None, field: str) -> Any:
    if not isinstance(label, Mapping):
        return None
    if field in label:
        return label.get(field)
    for side in ("recto", "verso"):
        nested = label.get(side)
        if isinstance(nested, Mapping) and field in nested:
            return nested.get(field)
    # QlickEER reste stocké dans sa forme de réponse d'origine. On ne crée pas
    # de copie normalisée : ce petit adaptateur lit seulement la clé utile au
    # moment précis de comparer le résultat OCR.
    customer_data = _qlicker_customer_data(label)
    qlicker_keys = {
        "cin": "cin_id",
        "prenom": "first_name",
        "nom": "last_name",
        "date_naissance": "birth_date",
        # ``city`` est une ville de résidence dans l'exemple QlickEER ; le
        # champ CNI attendu est le lieu de naissance.
        "ville_naissance": "birth_place",
        "date_validite": "validity_date",
        "adresse": "address",
    }
    key = qlicker_keys.get(field)
    return customer_data.get(key) if key else None


def _label_confidence(label: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Retourne les confiances du label sans modifier sa réponse brute."""
    if not isinstance(label, Mapping):
        return {}
    explicit = label.get("field_confidence")
    if isinstance(explicit, Mapping):
        return explicit
    customer_data = _qlicker_customer_data(label)
    qlicker_keys = {
        "cin": "cin_id_confidence",
        "prenom": "first_name_confidence",
        "nom": "last_name_confidence",
        "date_naissance": "birth_date_confidence",
        "ville_naissance": "birth_place_confidence",
        "date_validite": "validity_date_confidence",
        "adresse": "address_confidence",
    }
    return {
        field: customer_data[key]
        for field, key in qlicker_keys.items()
        if key in customer_data
    }


def _qlicker_customer_data(label: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accède au bloc QlickEER, avec ou sans enveloppe HTTP sauvegardée."""
    if not isinstance(label, Mapping):
        return {}
    if isinstance(label.get("customer_data"), Mapping):
        return label["customer_data"]
    response = label.get("response")
    body = response.get("body") if isinstance(response, Mapping) else label.get("body")
    response_data = body.get("response_data") if isinstance(body, Mapping) else None
    customer = response_data.get("customer") if isinstance(response_data, Mapping) else None
    value = customer.get("customer_data") if isinstance(customer, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _extracted_value(extraction: Mapping[str, Any] | None, field: str) -> Any:
    if not isinstance(extraction, Mapping):
        return None
    aliases = {"cin": "cin_fusionne", "date_validite": "date_validite_fusionnee"}
    return extraction.get(aliases.get(field, field))


def _normalise(field: str, value: Any) -> str:
    text = str(value or "").strip()
    if field in {"date_naissance", "date_validite"}:
        for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(text[:10], pattern).date().isoformat()
            except ValueError:
                continue
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", without_accents)


def _is_empty(value: Any) -> bool:
    return value is None or not str(value).strip()
