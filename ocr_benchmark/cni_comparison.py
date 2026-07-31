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
    total_character_edits = 0
    total_reference_characters = 0
    total_word_edits = 0
    total_reference_words = 0

    for field in CNI_COMPARISON_FIELDS:
        expected = _label_value(label, field) if label_is_available else None
        actual = _extracted_value(extraction, field) if extraction_is_available else None
        expected_metric = (
            _normalise_metric_text(field, expected)
            if label_is_available and not _is_empty(expected)
            else ""
        )
        actual_metric = (
            _normalise_metric_text(field, actual)
            if extraction_is_available and not _is_empty(actual)
            else ""
        )
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

        # CER et WER sont calculés uniquement lorsqu'une référence existe et
        # que l'extraction technique est disponible. Une valeur modèle absente
        # devient naturellement 100 % d'erreur ; une panne technique reste N/A.
        if state not in {"reference_missing", "extraction_unavailable"}:
            character_edits = _levenshtein_distance(
                expected_metric,
                actual_metric,
            )
            expected_words = expected_metric.split()
            actual_words = actual_metric.split()
            word_edits = _levenshtein_distance(expected_words, actual_words)
            reference_characters = len(expected_metric)
            reference_words = len(expected_words)
            cer = (
                character_edits / reference_characters
                if reference_characters
                else None
            )
            wer = word_edits / reference_words if reference_words else None
            similarity = max(0.0, 1.0 - cer) if cer is not None else None
            total_character_edits += character_edits
            total_reference_characters += reference_characters
            total_word_edits += word_edits
            total_reference_words += reference_words
        else:
            character_edits = None
            word_edits = None
            reference_characters = None
            reference_words = None
            cer = None
            wer = None
            similarity = None
        rows.append(
            {
                "field": field,
                "expected": expected,
                "actual": actual,
                "expected_normalized": expected_metric or None,
                "actual_normalized": actual_metric or None,
                "reference_confidence": confidence.get(field),
                "state": state,
                "character_edits": character_edits,
                "reference_characters": reference_characters,
                "word_edits": word_edits,
                "reference_words": reference_words,
                "cer": cer,
                "wer": wer,
                "similarity": similarity,
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

    cer = (
        total_character_edits / total_reference_characters
        if total_reference_characters
        else None
    )
    wer = (
        total_word_edits / total_reference_words
        if total_reference_words
        else None
    )
    return {
        "rows": rows,
        "accuracy": score,
        # La similarité est bornée à zéro : CER peut dépasser 100 % lorsque
        # les insertions/hallucinations dépassent la longueur de référence.
        "text_similarity": max(0.0, 1.0 - cer) if cer is not None else None,
        "cer": cer,
        "wer": wer,
        "score_status": score_status,
        "correct_fields": sum(row["state"] == "correct" for row in rows),
        "comparable_fields": sum(row["state"] != "reference_missing" for row in rows),
        "character_edits": total_character_edits,
        "reference_characters": total_reference_characters,
        "word_edits": total_word_edits,
        "reference_words": total_reference_words,
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


def _normalise_metric_text(field: str, value: Any) -> str:
    """Normalise pour CER/WER tout en conservant les frontières de mots.

    La comparaison exacte compacte encore les séparateurs afin d'accepter
    ``XA 12-34`` et ``XA1234``. Pour WER, supprimer tous les espaces rendrait
    les noms et adresses artificiellement mono-mot ; on conserve donc ici un
    espace normalisé entre les groupes alphanumériques.
    """
    text = str(value or "").strip()
    if field in {"date_naissance", "date_validite"}:
        for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                text = datetime.strptime(text[:10], pattern).date().isoformat()
                break
            except ValueError:
                continue
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    groups = re.findall(r"[a-z0-9]+", without_accents)
    return " ".join(groups)


def _levenshtein_distance(first: Any, second: Any) -> int:
    """Calcule la distance d'édition avec une mémoire linéaire.

    Les entrées peuvent être des chaînes (CER) ou des listes de mots (WER).
    Une édition correspond à une insertion, une suppression ou une substitution.
    """
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for first_index, first_value in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_value in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1]
                    + (first_value != second_value),
                )
            )
        previous = current
    return previous[-1]


def _is_empty(value: Any) -> bool:
    return value is None or not str(value).strip()
