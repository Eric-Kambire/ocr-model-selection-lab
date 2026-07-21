"""Client HTTP minimal, sans persistance, pour explorer l'API Qlicker interne.

Le contrat Qlicker n'est pas encore documenté complètement. Ce service ne
suppose donc ni token, ni nom d'endpoint, ni structure de réponse. Il construit
des GET à partir d'une Base URL, d'un segment d'endpoint et de paramètres, puis
retourne un diagnostic prêt à afficher dans Gradio.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin

import requests


def parse_extra_query_params(raw_value: str | None) -> dict[str, Any]:
    """Lit des paramètres JSON additionnels en conservant `null` et `""`.

    `null` signifie « ne pas envoyer le paramètre », car `requests` omet cette
    valeur dans une query string. Une chaîne vide est elle envoyée comme
    `nom_parametre=`. Cette distinction est affichée dans l'interface.
    """
    text = str(raw_value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Paramètres supplémentaires : JSON invalide ({exc.msg}).") from exc
    if not isinstance(payload, dict):
        raise ValueError("Paramètres supplémentaires : fournissez un objet JSON, par exemple {\"tri\": null}.")
    return {str(key): value for key, value in payload.items()}


def merge_query_params(explicit: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Fusionne les paramètres ; les champs guidés priment sur le JSON libre."""
    result = dict(extra)
    result.update({key: value for key, value in explicit.items() if value is not None})
    return result


def build_qlicker_url(base_url: str, endpoint: str) -> str:
    """Assemble la Base URL commune et un segment d'endpoint configurable."""
    base = str(base_url or "").strip().rstrip("/")
    path = str(endpoint or "").strip().lstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("La Base URL doit commencer par http:// ou https://.")
    if not path:
        raise ValueError("Le segment endpoint/fonction est obligatoire.")
    return urljoin(base + "/", path)


def execute_qlicker_get(
    base_url: str,
    endpoint: str,
    query_params: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Envoie un GET interne et retourne la requête/réponse sans écrire sur disque."""
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise ValueError("Le timeout doit être supérieur à zéro.")
    url = build_qlicker_url(base_url, endpoint)
    response = requests.get(url, params=query_params, timeout=timeout)
    content_type = response.headers.get("content-type", "").lower()
    is_json = "json" in content_type
    is_text = is_json or content_type.startswith("text/") or not content_type
    if is_json:
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text[:100_000]
    elif is_text:
        body = response.text[:100_000]
    else:
        body = {
            "binary": True,
            "message": "Réponse binaire reçue ; elle n'est pas enregistrée pendant ce test API.",
            "bytes": len(response.content),
        }
    return {
        "request": {
            "method": "GET",
            "url": response.url,
            "params": query_params,
            "omitted_null_parameters": [key for key, value in query_params.items() if value is None],
        },
        "response": {
            "status_code": response.status_code,
            "content_type": content_type or "absent",
            "bytes": len(response.content),
            "body": body,
        },
    }
