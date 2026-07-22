"""Client HTTP minimal, sans persistance, pour explorer l'API QlickEER interne.

Le contrat QlickEER n'est pas encore documenté complètement. Ce service ne
suppose donc ni token, ni nom d'endpoint, ni structure de réponse. Il construit
des GET à partir d'une Base URL, d'un segment d'endpoint et de paramètres, puis
retourne un diagnostic prêt à afficher dans Gradio.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

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


def parse_qlicker_url(raw_url: str) -> tuple[str, str, list[list[Any]]]:
    """Découpe une URL QlickEER complète en base, endpoint et table éditable.

    Les query parameters sont conservés dans l'ordre, y compris les doublons
    et les valeurs vides. Cela permet à l'interface de reproduire une URL
    collée depuis Postman avant que l'utilisateur ne modifie une ligne.
    """
    parsed = urlsplit(str(raw_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL invalide : indiquez une URL complète http:// ou https://.")
    if not parsed.path or parsed.path == "/":
        raise ValueError("URL incomplète : le chemin de la fonction QlickEER est absent.")
    base_url = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    endpoint = parsed.path.lstrip("/")
    rows = [[name, value, True] for name, value in parse_qsl(parsed.query, keep_blank_values=True)]
    return base_url, endpoint, rows


def editable_rows_to_query_pairs(rows: Sequence[Sequence[Any]] | None) -> list[tuple[str, str]]:
    """Convertit le tableau Gradio en paramètres GET sans perdre les doublons.

    Une ligne décochée est omise. Une valeur vide reste envoyée sous la forme
    ``param=`` : c'est différent d'un paramètre absent.
    """
    pairs: list[tuple[str, str]] = []
    for row in rows or []:
        if len(row) < 3:
            continue
        name, value, enabled = row[0], row[1], row[2]
        clean_name = str(name or "").strip()
        if clean_name and bool(enabled):
            pairs.append((clean_name, "" if value is None else str(value)))
    return pairs


def _proxy_mapping(proxy_url: str | None) -> dict[str, str] | None:
    """Valide un proxy explicite, sans le journaliser avec son mot de passe."""
    candidate = str(proxy_url or "").strip()
    if not candidate:
        return None
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Proxy invalide : utilisez par exemple http://proxy.interne.local:8080.")
    return {"http": candidate, "https": candidate}


def _masked_proxy(proxy_url: str | None) -> str:
    """Masque un éventuel mot de passe avant retour vers Gradio."""
    parsed = urlsplit(str(proxy_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    if parsed.password is None:
        return str(proxy_url)
    username = parsed.username or ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{username}:***@{parsed.hostname}{port}"


def execute_qlicker_get(
    base_url: str,
    endpoint: str,
    query_params: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    timeout_seconds: float = 30.0,
    proxy_url: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    """Envoie un GET interne et retourne la requête/réponse sans écrire sur disque.

    ``proxy_url`` et ``verify_ssl`` viennent de la configuration UI ; aucun
    secret n'est enregistré. La vérification SSL est active par défaut et ne
    doit être désactivée que pour un certificat interne connu.
    """
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise ValueError("Le timeout doit être supérieur à zéro.")
    url = build_qlicker_url(base_url, endpoint)
    proxy_mapping = _proxy_mapping(proxy_url)
    response = requests.get(
        url,
        params=query_params,
        timeout=timeout,
        proxies=proxy_mapping,
        verify=bool(verify_ssl),
    )
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
            "omitted_null_parameters": (
                [key for key, value in query_params.items() if value is None]
                if isinstance(query_params, Mapping)
                else []
            ),
            "proxy": _masked_proxy(proxy_url),
            "verification_ssl": "active" if verify_ssl else "désactivée",
        },
        "response": {
            "status_code": response.status_code,
            "content_type": content_type or "absent",
            "bytes": len(response.content),
            "body": body,
        },
    }
