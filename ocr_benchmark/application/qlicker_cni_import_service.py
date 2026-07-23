"""Préparation séquentielle de lots CNI depuis les routes QlickEER configurées.

Le module ne connaît pas Gradio. Il reçoit des clients sélectionnés et les
quatre routes déjà validées dans les paramètres, puis matérialise un dossier
local par client compatible avec le scanner CNI existant.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cni_ingestion import DEFAULT_RECTO_SUFFIX, DEFAULT_VERSO_SUFFIX, write_cni_json
from .qlicker_api_service import (
    download_qlicker_file,
    execute_qlicker_get,
    extract_customer_cni_label,
    find_cni_documents,
)


@dataclass(frozen=True)
class QlickerCniRoutes:
    """Routes et paramètres modèles issus des quatre onglets API QlickEER."""

    customer_endpoint: str
    customer_params: list[tuple[str, str]]
    documents_endpoint: str
    documents_params: list[tuple[str, str]]
    file_endpoint: str
    file_params: list[tuple[str, str]]


def build_qlicker_cni_routes(
    customer_endpoint: str,
    customer_rows: Sequence[Sequence[Any]] | None,
    documents_endpoint: str,
    documents_rows: Sequence[Sequence[Any]] | None,
    file_endpoint: str,
    file_rows: Sequence[Sequence[Any]] | None,
) -> QlickerCniRoutes:
    """Valide les trois routes nécessaires sans réinterpréter leurs paramètres."""
    routes = QlickerCniRoutes(
        customer_endpoint=str(customer_endpoint or "").strip(),
        customer_params=_enabled_pairs(customer_rows),
        documents_endpoint=str(documents_endpoint or "").strip(),
        documents_params=_enabled_pairs(documents_rows),
        file_endpoint=str(file_endpoint or "").strip(),
        file_params=_enabled_pairs(file_rows),
    )
    missing = [
        name for name, value in (
            ("Détail client", routes.customer_endpoint),
            ("Documents", routes.documents_endpoint),
            ("Fichier", routes.file_endpoint),
        ) if not value
    ]
    if missing:
        raise ValueError("Route(s) API non configurée(s) : " + ", ".join(missing) + ".")
    return routes


def iter_prepare_qlicker_cni_clients(
    customers: Sequence[Mapping[str, Any]],
    destination_root: Path,
    *,
    base_url: str,
    routes: QlickerCniRoutes,
    timeout_seconds: float,
    proxy_url: str | None,
    use_system_proxy: bool,
    verify_ssl: bool,
    recto_suffix: str = DEFAULT_RECTO_SUFFIX,
    verso_suffix: str = DEFAULT_VERSO_SUFFIX,
) -> Iterator[dict[str, Any]]:
    """Prépare les CNI des clients un par un et émet leur état après chaque étape.

    Les fichiers restent au format fourni par l'API (PDF, JPEG ou PNG). Les
    erreurs d'un client sont isolées : elles ne stoppent jamais les suivants.
    """
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    total = len(customers)
    for index, customer in enumerate(customers, start=1):
        raw_client_id = str(customer.get("id") or customer.get("customer_id") or "").strip()
        client_id = _safe_client_id(raw_client_id)
        event: dict[str, Any] = {
            "index": index,
            "total": total,
            "client_id": raw_client_id or "inconnu",
            "status": "discovered",
            "message": "Client sélectionné dans la file.",
            "issues": [],
        }
        if client_id is None:
            event.update(status="failed", message="Identifiant client invalide.", issues=["client_id_invalid"])
            yield event
            continue

        client_dir = root / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        event["client_dir"] = str(client_dir)
        yield dict(event)

        try:
            documents_payload = execute_qlicker_get(
                base_url,
                routes.documents_endpoint,
                _replace_route_parameters(routes.documents_params, {"customerID": raw_client_id}),
                timeout_seconds=timeout_seconds,
                proxy_url=proxy_url,
                use_system_proxy=use_system_proxy,
                verify_ssl=verify_ssl,
            )
            response_data = _response_data(documents_payload, "get_signed_documents_list")
            documents = response_data.get("documents_list", [])
            if not isinstance(documents, list):
                raise ValueError("get_signed_documents_list : documents_list est absent ou invalide.")
            pair = find_cni_documents(documents)
            if not pair["recto"] or not pair["verso"]:
                missing = ", ".join(side for side, name in pair.items() if not name)
                raise ValueError(f"CNI incomplète : face(s) absente(s) : {missing}.")
            event.update(
                status="documents_detected",
                message="Recto et verso CNI détectés.",
                recto_name=pair["recto"],
                verso_name=pair["verso"],
            )
            yield dict(event)

            recto = download_qlicker_file(
                base_url,
                routes.file_endpoint,
                _replace_route_parameters(
                    routes.file_params,
                    {"customerID": raw_client_id, "file": pair["recto"], "page": "1"},
                ),
                client_dir / f"{client_id}{recto_suffix}",
                timeout_seconds=timeout_seconds,
                proxy_url=proxy_url,
                use_system_proxy=use_system_proxy,
                verify_ssl=verify_ssl,
            )
            verso = download_qlicker_file(
                base_url,
                routes.file_endpoint,
                _replace_route_parameters(
                    routes.file_params,
                    {"customerID": raw_client_id, "file": pair["verso"], "page": "1"},
                ),
                client_dir / f"{client_id}{verso_suffix}",
                timeout_seconds=timeout_seconds,
                proxy_url=proxy_url,
                use_system_proxy=use_system_proxy,
                verify_ssl=verify_ssl,
            )
            event.update(
                status="downloaded",
                message="Recto et verso téléchargés.",
                recto_source=recto["path"],
                verso_source=verso["path"],
            )
            yield dict(event)

            try:
                customer_payload = execute_qlicker_get(
                    base_url,
                    routes.customer_endpoint,
                    _replace_route_parameters(routes.customer_params, {"customerID": raw_client_id}),
                    timeout_seconds=timeout_seconds,
                    proxy_url=proxy_url,
                    use_system_proxy=use_system_proxy,
                    verify_ssl=verify_ssl,
                )
                _response_data(customer_payload, "get_customer_data")
                label = extract_customer_cni_label(customer_payload)
                write_cni_json(client_dir / f"{client_id}.json", label)
                event.update(
                    status="label_normalized",
                    message="Label QlickEER normalisé.",
                    label_path=str(client_dir / f"{client_id}.json"),
                )
                yield dict(event)
                event.update(status="ready", message="Client prêt pour le benchmark CNI.")
            except Exception as label_error:
                # Une CNI complète reste exploitable sans score, conformément
                # à l'option existante « Continuer sans labels ».
                event.update(
                    status="ready_without_label",
                    message="Documents prêts, mais label indisponible.",
                    issues=[*event.get("issues", []), f"label:{type(label_error).__name__}: {label_error}"],
                )
            yield dict(event)
        except Exception as exc:
            event.update(
                status="failed",
                message="Préparation API interrompue pour ce client.",
                issues=[*event.get("issues", []), f"{type(exc).__name__}: {exc}"],
            )
            yield dict(event)


def _enabled_pairs(rows: Sequence[Sequence[Any]] | None) -> list[tuple[str, str]]:
    """Conserve l'ordre, les doublons et les valeurs vides des lignes cochées."""
    pairs: list[tuple[str, str]] = []
    for row in rows or []:
        if len(row) < 3 or not bool(row[2]):
            continue
        name = str(row[0] or "").strip()
        if name:
            pairs.append((name, "" if row[1] is None else str(row[1])))
    return pairs


def _replace_route_parameters(template: Sequence[tuple[str, str]], overrides: Mapping[str, str]) -> list[tuple[str, str]]:
    """Remplace customerID/file/page sans perdre les paramètres inconnus.

    Les clés non connues de ``view_file`` restent exactement celles que
    l'utilisateur a parsées depuis Postman. Si une clé obligatoire n'était pas
    présente dans l'URL collée, elle est ajoutée en fin de query string.
    """
    expected = {str(name).casefold(): str(value) for name, value in overrides.items()}
    seen: set[str] = set()
    output: list[tuple[str, str]] = []
    for name, value in template:
        normalized = name.casefold()
        if normalized in expected:
            output.append((name, expected[normalized]))
            seen.add(normalized)
        else:
            output.append((name, value))
    for name, value in overrides.items():
        if name.casefold() not in seen:
            output.append((name, str(value)))
    return output


def _response_data(payload: Mapping[str, Any], route_name: str) -> Mapping[str, Any]:
    """Valide le statut HTTP puis retourne ``body.response_data``."""
    response = payload.get("response", {}) if isinstance(payload, Mapping) else {}
    status_code = int(response.get("status_code") or 0) if isinstance(response, Mapping) else 0
    if not 200 <= status_code < 300:
        raise RuntimeError(f"{route_name} : HTTP {status_code}.")
    body = response.get("body", {}) if isinstance(response, Mapping) else {}
    response_data = body.get("response_data", {}) if isinstance(body, Mapping) else {}
    if not isinstance(response_data, Mapping):
        raise ValueError(f"{route_name} : response_data est absent ou invalide.")
    return response_data


def _safe_client_id(value: str) -> str | None:
    """Empêche qu'un identifiant distant puisse modifier le chemin local."""
    candidate = str(value or "").strip()
    return candidate if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate) else None
