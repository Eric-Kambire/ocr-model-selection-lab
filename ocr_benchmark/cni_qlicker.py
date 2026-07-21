"""Import REST configurable de documents et labels CNI depuis Qlicker.

Les routes Qlicker ne sont pas publiées dans ce projet. Ce module ne les
invente donc pas : les modèles d'URL et la clé JSON de téléchargement sont
configurés explicitement avant tout appel réseau.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


@dataclass(frozen=True)
class QlickerImportConfig:
    """Contrat minimal à compléter dès que l'API Qlicker est documentée."""

    base_url: str
    recto_path_template: str
    verso_path_template: str
    label_path_template: str
    document_url_key: str = ""
    token_env_name: str = "QLICKER_API_TOKEN"
    auth_header_name: str = "Authorization"
    auth_prefix: str = "Bearer "
    timeout_seconds: float = 30.0


def import_qlicker_clients(
    client_ids: list[str],
    destination_root: Path,
    config: QlickerImportConfig,
    *,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Télécharge les deux faces et le label de chaque client localement."""
    token = os.environ.get(config.token_env_name)
    if not token:
        raise RuntimeError(f"Le secret d'API est absent : définissez {config.token_env_name} dans l'environnement.")
    http = session or requests.Session()
    headers = {config.auth_header_name: f"{config.auth_prefix}{token}".strip()}
    destination_root.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []

    for raw_client_id in client_ids:
        client_id = str(raw_client_id).strip()
        if not client_id:
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", client_id):
            report.append({"client_id": client_id, "status": "failed", "issues": ["client_id_invalide"]})
            continue
        client_dir = destination_root / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        item: dict[str, Any] = {"client_id": client_id, "status": "ready", "issues": []}
        try:
            recto = _download_document(http, headers, config, config.recto_path_template, client_id, client_dir / f"{client_id}_CIN_Recto")
            verso = _download_document(http, headers, config, config.verso_path_template, client_id, client_dir / f"{client_id}_CIN_Verso")
            label = _download_label(http, headers, config, client_id)
            label_path = client_dir / f"{client_id}.json"
            label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
            item.update({"recto_source": str(recto), "verso_source": str(verso), "label_path": str(label_path)})
        except (requests.RequestException, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            item["status"] = "failed"
            item["issues"].append(f"{type(exc).__name__}: {exc}")
        report.append(item)
    return report


def _request_url(config: QlickerImportConfig, path_template: str, client_id: str) -> str:
    """Construit une URL en ne permettant que le paramètre client_id."""
    return urljoin(config.base_url.rstrip("/") + "/", path_template.format(client_id=client_id).lstrip("/"))


def _download_document(
    http: requests.Session, headers: dict[str, str], config: QlickerImportConfig,
    path_template: str, client_id: str, output_stem: Path,
) -> Path:
    """Télécharge un PDF, JPEG ou PNG et conserve son format d'origine."""
    response = http.get(_request_url(config, path_template, client_id), headers=headers, timeout=config.timeout_seconds)
    response.raise_for_status()
    if config.document_url_key:
        payload = response.json()
        url_value = payload.get(config.document_url_key) if isinstance(payload, dict) else None
        if not isinstance(url_value, str) or not url_value.strip():
            raise ValueError(f"Clé document_url_key absente ou invalide : {config.document_url_key}")
        response = http.get(urljoin(config.base_url.rstrip("/") + "/", url_value), headers=headers, timeout=config.timeout_seconds)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    extension = ".pdf" if "pdf" in content_type else ".jpg" if "jpeg" in content_type else ".png" if "png" in content_type else ""
    if not extension:
        raise ValueError(f"Format document non reconnu (Content-Type={content_type or 'absent'})")
    output_path = output_stem.with_suffix(extension)
    output_path.write_bytes(response.content)
    return output_path


def _download_label(http: requests.Session, headers: dict[str, str], config: QlickerImportConfig, client_id: str) -> dict[str, Any]:
    """Télécharge et valide que le label distant est un objet JSON."""
    response = http.get(_request_url(config, config.label_path_template, client_id), headers=headers, timeout=config.timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Le label Qlicker doit être un objet JSON.")
    return payload
