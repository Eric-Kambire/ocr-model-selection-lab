"""Passerelle FastAPI locale et sécurisée vers les API QlickEER.

Ce script est séparé de Gradio : le navigateur appelle FastAPI sur le PC ou le
serveur interne, puis FastAPI appelle QlickEER avec le proxy Windows local si
nécessaire. Il ne constitue pas un proxy HTTP général : seuls les hôtes placés
explicitement dans ``QLICKER_ALLOWED_HOSTS`` sont autorisés.

Exemple PowerShell (sur le PC interne) :

    $env:QLICKER_ALLOWED_HOSTS = "qlicker.intra.local,10.20.30.40"
    $env:QLICKER_PROXY_URL = "http://proxy.entreprise.local:8080"
    $env:QLICKER_VERIFY_SSL = "false"  # seulement si le certificat interne est non reconnu
    python scripts/qlicker_fastapi_gateway.py

Consulter ensuite http://127.0.0.1:8120/docs. Le serveur écoute uniquement sur
127.0.0.1 par défaut ; il n'est donc pas exposé aux autres machines du réseau.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Any
from urllib.parse import urlsplit

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scripts.qlicker_url_parser_lab import (
    explicit_proxy_mapping,
    mask_proxy_url,
    windows_manual_proxy_mapping,
    windows_proxy_summary,
)


LOGGER = logging.getLogger("qlicker_gateway")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(
    title="QlickEER Internal Gateway",
    version="0.1.0",
    description="Passerelle locale contrôlée pour les GET QlickEER. Aucun hôte non autorisé n'est accepté.",
    docs_url="/docs",
    redoc_url=None,
)


class QueryParameter(BaseModel):
    """Une paire query-string. Plusieurs lignes portant le même nom sont admises."""

    name: str = Field(min_length=1, max_length=100)
    value: str = Field(default="", max_length=5_000)


class QlickerGetRequest(BaseModel):
    """Contrat de l'unique appel QlickEER autorisé par cette passerelle."""

    endpoint: str = Field(description="URL complète de l'API QlickEER, dont l'hôte doit être autorisé.")
    parameters: list[QueryParameter] = Field(default_factory=list)
    connect_timeout_seconds: float = Field(default=30, gt=0, le=300)
    read_timeout_seconds: float = Field(default=300, gt=0, le=900)
    use_system_proxy: bool = Field(default=True)


def configured_allowed_hosts() -> set[str]:
    """Lit la liste blanche d'hôtes sans valeur par défaut dangereuse."""
    return {
        value.strip().lower()
        for value in os.environ.get("QLICKER_ALLOWED_HOSTS", "").split(",")
        if value.strip()
    }


def ssl_verification_enabled() -> bool:
    """Lit l'option TLS du serveur, sécurisée par défaut.

    La vérification reste active tant que ``QLICKER_VERIFY_SSL`` n'est pas
    explicitement positionnée à ``false``/``0``. Ce réglage est volontairement
    une variable du serveur et non un paramètre fourni par le navigateur.
    """
    value = os.environ.get("QLICKER_VERIFY_SSL", "true").strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError("QLICKER_VERIFY_SSL doit valoir true ou false")


def validate_internal_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Valide schéma, hôte autorisé et port avant toute connexion sortante."""
    target = str(endpoint or "").strip()
    parsed = urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=422, detail="endpoint doit être une URL http:// ou https:// valide")
    allowed_hosts = configured_allowed_hosts()
    if not allowed_hosts:
        raise HTTPException(
            status_code=503,
            detail="QLICKER_ALLOWED_HOSTS n'est pas configuré : la passerelle refuse tout appel par sécurité.",
        )
    if parsed.hostname.lower() not in allowed_hosts:
        raise HTTPException(
            status_code=403,
            detail=f"Hôte refusé. Autorisés : {sorted(allowed_hosts)}",
        )
    return target, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def gateway_proxy_mapping(endpoint: str, use_system_proxy: bool) -> tuple[dict[str, str] | None, str]:
    """Choisit le proxy sans laisser un client HTTP imposer sa propre route."""
    explicit = os.environ.get("QLICKER_PROXY_URL", "").strip()
    if explicit:
        try:
            return explicit_proxy_mapping(explicit), "proxy configuré par QLICKER_PROXY_URL"
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=f"QLICKER_PROXY_URL invalide : {exc}") from exc
    if not use_system_proxy:
        return None, "connexion directe demandée"
    windows_proxy = windows_manual_proxy_mapping(endpoint)
    if windows_proxy:
        return windows_proxy, "proxy manuel Windows"
    environment_proxy = {
        name: value
        for name, value in requests.utils.get_environ_proxies(endpoint).items()
        if name in {"http", "https", "all"}
    }
    return environment_proxy or None, "proxy détecté par Python" if environment_proxy else "connexion directe"


def response_body(response: requests.Response) -> Any:
    """Retourne JSON/texte, sans rapatrier un binaire volumineux dans la réponse API."""
    content_type = response.headers.get("content-type", "").lower()
    if "json" in content_type:
        try:
            return response.json()
        except ValueError:
            pass
    if content_type.startswith("text/") or not content_type:
        return response.text[:100_000]
    return {"binary": True, "content_type": content_type, "bytes": len(response.content)}


@app.get("/health")
def health() -> dict[str, Any]:
    """Expose seulement l'état local, sans contacter QlickEER."""
    try:
        ssl_verification = ssl_verification_enabled()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "emetteur": "processus FastAPI local sur ce PC/serveur interne",
        "poste": socket.gethostname(),
        "allowed_hosts": sorted(configured_allowed_hosts()),
        "proxy_windows": windows_proxy_summary(),
        "verification_ssl": "active" if ssl_verification else "DÉSACTIVÉE — uniquement pour certificat interne non reconnu",
    }


@app.post("/v1/qlicker/get")
def qlicker_get(payload: QlickerGetRequest) -> dict[str, Any]:
    """Exécute un GET QlickEER local, limité à la liste blanche configurée."""
    target, host, port = validate_internal_endpoint(payload.endpoint)
    proxy_mapping, proxy_mode = gateway_proxy_mapping(target, payload.use_system_proxy)
    try:
        verify_ssl = ssl_verification_enabled()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    query_pairs = [(item.name, item.value) for item in payload.parameters]
    started = time.perf_counter()
    try:
        with requests.Session() as session:
            session.trust_env = bool(payload.use_system_proxy)
            LOGGER.info(
                "QlickEER GET | host=%s | port=%s | params=%s | proxy=%s | verify_ssl=%s",
                host, port, [name for name, _value in query_pairs], proxy_mode, verify_ssl,
            )
            if not verify_ssl:
                LOGGER.warning("Vérification SSL désactivée pour QlickEER via QLICKER_VERIFY_SSL=false")
            response = session.get(
                target,
                params=query_pairs,
                proxies=proxy_mapping,
                timeout=(payload.connect_timeout_seconds, payload.read_timeout_seconds),
                verify=verify_ssl,
            )
    except requests.RequestException as exc:
        LOGGER.exception("QlickEER GET échoué")
        raise HTTPException(
            status_code=502,
            detail={
                "type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "proxy_mode": proxy_mode,
                "proxy": {name: mask_proxy_url(value) for name, value in (proxy_mapping or {}).items()},
            },
        ) from exc

    return {
        "execution": "FastAPI local → proxy éventuel → QlickEER",
        "target": f"{host}:{port}",
        "proxy_mode": proxy_mode,
        "proxy": {name: mask_proxy_url(value) for name, value in (proxy_mapping or {}).items()},
        "verification_ssl": "active" if verify_ssl else "désactivée",
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "body": response_body(response),
    }


if __name__ == "__main__":
    # Conserver 127.0.0.1 limite l'accès au seul PC interne qui exécute le code.
    # Une exposition réseau devra être décidée explicitement avec authentification.
    uvicorn.run(app, host=os.environ.get("QLICKER_GATEWAY_HOST", "127.0.0.1"), port=8120)
