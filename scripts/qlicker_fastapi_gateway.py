"""Passerelle FastAPI locale vers Qlicker, avec HTTPX et sans Requests.

FastAPI reçoit les appels du navigateur local. HTTPX effectue ensuite l'appel
sortant vers Qlicker depuis ce même PC/serveur interne. FastAPI n'est pas un
client HTTP : HTTPX est donc la bibliothèque qui ouvre réellement le socket.

Lancement PowerShell :

    $env:QLICKER_ALLOWED_HOSTS = "qlicker.intra.local,10.20.30.40"
    python scripts/qlicker_fastapi_gateway.py

Ouvrir ensuite http://127.0.0.1:8120/docs.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import socket
import time
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


LOGGER = logging.getLogger("qlicker_gateway")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(
    title="Qlicker Internal Gateway",
    version="0.2.0",
    description="Passerelle locale FastAPI + HTTPX, limitée aux hôtes Qlicker autorisés.",
    docs_url="/docs",
    redoc_url=None,
)


class QueryParameter(BaseModel):
    """Une paire query-string ; les noms dupliqués restent admis."""

    name: str = Field(min_length=1, max_length=100)
    value: str = Field(default="", max_length=5_000)


class QlickerGetRequest(BaseModel):
    """Contrat du GET Qlicker autorisé par la passerelle locale."""

    endpoint: str = Field(description="URL complète ; son hôte doit appartenir à QLICKER_ALLOWED_HOSTS.")
    parameters: list[QueryParameter] = Field(default_factory=list)
    connect_timeout_seconds: float = Field(default=30, gt=0, le=300)
    read_timeout_seconds: float = Field(default=300, gt=0, le=900)
    use_system_proxy: bool = Field(default=True)


class RawUrlRequest(BaseModel):
    """URL brute à découper avant de l'éditer dans le client appelant."""

    url: str


def mask_proxy_url(proxy_url: str) -> str:
    """Masque le mot de passe éventuel d'une URL de proxy dans les journaux."""
    parsed = urlsplit(str(proxy_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return "(proxy invalide)"
    if parsed.password is None:
        return str(proxy_url)
    username = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{username}:***@{host}{port}"


def parse_url_for_gateway(raw_url: str) -> tuple[str, list[dict[str, str]]]:
    """Découpe une URL façon Postman, en préservant vides et paramètres répétés."""
    parsed = urlsplit(str(raw_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="url doit être une URL http:// ou https:// valide")
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return endpoint, [{"name": name, "value": value} for name, value in parse_qsl(parsed.query, keep_blank_values=True)]


def configured_allowed_hosts() -> set[str]:
    """Lit la liste blanche ; aucune valeur par défaut ouverte n'est autorisée."""
    return {
        value.strip().lower()
        for value in os.environ.get("QLICKER_ALLOWED_HOSTS", "").split(",")
        if value.strip()
    }


def validate_internal_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Bloque l'usage de la passerelle comme proxy arbitraire (risque SSRF)."""
    target = str(endpoint or "").strip()
    parsed = urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=422, detail="endpoint doit être une URL http:// ou https:// valide")
    allowed_hosts = configured_allowed_hosts()
    if not allowed_hosts:
        raise HTTPException(status_code=503, detail="QLICKER_ALLOWED_HOSTS n'est pas configuré : appel refusé par sécurité.")
    if parsed.hostname.lower() not in allowed_hosts:
        raise HTTPException(status_code=403, detail=f"Hôte refusé. Autorisés : {sorted(allowed_hosts)}")
    return target, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def _normalise_proxy(value: str) -> str:
    """Le registre Windows omet souvent le protocole, HTTPX l'exige."""
    candidate = str(value or "").strip()
    return candidate if "://" in candidate else f"http://{candidate}"


def parse_windows_proxy_server(proxy_server: str) -> dict[str, str] | None:
    """Convertit ProxyServer Windows vers des proxys HTTPX par protocole."""
    raw = str(proxy_server or "").strip()
    if not raw:
        return None
    if "=" not in raw:
        proxy = _normalise_proxy(raw)
        return {"http": proxy, "https": proxy}
    mapping: dict[str, str] = {}
    for item in raw.split(";"):
        protocol, separator, address = item.partition("=")
        if separator and protocol.strip().lower() in {"http", "https"} and address.strip():
            mapping[protocol.strip().lower()] = _normalise_proxy(address)
    return mapping or None


def _windows_proxy_settings() -> dict[str, Any] | None:
    """Lit la configuration WinINET du PC exécutant réellement FastAPI."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            def value(name: str, default: Any = "") -> Any:
                try:
                    result, _ = winreg.QueryValueEx(key, name)
                    return result
                except FileNotFoundError:
                    return default

            return {
                "enabled": bool(value("ProxyEnable", 0)),
                "server": str(value("ProxyServer", "") or ""),
                "override": str(value("ProxyOverride", "") or ""),
                "pac": str(value("AutoConfigURL", "") or ""),
                "auto_detect": bool(value("AutoDetect", 0)),
            }
    except (ImportError, OSError):
        return None


def _bypasses_windows_proxy(hostname: str, override: str) -> bool:
    """Respecte les exceptions WinINET usuelles sans interpréter un PAC."""
    host = hostname.lower()
    for pattern in (item.strip() for item in str(override or "").split(";")):
        if not pattern:
            continue
        if pattern.lower() == "<local>" and "." not in host:
            return True
        if fnmatch.fnmatch(host, pattern.lower()):
            return True
    return False


def system_proxy_mapping(endpoint: str, use_system_proxy: bool) -> tuple[dict[str, str] | None, str, dict[str, str]]:
    """Construit le proxy HTTPX : config explicite, Windows manuel, env ou direct.

    Un PAC est déclaré dans le statut mais n'est pas exécuté ici. HTTPX sait
    utiliser les variables HTTP_PROXY/HTTPS_PROXY avec ``trust_env=True``.
    """
    explicit = os.environ.get("QLICKER_PROXY_URL", "").strip()
    if explicit:
        parsed = urlsplit(explicit)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise HTTPException(status_code=503, detail="QLICKER_PROXY_URL doit être une URL HTTP(S) valide")
        return {"http": explicit, "https": explicit}, "proxy QLICKER_PROXY_URL", {}
    if not use_system_proxy:
        return None, "connexion directe demandée", {}

    settings = _windows_proxy_settings()
    if settings and settings["enabled"] and not _bypasses_windows_proxy(urlsplit(endpoint).hostname or "", settings["override"]):
        mapping = parse_windows_proxy_server(settings["server"])
        if mapping:
            return mapping, "proxy manuel Windows", {"pac": "configuré" if settings["pac"] else "absent"}

    environment = {
        name: value
        for name, value in __import__("urllib.request", fromlist=["getproxies"]).getproxies().items()
        if name in {"http", "https", "all"}
    }
    return environment or None, "proxy environnement HTTPX" if environment else "connexion directe", {
        "pac": "configuré" if settings and settings["pac"] else "absent"
    }


def httpx_mounts(proxy_mapping: dict[str, str] | None) -> dict[str, httpx.AsyncBaseTransport] | None:
    """Adapte les proxys séparés Windows au format de montage HTTPX."""
    if not proxy_mapping:
        return None
    mounts: dict[str, httpx.AsyncBaseTransport] = {}
    for scheme in ("http", "https"):
        proxy = proxy_mapping.get(scheme) or proxy_mapping.get("all")
        if proxy:
            mounts[f"{scheme}://"] = httpx.AsyncHTTPTransport(proxy=proxy)
    return mounts or None


def response_body(response: httpx.Response) -> Any:
    """Retourne JSON/texte ; les binaires restent des métadonnées."""
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
async def health() -> dict[str, Any]:
    """État local sans appel sortant Qlicker."""
    settings = _windows_proxy_settings()
    return {
        "status": "ok",
        "emetteur": "processus FastAPI local sur ce PC/serveur interne",
        "poste": socket.gethostname(),
        "allowed_hosts": sorted(configured_allowed_hosts()),
        "proxy_windows": {
            "manuel": "actif" if settings and settings["enabled"] else "désactivé",
            "pac": "configuré" if settings and settings["pac"] else "absent",
        },
        "client_sortant": "httpx.AsyncClient (aucun import requests dans cette passerelle)",
    }


@app.post("/v1/qlicker/parse-url")
async def parse_url(payload: RawUrlRequest) -> dict[str, Any]:
    """Étape 1 : le parser URL fournit endpoint et paramètres éditables."""
    endpoint, parameters = parse_url_for_gateway(payload.url)
    # Valider ici évite de présenter une URL externe comme si elle était Qlicker.
    validate_internal_endpoint(endpoint)
    return {"endpoint": endpoint, "parameters": parameters, "count": len(parameters)}


@app.post("/v1/qlicker/get")
async def qlicker_get(payload: QlickerGetRequest) -> dict[str, Any]:
    """Étape 2 : GET asynchrone HTTPX depuis FastAPI vers Qlicker interne."""
    target, host, port = validate_internal_endpoint(payload.endpoint)
    proxy_mapping, proxy_mode, proxy_notes = system_proxy_mapping(target, payload.use_system_proxy)
    timeout = httpx.Timeout(
        connect=payload.connect_timeout_seconds,
        read=payload.read_timeout_seconds,
        write=payload.read_timeout_seconds,
        pool=payload.connect_timeout_seconds,
    )
    parameters = [(item.name, item.value) for item in payload.parameters]
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            mounts=httpx_mounts(proxy_mapping),
            trust_env=bool(payload.use_system_proxy),
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            LOGGER.info("HTTPX GET | host=%s | port=%s | params=%s | proxy=%s", host, port, [name for name, _ in parameters], proxy_mode)
            response = await client.get(target, params=parameters)
    except httpx.HTTPError as exc:
        LOGGER.exception("HTTPX GET Qlicker échoué")
        raise HTTPException(
            status_code=502,
            detail={
                "type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "proxy_mode": proxy_mode,
                "proxy": {name: mask_proxy_url(value) for name, value in (proxy_mapping or {}).items()},
                "proxy_notes": proxy_notes,
            },
        ) from exc

    return {
        "execution": "FastAPI local → HTTPX → proxy éventuel → Qlicker",
        "target": f"{host}:{port}",
        "proxy_mode": proxy_mode,
        "proxy": {name: mask_proxy_url(value) for name, value in (proxy_mapping or {}).items()},
        "proxy_notes": proxy_notes,
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "body": response_body(response),
    }


if __name__ == "__main__":
    # Par défaut, uniquement le PC interne qui exécute le script peut joindre FastAPI.
    uvicorn.run(app, host=os.environ.get("QLICKER_GATEWAY_HOST", "127.0.0.1"), port=8120)
