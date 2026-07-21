"""Laboratoire pédagogique pour comprendre puis connecter les quatre API Qlicker.

Ce script est volontairement indépendant du benchmark. Il permet de vérifier une
route à la fois, d'observer la requête réellement envoyée et de télécharger le
document retourné. Aucune route Qlicker n'est inventée : les quatre modèles
d'URL restent modifiables dans l'interface.

Lancement :
    python scripts/qlicker_api_lab.py
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr
import requests


# Les téléchargements de cette session sont isolés dans un dossier temporaire.
# Le dossier est supprimé à la fermeture du script : aucun document n'est ajouté
# au dépôt, au dataset ou aux runs de benchmark.
SESSION_DIR = Path(tempfile.mkdtemp(prefix="qlicker_api_lab_"))
atexit.register(lambda: shutil.rmtree(SESSION_DIR, ignore_errors=True))


@dataclass(frozen=True)
class ApiSettings:
    """Configuration réutilisée pour un appel HTTP Qlicker.

    Les routes utilisent des placeholders : `{client_id}` et `{document_id}`.
    Elles sont remplacées et encodées avant l'appel réseau.
    """

    base_url: str
    token: str
    auth_header_name: str
    auth_prefix: str
    timeout_seconds: float


def _resolve_token(token_from_ui: str, token_env_name: str) -> str:
    """Prend le token saisi pour cette session, sinon lit une variable d'environnement.

    Entrées : token temporaire de Gradio, nom de variable d'environnement.
    Sortie : token non vide, jamais affiché ni enregistré sur disque.
    """
    direct_token = str(token_from_ui or "").strip()
    if direct_token:
        return direct_token
    variable_name = str(token_env_name or "QLICKER_API_TOKEN").strip()
    return os.environ.get(variable_name, "").strip()


def _settings(
    base_url: str,
    token_from_ui: str,
    token_env_name: str,
    auth_header_name: str,
    auth_prefix: str,
    timeout_seconds: float,
) -> ApiSettings:
    """Valide les paramètres généraux avant tout appel HTTP."""
    normalized_url = str(base_url or "").strip().rstrip("/")
    if not normalized_url.startswith(("https://", "http://")):
        raise ValueError("Base URL invalide : elle doit commencer par http:// ou https://")
    token = _resolve_token(token_from_ui, token_env_name)
    if not token:
        variable_name = str(token_env_name or "QLICKER_API_TOKEN").strip()
        raise ValueError(f"Token absent : saisissez-le pour cette session ou définissez ${variable_name}.")
    if not str(auth_header_name or "").strip():
        raise ValueError("Le nom du header d'authentification est obligatoire.")
    if float(timeout_seconds) <= 0:
        raise ValueError("Le timeout doit être strictement positif.")
    return ApiSettings(
        base_url=normalized_url,
        token=token,
        auth_header_name=str(auth_header_name).strip(),
        auth_prefix=str(auth_prefix or ""),
        timeout_seconds=float(timeout_seconds),
    )


def _format_route(route_template: str, **identifiers: str) -> str:
    """Remplace et encode les identifiants dans un modèle de route.

    Exemple : `/clients/{client_id}` devient `/clients/abc%201`.
    Un placeholder non fourni est une erreur explicite, pas une requête ambiguë.
    """
    route = str(route_template or "").strip()
    if not route:
        raise ValueError("Le modèle de route est vide.")
    encoded = {name: quote(str(value or ""), safe="") for name, value in identifiers.items()}
    try:
        return route.format(**encoded)
    except KeyError as exc:
        raise ValueError(f"Placeholder absent dans la route : {exc.args[0]}") from exc


def _request_preview(url: str, params: dict[str, Any], headers: dict[str, str]) -> str:
    """Construit une vue lisible de la requête sans révéler le token."""
    safe_headers = dict(headers)
    for key in list(safe_headers):
        if key.lower() in {"authorization", "x-api-key", "api-key"}:
            safe_headers[key] = "•••••••• (masqué)"
    return (
        "### Requête envoyée\n"
        f"- **Méthode :** `GET`\n"
        f"- **URL :** `{url}`\n"
        f"- **Query params :** `{json.dumps(params, ensure_ascii=False)}`\n"
        f"- **Headers :** `{json.dumps(safe_headers, ensure_ascii=False)}`"
    )


def _perform_get(
    settings: ApiSettings,
    route_template: str,
    *,
    params: dict[str, Any] | None = None,
    **identifiers: str,
) -> tuple[requests.Response, str]:
    """Exécute un GET et retourne la réponse avec son aperçu masqué.

    Les erreurs HTTP 4xx/5xx ne sont pas perdues : la réponse est retournée afin
    que l'interface affiche le code, le contenu et les éventuels détails JSON.
    """
    route = _format_route(route_template, **identifiers)
    url = f"{settings.base_url}/{route.lstrip('/')}"
    headers = {settings.auth_header_name: f"{settings.auth_prefix}{settings.token}".strip()}
    clean_params = {key: value for key, value in (params or {}).items() if value not in (None, "")}
    response = requests.get(url, headers=headers, params=clean_params, timeout=settings.timeout_seconds)
    return response, _request_preview(url, clean_params, headers)


def _response_payload(response: requests.Response) -> tuple[str, str]:
    """Transforme une réponse JSON ou texte en contenu affichable dans Gradio."""
    content_type = response.headers.get("content-type", "").lower()
    status = f"### Réponse\n- **HTTP :** `{response.status_code}`\n- **Content-Type :** `{content_type or 'absent'}`\n- **Taille :** `{len(response.content):,} octets`"
    if "json" in content_type:
        try:
            return status, json.dumps(response.json(), ensure_ascii=False, indent=2)
        except ValueError:
            return status, response.text
    return status, response.text[:20_000] or "(réponse sans corps)"


def _call_json_endpoint(
    base_url: str, token_from_ui: str, token_env_name: str,
    auth_header_name: str, auth_prefix: str, timeout_seconds: float,
    route_template: str, params: dict[str, Any], **identifiers: str,
) -> tuple[str, str, str]:
    """Gère les trois routes JSON et convertit les exceptions en diagnostic UI."""
    try:
        settings = _settings(base_url, token_from_ui, token_env_name, auth_header_name, auth_prefix, timeout_seconds)
        response, preview = _perform_get(settings, route_template, params=params, **identifiers)
        status, payload = _response_payload(response)
        return preview, status, payload
    except (ValueError, requests.RequestException) as exc:
        return "### Requête non envoyée", f"### Erreur\n`{type(exc).__name__}: {exc}`", ""


def list_clients(
    base_url: str, token_from_ui: str, token_env_name: str,
    auth_header_name: str, auth_prefix: str, timeout_seconds: float,
    route_template: str, step: str, page: int, page_size: int,
) -> tuple[str, str, str]:
    """Étape 1 : liste paginée des clients, avec filtre `step` facultatif."""
    return _call_json_endpoint(
        base_url, token_from_ui, token_env_name, auth_header_name, auth_prefix, timeout_seconds,
        route_template, {"step": step, "page": int(page), "pageSize": int(page_size)},
    )


def get_client_info(
    base_url: str, token_from_ui: str, token_env_name: str,
    auth_header_name: str, auth_prefix: str, timeout_seconds: float,
    route_template: str, client_id: str,
) -> tuple[str, str, str]:
    """Étape 2 : récupère les informations d'un client identifié."""
    return _call_json_endpoint(
        base_url, token_from_ui, token_env_name, auth_header_name, auth_prefix, timeout_seconds,
        route_template, {}, client_id=client_id,
    )


def list_documents(
    base_url: str, token_from_ui: str, token_env_name: str,
    auth_header_name: str, auth_prefix: str, timeout_seconds: float,
    route_template: str, client_id: str,
) -> tuple[str, str, str]:
    """Étape 3 : récupère la liste de documents d'un client."""
    return _call_json_endpoint(
        base_url, token_from_ui, token_env_name, auth_header_name, auth_prefix, timeout_seconds,
        route_template, {}, client_id=client_id,
    )


def view_document(
    base_url: str, token_from_ui: str, token_env_name: str,
    auth_header_name: str, auth_prefix: str, timeout_seconds: float,
    route_template: str, client_id: str, document_id: str,
) -> tuple[str, str, str, str | None]:
    """Étape 4 : télécharge un PDF ou une image et expose un fichier Gradio."""
    try:
        settings = _settings(base_url, token_from_ui, token_env_name, auth_header_name, auth_prefix, timeout_seconds)
        response, preview = _perform_get(settings, route_template, client_id=client_id, document_id=document_id)
        status, payload = _response_payload(response)
        if not response.ok:
            return preview, status, payload, None
        content_type = response.headers.get("content-type", "").lower()
        extension = ".pdf" if "pdf" in content_type else ".jpg" if "jpeg" in content_type else ".png" if "png" in content_type else ".bin"
        output_path = SESSION_DIR / f"document_{quote(str(document_id), safe='')}{extension}"
        output_path.write_bytes(response.content)
        return preview, status, f"Document téléchargé temporairement : `{output_path.name}`", str(output_path)
    except (ValueError, requests.RequestException, OSError) as exc:
        return "### Requête non envoyée", f"### Erreur\n`{type(exc).__name__}: {exc}`", "", None


def build_ui() -> gr.Blocks:
    """Construit l'interface pédagogique sans dépendre de l'application principale."""
    # L'interface reste volontairement native et compacte : elle doit servir à
    # lire le protocole HTTP, pas à reproduire l'interface du benchmark.
    with gr.Blocks(title="Qlicker API Lab", fill_width=True) as app:
        with gr.Column(elem_classes=["api-lab"]):
            gr.Markdown(
                "# Qlicker API Lab\n"
                "Un appel à la fois : configure la route, envoie la requête, puis lis la réponse avant de passer à l'étape suivante. "
                "Le token reste dans la session Gradio et est toujours masqué dans l'aperçu."
            )
            with gr.Accordion("0. Connexion et modèles de routes", open=True):
                gr.Markdown(
                    "Les chemins ci-dessous sont des **hypothèses éditables**. Remplace-les par les routes exactes données par Qlicker. "
                    "`{client_id}` et `{document_id}` seront remplacés automatiquement."
                )
                with gr.Row():
                    base_url = gr.Textbox(label="Base URL", placeholder="https://api.qlicker.example/v1")
                    token_env = gr.Textbox(label="Variable token", value="QLICKER_API_TOKEN")
                    timeout = gr.Number(label="Timeout (secondes)", value=30, precision=0)
                with gr.Row():
                    token = gr.Textbox(label="Token de session (optionnel)", type="password", placeholder="Sinon lecture de QLICKER_API_TOKEN")
                    auth_header = gr.Textbox(label="Nom du header", value="Authorization")
                    auth_prefix = gr.Textbox(label="Préfixe", value="Bearer ")
                clients_route = gr.Textbox(label="1. Liste clients", value="/clients")
                client_route = gr.Textbox(label="2. Info client", value="/clients/{client_id}")
                documents_route = gr.Textbox(label="3. Liste documents", value="/clients/{client_id}/documents")
                document_route = gr.Textbox(label="4. Voir / télécharger document", value="/clients/{client_id}/documents/{document_id}/view")

            def outputs() -> tuple[gr.Markdown, gr.Markdown, gr.Code]:
                return (
                    gr.Markdown(label="Requête sûre"),
                    gr.Markdown(label="Statut HTTP"),
                    gr.Code(label="Corps de réponse", language="json", lines=16),
                )

            with gr.Tabs():
                with gr.Tab("1. Liste clients"):
                    gr.Markdown("**But :** explorer les clients par statut (`step`) et pagination. Commence avec `page=1`, `pageSize=10`.")
                    with gr.Row():
                        step = gr.Textbox(label="step / statut", placeholder="Ex. validated")
                        page = gr.Number(label="page", value=1, precision=0)
                        page_size = gr.Number(label="pageSize", value=10, precision=0)
                    list_button = gr.Button("Envoyer GET liste clients", variant="primary")
                    request_1, status_1, body_1 = outputs()
                    list_button.click(list_clients, [base_url, token, token_env, auth_header, auth_prefix, timeout, clients_route, step, page, page_size], [request_1, status_1, body_1])

                with gr.Tab("2. Informations client"):
                    gr.Markdown("**But :** vérifier la structure d'un client avant de chercher ses documents.")
                    client_id_2 = gr.Textbox(label="client_id")
                    client_button = gr.Button("Envoyer GET info client", variant="primary")
                    request_2, status_2, body_2 = outputs()
                    client_button.click(get_client_info, [base_url, token, token_env, auth_header, auth_prefix, timeout, client_route, client_id_2], [request_2, status_2, body_2])

                with gr.Tab("3. Liste documents"):
                    gr.Markdown("**But :** récupérer les `document_id` et le type de chaque document du client.")
                    client_id_3 = gr.Textbox(label="client_id")
                    documents_button = gr.Button("Envoyer GET liste documents", variant="primary")
                    request_3, status_3, body_3 = outputs()
                    documents_button.click(list_documents, [base_url, token, token_env, auth_header, auth_prefix, timeout, documents_route, client_id_3], [request_3, status_3, body_3])

                with gr.Tab("4. Voir un document"):
                    gr.Markdown("**But :** télécharger le PDF, JPEG ou PNG retourné. Le fichier est temporaire et disparaît lorsque le laboratoire est arrêté.")
                    with gr.Row():
                        client_id_4 = gr.Textbox(label="client_id")
                        document_id_4 = gr.Textbox(label="document_id")
                    document_button = gr.Button("Envoyer GET document", variant="primary")
                    request_4, status_4, body_4 = outputs()
                    downloaded_document = gr.File(label="Document reçu", interactive=False)
                    document_button.click(view_document, [base_url, token, token_env, auth_header, auth_prefix, timeout, document_route, client_id_4, document_id_4], [request_4, status_4, body_4, downloaded_document])

            gr.Markdown(
                "### Lecture rapide\n"
                "1. **HTTP 200** : lis le JSON et relève les clés utiles.  \\n"
                "2. **HTTP 401/403** : le token ou le header d'authentification ne correspond pas.  \\n"
                "3. **HTTP 404** : vérifie le chemin ou l'identifiant.  \\n"
                "4. Une fois les quatre réponses confirmées, nous créerons le connecteur de production avec les vraies structures JSON."
            )
    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=8110, inbrowser=True)
