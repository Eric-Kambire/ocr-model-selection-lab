"""Mini laboratoire Gradio : coller une URL Qlicker, éditer ses paramètres, lancer GET.

Le script reproduit uniquement l'étape utile de Postman pour les API GET :
analyse d'une URL, édition des query parameters et visualisation de la réponse.
Il n'écrit aucun document ni résultat dans le benchmark.

Lancement :
    python scripts/qlicker_url_parser_lab.py
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import gradio as gr
import requests


def parse_url_to_rows(raw_url: str) -> tuple[str, list[list[Any]], str]:
    """Découpe une URL GET en endpoint et tableau de paramètres éditables.

    Entrée : URL complète, par exemple
        http://serveur/api/GetCustomers?page=1&pageSize=20&filter=

    Sorties : endpoint sans query string, lignes [nom, valeur, envoyer], message.
    `keep_blank_values=True` préserve `filter=` : une valeur vide est différente
    d'un paramètre absent.
    """
    candidate = str(raw_url or "").strip()
    if not candidate:
        return "", [], "Collez une URL complète pour analyser ses paramètres."
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "", [], "URL invalide : utilisez par exemple http://serveur/api/GetCustomers?page=1"
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    parameters = parse_qsl(parsed.query, keep_blank_values=True)
    rows = [[name, value, True] for name, value in parameters]
    return endpoint, rows, f"URL analysée : {len(rows)} paramètre(s) détecté(s)."


def rows_to_query_pairs(rows: list[list[Any]] | None) -> list[tuple[str, str]]:
    """Transforme le tableau Gradio en paires query string, en gardant les doublons.

    La colonne Envoyer est importante : décocher une ligne supprime le paramètre
    de la requête. Écrire `null` reste le texte `null` ; pour omettre réellement
    le paramètre, décochez la ligne.
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


def execute_get(endpoint: str, rows: list[list[Any]] | None, timeout_seconds: float) -> tuple[str, str, str]:
    """Reconstruit l'URL et exécute un GET sans authentification ni persistance.

    Entrées : endpoint sans query string, tableau éditable, timeout.
    Sorties : aperçu de requête, état HTTP, réponse lisible ou diagnostic.
    """
    target = str(endpoint or "").strip()
    parsed = urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "### Requête non envoyée", "### Erreur", "Renseignez une Base URL / endpoint HTTP valide."
    if float(timeout_seconds or 0) <= 0:
        return "### Requête non envoyée", "### Erreur", "Le timeout doit être supérieur à zéro."

    pairs = rows_to_query_pairs(rows)
    try:
        response = requests.get(target, params=pairs, timeout=float(timeout_seconds))
    except requests.RequestException as exc:
        return (
            f"### GET prévu\n- Endpoint : `{target}`\n- Paramètres : `{json.dumps(pairs, ensure_ascii=False)}`",
            "### Erreur réseau",
            f"{type(exc).__name__}: {exc}",
        )

    content_type = response.headers.get("content-type", "").lower()
    if "json" in content_type:
        try:
            body = json.dumps(response.json(), ensure_ascii=False, indent=2)
        except ValueError:
            body = response.text[:100_000]
    elif content_type.startswith("text/") or not content_type:
        body = response.text[:100_000] or "(réponse sans corps)"
    else:
        body = json.dumps(
            {
                "binary": True,
                "content_type": content_type,
                "bytes": len(response.content),
                "message": "Le fichier n'est pas téléchargé dans ce laboratoire.",
            },
            ensure_ascii=False,
            indent=2,
        )
    return (
        f"### Requête réellement envoyée\n- Méthode : `GET`\n- URL : `{response.url}`\n- Paramètres actifs : `{json.dumps(pairs, ensure_ascii=False)}`",
        f"### Réponse\n- HTTP : `{response.status_code}`\n- Content-Type : `{content_type or 'absent'}`\n- Taille : `{len(response.content):,} octets`",
        body,
    )


def build_ui() -> gr.Blocks:
    """Construit un petit équivalent Postman destiné aux GET Qlicker."""
    with gr.Blocks(title="Qlicker URL Parser Lab", fill_width=True) as app:
        gr.Markdown(
            "# Qlicker URL Parser Lab\n"
            "Collez l'URL fournie par Qlicker. Les paramètres deviennent éditables, puis vous lancez un GET sans enregistrer de document."
        )
        with gr.Row():
            raw_url = gr.Textbox(
                label="URL complète à analyser",
                lines=2,
                placeholder="http://serveur/api/GetCustomers?from_date=2026-01-01&page=1&pageSize=20",
                scale=5,
            )
            parse_button = gr.Button("Analyser l'URL", variant="primary", scale=1)
        parse_status = gr.Markdown("Collez une URL, puis cliquez sur Analyser l'URL.")
        endpoint = gr.Textbox(
            label="Endpoint sans paramètres",
            info="Construit automatiquement ; vous pouvez le modifier si nécessaire.",
        )
        parameters = gr.Dataframe(
            headers=["Paramètre", "Valeur", "Envoyer"],
            datatype=["str", "str", "bool"],
            row_count=(1, "dynamic"),
            column_count=(3, "fixed"),
            interactive=True,
            type="array",
            label="Paramètres query éditables",
        )
        gr.Markdown("Ajoutez les paramètres absents dans une nouvelle ligne. Décochez **Envoyer** pour exclure une ligne.")
        with gr.Row():
            timeout_seconds = gr.Number(label="Timeout (secondes)", value=30, precision=0, minimum=1)
            execute_button = gr.Button("Envoyer GET", variant="primary")
        request_preview = gr.Markdown(label="Requête")
        response_status = gr.Markdown(label="Statut")
        response_body = gr.Code(label="Réponse", language="json", lines=18, interactive=False)

        parse_button.click(
            parse_url_to_rows,
            inputs=[raw_url],
            outputs=[endpoint, parameters, parse_status],
            queue=False,
        )
        execute_button.click(
            execute_get,
            inputs=[endpoint, parameters, timeout_seconds],
            outputs=[request_preview, response_status, response_body],
            queue=False,
        )
        gr.Markdown(
            "### Lecture rapide\n"
            "- Coller une URL ne déclenche aucune requête : cela remplit seulement le tableau.  \n"
            "- `param=` est une valeur vide envoyée. Décochez **Envoyer** pour omettre réellement le paramètre.  \n"
            "- Les paramètres en double sont conservés, contrairement à une simple structure dictionnaire."
        )
    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=8112, inbrowser=True)
