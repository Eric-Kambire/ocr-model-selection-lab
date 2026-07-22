"""Mini laboratoire Gradio : coller une URL Qlicker, éditer ses paramètres, lancer GET.

Le script reproduit uniquement l'étape utile de Postman pour les API GET :
analyse d'une URL, édition des query parameters et visualisation de la réponse.
Il n'écrit aucun document ni résultat dans le benchmark.

Lancement :
    python scripts/qlicker_url_parser_lab.py
"""

from __future__ import annotations

import json
import socket
import time
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


def mask_proxy_url(proxy_url: str) -> str:
    """Masque un éventuel mot de passe présent dans une URL de proxy.

    Exemple : ``http://alice:secret@proxy.local:8080`` devient
    ``http://alice:***@proxy.local:8080``. Le journal de l'interface reste
    donc exploitable sans exposer involontairement un secret.
    """
    candidate = str(proxy_url or "").strip()
    if not candidate:
        return ""
    parsed = urlsplit(candidate)
    if not parsed.scheme or not parsed.netloc:
        return "(proxy invalide)"
    if parsed.password is None:
        return candidate
    username = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{username}:***@{host}{port}"


def explicit_proxy_mapping(proxy_url: str) -> dict[str, str] | None:
    """Valide un proxy explicite et le prépare pour HTTP et HTTPS.

    Requests exige une URL complète, protocole inclus. Le support SOCKS est
    possible seulement si ``requests[socks]`` est installé ; ce laboratoire
    privilégie donc les proxys HTTP/HTTPS d'entreprise.
    """
    candidate = str(proxy_url or "").strip()
    if not candidate:
        return None
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Le proxy explicite doit ressembler à http://proxy.entreprise.local:8080")
    return {"http": candidate, "https": candidate}


def windows_proxy_summary() -> dict[str, str]:
    """Retourne un état non sensible du proxy Windows, sans afficher son URL.

    Postman Desktop peut suivre les réglages WinINET de Windows. Requests peut
    détecter un proxy manuel dans certains contextes, mais ne sait pas évaluer
    fiablement un script PAC. On affiche donc le type de réglage plutôt que
    l'URL complète, qui pourrait contenir un identifiant ou un mot de passe.
    """
    try:
        import winreg  # Disponible uniquement sous Windows.

        registry_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path) as key:
            def state(name: str, *, enabled: bool = False) -> str:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    return "absent"
                if enabled:
                    return "actif" if bool(value) else "désactivé"
                return "configuré" if value else "vide"

            return {
                "proxy_manuel": state("ProxyEnable", enabled=True),
                "serveur_proxy": state("ProxyServer"),
                "exceptions_proxy": state("ProxyOverride"),
                "script_pac": state("AutoConfigURL"),
                "detection_automatique": state("AutoDetect", enabled=True),
            }
    except (ImportError, OSError):
        return {"windows": "non disponible (système non Windows ou registre inaccessible)"}


def network_diagnostics(
    endpoint: str,
    connect_timeout_seconds: float,
    use_environment_proxy: bool,
    explicit_proxy_url: str,
) -> tuple[str, str]:
    """Diagnostique DNS et TCP sans envoyer la requête API elle-même.

    Le test TCP est volontairement direct : il vérifie si le PC peut joindre
    l'hôte Qlicker sans proxy. Si Postman passe par un proxy, un échec ici ne
    prouve pas que l'API est indisponible ; il indique simplement que la route
    directe n'est pas utilisable.
    """
    target = str(endpoint or "").strip()
    parsed = urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "### Diagnostic non lancé", "Renseignez un endpoint HTTP/HTTPS valide."

    try:
        explicit_mapping = explicit_proxy_mapping(explicit_proxy_url)
    except ValueError as exc:
        return "### Diagnostic non lancé", str(exc)

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Le diagnostic doit rester rapide, même si l'utilisateur a choisi un long
    # timeout de production pour les réponses de l'API.
    timeout = min(max(float(connect_timeout_seconds or 0), 1.0), 10.0)
    # Cette fonction expose ce que Requests voit effectivement avant envoi.
    # Sous Windows, la détection peut inclure le proxy manuel de WinINET ; un
    # PAC reste toutefois un programme JavaScript qui n'est pas exécuté ici.
    environment_proxies = requests.utils.get_environ_proxies(target)
    report: dict[str, Any] = {
        "hote": parsed.hostname,
        "port": port,
        "timeout_tcp_direct_s": timeout,
        "mode_proxy_demande": (
            f"proxy explicite : {mask_proxy_url(explicit_proxy_url)}"
            if explicit_mapping
            else ("variables Python" if use_environment_proxy else "aucun proxy Python")
        ),
        "variables_proxy_detectees": {
            name: mask_proxy_url(value) for name, value in environment_proxies.items()
        },
        "configuration_proxy_windows": windows_proxy_summary(),
    }
    try:
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        report["dns"] = {"statut": "ok", "adresses": sorted({item[4][0] for item in addresses})}
    except OSError as exc:
        report["dns"] = {"statut": "erreur", "detail": repr(exc)}
        return "### Diagnostic réseau", json.dumps(report, ensure_ascii=False, indent=2)

    started = time.perf_counter()
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            pass
        report["tcp_direct"] = {"statut": "ok", "duree_s": round(time.perf_counter() - started, 3)}
    except OSError as exc:
        report["tcp_direct"] = {
            "statut": "erreur",
            "duree_s": round(time.perf_counter() - started, 3),
            "detail": repr(exc),
        }
    return (
        "### Diagnostic réseau\n"
        "DNS puis connexion TCP directe terminés. Ce test ne traverse pas un proxy Windows/PAC.",
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def execute_get(
    endpoint: str,
    rows: list[list[Any]] | None,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    use_environment_proxy: bool,
    explicit_proxy_url: str,
) -> tuple[str, str, str]:
    """Reconstruit l'URL et exécute un GET sans authentification ni persistance.

    Entrées : endpoint sans query string, tableau éditable, deux délais et proxy.
    Sorties : aperçu de requête, état HTTP, réponse lisible ou diagnostic.
    """
    target = str(endpoint or "").strip()
    parsed = urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "### Requête non envoyée", "### Erreur", "Renseignez une Base URL / endpoint HTTP valide."
    connect_timeout = float(connect_timeout_seconds or 0)
    read_timeout = float(read_timeout_seconds or 0)
    if connect_timeout <= 0 or read_timeout <= 0:
        return "### Requête non envoyée", "### Erreur", "Les deux délais doivent être supérieurs à zéro."

    try:
        explicit_mapping = explicit_proxy_mapping(explicit_proxy_url)
    except ValueError as exc:
        return "### Requête non envoyée", "### Erreur de proxy", str(exc)

    pairs = rows_to_query_pairs(rows)
    proxy_mode = (
        f"proxy explicite : `{mask_proxy_url(explicit_proxy_url)}`"
        if explicit_mapping
        else ("variables proxy Python" if use_environment_proxy else "aucun proxy Python")
    )
    preview = (
        "### GET prévu\n"
        f"- Endpoint : `{target}`\n"
        f"- Paramètres : `{json.dumps(pairs, ensure_ascii=False)}`\n"
        f"- Timeout connexion : `{connect_timeout:g} s`\n"
        f"- Timeout réponse : `{read_timeout:g} s`\n"
        f"- Mode proxy : `{proxy_mode}`"
    )
    started = time.perf_counter()
    try:
        # `trust_env` laisse Requests employer les proxys qu'il détecte dans
        # l'environnement. Il ne sait pas évaluer un script PAC Windows ; le
        # proxy explicite est transmis à l'appel pour dominer cette détection.
        with requests.Session() as session:
            session.trust_env = bool(use_environment_proxy)
            response = session.get(
                target,
                params=pairs,
                timeout=(connect_timeout, read_timeout),
                proxies=explicit_mapping,
            )
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - started
        return (
            preview,
            f"### Erreur réseau après `{elapsed:.1f} s`",
            f"{type(exc).__name__}: {exc}\n\n"
            "Un `ConnectTimeout` signifie que la connexion TCP/TLS n'a pas été établie. "
            "Vérifiez l'hôte, le port, le VPN/réseau interne et le mode proxy. "
            "Le diagnostic DNS/TCP peut aider à isoler la route en cause.",
        )

    elapsed = time.perf_counter() - started

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
        preview + f"\n- URL finale : `{response.url}`",
        f"### Réponse\n- HTTP : `{response.status_code}`\n- Content-Type : `{content_type or 'absent'}`\n- Taille : `{len(response.content):,} octets\n- Durée réelle : `{elapsed:.2f} s`",
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
            connect_timeout = gr.Number(
                label="Timeout connexion (s)",
                value=30,
                precision=0,
                minimum=1,
                info="Temps maximal pour joindre le serveur et établir HTTPS.",
            )
            read_timeout = gr.Number(
                label="Timeout réponse (s)",
                value=300,
                precision=0,
                minimum=1,
                info="Temps maximal après connexion, pendant le traitement Qlicker.",
            )
            use_environment_proxy = gr.Checkbox(
                label="Utiliser le proxy détecté par Python",
                value=True,
                info="Utilise les variables proxy et, selon Windows, le proxy manuel. Un script PAC Windows n'est pas interprété.",
            )
            explicit_proxy_url = gr.Textbox(
                label="Proxy explicite (facultatif)",
                type="password",
                placeholder="http://proxy.entreprise.local:8080",
                info="À recopier depuis Postman si un proxy personnalisé est utilisé. Il remplace les variables proxy.",
            )
            execute_button = gr.Button("Envoyer GET", variant="primary")
        diagnostic_button = gr.Button("Diagnostiquer DNS / TCP", variant="secondary")
        request_preview = gr.Markdown(label="Requête")
        response_status = gr.Markdown(label="Statut")
        response_body = gr.Code(label="Réponse", language="json", lines=18, interactive=False)
        diagnostic_status = gr.Markdown(label="Diagnostic réseau")
        diagnostic_report = gr.Code(label="Rapport DNS / TCP", language="json", lines=14, interactive=False)

        parse_button.click(
            parse_url_to_rows,
            inputs=[raw_url],
            outputs=[endpoint, parameters, parse_status],
            queue=False,
        )
        execute_button.click(
            execute_get,
            inputs=[endpoint, parameters, connect_timeout, read_timeout, use_environment_proxy, explicit_proxy_url],
            outputs=[request_preview, response_status, response_body],
            queue=False,
        )
        diagnostic_button.click(
            network_diagnostics,
            inputs=[endpoint, connect_timeout, use_environment_proxy, explicit_proxy_url],
            outputs=[diagnostic_status, diagnostic_report],
            queue=False,
        )
        gr.Markdown(
            "### Lecture rapide\n"
            "- Coller une URL ne déclenche aucune requête : cela remplit seulement le tableau.  \n"
            "- `param=` est une valeur vide envoyée. Décochez **Envoyer** pour omettre réellement le paramètre.  \n"
            "- Les paramètres en double sont conservés, contrairement à une simple structure dictionnaire.  \n"
            "- Postman peut utiliser un proxy Windows/PAC ; Python Requests utilise ce qu'il détecte ou le proxy explicite ci-dessus. Le rapport indique la différence."
        )
    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=8112, inbrowser=True)
