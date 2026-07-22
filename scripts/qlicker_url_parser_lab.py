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
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
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


def _windows_proxy_values() -> dict[str, Any] | None:
    """Lit les valeurs WinINET nécessaires, uniquement sur Windows.

    Cette fonction privée ne journalise jamais l'adresse brute du proxy. Elle
    sert à reproduire, pour un proxy manuel, le choix « System proxy » de
    Postman sur la machine où le script est réellement exécuté.
    """
    try:
        import winreg

        registry_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path) as key:
            def get_value(name: str, default: Any = "") -> Any:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                    return value
                except FileNotFoundError:
                    return default

            return {
                "enabled": bool(get_value("ProxyEnable", 0)),
                "server": str(get_value("ProxyServer", "") or "").strip(),
                "override": str(get_value("ProxyOverride", "") or "").strip(),
                "pac": str(get_value("AutoConfigURL", "") or "").strip(),
                "auto_detect": bool(get_value("AutoDetect", 0)),
            }
    except (ImportError, OSError):
        return None


def _normalise_windows_proxy_address(address: str) -> str:
    """Ajoute le protocole requis par Requests à une adresse WinINET."""
    candidate = str(address or "").strip()
    if not candidate:
        return ""
    return candidate if "://" in candidate else f"http://{candidate}"


def parse_windows_proxy_server(proxy_server: str) -> dict[str, str] | None:
    """Convertit ``ProxyServer`` Windows vers le format ``requests``.

    Windows accepte soit ``proxy.local:8080`` pour tous les protocoles, soit
    ``http=proxy-http:8080;https=proxy-https:8443``. Requests attend un
    dictionnaire par protocole, avec une URL complète.
    """
    raw = str(proxy_server or "").strip()
    if not raw:
        return None
    if "=" not in raw:
        address = _normalise_windows_proxy_address(raw)
        return {"http": address, "https": address}

    mapping: dict[str, str] = {}
    for item in raw.split(";"):
        protocol, separator, address = item.partition("=")
        if not separator:
            continue
        protocol = protocol.strip().lower()
        if protocol in {"http", "https"} and address.strip():
            mapping[protocol] = _normalise_windows_proxy_address(address)
    return mapping or None


def windows_manual_proxy_mapping(endpoint: str) -> dict[str, str] | None:
    """Retourne le proxy manuel Windows applicable à cet endpoint.

    Les exceptions simples de Windows (``localhost``, suffixes et ``<local>``)
    sont respectées. Un PAC reste volontairement hors périmètre : c'est du
    JavaScript dépendant de l'URL, que Requests ne peut pas exécuter seul.
    """
    settings = _windows_proxy_values()
    if not settings or not settings["enabled"]:
        return None
    hostname = (urlsplit(endpoint).hostname or "").lower()
    override = settings["override"]
    if "<local>" in override.lower() and "." not in hostname:
        return None
    no_proxy = ",".join(item.strip() for item in override.split(";") if item.strip() and item.strip() != "<local>")
    if no_proxy and requests.utils.should_bypass_proxies(endpoint, no_proxy=no_proxy):
        return None
    return parse_windows_proxy_server(settings["server"])


def windows_proxy_summary() -> dict[str, str]:
    """Retourne un état non sensible du proxy Windows, sans afficher son URL.

    Postman Desktop peut suivre les réglages WinINET de Windows. Requests peut
    détecter un proxy manuel dans certains contextes, mais ne sait pas évaluer
    fiablement un script PAC. On affiche donc le type de réglage plutôt que
    l'URL complète, qui pourrait contenir un identifiant ou un mot de passe.
    """
    settings = _windows_proxy_values()
    if settings is None:
        return {"windows": "non disponible (système non Windows ou registre inaccessible)"}
    return {
        "proxy_manuel": "actif" if settings["enabled"] else "désactivé",
        "serveur_proxy": "configuré" if settings["server"] else "vide",
        "exceptions_proxy": "configuré" if settings["override"] else "vide",
        "script_pac": "configuré" if settings["pac"] else "absent",
        "detection_automatique": "active" if settings["auto_detect"] else "désactivée",
    }


def _tcp_probe(host: str, port: int, timeout_seconds: float) -> dict[str, Any]:
    """Teste une connexion TCP et retourne toujours un résultat sérialisable."""
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            pass
        return {"statut": "ok", "duree_s": round(time.perf_counter() - started, 3)}
    except OSError as exc:
        return {
            "statut": "erreur",
            "duree_s": round(time.perf_counter() - started, 3),
            "detail": repr(exc),
        }


def _tls_probe(host: str, port: int, timeout_seconds: float) -> dict[str, Any]:
    """Vérifie le handshake TLS et la chaîne de certificats, sans HTTP."""
    started = time.perf_counter()
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout_seconds) as connection:
            with context.wrap_socket(connection, server_hostname=host) as secured:
                certificate = secured.getpeercert()
                return {
                    "statut": "ok",
                    "duree_s": round(time.perf_counter() - started, 3),
                    "tls": secured.version(),
                    "cipher": secured.cipher()[0] if secured.cipher() else "inconnue",
                    "subject": str(certificate.get("subject", "non fourni")),
                    "issuer": str(certificate.get("issuer", "non fourni")),
                }
    except (OSError, ssl.SSLError) as exc:
        return {
            "statut": "erreur",
            "duree_s": round(time.perf_counter() - started, 3),
            "detail": repr(exc),
            "interpretation": "Le réseau est peut-être joignable, mais Python ne fait pas confiance au certificat ou le handshake TLS échoue.",
        }


def _proxy_endpoint(mapping: dict[str, str] | None, scheme: str) -> tuple[str, int] | None:
    """Extrait hôte/port du proxy applicable au protocole demandé."""
    if not mapping:
        return None
    raw_proxy = mapping.get(scheme) or mapping.get("https") or mapping.get("http")
    parsed = urlsplit(raw_proxy or "")
    if not parsed.hostname:
        return None
    return parsed.hostname, parsed.port or 8080


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
    windows_mapping = windows_manual_proxy_mapping(target) if use_environment_proxy else None
    effective_mapping = explicit_mapping or windows_mapping or environment_proxies or None
    report: dict[str, Any] = {
        "hote": parsed.hostname,
        "port": port,
        "timeout_tcp_direct_s": timeout,
        "mode_proxy_demande": (
            f"proxy explicite : {mask_proxy_url(explicit_proxy_url)}"
            if explicit_mapping
            else ("proxy manuel Windows" if windows_mapping else ("proxy détecté par Python" if use_environment_proxy else "aucun proxy"))
        ),
        "variables_proxy_detectees": {
            name: mask_proxy_url(value) for name, value in environment_proxies.items()
        },
        "configuration_proxy_windows": windows_proxy_summary(),
        "proxy_manuel_windows_utilise": sorted(windows_mapping) if windows_mapping else [],
    }
    try:
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        unique_addresses = sorted({item[4][0] for item in addresses})
        report["dns"] = {"statut": "ok", "adresses": unique_addresses}
    except OSError as exc:
        report["dns"] = {"statut": "erreur", "detail": repr(exc)}
        return "### Diagnostic réseau", json.dumps(report, ensure_ascii=False, indent=2)

    # Tester chaque IP explique les cas fréquents où IPv6 échoue alors qu'IPv4
    # fonctionne (ou l'inverse), un détail que Postman masque souvent.
    addresses_to_probe = unique_addresses[:8]
    with ThreadPoolExecutor(max_workers=min(len(addresses_to_probe), 8)) as executor:
        probes = executor.map(lambda address: _tcp_probe(address, port, timeout), addresses_to_probe)
        report["tcp_direct_par_ip"] = dict(zip(addresses_to_probe, probes, strict=True))
    if parsed.scheme == "https":
        report["tls_direct"] = _tls_probe(parsed.hostname, port, timeout)

    proxy_target = _proxy_endpoint(effective_mapping, parsed.scheme)
    if proxy_target:
        proxy_host, proxy_port = proxy_target
        report["tcp_proxy"] = {
            "proxy": f"{proxy_host}:{proxy_port}",
            "resultat": _tcp_probe(proxy_host, proxy_port, timeout),
            "note": "Ce test joint le proxy ; il ne confirme pas encore le tunnel HTTPS vers Qlicker.",
        }
    else:
        report["tcp_proxy"] = {"statut": "non teste", "raison": "aucun proxy utilisable détecté par Python"}

    report["lecture"] = [
        "DNS en erreur : Python ne trouve pas le nom du serveur ; Internet/Postman peuvent fonctionner pour d'autres noms.",
        "TCP direct en erreur : port bloqué, VPN/routage absent, mauvais hôte ou accès autorisé seulement via proxy.",
        "TLS en erreur : certificat interne, inspection TLS d'entreprise ou incompatibilité TLS ; ne désactivez pas verify=False.",
        "TCP proxy en erreur : Python ne peut pas joindre le proxy configuré par Windows/Postman.",
        "TCP/TLS OK mais GET en erreur : comparer URL, paramètres, headers, cookies, certificat client et éventuelle authentification proxy dans Postman.",
    ]
    return (
        "### Diagnostic réseau\n"
        "Diagnostic terminé : chaque étape indique la couche réseau qui échoue. Le GET reste le test final de l'API.",
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

    windows_mapping = windows_manual_proxy_mapping(target) if use_environment_proxy else None
    effective_mapping = explicit_mapping or windows_mapping
    pairs = rows_to_query_pairs(rows)
    proxy_mode = (
        f"proxy explicite : `{mask_proxy_url(explicit_proxy_url)}`"
        if explicit_mapping
        else ("proxy manuel Windows" if windows_mapping else ("proxy détecté par Python" if use_environment_proxy else "aucun proxy"))
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
        # Pour un proxy manuel Windows, on transmet la configuration à la
        # requête : cela reproduit le comportement de Postman sans dépendre
        # du mécanisme de détection implicite de Requests. Un PAC est signalé
        # dans le diagnostic mais ne peut pas être évalué par Requests seul.
        with requests.Session() as session:
            session.trust_env = bool(use_environment_proxy)
            response = session.get(
                target,
                params=pairs,
                timeout=(connect_timeout, read_timeout),
                proxies=effective_mapping,
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
        gr.Markdown(
            "**Pourquoi Postman/navigateur peuvent réussir alors que Python échoue ?**  \n"
            "Le navigateur et Postman peuvent utiliser le proxy Windows, un PAC, des cookies, un certificat client ou des réglages TLS différents. "
            "Le bouton de diagnostic sépare donc DNS, TCP, TLS et proxy au lieu d'afficher seulement « timeout ».")
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
                label="Utiliser le proxy système Windows / Python",
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
