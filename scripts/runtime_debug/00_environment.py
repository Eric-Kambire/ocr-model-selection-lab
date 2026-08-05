"""Affiche l'environnement qui peut influencer les appels Ollama."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


## Liste blanche des variables susceptibles de modifier silencieusement
## l'adresse d'Ollama, son contexte, sa durée de chargement ou le réseau HTTP.
VARIABLES = [
    "OLLAMA_HOST",
    "OLLAMA_LOAD_TIMEOUT",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_CONTEXT_LENGTH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
]


def mask_url(value: str | None) -> str | None:
    """Masque les identifiants éventuels contenus dans une URL de proxy."""

    ## Une variable absente reste absente dans le rapport.
    if not value:
        return value
    ## urlsplit sépare proprement schéma, utilisateur, mot de passe, hôte et port.
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<valeur définie mais URL invalide>"
    ## Une URL sans identifiants peut être affichée telle quelle.
    if not parsed.username and not parsed.password:
        return value
    ## Les identifiants sont remplacés, mais l'adresse utile au diagnostic reste visible.
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, f"***:***@{host}", parsed.path, parsed.query, parsed.fragment))


def package_version(name: str) -> str:
    """Retourne la version installée sans faire échouer le diagnostic."""

    ## importlib.metadata interroge l'environnement Python réellement actif.
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "non installé"


def launchctl_value(name: str) -> str | None:
    """Lit l'environnement transmis aux applications macOS."""

    ## launchctl concerne macOS. Sur Windows et Linux, cette source n'existe pas.
    if platform.system() != "Darwin":
        return None
    ## Une application macOS lancée graphiquement peut recevoir des variables
    ## différentes de celles du terminal : ce test rend cette différence visible.
    completed = subprocess.run(
        ["launchctl", "getenv", name],
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value or None


def collect_environment() -> dict:
    """Construit un rapport sans exposer de clé API."""

    ## Lecture de chaque variable une seule fois depuis launchctl.
    launch_values = {name: launchctl_value(name) for name in VARIABLES}
    ## Variables visibles par le processus Python actuel.
    shell_environment = {
        name: mask_url(os.getenv(name))
        for name in VARIABLES
        if os.getenv(name) is not None
    }
    launch_environment = {
        name: mask_url(value)
        for name, value in launch_values.items()
        if value is not None
    }
    ## Le dictionnaire final est volontairement sérialisable directement en JSON.
    return {
        ## Confirme quel interpréteur et quel environnement virtuel sont utilisés.
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "virtual_env": os.getenv("VIRTUAL_ENV"),
            "cwd": str(Path.cwd()),
        },
        ## Donne l'OS et l'architecture, utiles pour comparer Mac Intel/Apple Silicon.
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        ## Versions qui peuvent expliquer un comportement différent entre deux postes.
        "packages": {
            "ollama": package_version("ollama"),
            "httpx": package_version("httpx"),
            "gradio": package_version("gradio"),
            "pillow": package_version("pillow"),
            "pymupdf": package_version("pymupdf"),
        },
        "shell_environment": shell_environment,
        "launchctl_environment": launch_environment,
        ## Ne jamais écrire la valeur d'une éventuelle clé secrète.
        "ollama_api_key": "définie" if os.getenv("OLLAMA_API_KEY") else "absente",
    }


def main() -> None:
    ## argparse transforme les options du terminal en valeurs Python validées.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Fichier JSON facultatif.")
    args = parser.parse_args()

    ## Le même rapport est affiché dans le terminal pour une lecture immédiate.
    report = collect_environment()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    ## --output permet de garder une preuve comparable entre deux machines.
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"\nRapport écrit dans : {args.output}")


## Ce garde-fou lance main() uniquement lors de l'exécution directe du fichier.
## Il évite une exécution involontaire si le fichier est importé par un autre script.
if __name__ == "__main__":
    main()
