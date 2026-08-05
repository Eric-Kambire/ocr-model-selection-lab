"""Teste le serveur Ollama au niveau HTTP, sans charger de modèle."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import httpx


def request_endpoint(
    client: httpx.Client,
    endpoint: str,
) -> dict:
    """Mesure un endpoint et conserve le type exact d'exception."""

    ## perf_counter mesure une durée monotone et précise, indépendante de l'heure système.
    started = time.perf_counter()
    try:
        ## GET envoie une requête légère : aucune image et aucun modèle ne sont chargés.
        response = client.get(endpoint)
        elapsed = time.perf_counter() - started
        ## Une réponse 4xx/5xx devient une exception explicite au lieu d'un faux succès.
        response.raise_for_status()
        return {
            "endpoint": endpoint,
            "status": "success",
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "body": response.json(),
        }
    except Exception as exc:
        ## La traceback permet de distinguer DNS, proxy, connexion TCP et timeout.
        return {
            "endpoint": endpoint,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }


def run_health_check(
    host: str,
    timeout_seconds: float,
    ignore_environment_proxy: bool,
) -> dict:
    """Teste version et liste des modèles avec un timeout HTTPX explicite."""

    ## Ce timeout est appliqué explicitement à connexion, lecture, écriture et pool.
    timeout = httpx.Timeout(float(timeout_seconds))
    ## base_url évite de répéter l'hôte pour chaque endpoint.
    ## trust_env=False neutralise HTTP_PROXY/HTTPS_PROXY/ALL_PROXY pour ce test.
    with httpx.Client(
        base_url=host.rstrip("/"),
        timeout=timeout,
        trust_env=not ignore_environment_proxy,
    ) as client:
        ## Les deux appels partagent exactement le même client et la même configuration.
        return {
            "host": str(client.base_url),
            "timeout": str(client.timeout),
            "trust_environment": not ignore_environment_proxy,
            "tests": [
                request_endpoint(client, "/api/version"),
                request_endpoint(client, "/api/tags"),
            ],
        }


def main() -> None:
    ## Définition des paramètres disponibles dans le terminal.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Adresse du serveur Ollama ; utilise OLLAMA_HOST si elle existe.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Durée HTTP maximale en secondes pour chaque endpoint.",
    )
    parser.add_argument(
        "--ignore-env-proxy",
        action="store_true",
        help="Passe trust_env=False à HTTPX pour ignorer les variables proxy.",
    )
    parser.add_argument("--output", type=Path, help="Chemin facultatif du rapport JSON.")
    args = parser.parse_args()

    ## Exécution puis transformation du dictionnaire en JSON lisible.
    report = run_health_check(args.host, args.timeout, args.ignore_env_proxy)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    ## Le dossier parent est créé seulement lorsqu'un fichier de sortie est demandé.
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"\nRapport écrit dans : {args.output}")


## Point d'entrée lorsque la commande `python 01_network_health.py` est utilisée.
if __name__ == "__main__":
    main()
