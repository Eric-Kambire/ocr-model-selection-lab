"""Exécute ollama list/show et affiche les informations du modèle."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import ollama


## Le script est situé deux niveaux sous la racine du dépôt.
## Ajouter cette racine au chemin Python permet de réutiliser le contrôle vision
## de l'application sans copier sa logique.
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

## Import volontairement effectué après l'ajout de ROOT_DIR.
from models.ollama_capabilities import inspect_ollama_vision_capability


def plain_value(value: Any) -> Any:
    """Convertit une réponse SDK Ollama en types JSON simples."""

    ## Les versions récentes du SDK retournent des modèles Pydantic.
    if hasattr(value, "model_dump"):
        return value.model_dump()
    ## Les anciennes versions peuvent déjà retourner un dictionnaire.
    if isinstance(value, dict):
        return value
    ## Dernier recours : conversion tolérante pour pouvoir écrire le diagnostic.
    return json.loads(json.dumps(value, default=str))


def model_names(response: Any) -> list[str]:
    """Extrait les noms quelle que soit la version du SDK."""

    ## Normalise d'abord la réponse, puis accepte les clés `model` et `name`.
    payload = plain_value(response)
    values = payload.get("models", []) if isinstance(payload, dict) else []
    return [
        str(item.get("model") or item.get("name"))
        for item in values
        if isinstance(item, dict) and (item.get("model") or item.get("name"))
    ]


def inspect_model(
    host: str,
    model_name: str | None,
    timeout_seconds: float,
    ignore_environment_proxy: bool,
) -> dict:
    """Liste les modèles puis inspecte le modèle demandé."""

    ## Client identique au SDK employé dans l'application.
    ## Le timeout est explicite : aucune valeur implicite de HTTPX n'est utilisée.
    client = ollama.Client(
        host=host,
        timeout=float(timeout_seconds),
        trust_env=not ignore_environment_proxy,
    )
    ## Mesure séparée de `list()` : ce premier appel ne charge pas le modèle en mémoire.
    list_started = time.perf_counter()
    listed = client.list()
    list_elapsed = time.perf_counter() - list_started
    names = model_names(listed)
    ## Si --model est absent, le premier modèle installé sert uniquement au diagnostic.
    selected = model_name or (names[0] if names else None)
    report = {
        "host": host,
        "http_timeout_seconds": timeout_seconds,
        "trust_environment": not ignore_environment_proxy,
        "list_elapsed_seconds": list_elapsed,
        "models": names,
        "selected_model": selected,
    }
    ## Cas normal d'une installation Ollama fonctionnelle mais sans modèle téléchargé.
    if not selected:
        report["status"] = "no_model"
        return report

    ## `show()` retourne le Modelfile, les paramètres, les familles et les capacités.
    show_started = time.perf_counter()
    shown = client.show(model=selected)
    report["show_elapsed_seconds"] = time.perf_counter() - show_started
    report["show"] = plain_value(shown)
    ## Cette fonction applique le même garde-fou LLM/VLM que l'application.
    capability = inspect_ollama_vision_capability(client, selected)
    report["vision"] = {
        "supported": capability.supported,
        "capabilities": list(capability.capabilities),
        "reason": capability.reason,
    }
    report["status"] = "success"
    return report


def main() -> None:
    ## Paramètres modifiables depuis le terminal.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Nom exact ; sinon le premier modèle installé est utilisé.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Adresse Ollama ; OLLAMA_HOST est prioritaire sur la valeur locale.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Timeout HTTP en secondes pour list/show.",
    )
    parser.add_argument(
        "--ignore-env-proxy",
        action="store_true",
        help="Ignore les variables proxy du système pendant ce diagnostic.",
    )
    parser.add_argument("--output", type=Path, help="Rapport JSON facultatif.")
    args = parser.parse_args()

    ## Toute exception est convertie en rapport : le terminal garde la cause complète.
    try:
        report = inspect_model(
            args.host,
            args.model,
            args.timeout,
            args.ignore_env_proxy,
        )
    except Exception as exc:
        report = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
    ## Affichage humain et sauvegarde facultative du même contenu.
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"\nRapport écrit dans : {args.output}")


## Point d'entrée du script autonome.
if __name__ == "__main__":
    main()
