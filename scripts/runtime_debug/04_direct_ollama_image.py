"""Envoie directement une image au SDK Ollama, sans le runner applicatif."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import ollama


## Prompts volontairement courts pour tester le transport de l'image avant
## d'introduire le prompt CNI complet et ses nombreuses règles.
DEFAULT_SYSTEM = (
    "You are a vision OCR engine. Read only visible information. "
    "Never invent values. Return only the requested result."
)
DEFAULT_USER = "Analyze this image and return the visible text."


def plain_value(value: Any) -> Any:
    """Convertit une réponse Ollama en dictionnaire JSON."""

    ## Réponse Pydantic des versions récentes du SDK.
    if hasattr(value, "model_dump"):
        return value.model_dump()
    ## Compatibilité avec les versions retournant déjà un dictionnaire.
    if isinstance(value, dict):
        return value
    ## Conversion de secours pour garantir l'écriture du rapport.
    return json.loads(json.dumps(value, default=str))


def read_text(direct_value: str, file_path: Path | None) -> str:
    """Lit une consigne directe ou un fichier UTF-8."""

    ## Un fichier est prioritaire lorsqu'il est fourni ; sinon la valeur CLI est utilisée.
    return file_path.read_text(encoding="utf-8") if file_path else direct_value


def build_messages(
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    image_only: bool,
) -> list[dict]:
    """Construit exactement les messages transmis à ``client.chat``."""

    ## Mode expérimental : aucune instruction texte, seulement l'image.
    ## Il sert à tester un Modelfile qui contient déjà toutes les consignes système.
    if image_only:
        return [{"role": "user", "content": "", "images": [str(image_path.resolve())]}]
    ## Liste ordonnée des messages comprise par l'API chat d'Ollama.
    messages = []
    ## Le rôle system fixe les règles générales si la consigne n'est pas vide.
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    ## Le rôle user porte la tâche et le chemin absolu de l'image à encoder.
    messages.append(
        {
            "role": "user",
            "content": user_prompt.strip(),
            "images": [str(image_path.resolve())],
        }
    )
    return messages


def run_direct_call(args: argparse.Namespace) -> dict:
    """Mesure uniquement le client Ollama et affiche les chunks si demandé."""

    ## Ce client contourne Gradio et BenchmarkRunner : il isole le SDK Ollama.
    ## `timeout` contrôle la requête HTTP ; `trust_env` contrôle les proxy système.
    client = ollama.Client(
        host=args.host,
        timeout=float(args.timeout),
        trust_env=not args.ignore_env_proxy,
    )
    ## Résolution des prompts, qu'ils viennent du terminal ou d'un fichier long.
    system_prompt = read_text(args.system, args.system_file)
    user_prompt = read_text(args.prompt, args.prompt_file)
    messages = build_messages(args.image, system_prompt, user_prompt, args.image_only)
    ## Options d'inférence réellement envoyées au serveur Ollama.
    options = {
        ## 0 réduit la variabilité et facilite la reproduction d'un test OCR.
        "temperature": float(args.temperature),
        ## Taille maximale du contexte : prompt + image encodée + sortie.
        "num_ctx": int(args.num_ctx),
        ## Nombre maximal de tokens que le modèle peut générer.
        "num_predict": int(args.num_predict),
    }
    ## num_thread est facultatif : absent, Ollama choisit lui-même la valeur.
    if args.num_thread:
        options["num_thread"] = int(args.num_thread)
    ## Corps final remis à `client.chat`.
    request = {
        "model": args.model,
        "messages": messages,
        "stream": bool(args.stream),
        "options": options,
    }
    ## format=json demande à Ollama d'appliquer son mode de sortie JSON.
    if args.output_format == "json":
        request["format"] = "json"
    ## Certains modèles récents acceptent un niveau de raisonnement.
    if args.think != "off":
        request["think"] = True if args.think == "true" else args.think

    ## Ces logs affichent les valeurs effectives juste avant l'appel.
    print("Appel direct Ollama démarré.", flush=True)
    print(f"host={args.host}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"timeout_http={args.timeout}s", flush=True)
    print(f"stream={args.stream} image_only={args.image_only}", flush=True)
    ## Point zéro commun pour latence totale et arrivée de chaque fragment.
    started = time.perf_counter()

    ## Mode non streaming : `chat` rend la main uniquement à la réponse finale.
    if not args.stream:
        response = client.chat(**request)
        elapsed = time.perf_counter() - started
        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "request": {
                **request,
                "messages": messages,
            },
            "response": plain_value(response),
        }

    ## Mode streaming : les fragments sont affichés et accumulés en direct.
    content_parts = []
    thinking_parts = []
    chunk_count = 0
    last_chunk = None
    first_chunk_seconds = None
    last_chunk_seconds = None
    try:
        ## Le générateur produit un fragment chaque fois qu'Ollama en envoie un.
        for chunk in client.chat(**request):
            chunk_count += 1
            chunk_elapsed = time.perf_counter() - started
            ## Le premier fragment mesure le "time to first token/chunk".
            if first_chunk_seconds is None:
                first_chunk_seconds = chunk_elapsed
            last_chunk_seconds = chunk_elapsed

            ## Selon le modèle, le texte peut arriver dans thinking ou content.
            last_chunk = plain_value(chunk)
            message = last_chunk.get("message", {}) if isinstance(last_chunk, dict) else {}
            thinking = message.get("thinking") or ""
            content = message.get("content") or ""
            print(
                f"\n[fragment {chunk_count} reçu à {chunk_elapsed:.2f}s] "
                f"thinking={len(thinking)} content={len(content)}",
                flush=True,
            )
            if thinking:
                thinking_parts.append(thinking)
                print(thinking, end="", flush=True)
            if content:
                content_parts.append(content)
                print(content, end="", flush=True)
    except Exception as exc:
        ## Même en cas de coupure, on garde ce que le modèle avait déjà envoyé.
        return {
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "request": {
                **request,
                "messages": messages,
            },
            "stream": {
                "chunks": chunk_count,
                "first_chunk_seconds": first_chunk_seconds,
                "last_chunk_seconds": last_chunk_seconds,
                "thinking": "".join(thinking_parts),
                "content": "".join(content_parts),
                "last_chunk": last_chunk,
            },
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
    ## Saut de ligne après l'affichage progressif des fragments.
    print(flush=True)
    return {
        "status": "success",
        "elapsed_seconds": time.perf_counter() - started,
        "request": {
            **request,
            "messages": messages,
        },
        "stream": {
            "chunks": chunk_count,
            "first_chunk_seconds": first_chunk_seconds,
            "last_chunk_seconds": last_chunk_seconds,
            "thinking": "".join(thinking_parts),
            "content": "".join(content_parts),
            "last_chunk": last_chunk,
        },
    }


def main() -> None:
    ## Paramètres du test direct. Chaque option correspond à une hypothèse testable.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image préparée à envoyer au modèle.")
    parser.add_argument("--model", required=True, help="Nom Ollama exact, avec son tag.")
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Adresse du serveur Ollama.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Timeout HTTP explicite en secondes.",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="Prompt système direct.")
    parser.add_argument("--system-file", type=Path, help="Fichier UTF-8 remplaçant --system.")
    parser.add_argument("--prompt", default=DEFAULT_USER, help="Prompt utilisateur direct.")
    parser.add_argument("--prompt-file", type=Path, help="Fichier UTF-8 remplaçant --prompt.")
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Envoie l'image sans texte ; utile avec un SYSTEM intégré au Modelfile.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Affiche les fragments dès leur réception.",
    )
    parser.add_argument(
        "--think",
        choices=["off", "true", "low", "medium", "high"],
        default="off",
        help="Active ou règle le raisonnement si le modèle le supporte.",
    )
    parser.add_argument(
        "--output-format",
        choices=["prompt", "json"],
        default="prompt",
        help="`json` utilise format=json ; `prompt` laisse le prompt guider la sortie.",
    )
    parser.add_argument("--temperature", type=float, default=0, help="Aléatoire de génération.")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Fenêtre de contexte.")
    parser.add_argument("--num-predict", type=int, default=4096, help="Limite de génération.")
    parser.add_argument("--num-thread", type=int, help="Nombre de threads CPU facultatif.")
    parser.add_argument(
        "--ignore-env-proxy",
        action="store_true",
        help="Ignore les proxy provenant de l'environnement.",
    )
    parser.add_argument("--output", type=Path, help="Chemin du rapport JSON.")
    args = parser.parse_args()

    ## Un nom horodaté empêche d'écraser automatiquement un diagnostic précédent.
    output = args.output or (
        Path("runs")
        / "runtime_debug"
        / f"direct_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    ## Chronomètre externe : il mesure également une erreur avant l'entrée dans l'appel.
    wall_started = time.perf_counter()
    try:
        ## Échec immédiat et explicite si le chemin image est incorrect.
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        report = run_direct_call(args)
    except Exception as exc:
        ## Les erreurs non streaming et de préparation sont elles aussi sérialisées.
        report = {
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - wall_started,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
        print(
            f"ÉCHEC {report['exception_type']}: {report['exception']}",
            flush=True,
        )
    ## Le rapport est écrit avant de retourner un code d'échec au terminal.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Rapport écrit dans : {output}")
    ## Un exit code 1 permet aux scripts automatisés de détecter l'échec.
    if report["status"] != "success":
        raise SystemExit(1)


## Point d'entrée du script autonome.
if __name__ == "__main__":
    main()
