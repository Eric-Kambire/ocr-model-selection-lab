"""Appelle /api/chat avec HTTPX, sans utiliser le SDK Python Ollama."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import httpx


## Prompts courts : cette étape teste d'abord le transport HTTP de l'image.
DEFAULT_SYSTEM = (
    "You are a vision OCR engine. Read only visible information. "
    "Never invent values."
)
DEFAULT_USER = "Analyze this image and return only the visible text."


def encode_image(image_path: Path) -> str:
    """Transforme les octets de l'image en Base64 accepté par l'API Ollama."""

    ## Contrairement au SDK, l'API REST ne peut pas lire notre chemin local.
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def read_text(direct_value: str, file_path: Path | None) -> str:
    """Utilise un texte direct ou le contenu UTF-8 d'un fichier."""

    ## Un fichier permet de tester un prompt long sans l'échapper dans le terminal.
    return file_path.read_text(encoding="utf-8") if file_path else direct_value


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Construit le JSON exact envoyé à POST /api/chat."""

    ## L'encodage est terminé avant le chronomètre réseau.
    encoded_image = encode_image(args.image)
    messages: list[dict[str, Any]] = []
    if not args.image_only:
        system_prompt = read_text(args.system, args.system_file).strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": (
                ""
                if args.image_only
                else read_text(args.prompt, args.prompt_file).strip()
            ),
            "images": [encoded_image],
        }
    )

    ## Paramètres fondamentaux de génération, identiques au test SDK.
    options: dict[str, Any] = {
        "temperature": float(args.temperature),
        "num_ctx": int(args.num_ctx),
        "num_predict": int(args.num_predict),
    }
    if args.num_thread:
        options["num_thread"] = int(args.num_thread)

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": bool(args.stream),
        "options": options,
    }
    if args.output_format == "json":
        payload["format"] = "json"
    if args.think != "off":
        payload["think"] = True if args.think == "true" else args.think
    return payload


def public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Masque l'image Base64 dans la copie écrite dans le rapport."""

    ## Copie superficielle ciblée : on évite de dupliquer la grande chaîne Base64.
    copied = {key: value for key, value in payload.items() if key != "messages"}
    copied["messages"] = []
    for message in payload.get("messages", []):
        public_message = {key: value for key, value in message.items() if key != "images"}
        public_message["images"] = [
            f"<image Base64 masquée : {len(value)} caractères>"
            for value in (message.get("images") or [])
        ]
        copied["messages"].append(public_message)
    return copied


def build_client(args: argparse.Namespace) -> httpx.Client:
    """Crée HTTPX avec quatre timeouts explicites."""

    ## connect : DNS + TCP + TLS.
    ## read : attente entre deux lectures de la réponse.
    ## write : envoi du JSON et de l'image Base64.
    ## pool : attente d'une connexion disponible.
    timeout = httpx.Timeout(
        connect=float(args.connect_timeout),
        read=float(args.read_timeout),
        write=float(args.write_timeout),
        pool=float(args.pool_timeout),
    )
    return httpx.Client(
        base_url=args.host.rstrip("/"),
        timeout=timeout,
        trust_env=not args.ignore_env_proxy,
    )


def run_non_streaming(
    client: httpx.Client,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Attend la réponse finale sans fragments intermédiaires."""

    ## Un intermédiaire réseau peut couper cette connexion jugée inactive.
    started = time.perf_counter()
    response = client.post("/api/chat", json=payload)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return {
        "status": "success",
        "elapsed_seconds": elapsed,
        "http_status": response.status_code,
        "response_headers": dict(response.headers),
        "response": response.json(),
    }


def run_streaming(
    client: httpx.Client,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Lit chaque ligne NDJSON et mesure son instant d'arrivée."""

    ## Ces listes conservent la sortie partielle même si la connexion tombe.
    chunks: list[dict[str, Any]] = []
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    first_chunk_seconds: float | None = None
    last_chunk_seconds: float | None = None
    headers: dict[str, str] = {}
    started = time.perf_counter()

    try:
        ## stream() rend les octets disponibles avant la fin de la génération.
        with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            headers = dict(response.headers)
            for line in response.iter_lines():
                if not line.strip():
                    continue
                elapsed = time.perf_counter() - started
                if first_chunk_seconds is None:
                    first_chunk_seconds = elapsed
                last_chunk_seconds = elapsed

                ## Ollama utilise un objet JSON indépendant par ligne.
                chunk = json.loads(line)
                chunks.append(chunk)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                message = chunk.get("message") or {}
                thinking = str(message.get("thinking") or "")
                content = str(message.get("content") or "")
                thinking_parts.append(thinking)
                content_parts.append(content)
                print(
                    f"[HTTP brut fragment {len(chunks)} à {elapsed:.2f}s] "
                    f"thinking={len(thinking)} content={len(content)}",
                    flush=True,
                )
                if thinking:
                    print(thinking, end="", flush=True)
                if content:
                    print(content, end="", flush=True)
    except Exception as exc:
        ## Le type exact distingue ReadTimeout, coupure TLS et erreur serveur.
        return {
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "first_chunk_seconds": first_chunk_seconds,
            "last_chunk_seconds": last_chunk_seconds,
            "chunks_received": len(chunks),
            "thinking": "".join(thinking_parts),
            "content": "".join(content_parts),
            "chunks": chunks,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }

    return {
        "status": "success",
        "elapsed_seconds": time.perf_counter() - started,
        "first_chunk_seconds": first_chunk_seconds,
        "last_chunk_seconds": last_chunk_seconds,
        "chunks_received": len(chunks),
        "thinking": "".join(thinking_parts),
        "content": "".join(content_parts),
        "chunks": chunks,
        "response_headers": headers,
    }


def main() -> None:
    ## Chaque timeout est réglable séparément pour trouver celui qui coupe.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image à envoyer au VLM.")
    parser.add_argument("--model", required=True, help="Nom exact du modèle Ollama.")
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Adresse du serveur Ollama.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=30,
        help="Secondes maximales pour établir DNS/TCP/TLS.",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=300,
        help="Secondes maximales sans recevoir de nouvelles données.",
    )
    parser.add_argument(
        "--write-timeout",
        type=float,
        default=300,
        help="Secondes maximales pour envoyer le corps et l'image.",
    )
    parser.add_argument(
        "--pool-timeout",
        type=float,
        default=30,
        help="Secondes maximales pour obtenir une connexion HTTPX.",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="Prompt système.")
    parser.add_argument("--system-file", type=Path, help="Fichier remplaçant --system.")
    parser.add_argument("--prompt", default=DEFAULT_USER, help="Prompt utilisateur.")
    parser.add_argument("--prompt-file", type=Path, help="Fichier remplaçant --prompt.")
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Envoie l'image sans prompt texte.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Lit et affiche chaque fragment NDJSON en direct.",
    )
    parser.add_argument(
        "--think",
        choices=["off", "true", "low", "medium", "high"],
        default="off",
        help="Niveau de raisonnement, si le modèle le supporte.",
    )
    parser.add_argument(
        "--output-format",
        choices=["prompt", "json"],
        default="prompt",
        help="Mode de contrainte de la réponse Ollama.",
    )
    parser.add_argument("--temperature", type=float, default=0, help="Variabilité.")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Taille du contexte.")
    parser.add_argument(
        "--num-predict",
        type=int,
        default=4096,
        help="Nombre maximal de tokens générés.",
    )
    parser.add_argument("--num-thread", type=int, help="Nombre de threads CPU.")
    parser.add_argument(
        "--ignore-env-proxy",
        action="store_true",
        help="Passe trust_env=False à HTTPX.",
    )
    parser.add_argument("--output", type=Path, help="Chemin du rapport JSON.")
    args = parser.parse_args()

    ## Le rapport est sauvegardé même si aucun fragment n'arrive.
    output = args.output or (
        Path("runs")
        / "runtime_debug"
        / f"raw_http_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    started = time.perf_counter()
    try:
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        payload = build_payload(args)
        print(
            "Appel HTTP brut | "
            f"host={args.host} | stream={args.stream} | "
            f"connect={args.connect_timeout}s | read={args.read_timeout}s | "
            f"write={args.write_timeout}s | pool={args.pool_timeout}s",
            flush=True,
        )
        with build_client(args) as client:
            report = (
                run_streaming(client, payload)
                if args.stream
                else run_non_streaming(client, payload)
            )
        report["request"] = public_payload(payload)
        report["trust_environment"] = not args.ignore_env_proxy
    except Exception as exc:
        report = {
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }

    ## Écriture avant le code d'échec afin de ne jamais perdre le diagnostic.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nRapport écrit dans : {output}")
    if report["status"] != "success":
        raise SystemExit(1)


## Point d'entrée du script autonome.
if __name__ == "__main__":
    main()
