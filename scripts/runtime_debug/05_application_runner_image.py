"""Teste la même image avec OllamaOCRModel et le timeout du runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


## Racine du dépôt : elle rend accessibles les vrais modules de l'application.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

## Ces imports sont précisément ceux utilisés par le benchmark.
from models.ollama_model import OllamaOCRModel
from ocr_benchmark.domain import InferenceResult
from ocr_benchmark.runner import BenchmarkRunner


## Prompts courts de contrôle. Ils peuvent être remplacés par des fichiers.
DEFAULT_SYSTEM = (
    "You are a vision OCR engine. Read only visible information. "
    "Never invent values."
)
DEFAULT_USER = "Analyze this image and return only the visible text."


def read_text(direct_value: str, file_path: Path | None) -> str:
    """Lit un prompt direct ou un fichier UTF-8."""

    ## Le fichier est prioritaire pour pouvoir tester un prompt long sans l'échapper.
    return file_path.read_text(encoding="utf-8") if file_path else direct_value


def run_application_call(args: argparse.Namespace) -> dict:
    """Exécute exactement l'adaptateur et le garde-fou de l'application."""

    ## OllamaOCRModel lit l'hôte depuis cette variable comme dans l'application.
    os.environ["OLLAMA_HOST"] = args.host
    ## Construction du même adaptateur que celui sélectionné dans Gradio.
    model = OllamaOCRModel(
        args.model,
        ## None laisse Ollama choisir ; un entier force le nombre de threads CPU.
        cpu_threads=args.num_thread,
        ## Par défaut, le modèle est déchargé après la tâche pour libérer la mémoire.
        unload_after_task=not args.keep_loaded,
        ## Timeout interne du client HTTP Ollama.
        request_timeout=args.http_timeout,
    )
    ## En mode image-only, aucun texte n'est ajouté par le runner.
    prompt = "" if args.image_only else read_text(args.prompt, args.prompt_file)
    system = None if args.image_only else read_text(args.system, args.system_file)
    started = time.perf_counter()
    try:
        ## Deux limites sont testées séparément :
        ## - request_timeout dans le client HTTP ;
        ## - runner_timeout dans le garde-fou de l'application.
        raw = BenchmarkRunner._perform_with_timeout(
            model,
            str(args.image.resolve()),
            args.runner_timeout,
            prompt=prompt,
            system_prompt=system,
            output_format=args.output_format,
            image_only=args.image_only,
        )
        elapsed = time.perf_counter() - started
        ## La couche historique peut rendre un dictionnaire ; on le normalise.
        inference = (
            raw
            if isinstance(raw, InferenceResult)
            else InferenceResult.from_legacy_dict(raw)
        )
        ## Le rapport met les deux timeouts côte à côte pour identifier celui qui coupe.
        return {
            "status": inference.status.value,
            "wall_elapsed_seconds": elapsed,
            "http_timeout_seconds": args.http_timeout,
            "runner_timeout_seconds": args.runner_timeout,
            "image_only": args.image_only,
            "result": asdict(inference),
        }
    finally:
        ## close() est toujours appelé, succès ou exception, pour libérer les ressources.
        model.close()


def main() -> None:
    ## Paramètres du test applicatif, volontairement proches du script direct.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image préparée à analyser.")
    parser.add_argument("--model", required=True, help="Nom Ollama exact, avec son tag.")
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Adresse du serveur Ollama.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=300,
        help="Timeout du SDK Ollama en secondes.",
    )
    parser.add_argument(
        "--runner-timeout",
        type=float,
        default=300,
        help="Timeout global imposé par BenchmarkRunner en secondes.",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="Prompt système.")
    parser.add_argument("--system-file", type=Path, help="Fichier remplaçant --system.")
    parser.add_argument("--prompt", default=DEFAULT_USER, help="Prompt utilisateur.")
    parser.add_argument("--prompt-file", type=Path, help="Fichier remplaçant --prompt.")
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Envoie uniquement l'image, sans prompts.",
    )
    parser.add_argument(
        "--output-format",
        choices=["prompt", "json"],
        default="prompt",
        help="Contrat de sortie transmis à l'adaptateur.",
    )
    parser.add_argument("--num-thread", type=int, help="Nombre de threads CPU.")
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Demande de garder le modèle chargé après la tâche.",
    )
    parser.add_argument("--output", type=Path, help="Chemin du rapport JSON.")
    args = parser.parse_args()

    ## Par défaut, chaque essai reçoit un fichier horodaté distinct.
    output = args.output or (
        ROOT_DIR
        / "runs"
        / "runtime_debug"
        / f"application_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    ## Le bloc capture les erreurs avant, pendant et après l'inférence.
    try:
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        ## Log des deux limites effectives juste avant l'appel.
        print(
            "Appel application démarré | "
            f"http_timeout={args.http_timeout}s | "
            f"runner_timeout={args.runner_timeout}s",
            flush=True,
        )
        report = run_application_call(args)
    except Exception as exc:
        report = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
    ## Sauvegarde systématique du diagnostic avant de quitter.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"Rapport écrit dans : {output}")
    ## Le code de sortie 1 signale un échec aux outils d'automatisation.
    if report["status"] not in {"success"}:
        raise SystemExit(1)


## Point d'entrée du script autonome.
if __name__ == "__main__":
    main()
