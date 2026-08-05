"""Interface locale pour créer un modèle Ollama dérivé avec un Modelfile.

Lancement depuis la racine du dépôt :
    python scripts/ollama_modelfile_lab.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import gradio as gr


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ocr_benchmark.application.benchmark_service import list_ollama_models
from ocr_benchmark.application.ollama_modelfile_service import (
    DEFAULT_CNI_MODEL_SYSTEM_PROMPT,
    build_modelfile_template,
    create_ollama_model,
    save_modelfile_artifact,
)
from models.ollama_capabilities import inspect_ollama_vision_capability


LOGGER = logging.getLogger("ollama_modelfile_lab")
ARTEFACTS_DIR = ROOT_DIR / "runs" / "ollama_modelfiles"


def _models() -> list[str]:
    """Retourne une liste stable des modèles installés."""

    return sorted(set(list_ollama_models()), key=str.casefold)


def _initial_template(models: list[str]) -> str:
    """Génère un exemple exploitable, même si Ollama n'est pas démarré."""

    source = models[0] if models else "qwen3-vl:4b"
    return build_modelfile_template(source)


def refresh_models(current: str | None):
    """Rafraîchit le sélecteur après un pull ou une création externe."""

    models = _models()
    selected = current if current in models else (models[0] if models else None)
    return (
        gr.update(choices=models, value=selected),
        (
            f"{len(models)} modèle(s) Ollama détecté(s)."
            if models
            else "Aucun modèle détecté. Vérifiez que le serveur Ollama est démarré."
        ),
        inspect_base_model(selected),
    )


def inspect_base_model(model_name: str | None) -> str:
    """Indique si le modèle source pourra réellement recevoir des images."""

    if not model_name:
        return "Capacité vision non vérifiée : aucun modèle sélectionné."
    try:
        import ollama

        client = ollama.Client(
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout=10,
        )
        capability = inspect_ollama_vision_capability(client, model_name)
    except Exception as exc:
        return (
            "Capacité vision non vérifiée : "
            f"{type(exc).__name__}: {exc}"
        )
    if capability.supported:
        return (
            f"Capacité vision confirmée pour `{model_name}`. "
            "Ce modèle peut être utilisé par le Benchmark CNI."
        )
    return (
        f"Attention : `{model_name}` n’est pas déclaré vision. "
        "La variante pourra être créée, mais le Benchmark CNI la refusera "
        f"pour éviter un faux succès. Détail : {capability.reason}"
    )


def regenerate_template(
    base_model: str,
    system_prompt: str,
    num_ctx: int,
    num_predict: int,
    temperature: float,
) -> str:
    """Reconstruit le Modelfile sans créer de modèle."""

    return build_modelfile_template(
        base_model,
        system_prompt=system_prompt,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=temperature,
    )


def create_model(
    base_model: str,
    new_model_name: str,
    modelfile: str,
    timeout_seconds: float,
):
    """Crée le modèle et expose le blueprint exact au téléchargement."""

    try:
        result = create_ollama_model(
            base_model=base_model,
            new_model_name=new_model_name,
            modelfile=modelfile,
            timeout_seconds=timeout_seconds,
        )
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Modelfile validation failed | error=%s", exc)
        return f"Erreur de validation : {exc}", modelfile, None

    artifact = save_modelfile_artifact(
        ARTEFACTS_DIR,
        result.model_name,
        result.modelfile,
    )
    if result.success:
        LOGGER.info(
            "Ollama model created | source=%s | target=%s | artifact=%s",
            base_model,
            result.model_name,
            artifact,
        )
        detail = result.stdout or "Création terminée."
        status = (
            f"Modèle `{result.model_name}` créé avec succès.\n\n"
            f"Sortie Ollama :\n{detail}"
        )
    else:
        LOGGER.error(
            "Ollama model creation failed | source=%s | target=%s | error=%s",
            base_model,
            result.model_name,
            result.error,
        )
        status = (
            f"Échec de création de `{result.model_name}`.\n\n"
            f"{result.error or 'Erreur Ollama non précisée.'}"
        )
    return status, result.modelfile, str(artifact)


def build_app() -> gr.Blocks:
    """Construit l'interface sans la lancer, afin de faciliter les tests."""

    models = _models()
    selected = models[0] if models else None
    with gr.Blocks(title="Ollama Modelfile Lab") as app:
        gr.Markdown(
            "# Ollama Modelfile Lab\n"
            "Créez une variante locale d’un modèle existant. Les poids ne sont "
            "pas recopiés : Ollama réutilise les couches du modèle source et "
            "enregistre la nouvelle configuration."
        )
        with gr.Row():
            base_model = gr.Dropdown(
                choices=models,
                value=selected,
                label="1. Modèle source",
                info="Choisissez un VLM si le nouveau modèle doit recevoir des images.",
                scale=4,
            )
            refresh = gr.Button("↻", size="sm", scale=0)
            new_model_name = gr.Textbox(
                label="2. Nom du nouveau modèle",
                placeholder="cni-ocr-local:latest",
                scale=4,
            )
            timeout = gr.Number(
                value=1800,
                minimum=1,
                precision=0,
                label="Timeout création (s)",
                scale=2,
            )
        base_model_info = gr.Markdown(inspect_base_model(selected))

        with gr.Accordion("Assistant de template", open=True):
            system_prompt = gr.Textbox(
                value=DEFAULT_CNI_MODEL_SYSTEM_PROMPT,
                label="Prompt système à embarquer",
                lines=14,
                info=(
                    "Ce texte devient le SYSTEM du modèle. Il peut être long ; "
                    "évitez seulement trois guillemets consécutifs."
                ),
            )
            with gr.Row():
                num_ctx = gr.Number(value=8192, minimum=256, precision=0, label="num_ctx")
                num_predict = gr.Number(
                    value=4096, minimum=1, precision=0, label="num_predict"
                )
                temperature = gr.Number(
                    value=0.0, minimum=0, maximum=2, label="temperature"
                )
                regenerate = gr.Button("Regénérer le template", variant="secondary")

        modelfile = gr.Code(
            value=_initial_template(models),
            label="3. Modelfile éditable",
            language=None,
            lines=28,
        )
        gr.Markdown(
            "Au moment de créer, la ligne `FROM` est automatiquement alignée "
            "sur le modèle choisi au-dessus. Le reste du fichier est conservé."
        )
        create = gr.Button("Créer le modèle Ollama", variant="primary")
        status = gr.Markdown("Prêt.")
        with gr.Accordion("Modelfile réellement utilisé", open=False):
            effective_modelfile = gr.Code(language=None, lines=22, interactive=False)
            downloadable = gr.File(label="Télécharger le Modelfile", interactive=False)

        refresh.click(
            refresh_models,
            inputs=[base_model],
            outputs=[base_model, status, base_model_info],
            queue=False,
        )
        base_model.change(
            inspect_base_model,
            inputs=[base_model],
            outputs=[base_model_info],
            queue=False,
        )
        regenerate.click(
            regenerate_template,
            inputs=[base_model, system_prompt, num_ctx, num_predict, temperature],
            outputs=[modelfile],
            queue=False,
        )
        create.click(
            create_model,
            inputs=[base_model, new_model_name, modelfile, timeout],
            outputs=[status, effective_modelfile, downloadable],
            concurrency_limit=1,
            concurrency_id="ollama-model-create",
        )
    return app


def parse_args() -> argparse.Namespace:
    """Lit les options minimales de lancement local."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
