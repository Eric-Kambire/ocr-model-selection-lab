"""Laboratoire Gradio pour comparer les méthodes de crop CNI.

Lancement :
    python scripts/cni_crop_methods_lab.py --port 8101
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark.cni_crop_methods import (
    METHOD_DESCRIPTIONS,
    METHOD_LABELS,
    normalise_crop_lab_source,
    run_crop_method,
)


TEMP_ROOT = Path(tempfile.gettempdir()) / "cni-crop-methods-lab"

APP_CSS = """
.gradio-container { max-width: 1480px !important; padding: 18px 24px 30px !important; }
#lab-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 14px; margin-bottom: 14px; }
#lab-header h1 { margin: 0; font-size: 28px; }
#lab-header p { margin: 5px 0 0; color: var(--body-text-color-subdued); }
#method-bar { align-items: end; border-bottom: 1px solid var(--border-color-primary); padding-bottom: 14px; }
#method-settings { padding: 8px 0 4px; }
#lab-workspace { gap: 24px !important; align-items: flex-start; }
#lab-preview { min-height: 620px; background: var(--background-fill-secondary); }
#lab-preview .image-container { min-height: 600px !important; }
#lab-preview img { object-fit: contain !important; }
#lab-inspector { border-left: 1px solid var(--border-color-primary); padding-left: 22px; }
#stage-nav button { min-height: 38px; }
.method-copy { color: var(--body-text-color-subdued); line-height: 1.45; }
"""


def method_visibility(method: str) -> tuple[Any, ...]:
    """N'affiche que les réglages de la méthode active."""
    return (
        f"**{METHOD_LABELS[method]}** — {METHOD_DESCRIPTIONS[method]}",
        gr.update(visible=method == "connected_components"),
        gr.update(visible=method == "canny_contours"),
        gr.update(visible=method == "min_area_rect"),
        gr.update(visible=method == "pillow_ratio"),
    )


def analyse_document(
    source_value: str | None,
    method: str,
    dpi: int,
    page_number: int,
    final_rotation: int,
    component_mask_mode: str,
    component_threshold: int,
    component_kernel: int,
    component_min_area_pct: float,
    component_selection: str,
    canny_low: int,
    canny_high: int,
    contour_kernel: int,
    contour_min_area_pct: float,
    contour_min_score: float,
    global_threshold: int,
    ignore_border_pct: float,
    pillow_threshold: int,
    pillow_coarse_step: int,
    pillow_fine_radius: int,
) -> tuple[Any, ...]:
    """Prépare la source, exécute la méthode et ouvre sa première étape."""
    if not source_value:
        raise gr.Error("Chargez d'abord un PDF ou une image.")
    source = Path(source_value)
    if not source.is_file():
        raise gr.Error("Le fichier chargé n'est plus disponible.")
    session_dir = TEMP_ROOT / f"session-{uuid.uuid4().hex}"
    session_dir.mkdir(parents=True, exist_ok=False)
    normalised = normalise_crop_lab_source(
        source,
        session_dir / "00_source.png",
        dpi=int(dpi),
        page_number=int(page_number),
    )
    parameters = {
        "final_rotation": int(final_rotation),
        "component_mask_mode": component_mask_mode,
        "component_threshold": int(component_threshold),
        "component_kernel": int(component_kernel),
        "component_min_area_pct": float(component_min_area_pct),
        "component_selection": component_selection,
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
        "contour_kernel": int(contour_kernel),
        "contour_min_area_pct": float(contour_min_area_pct),
        "contour_min_score": float(contour_min_score),
        "global_threshold": int(global_threshold),
        "ignore_border_pct": float(ignore_border_pct),
        "pillow_threshold": int(pillow_threshold),
        "pillow_coarse_step": int(pillow_coarse_step),
        "pillow_fine_radius": int(pillow_fine_radius),
    }
    result = run_crop_method(
        Path(normalised["image_path"]),
        session_dir / method,
        method=method,
        parameters=parameters,
    )
    result["source"] = normalised
    report_path = Path(result["report_path"])
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    state = {
        "result": result,
        "stages": result["stages"],
        "session_dir": str(session_dir),
    }
    choices = [
        (f"{stage['index'] + 1}. {stage['name']}", stage["index"])
        for stage in result["stages"]
    ]
    summary = _summary_markdown(result)
    first = _stage_outputs(state, 0)
    return (
        state,
        0,
        gr.update(choices=choices, value=0),
        *first,
        result["final_path"],
        result["report_path"],
        summary,
        gr.update(interactive=len(choices) > 1),
        gr.update(visible=True, interactive=bool(choices)),
    )


def _summary_markdown(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status == "crop_detected":
        decision = "Crop accepté : le fichier final est redressé."
    else:
        reason = result.get("summary", {}).get("reason", "candidat insuffisant")
        decision = f"Crop refusé : l'image source complète est conservée. Raison : {reason}"
    return (
        f"### Décision\n\n**Méthode :** {result.get('method_label')}  \n"
        f"**Durée :** {result.get('elapsed_ms')} ms  \n"
        f"**Résultat :** {decision}"
    )


def _stage_outputs(state: dict[str, Any], index: int) -> tuple[Any, ...]:
    stages = state.get("stages", []) if isinstance(state, dict) else []
    if not stages:
        return None, "### Aucune étape", "Lancez une analyse.", {}, None
    position = max(0, min(int(index), len(stages) - 1))
    stage = stages[position]
    title = f"### {position + 1}/{len(stages)} · {stage['name']}"
    return (
        stage["image_path"],
        title,
        stage["explanation"],
        stage.get("metrics", {}),
        stage["image_path"],
    )


def select_stage(index: int, state: dict[str, Any]) -> tuple[Any, ...]:
    return int(index), *_stage_outputs(state, int(index))


def move_stage(index: int, state: dict[str, Any], direction: int) -> tuple[Any, ...]:
    stages = state.get("stages", []) if isinstance(state, dict) else []
    position = max(0, min(int(index) + int(direction), max(0, len(stages) - 1)))
    return position, gr.update(value=position), *_stage_outputs(state, position)


def play_stages(
    state: dict[str, Any],
    delay_ms: int,
) -> Iterator[tuple[Any, ...]]:
    """Anime les vrais artefacts calculés, sans recalculer ni compresser."""
    stages = state.get("stages", []) if isinstance(state, dict) else []
    if not stages:
        raise gr.Error("Lancez une analyse avant la lecture.")
    for position, _ in enumerate(stages):
        yield (
            position,
            gr.update(value=position),
            *_stage_outputs(state, position),
            f"Lecture : étape {position + 1}/{len(stages)}",
        )
        time.sleep(max(0.05, int(delay_ms) / 1000.0))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CNI Crop Methods Lab") as app:
        gr.HTML(
            "<header id='lab-header'><h1>CNI Crop Methods Lab</h1>"
            "<p>Comparez les masques, composants, contours, rectangles et rotations sur le même document.</p></header>"
        )
        state = gr.State({})
        stage_index = gr.State(0)
        with gr.Row(elem_id="method-bar"):
            source = gr.File(
                label="PDF ou image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"],
                type="filepath",
                scale=4,
            )
            method = gr.Dropdown(
                choices=[(label, key) for key, label in METHOD_LABELS.items()],
                value="connected_components",
                label="Méthode",
                filterable=False,
                scale=3,
            )
            dpi = gr.Slider(72, 600, value=300, step=1, label="DPI du PDF", scale=2)
            page_number = gr.Number(value=1, minimum=1, precision=0, label="Page", scale=1)
            final_rotation = gr.Dropdown(
                choices=[("Aucune", 0), ("90°", 90), ("180°", 180), ("270°", 270)],
                value=0,
                label="Rotation finale",
                filterable=False,
                scale=2,
            )
            analyse = gr.Button("Analyser", variant="primary", scale=2)

        method_copy = gr.Markdown(
            f"**{METHOD_LABELS['connected_components']}** — "
            f"{METHOD_DESCRIPTIONS['connected_components']}",
            elem_classes=["method-copy"],
        )
        with gr.Column(elem_id="method-settings"):
            with gr.Group(visible=True) as component_group:
                with gr.Row():
                    component_mask_mode = gr.Dropdown(
                        [("Seuil adaptatif", "adaptive"), ("Otsu", "otsu"), ("Bords Canny", "canny")],
                        value="adaptive",
                        label="Construction du masque",
                        filterable=False,
                    )
                    component_threshold = gr.Slider(170, 252, value=235, step=1, label="Seuil indicatif")
                    component_kernel = gr.Slider(3, 41, value=11, step=2, label="Connexion des pixels")
                    component_min_area_pct = gr.Slider(0.01, 5.0, value=0.15, step=0.01, label="Aire minimale (%)")
                    component_selection = gr.Dropdown(
                        [("Meilleur score — recommandé", "scored"), ("Plus grande aire — méthode stricte", "largest")],
                        value="scored",
                        label="Choix du composant",
                        filterable=False,
                    )
            with gr.Group(visible=False) as contour_group:
                with gr.Row():
                    canny_low = gr.Slider(1, 200, value=45, step=1, label="Canny bas")
                    canny_high = gr.Slider(2, 300, value=135, step=1, label="Canny haut")
                    contour_kernel = gr.Slider(3, 31, value=7, step=2, label="Fermeture")
                    contour_min_area_pct = gr.Slider(0.01, 5.0, value=0.2, step=0.01, label="Aire minimale (%)")
                    contour_min_score = gr.Slider(0.0, 1.0, value=0.64, step=0.01, label="Score minimal")
            with gr.Group(visible=False) as minrect_group:
                with gr.Row():
                    global_threshold = gr.Slider(170, 252, value=235, step=1, label="Seuil global")
                    ignore_border_pct = gr.Slider(0, 15, value=0, step=0.5, label="Marge ignorée (%)")
                    gr.Markdown(
                        "Cette méthode utilise tous les pixels blancs ensemble. Elle permet de reproduire "
                        "et comprendre le rectangle trop grand causé par du bruit éloigné."
                    )
            with gr.Group(visible=False) as pillow_group:
                with gr.Row():
                    pillow_threshold = gr.Slider(170, 252, value=235, step=1, label="Seuil blanc")
                    pillow_coarse_step = gr.Slider(3, 15, value=9, step=2, label="Pas recherche large (°)")
                    pillow_fine_radius = gr.Slider(1, 8, value=3, step=1, label="Rayon affinage (°)")

        with gr.Row(elem_id="lab-workspace"):
            with gr.Column(scale=7):
                stage_image = gr.Image(
                    label="Visualisation de l'étape",
                    type="filepath",
                    interactive=False,
                    elem_id="lab-preview",
                )
                stage_selector = gr.Dropdown(
                    choices=[],
                    label="Étape affichée",
                    interactive=True,
                    filterable=False,
                )
                with gr.Row(elem_id="stage-nav"):
                    previous = gr.Button("Précédent")
                    next_button = gr.Button("Suivant", interactive=False)
                    play = gr.Button("Lire les étapes", visible=False)
                    delay = gr.Slider(80, 1500, value=350, step=20, label="Pause (ms)")
                playback_status = gr.Markdown("En attente d'une analyse.")
            with gr.Column(scale=4, elem_id="lab-inspector"):
                stage_title = gr.Markdown("### Aucune étape")
                stage_explanation = gr.Markdown("Chargez un document et choisissez une méthode.")
                stage_metrics = gr.JSON(label="Mesures de l'étape")
                current_download = gr.File(label="Télécharger cette étape", interactive=False)
                final_download = gr.File(label="Télécharger le résultat final", interactive=False)
                report_download = gr.File(label="Télécharger le rapport JSON", interactive=False)
                summary = gr.Markdown("### Décision\n\nAucune analyse.")

        method.change(
            method_visibility,
            inputs=[method],
            outputs=[method_copy, component_group, contour_group, minrect_group, pillow_group],
            queue=False,
        )
        analyse.click(
            analyse_document,
            inputs=[
                source, method, dpi, page_number, final_rotation,
                component_mask_mode, component_threshold, component_kernel,
                component_min_area_pct, component_selection,
                canny_low, canny_high, contour_kernel, contour_min_area_pct,
                contour_min_score, global_threshold, ignore_border_pct,
                pillow_threshold, pillow_coarse_step, pillow_fine_radius,
            ],
            outputs=[
                state, stage_index, stage_selector, stage_image, stage_title,
                stage_explanation, stage_metrics, current_download,
                final_download, report_download, summary, next_button, play,
            ],
        )
        stage_selector.change(
            select_stage,
            inputs=[stage_selector, state],
            outputs=[
                stage_index, stage_image, stage_title, stage_explanation,
                stage_metrics, current_download,
            ],
            queue=False,
        )
        previous.click(
            lambda index, value: move_stage(index, value, -1),
            inputs=[stage_index, state],
            outputs=[
                stage_index, stage_selector, stage_image, stage_title,
                stage_explanation, stage_metrics, current_download,
            ],
            queue=False,
        )
        next_button.click(
            lambda index, value: move_stage(index, value, 1),
            inputs=[stage_index, state],
            outputs=[
                stage_index, stage_selector, stage_image, stage_title,
                stage_explanation, stage_metrics, current_download,
            ],
            queue=False,
        )
        play.click(
            play_stages,
            inputs=[state, delay],
            outputs=[
                stage_index, stage_selector, stage_image, stage_title,
                stage_explanation, stage_metrics, current_download,
                playback_status,
            ],
            queue=True,
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare les méthodes de crop CNI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--share", action="store_true")
    arguments = parser.parse_args()
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name=arguments.host,
        server_port=arguments.port,
        share=arguments.share,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
