"""Laboratoire visuel de préparation d'une CNI depuis un PDF ou une image.

Le script ne modifie aucun PDF source. Il produit des PNG explicatifs dans un
dossier temporaire et permet de parcourir chaque étape via une petite interface
Gradio : rendu, gris, masque, rotation, contour et crop final.

Lancement :
    python scripts/cni_crop_stepper_app.py --port 8100
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

# Rend le script exécutable depuis n'importe quel dossier PowerShell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark.cni_images import render_single_page_pdf


APP_CSS = """
.gradio-container { max-width: 1440px !important; background: #f7f8fb !important; }
#crop-lab-header { border-bottom: 1px solid #dce2ea; padding: 8px 0 18px; }
#crop-lab-header h1 { margin: 0; color: #12233f; font-size: 30px; }
#crop-lab-header p { margin: 6px 0 0; color: #58677d; }
#crop-stage-name { font-size: 18px; font-weight: 700; color: #173a72; }
#crop-stage-note { min-height: 80px; color: #3d4b60; }
#crop-toolbar { align-items: end; border-bottom: 1px solid #dce2ea; padding-bottom: 12px; }
#crop-workspace { min-height: 590px; }
#crop-workspace .image-container { background: white; }
"""

STEPS = (
    ("1. Source", "Le PDF est rendu en PNG, ou l'image est normalisée en RGB."),
    ("2. Niveaux de gris", "Chaque pixel devient une intensité entre 0 (noir) et 255 (blanc)."),
    ("3. Masque binaire", "Les pixels plus sombres que le seuil sont conservés ; le fond A4 blanc est ignoré."),
    ("4. Rotation", "Une rotation optionnelle cherche l'angle qui rend le rectangle détecté le plus proche d'une carte horizontale."),
    ("5. Contour détecté", "Le rectangle bleu est le rectangle englobant des pixels non blancs, avec une marge de sécurité."),
    ("6. Crop final", "Seule la zone validée comme carte est conservée. En cas de doute, la page complète est gardée."),
)


def _write_image(image: Image.Image, path: Path, *, mode: str | None = None) -> str:
    """Enregistre une image PNG et retourne son chemin pour Gradio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    (image.convert(mode) if mode else image).save(path, format="PNG")
    return str(path)


def _threshold_mask(gray: Image.Image, threshold: int) -> Image.Image:
    """Crée le même masque que la pipeline : contenu sombre=blanc, fond=noir."""
    return gray.point(lambda pixel: 255 if pixel < threshold else 0)


def _validated_box(mask: Image.Image, source_size: tuple[int, int]) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    """Retourne un rectangle CNI plausible et ses métriques géométriques."""
    bbox = mask.getbbox()
    if bbox is None:
        return None, {"status": "crop_not_detected", "reason": "Aucun pixel non blanc après seuillage."}

    left, top, right, bottom = bbox
    padding = max(12, int(max(source_size) * 0.015))
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(source_size[0], right + padding), min(source_size[1], bottom + padding)
    width, height = right - left, bottom - top
    ratio = width / height if height else 0.0
    coverage = width * height / (source_size[0] * source_size[1])
    valid = 1.20 <= ratio <= 2.05 and 0.02 <= coverage <= 0.65
    info = {
        "status": "crop_detected" if valid else "crop_fallback_full_page",
        "crop_box": [left, top, right, bottom],
        "ratio": round(ratio, 4),
        "coverage": round(coverage, 4),
        "padding_px": padding,
    }
    return ((left, top, right, bottom) if valid else None), info


def _estimate_rotation(source: Image.Image, threshold: int, enabled: bool) -> tuple[Image.Image, float]:
    """Corrige une légère rotation sans dépendre d'OpenCV.

    Pour chaque angle entier entre -12° et +12°, un masque est calculé sur une
    miniature. L'angle dont le rectangle a le ratio le plus proche de 1.586
    (format de carte ISO ID-1) est retenu. C'est une estimation, pas une
    correction de perspective à quatre coins.
    """
    if not enabled:
        return source.copy(), 0.0

    preview = source.copy()
    preview.thumbnail((900, 900))
    target_ratio = 1.586
    best_angle, best_score = 0.0, float("inf")
    for angle in range(-12, 13):
        candidate = preview.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor="white")
        box, info = _validated_box(_threshold_mask(ImageOps.grayscale(candidate), threshold), candidate.size)
        if box is None:
            continue
        # Le score favorise une forme de carte plausible et une zone compacte.
        score = abs(float(info["ratio"]) - target_ratio) + abs(float(info["coverage"]) - 0.18) * 0.25
        if score < best_score:
            best_score, best_angle = score, float(angle)

    rotated = source.rotate(best_angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor="white")
    return rotated, best_angle


def _overlay_box(source: Image.Image, box: tuple[int, int, int, int] | None, metadata: dict[str, Any]) -> Image.Image:
    """Dessine le contour détecté et les mesures visibles pour l'apprentissage."""
    overlay = source.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    if box is None:
        draw.text((18, 18), "Aucun contour CNI fiable : page complete conservee", fill="#b42318")
        return overlay
    draw.rectangle(box, outline="#1769d1", width=max(3, max(source.size) // 500))
    label = f"ratio={metadata['ratio']} | couverture={metadata['coverage'] * 100:.1f}% | marge={metadata['padding_px']} px"
    draw.rectangle((box[0], max(0, box[1] - 30), min(source.width, box[0] + max(360, len(label) * 7)), box[1]), fill="#1769d1")
    draw.text((box[0] + 6, max(2, box[1] - 25)), label, fill="white")
    return overlay


def _load_source(input_path: str, output_dir: Path, dpi: int) -> Path:
    """Convertit une entrée PDF ou image vers le PNG source de la démonstration."""
    source = Path(input_path)
    target = output_dir / "01_source.png"
    if source.suffix.lower() == ".pdf":
        render_single_page_pdf(source, target, dpi=dpi)
        return target
    with Image.open(source) as image:
        _write_image(ImageOps.exif_transpose(image).convert("RGB"), target)
    return target


def build_pipeline(input_path: str | None, dpi: int, threshold: int, auto_rotate: bool) -> tuple[dict[str, Any], int, str, str, str]:
    """Produit toutes les étapes et initialise l'affichage sur la source."""
    if not input_path:
        raise gr.Error("Chargez un PDF ou une image avant de préparer les étapes.")
    if not 72 <= int(dpi) <= 600:
        raise gr.Error("Le DPI doit être compris entre 72 et 600.")

    output_dir = Path(tempfile.gettempdir()) / "cni-crop-lab" / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    source_path = _load_source(input_path, output_dir, int(dpi))
    with Image.open(source_path) as file:
        source = ImageOps.exif_transpose(file).convert("RGB")

    gray = ImageOps.grayscale(source)
    mask = _threshold_mask(gray, int(threshold))
    rotated, angle = _estimate_rotation(source, int(threshold), bool(auto_rotate))
    rotated_gray = ImageOps.grayscale(rotated)
    rotated_mask = _threshold_mask(rotated_gray, int(threshold))
    box, geometry = _validated_box(rotated_mask, rotated.size)
    contour = _overlay_box(rotated, box, geometry)
    crop = rotated.crop(box) if box else rotated

    paths = [
        str(source_path),
        _write_image(gray, output_dir / "02_grayscale.png"),
        _write_image(mask, output_dir / "03_binary_mask.png"),
        _write_image(rotated, output_dir / "04_rotated.png"),
        _write_image(contour, output_dir / "05_detected_contour.png"),
        _write_image(crop, output_dir / "06_cni_crop.png"),
    ]
    metadata = {
        "input": str(Path(input_path).resolve()),
        "dpi": int(dpi),
        "threshold": int(threshold),
        "auto_rotation": bool(auto_rotate),
        "rotation_degrees": angle,
        "geometry": geometry,
        "paths": paths,
    }
    report_path = output_dir / "analysis.json"
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["report_path"] = str(report_path)
    return metadata, 0, paths[0], STEPS[0][0], _stage_markdown(0, metadata)


def _stage_markdown(index: int, state: dict[str, Any]) -> str:
    """Construit l'explication affichée à côté de l'image de l'étape."""
    title, explanation = STEPS[index]
    if index == 2:
        explanation += f" Seuil utilisé : `{state['threshold']}` sur une échelle 0–255."
    if index == 3:
        explanation += f" Angle estimé : `{state['rotation_degrees']:+.0f}°`."
    if index in {4, 5}:
        geometry = state["geometry"]
        explanation += "\n\n```json\n" + json.dumps(geometry, ensure_ascii=False, indent=2) + "\n```"
    return f"## {title}\n\n{explanation}"


def show_stage(index: int, state: dict[str, Any]) -> tuple[int, str, str, str, str]:
    """Affiche une étape existante sans recalculer les transformations."""
    if not state or not state.get("paths"):
        raise gr.Error("Préparez d'abord une entrée.")
    safe_index = max(0, min(int(index), len(STEPS) - 1))
    return safe_index, state["paths"][safe_index], STEPS[safe_index][0], _stage_markdown(safe_index, state), state["paths"][safe_index]


def next_stage(index: int, state: dict[str, Any]):
    """Passe à l'artefact suivant."""
    return show_stage(int(index or 0) + 1, state)


def previous_stage(index: int, state: dict[str, Any]):
    """Revient à l'artefact précédent."""
    return show_stage(int(index or 0) - 1, state)


def build_ui() -> gr.Blocks:
    """Construit l'interface pas-à-pas autonome."""
    with gr.Blocks(title="CNI Crop Lab") as app:
        gr.HTML("<section id='crop-lab-header'><h1>CNI Crop Lab</h1><p>Visualiser, expliquer et télécharger chaque étape de préparation.</p></section>")
        state = gr.State({})
        stage_index = gr.State(0)
        with gr.Row(elem_id="crop-toolbar"):
            source = gr.File(label="PDF ou image source", type="filepath", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"])
            dpi = gr.Slider(72, 600, value=300, step=1, label="DPI PDF")
            threshold = gr.Slider(180, 252, value=242, step=1, label="Seuil blanc")
            auto_rotate = gr.Checkbox(value=False, label="Corriger une légère rotation")
            prepare = gr.Button("Préparer les étapes", variant="primary")
        with gr.Row(elem_id="crop-workspace"):
            with gr.Column(scale=3):
                stage_image = gr.Image(label="Artefact de l'étape", type="filepath", height=560)
                with gr.Row():
                    previous = gr.Button("← Précédent")
                    next_button = gr.Button("Suivant →", variant="secondary")
            with gr.Column(scale=2):
                stage_name = gr.HTML("<div id='crop-stage-name'>En attente d'une entrée</div>")
                stage_note = gr.Markdown("Chargez un fichier puis cliquez sur **Préparer les étapes**.", elem_id="crop-stage-note")
                download = gr.File(label="Télécharger l'artefact affiché", type="filepath", interactive=False)
                gr.Markdown("### Principe\n\n- Le masque détecte le contenu sombre sur le fond A4 blanc.\n- Le contour est un rectangle englobant, pas encore une détection de quatre coins.\n- La rotation est une estimation par recherche d'angle ; elle ne corrige pas la perspective.")

        prepare.click(
            build_pipeline,
            inputs=[source, dpi, threshold, auto_rotate],
            outputs=[state, stage_index, stage_image, stage_name, stage_note],
        ).then(lambda state_value: state_value["paths"][0], inputs=[state], outputs=[download])
        next_button.click(next_stage, inputs=[stage_index, state], outputs=[stage_index, stage_image, stage_name, stage_note, download])
        previous.click(previous_stage, inputs=[stage_index, state], outputs=[stage_index, stage_image, stage_name, stage_note, download])
    return app


def main() -> None:
    """Lance l'application locale Gradio."""
    parser = argparse.ArgumentParser(description="Laboratoire visuel de crop CNI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, ssr_mode=False, css=APP_CSS)


if __name__ == "__main__":
    main()
