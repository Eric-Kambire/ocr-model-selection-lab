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
    ("1. Source", "Le PDF est rendu en PNG, ou l'image est normalisée en RGB. En mode simulation, la carte est placée sur une feuille A4 blanche avant cette étape."),
    ("2. Niveaux de gris", "Chaque pixel couleur devient une intensité entre 0 (noir) et 255 (blanc). Les couleurs ne sont pas encore supprimées : seule la luminance est conservée."),
    ("3. Masque binaire", "Les pixels plus sombres que le seuil deviennent blancs ; le fond A4 blanc devient noir. Ce masque permet de localiser le contenu imprimé."),
    ("4. Rotation", "Si activée, une recherche d'angle teste de -12° à +12° pour rendre la zone détectée proche d'une carte horizontale. Ce n'est pas une correction de perspective."),
    ("5. Contour détecté", "Le rectangle bleu englobe les pixels détectés comme non blancs, avec une marge de sécurité. Ses dimensions et son ratio sont vérifiés."),
    ("6. Crop final", "Seule la zone validée comme carte est conservée. Si le ratio ou la surface semblent incohérents, la page entière est conservée afin de ne rien perdre."),
)

# Dimensions physiques de référence. Elles rendent l'effet du DPI mesurable :
# pixels = millimètres / 25,4 × DPI.
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DEFAULT_CARD_WIDTH_MM = 120.0


def _mm_to_px(millimeters: float, dpi: int) -> int:
    """Convertit une longueur physique en pixels pour le DPI demandé."""
    return max(1, round(float(millimeters) / 25.4 * int(dpi)))


def dpi_impact_markdown(dpi: int, card_width_mm: float = DEFAULT_CARD_WIDTH_MM) -> str:
    """Donne instantanément les conséquences géométriques du DPI sélectionné."""
    width, height = _mm_to_px(A4_WIDTH_MM, dpi), _mm_to_px(A4_HEIGHT_MM, dpi)
    megapixels = width * height / 1_000_000
    raw_rgb_mb = width * height * 3 / 1_000_000
    card_width_px = _mm_to_px(card_width_mm, dpi)
    return (
        f"**Aperçu DPI :** `{dpi}` DPI → A4 ≈ `{width} × {height}` px ({megapixels:.1f} Mpx) · "
        f"mémoire RGB non compressée ≈ `{raw_rgb_mb:.1f}` MB · carte cible ≈ `{card_width_px}` px de large.\n\n"
        "Réduire le DPI accélère le rendu et diminue le volume ; les petits caractères deviennent moins lisibles. "
        "Le poids PNG/PDF final dépend aussi des couleurs et de la compression."
    )


def _write_image(image: Image.Image, path: Path, *, mode: str | None = None) -> str:
    """Enregistre une image PNG et retourne son chemin pour Gradio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    (image.convert(mode) if mode else image).save(path, format="PNG")
    return str(path)


def _write_pdf(image: Image.Image, path: Path, dpi: int) -> str:
    """Écrit un PDF image sans modifier le fichier d'origine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PDF", resolution=float(dpi))
    return str(path)


def _human_size(size_bytes: int) -> str:
    """Formate un volume pour une lecture simple dans l'interface."""
    if size_bytes < 1024:
        return f"{size_bytes} o"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes / (1024 * 1024):.2f} MiB"


def _stage_metric(label: str, path: str | Path, elapsed_seconds: float, parameters: dict[str, Any]) -> dict[str, Any]:
    """Mesure l'artefact produit à une étape et garde ses paramètres de calcul."""
    artifact = Path(path)
    with Image.open(artifact) as image:
        width, height = image.size
    size_bytes = artifact.stat().st_size
    return {
        "step": label,
        "path": str(artifact),
        "width_px": width,
        "height_px": height,
        "size_bytes": size_bytes,
        "size_human": _human_size(size_bytes),
        "elapsed_ms": round(elapsed_seconds * 1000, 1),
        "parameters": parameters,
    }


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


def _prepare_direct_source(input_path: str, output_dir: Path, dpi: int) -> tuple[Path, str, dict[str, Any]]:
    """Prépare une entrée déjà scannée, sans imposer de simulation A4."""
    source = Path(input_path)
    target = output_dir / "01_source.png"
    if source.suffix.lower() == ".pdf":
        render_single_page_pdf(source, target, dpi=dpi)
        return target, str(source), {"mode": "direct_pdf", "source_pdf": str(source)}

    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        _write_image(normalized, target)
        prepared_pdf = _write_pdf(normalized, output_dir / "00_normalized_input.pdf", dpi)
    return target, prepared_pdf, {"mode": "direct_image", "source_image": str(source)}


def _prepare_simulated_a4(
    input_path: str,
    output_dir: Path,
    dpi: int,
    angle_degrees: float,
    card_width_mm: float,
) -> tuple[Path, str, dict[str, Any]]:
    """Place une image de carte inclinée sur une feuille A4 blanche.

    La carte est redimensionnée sans recadrage, puis tournée autour de son
    centre. Le canevas reste une feuille A4 : cette étape simule un document
    déposé de travers sur un scanner, sans toucher au fichier d'origine.
    """
    source = Path(input_path)
    if source.suffix.lower() == ".pdf":
        raise gr.Error("Le mode « image carte → PDF A4 » accepte une image, pas un PDF. Utilisez le mode direct pour un PDF déjà scanné.")

    with Image.open(source) as image:
        card = ImageOps.exif_transpose(image).convert("RGB")

    # La largeur cible est physique ; la hauteur est calculée pour conserver
    # exactement les proportions de la carte initiale (aucun crop, aucune déformation).
    target_width = _mm_to_px(card_width_mm, dpi)
    target_height = max(1, round(card.height * target_width / card.width))
    card = card.resize((target_width, target_height), Image.Resampling.LANCZOS)
    rotated_card = card.rotate(float(angle_degrees), expand=True, resample=Image.Resampling.BICUBIC, fillcolor="white")

    a4_size = (_mm_to_px(A4_WIDTH_MM, dpi), _mm_to_px(A4_HEIGHT_MM, dpi))
    canvas = Image.new("RGB", a4_size, "white")
    margin = _mm_to_px(12.0, dpi)
    canvas.paste(rotated_card, (margin, margin))

    source_png = Path(_write_image(canvas, output_dir / "01_source.png"))
    prepared_pdf = _write_pdf(canvas, output_dir / "00_simulated_a4.pdf", dpi)
    return source_png, prepared_pdf, {
        "mode": "simulate_a4",
        "source_image": str(source),
        "simulation_angle_degrees": float(angle_degrees),
        "card_width_mm": float(card_width_mm),
        "card_size_px_before_rotation": [target_width, target_height],
        "a4_size_px": list(a4_size),
        "top_left_margin_mm": 12.0,
    }


def _prepare_source(
    input_path: str,
    input_mode: str,
    output_dir: Path,
    dpi: int,
    simulation_angle: float,
    card_width_mm: float,
) -> tuple[Path, str, dict[str, Any]]:
    """Choisit explicitement entre un scan existant et la simulation d'un scan A4."""
    if input_mode == "simulate_a4":
        return _prepare_simulated_a4(input_path, output_dir, dpi, simulation_angle, card_width_mm)
    return _prepare_direct_source(input_path, output_dir, dpi)


def build_pipeline(
    input_path: str | None,
    input_mode: str,
    dpi: int,
    threshold: int,
    auto_rotate: bool,
    simulation_angle: float,
    card_width_mm: float,
) -> tuple[dict[str, Any], int, str, str, str, str, str, str, str]:
    """Produit toutes les étapes, leurs mesures et le journal réutilisable."""
    if not input_path:
        raise gr.Error("Chargez un PDF ou une image avant de préparer les étapes.")
    if not 72 <= int(dpi) <= 600:
        raise gr.Error("Le DPI doit être compris entre 72 et 600.")

    output_dir = Path(tempfile.gettempdir()) / "cni-crop-lab" / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    source_path, prepared_pdf, source_preparation = _prepare_source(
        input_path,
        str(input_mode),
        output_dir,
        int(dpi),
        float(simulation_angle),
        float(card_width_mm),
    )
    source_elapsed = time.perf_counter() - started
    with Image.open(source_path) as file:
        source = ImageOps.exif_transpose(file).convert("RGB")

    started = time.perf_counter()
    gray = ImageOps.grayscale(source)
    gray_path = _write_image(gray, output_dir / "02_grayscale.png")
    gray_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    mask = _threshold_mask(gray, int(threshold))
    mask_path = _write_image(mask, output_dir / "03_binary_mask.png")
    mask_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    rotated, angle = _estimate_rotation(source, int(threshold), bool(auto_rotate))
    rotated_path = _write_image(rotated, output_dir / "04_rotated.png")
    rotate_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    rotated_gray = ImageOps.grayscale(rotated)
    rotated_mask = _threshold_mask(rotated_gray, int(threshold))
    box, geometry = _validated_box(rotated_mask, rotated.size)
    contour = _overlay_box(rotated, box, geometry)
    contour_path = _write_image(contour, output_dir / "05_detected_contour.png")
    contour_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    crop = rotated.crop(box) if box else rotated
    crop_path = _write_image(crop, output_dir / "06_cni_crop.png")
    crop_elapsed = time.perf_counter() - started

    paths = [
        str(source_path),
        gray_path,
        mask_path,
        rotated_path,
        contour_path,
        crop_path,
    ]
    metrics = [
        _stage_metric(STEPS[0][0], paths[0], source_elapsed, {"dpi": int(dpi), **source_preparation}),
        _stage_metric(STEPS[1][0], paths[1], gray_elapsed, {"conversion": "RGB → niveaux de gris"}),
        _stage_metric(STEPS[2][0], paths[2], mask_elapsed, {"threshold": int(threshold)}),
        _stage_metric(STEPS[3][0], paths[3], rotate_elapsed, {"auto_rotation": bool(auto_rotate), "detected_angle_degrees": angle}),
        _stage_metric(STEPS[4][0], paths[4], contour_elapsed, geometry),
        _stage_metric(STEPS[5][0], paths[5], crop_elapsed, {"crop_box": geometry.get("crop_box"), "fallback": box is None}),
    ]
    metadata = {
        "input": str(Path(input_path).resolve()),
        "input_mode": str(input_mode),
        "dpi": int(dpi),
        "threshold": int(threshold),
        "auto_rotation": bool(auto_rotate),
        "simulation_angle_degrees": float(simulation_angle),
        "card_width_mm": float(card_width_mm),
        "prepared_pdf": prepared_pdf,
        "source_preparation": source_preparation,
        "rotation_degrees": angle,
        "geometry": geometry,
        "paths": paths,
        "metrics": metrics,
    }
    report_path = output_dir / "processing_log.json"
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["report_path"] = str(report_path)
    return (
        metadata,
        0,
        paths[0],
        _stage_html(0),
        _stage_markdown(0, metadata),
        paths[0],
        prepared_pdf,
        json.dumps(metadata, ensure_ascii=False, indent=2),
        str(report_path),
    )


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
    metric = state.get("metrics", [{}] * len(STEPS))[index]
    if metric:
        parameters = json.dumps(metric.get("parameters", {}), ensure_ascii=False, indent=2)
        explanation += (
            "\n\n### Mesures de cette étape"
            f"\n\n- Dimensions : `{metric['width_px']} × {metric['height_px']} px`"
            f"\n- Volume du PNG : `{metric['size_human']}`"
            f"\n- Durée de calcul + écriture : `{metric['elapsed_ms']} ms`"
            f"\n\nParamètres enregistrés :\n```json\n{parameters}\n```"
        )
    return f"## {title}\n\n{explanation}"


def _stage_html(index: int) -> str:
    """Garde un intitulé stable et stylable lorsque l'utilisateur navigue."""
    return f"<div id='crop-stage-name'>{STEPS[index][0]}</div>"


def show_stage(index: int, state: dict[str, Any]) -> tuple[int, str, str, str, str]:
    """Affiche une étape existante sans recalculer les transformations."""
    if not state or not state.get("paths"):
        raise gr.Error("Préparez d'abord une entrée.")
    safe_index = max(0, min(int(index), len(STEPS) - 1))
    return safe_index, state["paths"][safe_index], _stage_html(safe_index), _stage_markdown(safe_index, state), state["paths"][safe_index]


def next_stage(index: int, state: dict[str, Any]):
    """Passe à l'artefact suivant."""
    return show_stage(int(index or 0) + 1, state)


def previous_stage(index: int, state: dict[str, Any]):
    """Revient à l'artefact précédent."""
    return show_stage(int(index or 0) - 1, state)


def build_ui() -> gr.Blocks:
    """Construit l'interface pas-à-pas autonome."""
    with gr.Blocks(title="CNI Crop Lab") as app:
        gr.HTML(
            "<section id='crop-lab-header'><h1>CNI Crop Lab</h1>"
            "<p>Préparer un scan A4, visualiser chaque transformation et télécharger les artefacts avec leur journal de paramètres.</p></section>"
        )
        state = gr.State({})
        stage_index = gr.State(0)
        with gr.Row(elem_id="crop-toolbar"):
            source = gr.File(label="PDF A4 ou image de carte", type="filepath", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"])
            input_mode = gr.Radio(
                choices=[
                    ("Scan existant : analyser directement", "direct"),
                    ("Image carte → PDF A4 simulé", "simulate_a4"),
                ],
                value="direct",
                label="Chemin de préparation",
                info="Le second mode place l'image sur un A4 blanc incliné, comme un dépôt de travers au scanner.",
            )
            dpi = gr.Slider(72, 600, value=300, step=1, label="DPI PDF")
            threshold = gr.Slider(180, 252, value=242, step=1, label="Seuil blanc")
            auto_rotate = gr.Checkbox(value=False, label="Corriger une légère rotation")
            prepare = gr.Button("Préparer / régénérer", variant="primary")
        with gr.Accordion("Simulation A4 et mesures DPI", open=True):
            gr.Markdown(
                "**Optionnel.** Choisissez « Image carte → PDF A4 simulé » pour produire un A4. "
                "L'image est réduite sans déformation, tournée autour de son centre, puis posée dans le coin supérieur gauche. "
                "En mode direct, un PDF est analysé tel quel."
            )
            with gr.Row():
                simulation_angle = gr.Slider(-15, 15, value=0, step=0.5, label="Inclinaison simulée (degrés)")
                card_width_mm = gr.Slider(70, 150, value=DEFAULT_CARD_WIDTH_MM, step=1, label="Largeur de la carte sur l'A4 (mm)")
            dpi_impact = gr.Markdown(dpi_impact_markdown(300, DEFAULT_CARD_WIDTH_MM))
        with gr.Row(elem_id="crop-workspace"):
            with gr.Column(scale=3):
                stage_image = gr.Image(label="Artefact de l'étape", type="filepath", height=560)
                with gr.Row():
                    previous = gr.Button("← Précédent")
                    next_button = gr.Button("Suivant →", variant="secondary")
            with gr.Column(scale=2):
                stage_name = gr.HTML("<div id='crop-stage-name'>En attente d'une entrée</div>")
                stage_note = gr.Markdown(
                    "1. Chargez un fichier. 2. Choisissez le chemin de préparation. 3. Réglez le DPI et le seuil. 4. Cliquez sur **Préparer / régénérer**.",
                    elem_id="crop-stage-note",
                )
                download = gr.File(label="Télécharger l'artefact affiché", type="filepath", interactive=False)
                prepared_pdf = gr.File(label="PDF A4 préparé", type="filepath", interactive=False)
                with gr.Accordion("Journal des paramètres et mesures", open=False):
                    processing_log = gr.Code(label="Journal JSON", language=None, lines=14, interactive=False)
                    log_download = gr.File(label="Télécharger le journal JSON", type="filepath", interactive=False)
                gr.Markdown(
                    "### Comment lire le laboratoire\n\n"
                    "- **DPI** : nombre de pixels par pouce lors du rendu PDF. L'estimation au-dessus change immédiatement ; cliquez sur préparer pour recalculer les images.\n"
                    "- **Seuil blanc** : limite entre fond et contenu. Un seuil trop bas oublie des pixels clairs ; trop haut détecte le bruit.\n"
                    "- **Rotation** : cherche seulement une inclinaison. Les perspectives de photo ne sont pas rectifiées dans cette version.\n"
                    "- Chaque étape affiche ses dimensions, son volume, sa durée et ses paramètres ; le même contenu est exporté en JSON."
                )

        prepare.click(
            build_pipeline,
            inputs=[source, input_mode, dpi, threshold, auto_rotate, simulation_angle, card_width_mm],
            outputs=[state, stage_index, stage_image, stage_name, stage_note, download, prepared_pdf, processing_log, log_download],
        )
        next_button.click(next_stage, inputs=[stage_index, state], outputs=[stage_index, stage_image, stage_name, stage_note, download])
        previous.click(previous_stage, inputs=[stage_index, state], outputs=[stage_index, stage_image, stage_name, stage_note, download])
        dpi.change(dpi_impact_markdown, inputs=[dpi, card_width_mm], outputs=[dpi_impact], queue=False)
        card_width_mm.change(dpi_impact_markdown, inputs=[dpi, card_width_mm], outputs=[dpi_impact], queue=False)
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
