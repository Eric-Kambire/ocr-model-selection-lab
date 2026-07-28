from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Generator

import cv2
import gradio as gr
import numpy as np

try:
    # Import utilisé lorsque le laboratoire est chargé comme module du projet.
    from .degradation_lab import (
        apply_global_degradations,
        apply_local_effect,
        draw_annotations,
        editor_composite,
        export_lab,
        load_for_lab,
    )
    from .smart_crop import (
        DetectorConfig,
        detect_card,
        draw_debug,
        load_input,
        order_quad,
        save_image,
        warp_card,
    )
except ImportError:
    # Import conservé pour ``python app.py`` depuis le dossier décompressé.
    from degradation_lab import (
        apply_global_degradations,
        apply_local_effect,
        draw_annotations,
        editor_composite,
        export_lab,
        load_for_lab,
    )
    from smart_crop import (
        DetectorConfig,
        detect_card,
        draw_debug,
        load_input,
        order_quad,
        save_image,
        warp_card,
    )


def _as_path(uploaded: Any) -> Path:
    if uploaded is None:
        raise gr.Error("Charge d'abord une image ou un PDF.")
    if isinstance(uploaded, (str, Path)):
        return Path(uploaded)
    name = getattr(uploaded, "name", None)
    if name:
        return Path(name)
    raise gr.Error("Format de fichier reçu non reconnu.")


def _bgr_to_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _map_to_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return _bgr_to_rgb(image)


def _diagnostic_mosaic(maps: dict[str, np.ndarray]) -> np.ndarray:
    items = [
        ("Cadre noir détecté", maps.get("frame_artifact_mask")),
        ("Image nettoyée pour l'analyse", maps.get("analysis_image")),
        ("Gradient Sobel", maps.get("gradient")),
        ("Contours Canny + gradient", maps.get("edge_union")),
        ("Fermeture morphologique", maps.get("connected_edges")),
        ("Densité / texture", maps.get("density_mask")),
        ("Premier plan / fond LAB", maps.get("foreground_mask")),
        ("Distance de couleur LAB", maps.get("colour_distance")),
        ("Segments Hough/LSD", maps.get("line_mask")),
        ("Black-hat", maps.get("blackhat")),
    ]
    valid = [(title, image) for title, image in items if image is not None]
    if not valid:
        return np.zeros((300, 600, 3), dtype=np.uint8)

    tile_w, tile_h = 480, 310
    tiles: list[np.ndarray] = []
    for title, image in valid:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        h, w = tile.shape[:2]
        scale = min((tile_w - 20) / max(w, 1), (tile_h - 55) / max(h, 1))
        resized = cv2.resize(
            tile,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
        y = 45 + (tile_h - 45 - resized.shape[0]) // 2
        x = (tile_w - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            canvas,
            title,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        tiles.append(canvas)

    while len(tiles) % 2:
        tiles.append(np.full((tile_h, tile_w, 3), 245, dtype=np.uint8))
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)


def _selected_overlay(image: np.ndarray, best: Any | None) -> np.ndarray:
    output = image.copy()
    if best is None:
        cv2.putText(output, "Aucun candidat retenu", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return output
    pts = np.round(order_quad(best.quad)).astype(np.int32)
    cv2.polylines(output, [pts], True, (0, 220, 0), 4, cv2.LINE_AA)
    for index, (x, y) in enumerate(pts):
        cv2.circle(output, (int(x), int(y)), 7, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(output, str(index + 1), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        output,
        f"RETENU: {best.source}  score={best.score:.3f}",
        (16, max(30, output.shape[0] - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 130, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def _stage(image: np.ndarray | None, title: str, description: str) -> dict[str, Any] | None:
    if image is None:
        return None
    return {"image": _map_to_rgb(image), "title": title, "description": description}


def _build_stages(
    original: np.ndarray,
    working: np.ndarray,
    maps: dict[str, np.ndarray],
    debug: np.ndarray,
    selected: np.ndarray,
    crop: np.ndarray | None,
) -> list[dict[str, Any]]:
    frame_bands = maps.get("frame_bands") or {}
    frame_text = (
        f"Bandes neutralisées pour l'analyse — haut={frame_bands.get('top', 0)} px, "
        f"bas={frame_bands.get('bottom', 0)} px, "
        f"gauche={frame_bands.get('left', 0)} px, "
        f"droite={frame_bands.get('right', 0)} px. L'original n'est pas modifié."
    )
    stages = [
        _stage(original, "0 — Image d’entrée", "Image ou page PDF rendue en pixels. Aucun EXIF n’est utilisé."),
        _stage(maps.get("frame_artifact_mask"), "1 — Cadre noir détecté", frame_text),
        _stage(maps.get("analysis_image"), "2 — Copie de travail nettoyée", "Les bandes continues sont remplacées seulement pour la détection."),
        _stage(maps.get("gray"), "3 — Niveaux de gris", "Réduction de l’image à une intensité I(x,y)."),
        _stage(maps.get("smooth"), "4 — CLAHE + filtrage", "Contraste local renforcé, bruit fin atténué."),
        _stage(maps.get("gradient"), "5 — Gradient Sobel", "Amplitude √(Gx²+Gy²) : les changements d’intensité deviennent visibles."),
        _stage(maps.get("edge_union"), "6 — Contours", "Union Canny + gradients forts."),
        _stage(maps.get("connected_edges"), "7 — Bords reconnectés", "Fermeture morphologique pour tolérer de petites ruptures."),
        _stage(maps.get("foreground_mask"), "8 — Premier plan / fond", "Distance LAB au fond après neutralisation du cadre."),
        _stage(maps.get("density_mask"), "9 — Densité / texture", "Concentration locale de contours et variance."),
        _stage(maps.get("line_mask"), "10 — Segments de lignes", "Fragments Hough/LSD ; le cadre continu a déjà été neutralisé."),
        _stage(debug, "11 — Candidats", "Quadrilatères proposés par les détecteurs, avec leurs scores."),
        _stage(selected, "12 — Meilleur quadrilatère", "La fuite locale pénalise un crop qui laisse du document juste à l'extérieur."),
        _stage(crop, "13 — Homographie et crop", "Transformation projective des quatre coins vers un rectangle normalisé."),
    ]
    return [item for item in stages if item is not None]


def run_smart_crop(
    uploaded_file: Any,
    pdf_page: int,
    pdf_dpi: int,
    ratio: float,
    min_area: float,
    max_area: float,
    edge_tolerance: float,
    final_margin: float,
):
    input_path = _as_path(uploaded_file)
    if not input_path.exists():
        raise gr.Error(f"Fichier introuvable : {input_path}")

    config = DetectorConfig(
        expected_aspect_ratio=float(ratio),
        min_area_ratio=float(min_area),
        max_area_ratio=float(max_area),
        edge_tolerance_ratio=float(edge_tolerance),
        final_margin_ratio=float(final_margin),
    )

    try:
        image = load_input(input_path, pdf_page=int(pdf_page), pdf_dpi=int(pdf_dpi))
    except Exception as exc:
        raise gr.Error(f"Impossible de lire le fichier : {exc}") from exc

    started = time.perf_counter()
    best, candidates, maps, scale = detect_card(image, config)
    elapsed = time.perf_counter() - started
    working = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image.copy()
    )
    debug = draw_debug(working, candidates, limit=12)
    selected = _selected_overlay(working, best)
    diagnostics = _diagnostic_mosaic(maps)

    output_dir = Path(tempfile.mkdtemp(prefix="smart_crop_result_"))
    save_image(output_dir / "debug_candidates.jpg", debug)
    save_image(output_dir / "selected_candidate.jpg", selected)
    save_image(output_dir / "diagnostics.jpg", diagnostics)

    result: dict[str, Any] = {
        "input": input_path.name,
        "status": "NO_CARD_FOUND",
        "processing_seconds": round(float(elapsed), 4),
        "working_scale": float(scale),
        "candidate_count": len(candidates),
        "configuration": {
            "expected_aspect_ratio": float(ratio),
            "min_area_ratio": float(min_area),
            "max_area_ratio": float(max_area),
            "edge_tolerance_ratio": float(edge_tolerance),
            "final_margin_ratio": float(final_margin),
            "pdf_page": int(pdf_page),
            "pdf_dpi": int(pdf_dpi),
        },
        "top_candidates": [candidate.to_json() for candidate in candidates[:20]],
        "detected_frame_bands_px": maps.get("frame_bands", {}),
    }

    crop_bgr = None
    crop_rgb = None
    status_text = "❌ Aucune carte détectée. Consulte les étapes et les candidats."

    if best is not None:
        original_quad = order_quad(best.quad) / scale
        crop_strict = warp_card(image, original_quad, config.expected_aspect_ratio, margin_ratio=0.0)
        crop_margin = warp_card(
            image,
            original_quad,
            config.expected_aspect_ratio,
            margin_ratio=config.final_margin_ratio,
        )
        crop_bgr = crop_margin
        save_image(output_dir / "crop_strict.png", crop_strict)
        save_image(output_dir / "crop_margin.png", crop_margin)
        crop_rgb = _bgr_to_rgb(crop_margin)

        state = "SUCCESS" if best.score >= 0.55 else "LOW_CONFIDENCE"
        icon = "✅" if state == "SUCCESS" else "⚠️"
        leakage = float((best.metrics or {}).get("foreground_leakage_ratio", 0.0))
        bands = maps.get("frame_bands", {})
        status_text = (
            f"{icon} **{state}** — score `{best.score:.3f}` — détecteur `{best.source}` — "
            f"fuite locale `{leakage:.3f}` — {len(candidates)} candidat(s) — `{elapsed:.3f} s`. "
            f"Cadre : H `{bands.get('top', 0)}`, B `{bands.get('bottom', 0)}`, "
            f"G `{bands.get('left', 0)}`, D `{bands.get('right', 0)}` px."
        )
        result.update(
            {
                "status": state,
                "confidence": round(float(best.score), 6),
                "source": best.source,
                "quad_original": np.round(original_quad, 2).tolist(),
                "metrics": best.metrics,
            }
        )

    for name, map_image in maps.items():
        if isinstance(map_image, np.ndarray) and map_image.ndim in (2, 3):
            normalized = map_image
            if map_image.dtype not in (np.uint8, np.uint16):
                normalized = cv2.normalize(map_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            save_image(output_dir / f"step_{name}.png", normalized)

    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_path = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))

    stages = _build_stages(image, working, maps, debug, selected, crop_bgr)
    gallery = [(item["image"], item["title"]) for item in stages]

    return (
        _bgr_to_rgb(debug),
        crop_rgb,
        _bgr_to_rgb(diagnostics),
        status_text,
        result,
        str(zip_path),
        gallery,
        stages,
        stages[0]["image"] if stages else None,
        f"### {stages[0]['title']}\n{stages[0]['description']}" if stages else "Aucune étape.",
    )


def play_steps(stages: list[dict[str, Any]] | None, delay: float) -> Generator[tuple[Any, str], None, None]:
    if not stages:
        raise gr.Error("Lance d’abord la détection pour générer les étapes.")
    delay = max(0.0, float(delay))
    total = len(stages)
    for index, item in enumerate(stages, start=1):
        yield (
            item["image"],
            f"### Étape {index}/{total} — {item['title']}\n{item['description']}",
        )
        if delay > 0:
            time.sleep(delay)


def load_lab_file(uploaded_file: Any, pdf_page: int, pdf_dpi: int):
    path = _as_path(uploaded_file)
    try:
        rgb, state = load_for_lab(path, int(pdf_page), int(pdf_dpi))
    except Exception as exc:
        raise gr.Error(f"Impossible de charger le fichier dans le laboratoire : {exc}") from exc
    return rgb, rgb, state, "Image chargée. Dessine au pinceau ou choisis un effet local/global."


def sync_manual_editor(editor_value: Any, state: dict[str, Any] | None):
    image = editor_composite(editor_value)
    if image is None:
        raise gr.Error("Aucune image dans l’éditeur.")
    state = dict(state or {})
    state["current"] = image.copy()
    state.setdefault("original", image.copy())
    state.setdefault("operations", []).append({"type": "manual_editor_commit"})
    preview = draw_annotations(image, state.get("corners") or [])
    return image, preview, state, "Traits manuels validés."


def image_click(
    state: dict[str, Any] | None,
    annotation_mode: bool,
    evt: gr.SelectData,
):
    state = dict(state or {})
    image = state.get("current")
    if image is None:
        return 50.0, 50.0, state, None, "Charge d’abord une image."
    height, width = np.asarray(image).shape[:2]
    index = evt.index
    if isinstance(index, (tuple, list)) and len(index) >= 2:
        x, y = float(index[0]), float(index[1])
    else:
        x, y = width / 2.0, height / 2.0
    x = float(np.clip(x, 0, width - 1))
    y = float(np.clip(y, 0, height - 1))
    x_percent = 100.0 * x / max(width - 1, 1)
    y_percent = 100.0 * y / max(height - 1, 1)

    message = f"Position locale : x={x:.0f}px, y={y:.0f}px."
    if annotation_mode:
        corners = list(state.get("corners") or [])
        if len(corners) >= 4:
            message += " Les 4 coins sont déjà enregistrés ; réinitialise-les pour recommencer."
        else:
            corners.append([x, y])
            state["corners"] = corners
            message += f" Coin {len(corners)}/4 ajouté."
    preview = draw_annotations(np.asarray(image, dtype=np.uint8), state.get("corners") or [])
    return x_percent, y_percent, state, preview, message


def reset_corners(state: dict[str, Any] | None):
    state = dict(state or {})
    state["corners"] = []
    image = state.get("current")
    return state, image, "Coins réinitialisés. Clique dans l’ordre : haut-gauche, haut-droit, bas-droit, bas-gauche."


def apply_local_from_ui(
    editor_value: Any,
    state: dict[str, Any] | None,
    effect: str,
    x_percent: float,
    y_percent: float,
    size: int,
    thickness: int,
    opacity: float,
    angle: float,
    count: int,
    blur: int,
    color: str,
    text: str,
    seed: int,
):
    image = editor_composite(editor_value)
    if image is None:
        raise gr.Error("Aucune image dans le laboratoire.")
    try:
        output, operation = apply_local_effect(
            image,
            effect,
            x_percent,
            y_percent,
            size,
            thickness,
            opacity,
            angle,
            count,
            blur,
            color,
            text,
            seed,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    state = dict(state or {})
    state.setdefault("original", image.copy())
    state["current"] = output.copy()
    state.setdefault("operations", []).append(operation)
    preview = draw_annotations(output, state.get("corners") or [])
    return output, preview, state, f"Effet local « {effect} » appliqué."


def apply_global_from_ui(
    editor_value: Any,
    state: dict[str, Any] | None,
    rotation: float,
    perspective: float,
    translate_x: float,
    translate_y: float,
    brightness: float,
    contrast: float,
    shadow: float,
    shadow_angle: float,
    glare: float,
    glare_x: float,
    glare_y: float,
    focus_blur: int,
    motion_blur: int,
    motion_angle: float,
    gaussian_noise: float,
    jpeg_quality: int,
    downscale: float,
    nonuniform_background: float,
    seed: int,
):
    image = editor_composite(editor_value)
    if image is None:
        raise gr.Error("Aucune image dans le laboratoire.")
    state = dict(state or {})
    try:
        output, corners, operation = apply_global_degradations(
            image,
            state.get("corners") or [],
            rotation,
            perspective,
            translate_x,
            translate_y,
            brightness,
            contrast,
            shadow,
            shadow_angle,
            glare,
            glare_x,
            glare_y,
            focus_blur,
            motion_blur,
            motion_angle,
            gaussian_noise,
            jpeg_quality,
            downscale,
            nonuniform_background,
            seed,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    state.setdefault("original", image.copy())
    state["current"] = output.copy()
    state["corners"] = corners
    state.setdefault("operations", []).append(operation)
    preview = draw_annotations(output, corners)
    return output, preview, state, "Dégradations globales appliquées. Les coins ont été transformés avec la même matrice."


def restore_original(state: dict[str, Any] | None):
    state = dict(state or {})
    original = state.get("original")
    if original is None:
        raise gr.Error("Aucune image originale disponible.")
    output = np.asarray(original, dtype=np.uint8).copy()
    state["current"] = output.copy()
    state["corners"] = []
    state["operations"] = [item for item in state.get("operations", []) if item.get("type") == "load"]
    return output, output, state, "Image originale restaurée. Les coins ont été réinitialisés."


def export_lab_ui(editor_value: Any, state: dict[str, Any] | None):
    image = editor_composite(editor_value)
    state = dict(state or {})
    if image is not None:
        state["current"] = image.copy()
    try:
        exported = export_lab(state)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    message = "Export terminé."
    if not exported.mask_path:
        message += " Le masque n’a pas été créé, car les quatre coins n’ont pas été annotés."
    return exported.zip_path, exported.png_path, exported.pdf_path, exported.json_path, exported.mask_path, message


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Smart Crop & Degradation Lab") as demo:
        gr.Markdown(
            """
# Smart Crop & Degradation Lab — V4

Application locale CPU en trois parties : **détection**, **visualisation pas à pas** et **génération contrôlée de cas dégradés**.
Aucune hypothèse de feuille A4 et aucune utilisation des métadonnées EXIF.
            """
        )

        detection_stage_state = gr.State([])
        lab_state = gr.State({})

        with gr.Tabs():
            with gr.Tab("1 — Smart Crop"):
                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_file = gr.File(
                            label="Image ou PDF",
                            file_types=["image", ".pdf"],
                            type="filepath",
                        )
                        run_button = gr.Button("Détecter et recadrer", variant="primary")
                        with gr.Accordion("Paramètres", open=True):
                            ratio = gr.Number(label="Rapport largeur / hauteur attendu", value=1.586, minimum=1.0, maximum=3.0)
                            min_area = gr.Slider(label="Surface minimale", minimum=0.005, maximum=0.30, value=0.035, step=0.005)
                            max_area = gr.Slider(label="Surface maximale", minimum=0.40, maximum=0.99, value=0.92, step=0.01)
                            edge_tolerance = gr.Slider(label="Tolérance aux ruptures de bord", minimum=0.001, maximum=0.020, value=0.004, step=0.001)
                            final_margin = gr.Slider(label="Marge autour de la carte", minimum=0.0, maximum=0.06, value=0.012, step=0.002)
                            pdf_page = gr.Number(label="Page PDF (0 = première)", value=0, precision=0, minimum=0)
                            pdf_dpi = gr.Slider(label="DPI de conversion PDF", minimum=100, maximum=400, value=220, step=10)
                    with gr.Column(scale=2):
                        status = gr.Markdown("Charge un fichier puis clique sur **Détecter et recadrer**.")
                        with gr.Row():
                            detection_image = gr.Image(label="Détection et candidats", type="numpy")
                            crop_image = gr.Image(label="Crop corrigé", type="numpy")

                with gr.Accordion("Voir toutes les étapes", open=True):
                    stage_gallery = gr.Gallery(label="Étapes du pipeline", columns=3, height=520, allow_preview=True)
                    with gr.Row():
                        delay = gr.Slider(label="Pause entre les étapes (secondes)", minimum=0.0, maximum=3.0, value=0.6, step=0.1)
                        play_button = gr.Button("▶ Lire les étapes")
                    with gr.Row():
                        animated_stage = gr.Image(label="Étape en cours", type="numpy")
                        animated_text = gr.Markdown("Lance la détection, puis clique sur **Lire les étapes**.")

                diagnostic_image = gr.Image(label="Mosaïque des diagnostics", type="numpy")
                with gr.Row():
                    result_json = gr.JSON(label="Scores et résultat JSON")
                    result_zip = gr.File(label="Télécharger tous les résultats")

                run_button.click(
                    fn=run_smart_crop,
                    inputs=[uploaded_file, pdf_page, pdf_dpi, ratio, min_area, max_area, edge_tolerance, final_margin],
                    outputs=[
                        detection_image,
                        crop_image,
                        diagnostic_image,
                        status,
                        result_json,
                        result_zip,
                        stage_gallery,
                        detection_stage_state,
                        animated_stage,
                        animated_text,
                    ],
                )
                play_button.click(
                    fn=play_steps,
                    inputs=[detection_stage_state, delay],
                    outputs=[animated_stage, animated_text],
                )

            with gr.Tab("2 — Laboratoire de dégradation"):
                gr.Markdown(
                    """
### Principe d’utilisation

1. Charge une image ou une page PDF.
2. Dessine librement dans l’éditeur pour créer points, traits ou occultations.
3. Clique sur l’aperçu pour choisir la position d’un effet local.
4. Active **Annotation des coins**, puis clique dans l’ordre : haut-gauche, haut-droit, bas-droit, bas-gauche.
5. Applique les transformations globales et exporte le jeu de test reproductible.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        lab_file = gr.File(label="Image ou PDF", file_types=["image", ".pdf"], type="filepath")
                        lab_pdf_page = gr.Number(label="Page PDF", value=0, precision=0, minimum=0)
                        lab_pdf_dpi = gr.Slider(label="DPI PDF", minimum=100, maximum=400, value=300, step=10)
                        lab_load = gr.Button("Charger dans le laboratoire", variant="primary")
                        lab_status = gr.Markdown("Aucune image chargée.")
                    with gr.Column(scale=2):
                        editor = gr.ImageEditor(
                            label="Canvas manuel — pinceau, gomme, crop et calques",
                            type="numpy",
                            brush=gr.Brush(default_size=12, colors=["#000000", "#FFFFFF", "#777777", "#FF0000", "#FFFF00"], color_mode="defaults"),
                            eraser=gr.Eraser(default_size=18),
                            layers=True,
                            height=540,
                        )
                        with gr.Row():
                            commit_manual = gr.Button("Valider les traits manuels")
                            restore_button = gr.Button("Restaurer l’original")

                with gr.Row():
                    with gr.Column(scale=2):
                        preview = gr.Image(label="Aperçu cliquable — position des effets et annotation des coins", type="numpy", interactive=False)
                    with gr.Column(scale=1):
                        annotation_mode = gr.Checkbox(label="Annotation des quatre coins", value=False)
                        reset_corner_button = gr.Button("Réinitialiser les coins")
                        x_percent = gr.Slider(label="Position X (%)", minimum=0, maximum=100, value=50, step=0.1)
                        y_percent = gr.Slider(label="Position Y (%)", minimum=0, maximum=100, value=50, step=0.1)

                with gr.Accordion("Bruit localisé", open=True):
                    with gr.Row():
                        local_effect = gr.Dropdown(
                            label="Type",
                            choices=["Point", "Tache irrégulière", "Tiret", "Ligne", "Zone blanche", "Occultation noire", "Ombre localisée", "Reflet lumineux", "Poussière", "Bord noir du scanner", "Texte parasite"],
                            value="Tache irrégulière",
                        )
                        local_color = gr.ColorPicker(label="Couleur", value="#000000")
                        local_text = gr.Textbox(label="Texte parasite", value="COPY")
                    with gr.Row():
                        local_size = gr.Slider(label="Taille", minimum=1, maximum=150, value=18, step=1)
                        local_thickness = gr.Slider(label="Épaisseur", minimum=1, maximum=30, value=3, step=1)
                        local_opacity = gr.Slider(label="Opacité", minimum=0.0, maximum=1.0, value=0.65, step=0.05)
                        local_angle = gr.Slider(label="Orientation (°)", minimum=-180, maximum=180, value=0, step=1)
                    with gr.Row():
                        local_count = gr.Slider(label="Nombre", minimum=1, maximum=100, value=8, step=1)
                        local_blur = gr.Slider(label="Flou", minimum=0, maximum=61, value=9, step=2)
                        local_seed = gr.Number(label="Seed", value=1234, precision=0)
                        local_apply = gr.Button("Appliquer l’effet local", variant="primary")

                with gr.Accordion("Dégradations globales", open=False):
                    with gr.Row():
                        rotation = gr.Slider(label="Rotation (°)", minimum=-30, maximum=30, value=0, step=0.5)
                        perspective = gr.Slider(label="Perspective", minimum=0, maximum=0.20, value=0, step=0.005)
                        translate_x = gr.Slider(label="Décalage X (%)", minimum=-40, maximum=40, value=0, step=1)
                        translate_y = gr.Slider(label="Décalage Y (%)", minimum=-40, maximum=40, value=0, step=1)
                    with gr.Row():
                        brightness = gr.Slider(label="Luminosité", minimum=-100, maximum=100, value=0, step=1)
                        contrast = gr.Slider(label="Contraste", minimum=0.2, maximum=2.0, value=1.0, step=0.05)
                        shadow = gr.Slider(label="Ombre progressive", minimum=0, maximum=0.9, value=0, step=0.05)
                        shadow_angle = gr.Slider(label="Angle de l’ombre", minimum=-180, maximum=180, value=0, step=5)
                    with gr.Row():
                        glare = gr.Slider(label="Reflet global", minimum=0, maximum=0.95, value=0, step=0.05)
                        glare_x = gr.Slider(label="Reflet X (%)", minimum=0, maximum=100, value=50, step=1)
                        glare_y = gr.Slider(label="Reflet Y (%)", minimum=0, maximum=100, value=50, step=1)
                        nonuniform = gr.Slider(label="Fond / éclairage non uniforme", minimum=0, maximum=1, value=0, step=0.05)
                    with gr.Row():
                        focus_blur = gr.Slider(label="Flou de mise au point", minimum=0, maximum=31, value=0, step=2)
                        motion_blur = gr.Slider(label="Flou de mouvement", minimum=0, maximum=41, value=0, step=1)
                        motion_angle = gr.Slider(label="Angle du mouvement", minimum=-180, maximum=180, value=0, step=5)
                        gaussian_noise = gr.Slider(label="Bruit numérique σ", minimum=0, maximum=60, value=0, step=1)
                    with gr.Row():
                        jpeg_quality = gr.Slider(label="Qualité JPEG", minimum=10, maximum=100, value=100, step=1)
                        downscale = gr.Slider(label="Facteur de résolution", minimum=0.10, maximum=1.0, value=1.0, step=0.05)
                        global_seed = gr.Number(label="Seed", value=1234, precision=0)
                        global_apply = gr.Button("Appliquer les dégradations globales", variant="primary")

                with gr.Accordion("Exports", open=True):
                    export_button = gr.Button("Exporter le cas de test", variant="primary")
                    with gr.Row():
                        export_zip = gr.File(label="ZIP complet")
                        export_png = gr.File(label="Image dégradée PNG")
                        export_pdf = gr.File(label="PDF dégradé")
                    with gr.Row():
                        export_json = gr.File(label="Paramètres JSON")
                        export_mask = gr.File(label="Masque de la carte")

                lab_load.click(
                    fn=load_lab_file,
                    inputs=[lab_file, lab_pdf_page, lab_pdf_dpi],
                    outputs=[editor, preview, lab_state, lab_status],
                )
                commit_manual.click(
                    fn=sync_manual_editor,
                    inputs=[editor, lab_state],
                    outputs=[editor, preview, lab_state, lab_status],
                )
                preview.select(
                    fn=image_click,
                    inputs=[lab_state, annotation_mode],
                    outputs=[x_percent, y_percent, lab_state, preview, lab_status],
                )
                reset_corner_button.click(
                    fn=reset_corners,
                    inputs=[lab_state],
                    outputs=[lab_state, preview, lab_status],
                )
                local_apply.click(
                    fn=apply_local_from_ui,
                    inputs=[editor, lab_state, local_effect, x_percent, y_percent, local_size, local_thickness, local_opacity, local_angle, local_count, local_blur, local_color, local_text, local_seed],
                    outputs=[editor, preview, lab_state, lab_status],
                )
                global_apply.click(
                    fn=apply_global_from_ui,
                    inputs=[editor, lab_state, rotation, perspective, translate_x, translate_y, brightness, contrast, shadow, shadow_angle, glare, glare_x, glare_y, focus_blur, motion_blur, motion_angle, gaussian_noise, jpeg_quality, downscale, nonuniform, global_seed],
                    outputs=[editor, preview, lab_state, lab_status],
                )
                restore_button.click(
                    fn=restore_original,
                    inputs=[lab_state],
                    outputs=[editor, preview, lab_state, lab_status],
                )
                export_button.click(
                    fn=export_lab_ui,
                    inputs=[editor, lab_state],
                    outputs=[export_zip, export_png, export_pdf, export_json, export_mask, lab_status],
                )

            with gr.Tab("3 — Algorithme expliqué"):
                gr.Markdown(
                    r"""
## Quel algorithme ?

Ce n’est pas encore un réseau neuronal. C’est un **algorithme hybride de vision classique**, entièrement local et CPU :

1. prétraitement et contraste local ;
2. gradient Sobel et contours Canny ;
3. fermeture morphologique des petites ruptures ;
4. détection de lignes Hough et LSD ;
5. détection de régions par densité de texture ;
6. séparation premier plan/fond dans l’espace couleur LAB ;
7. neutralisation des bandes noires continues attachées au cadre ;
8. génération de plusieurs quadrilatères candidats ;
9. pénalité de fuite locale autour de chaque candidat ;
10. score multi-critères ;
11. homographie pour redresser le document.

## Principes mathématiques simples

**Gradient** :

\[
G_x = I * S_x,\qquad G_y = I * S_y,\qquad |G|=\sqrt{G_x^2+G_y^2}
\]

**Ligne de Hough** :

\[
\rho=x\cos\theta+y\sin\theta
\]

**Densité de contours** :

\[
d=\frac{\text{nombre de pixels de contour dans la région}}{\text{nombre total de pixels de la région}}
\]

**Continuité tolérante** : on échantillonne les quatre côtés et on mesure la proportion de points ayant un contour à une distance inférieure à une tolérance \(\varepsilon\).

**Score du candidat** :

\[
S=\sum_i w_i s_i-\sum_j \lambda_j p_j
\]

Les indices positifs sont le rapport largeur/hauteur, le gradient, la continuité,
les angles proches de 90°, la densité et la répartition du premier plan. La
fuite locale mesure le contenu présent dans une couronne autour du candidat :
si la carte continue juste après un bord proposé, ce candidat perd des points.

**Homographie** :

\[
s\begin{bmatrix}u\\v\\1\end{bmatrix}
=H\begin{bmatrix}x\\y\\1\end{bmatrix}
\]

Les quatre coins détectés déterminent la matrice projective \(H\), qui transforme la carte inclinée en rectangle frontal.

## Bibliothèques

- **OpenCV** : gradients, Canny, morphologie, contours, Hough/LSD et homographie ;
- **NumPy** : matrices, statistiques et calculs de score ;
- **PyMuPDF** : rendu d’une page PDF en image ;
- **Pillow** : export image/PDF dans le laboratoire ;
- **Gradio** : interface locale, canvas et lecture animée.
                    """
                )

        gr.Markdown(
            """
### Limite actuelle

Le laboratoire traite une page PDF à la fois. Les quatre coins et le masque sont exportés uniquement après annotation manuelle des quatre coins ; les transformations géométriques appliquées ensuite mettent automatiquement ces coordonnées à jour.
            """
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.queue()
    app.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)
