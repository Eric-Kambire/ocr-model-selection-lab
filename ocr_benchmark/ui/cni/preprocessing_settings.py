"""Composants Gradio relatifs au crop et au prétraitement CNI."""

from dataclasses import dataclass
from typing import Any

import gradio as gr


@dataclass(frozen=True)
class PreprocessingSettings:
    crop_method: Any
    smart_crop_min_score: Any
    smart_crop_margin: Any
    rotation_method: Any
    perspective_correction: Any
    preprocessing: Any


def build_preprocessing_settings(settings: dict[str, Any]) -> PreprocessingSettings:
    """Construit l'onglet des transformations appliquées avant le modèle."""
    with gr.Tab("Prétraitement"):
        crop_method = gr.Radio(
            [
                ("Smart Crop V4 · recommandé", "smart_crop_v4"),
                ("OpenCV historique · comparaison", "legacy_opencv"),
                ("Aucun crop · image entière", "original"),
            ],
            value=settings["crop_method"],
            label="Méthode de détection et de recadrage",
            info="Un score V4 insuffisant provoque l'envoi sûr de l'image entière.",
        )
        with gr.Row():
            smart_crop_min_score = gr.Slider(
                0.30, 0.90, value=settings["smart_crop_min_score"], step=0.01,
                label="Score minimum V4",
            )
            smart_crop_margin = gr.Slider(
                0.0, 0.05, value=settings["smart_crop_margin"], step=0.002,
                label="Marge autour de la carte",
            )
        gr.Markdown(
            "La détection peut utiliser une copie réduite, mais le redressement "
            "final conserve les pixels et le DPI du rendu original."
        )
        with gr.Row():
            rotation_method = gr.Radio(
                [
                    ("Aucune rotation automatique", "none"),
                    ("Pillow · recherche par ratio", "pillow"),
                    ("OpenCV · rectangle orienté", "opencv"),
                ],
                value=settings["rotation_method"],
                label="Rotation supplémentaire",
                info="Laissez « Aucune » avec Smart Crop V4, qui redresse déjà la carte.",
            )
            perspective_correction = gr.Checkbox(
                value=settings["perspective_correction"],
                label="Seconde correction de perspective",
                info="Option de comparaison, déconseillée avec Smart Crop V4.",
            )
        preprocessing = gr.CheckboxGroup(
            [("Améliorer le contraste", "contrast"), ("Réduire le bruit", "denoise")],
            value=settings["preprocessing"],
            label="Améliorations complémentaires",
        )
    return PreprocessingSettings(
        crop_method,
        smart_crop_min_score,
        smart_crop_margin,
        rotation_method,
        perspective_correction,
        preprocessing,
    )
