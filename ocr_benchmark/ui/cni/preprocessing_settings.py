"""Composants Gradio relatifs au crop et au prétraitement CNI."""

from dataclasses import dataclass
from typing import Any

import gradio as gr


@dataclass(frozen=True)
class PreprocessingSettings:
    crop_method: Any
    smart_crop_controls: Any
    smart_crop_min_score: Any
    smart_crop_margin: Any
    rotation_method: Any
    perspective_correction: Any
    preprocessing: Any


def build_preprocessing_settings(settings: dict[str, Any]) -> PreprocessingSettings:
    """Construit l'onglet des transformations appliquées avant le modèle."""
    with gr.Tab("Prétraitement"):
        gr.Markdown(
            "**1 · Détecter et recadrer la carte**\n\n"
            "Une seule méthode principale est exécutée. Smart Crop V4 inclut "
            "déjà le redressement de perspective et revient à l'image entière "
            "si sa détection n'est pas assez fiable."
        )
        crop_method = gr.Radio(
            [
                ("Smart Crop V4 · recommandé", "smart_crop_v4"),
                (
                    "Composants connectés · objets dominants",
                    "connected_components",
                ),
                (
                    "Canny + contours quadrilatères · bords visibles",
                    "canny_contours",
                ),
                ("OpenCV historique · comparaison", "legacy_opencv"),
                ("Aucun crop · image entière", "original"),
            ],
            value=settings["crop_method"],
            label="Méthode de détection et de recadrage",
            info=(
                "Chaque méthode refuse un candidat incohérent et transmet alors "
                "l’image entière."
            ),
        )
        with gr.Group(
            visible=settings["crop_method"] == "smart_crop_v4"
        ) as smart_crop_controls:
            with gr.Row():
                smart_crop_min_score = gr.Slider(
                    0.30, 0.90, value=settings["smart_crop_min_score"], step=0.01,
                    label="Score minimum V4",
                    info=(
                        "Défaut recommandé : 0,55. Plus la valeur est haute, plus "
                        "le détecteur préfère conserver l'image entière plutôt qu'un crop incertain."
                    ),
                )
                smart_crop_margin = gr.Slider(
                    0.0, 0.05, value=settings["smart_crop_margin"], step=0.002,
                    label="Marge autour de la carte",
                    info=(
                        "Défaut recommandé : 0,012, soit 1,2 % autour du quadrilatère. "
                        "Cette marge évite de couper un bord ou un caractère proche du contour."
                    ),
                )
            gr.Markdown(
                "La détection peut utiliser une copie réduite, mais le redressement "
                "final conserve les pixels et le DPI du rendu original."
            )
        gr.Markdown(
            "**Aide au choix :** Composants connectés sépare les objets isolés et "
            "tente de casser les ponts fins causés par une ligne ou une ombre. "
            "Canny recherche principalement quatre bords. V4 compare ces indices "
            "avec les lignes, la texture et le premier plan."
        )

        gr.Markdown("**2 · Améliorer l'image envoyée au modèle**")
        preprocessing = gr.CheckboxGroup(
            [("Améliorer le contraste", "contrast"), ("Réduire le bruit", "denoise")],
            value=settings["preprocessing"],
            label="Améliorations facultatives",
        )

        # Ces transformations peuvent répéter le travail de Smart Crop V4.
        # Elles restent disponibles pour comparer les anciennes méthodes, mais
        # sont repliées afin d'éviter une configuration contradictoire par erreur.
        with gr.Accordion("Comparaison avancée · rotation et seconde perspective", open=False):
            gr.Markdown(
                "À utiliser pour comparer les anciennes méthodes. Avec Smart "
                "Crop V4, conservez les deux options désactivées."
            )
            with gr.Row():
                rotation_method = gr.Radio(
                    [
                        ("Aucune rotation supplémentaire", "none"),
                        ("Pillow · recherche par ratio", "pillow"),
                        ("OpenCV · rectangle orienté", "opencv"),
                    ],
                    value=settings["rotation_method"],
                    label="Rotation supplémentaire",
                )
                perspective_correction = gr.Checkbox(
                    value=settings["perspective_correction"],
                    label="Appliquer une seconde correction de perspective",
                )
    return PreprocessingSettings(
        crop_method,
        smart_crop_controls,
        smart_crop_min_score,
        smart_crop_margin,
        rotation_method,
        perspective_correction,
        preprocessing,
    )
