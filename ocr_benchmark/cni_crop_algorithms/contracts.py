"""Noms stables et descriptions des méthodes de crop exposées à l'UI."""

METHOD_LABELS = {
    "connected_components": "Composants connectés",
    "canny_contours": "Canny + contours quadrilatères",
    "min_area_rect": "Rectangle orienté global (minAreaRect)",
    "pillow_ratio": "Pillow — recherche d'angle par ratio",
    "hybrid_v4": "Hybride V4 — cadre et fuite locale",
}

METHOD_DESCRIPTIONS = {
    "connected_components": (
        "Isole les blocs de pixels connectés, affiche leur aire, remplit le "
        "composant retenu, puis valide sa forme avant le redressement."
    ),
    "canny_contours": (
        "Détecte les bords, ferme les petites coupures, classe plusieurs "
        "quadrilatères et redresse le meilleur candidat crédible."
    ),
    "min_area_rect": (
        "Calcule un rectangle autour de tous les pixels du masque. Cette méthode "
        "est utile comme référence, mais elle reste sensible au bruit éloigné."
    ),
    "pillow_ratio": (
        "Tourne une copie réduite de la page à plusieurs angles et retient le "
        "rectangle englobant dont le ratio est le plus proche d'une CNI."
    ),
    "hybrid_v4": (
        "Combine contours, lignes Hough/LSD, texture et premier plan. Plusieurs "
        "quadrilatères sont classés par continuité des bords, ratio, angles, "
        "densité, cadre noir et contenu laissé juste hors du crop."
    ),
}
