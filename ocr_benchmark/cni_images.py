"""Préparation des images pour les entrées CNI marocaines.

Ce module ne connaît ni les labels ni les modèles. Il transforme le contrat
PDF mono-page en artefacts PNG reproductibles pour pouvoir diagnostiquer un OCR.
"""

# ─── POURQUOI CES IMPORTS ? ───────────────────────────────────────────────────

# 'annotations' : permet d'écrire les types modernes (ex: Path | None) sur
# les versions Python < 3.10 sans erreur.
from __future__ import annotations

# 'Path' : représentation moderne d'un chemin de fichier. Fonctionne sur
# Windows (\) et Linux/Mac (/) de façon transparente.
from pathlib import Path

# 'Any' : type générique pour annoter qu'une valeur peut être de n'importe
# quel type. Utilisé dans les types de retour des fonctions.
from typing import Any

# On importe trois outils de la bibliothèque Pillow (PIL) pour manipuler des images :
# - Image : l'objet principal représentant une image (ouvrir, sauvegarder, copier...).
# - ImageDraw : permet de dessiner sur une image (lignes, textes, formes...).
# - ImageOps : opérations utilitaires sur les images (conversion, niveaux de gris...).
from PIL import Image, ImageDraw, ImageOps

# NOTE : 'fitz' (PyMuPDF) est importé en "import tardif" DANS la fonction
# render_single_page_pdf et non ici au niveau du module. Cela permet à
# l'application de démarrer même si PyMuPDF n'est pas installé, tant qu'on
# n'utilise pas cette fonction spécifique.


# ─── FONCTION 1 : render_single_page_pdf ──────────────────────────────────────
# CE QU'ELLE FAIT : ouvre un fichier PDF qui contient UNE seule page (la face
# recto OU verso d'une CNI), et le convertit en image PNG à haute résolution.
# CE QU'ON FAIT AVEC : avant d'envoyer la CNI à un modèle OCR, on doit avoir
# une image. Le modèle ne peut pas lire un PDF directement.

def render_single_page_pdf(pdf_path: Path, output_path: Path, dpi: int = 300) -> dict[str, Any]:
    """Rend un PDF CNI mono-page en PNG et retourne ses métadonnées."""

    # On valide que le DPI demandé est dans une plage raisonnable.
    # En dessous de 72 : image trop floue pour l'OCR.
    # Au-dessus de 600 : fichier inutilement lourd sans gain de qualité réel.
    if dpi < 72 or dpi > 600:
        raise ValueError("CNI render DPI must be between 72 and 600.")

    # On tente d'importer 'fitz' (le vrai nom du module PyMuPDF).
    # Import tardif : on n'importe fitz que si on en a besoin, pas au démarrage
    # de l'application. Cela évite un crash si PyMuPDF n'est pas installé.
    try:
        import fitz
    except ImportError as exc:
        # Si PyMuPDF n'est pas installé, on lève une erreur claire avec instructions.
        raise RuntimeError("PyMuPDF is required to render CNI PDFs. Install requirements.txt.") from exc

    # On ouvre le document PDF avec fitz. Le bloc 'with' garantit la fermeture
    # du fichier même si une erreur survient.
    with fitz.open(pdf_path) as document:
        # On vérifie que le PDF a exactement 1 page.
        # Un PDF CNI ne doit contenir qu'une seule face. S'il en a plusieurs,
        # il s'agit probablement d'un mauvais fichier envoyé par erreur.
        if document.page_count != 1:
            raise ValueError(
                f"Expected exactly one PDF page, found {document.page_count}: {pdf_path.name}"
            )

        # On charge la première (et unique) page. L'index commence à 0 en Python.
        page = document.load_page(0)

        # PyMuPDF travaille nativement à 72 DPI. Pour augmenter la résolution,
        # on crée une "matrice de transformation" qui multiplie les pixels.
        # Exemple : DPI 300 → facteur de zoom = 300/72 ≈ 4.17 → image 4x plus grande.
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        # alpha=False : on n'a pas besoin de transparence, on veut une image RGB pure.

        # On crée le dossier parent du fichier de sortie s'il n'existe pas.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # On sauvegarde le pixmap (l'image en mémoire) sur le disque en PNG.
        pixmap.save(str(output_path))

    # On rouvre l'image sauvegardée pour lire ses dimensions exactes
    # (largeur et hauteur en pixels) et les inclure dans le résultat.
    with Image.open(output_path) as image:
        width, height = image.size  # image.size retourne le tuple (largeur, hauteur).

    # On retourne un dictionnaire avec les métadonnées de l'image créée.
    return {"image_path": str(output_path), "width": width, "height": height, "dpi": dpi}


# ─── FONCTION 2 : crop_cni_from_a4 ───────────────────────────────────────────
# CE QU'ELLE FAIT : la CNI est souvent scannée posée sur une grande feuille A4
# blanche. Cette fonction essaie de détecter la carte et de la "couper" de la
# feuille pour ne garder que la carte, sans le fond blanc autour.
# CE QU'ON FAIT AVEC : avant d'envoyer l'image au modèle OCR, on veut qu'il
# voie uniquement la CNI, pas tout l'espace vide autour.

def crop_cni_from_a4(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Tente de recadrer une CNI posée sur une feuille A4 blanche."""

    # On ouvre l'image source dans un bloc 'with' pour fermeture automatique.
    with Image.open(source_path) as source:
        # exif_transpose : corrige l'orientation si l'image a été prise avec un
        # téléphone et que ses métadonnées EXIF indiquent une rotation.
        # convert("RGB") : on force le format RGB (3 canaux) pour une manipulation uniforme.
        original = ImageOps.exif_transpose(source).convert("RGB")

    # On convertit l'image en niveaux de gris, puis on applique un seuil :
    # - Les pixels dont la valeur est inférieure à 242 (presque blanc) → 255 (blanc pur).
    # - Les autres pixels (plus foncés, donc potentiellement la carte) → 0 (noir).
    # Enfin, getbbox() retourne le rectangle englobant les pixels NON-blancs (=la carte).
    # Si l'image est entièrement blanche, getbbox() retourne None.
    bbox = ImageOps.grayscale(original).point(lambda pixel: 255 if pixel < 242 else 0).getbbox()

    # Si aucune zone non-blanche n'est détectée, il n'y a rien à recadrer.
    if bbox is None:
        return _copy_full_page(original, output_path, "crop_not_detected")

    # On décompose le résultat de getbbox() en coordonnées : gauche, haut, droite, bas.
    left, top, right, bottom = bbox

    # On calcule une marge de sécurité autour du contour détecté pour ne pas
    # couper les bords imprimés de la CNI. La marge est proportionnelle à la taille
    # de l'image mais toujours d'au moins 12 pixels.
    padding = max(12, int(max(original.size) * 0.015))

    # On élargit le cadre de détection en ajoutant la marge, en s'assurant de ne
    # pas dépasser les limites de l'image (max 0 pour les coins haut/gauche,
    # min(largeur/hauteur) pour les coins bas/droite).
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(original.width, right + padding), min(original.height, bottom + padding)

    # On calcule la largeur et la hauteur de la zone détectée.
    width, height = right - left, bottom - top

    # On calcule le ratio largeur/hauteur de la zone détectée.
    # Si height=0 (cas impossible mais défensif), on met 0 pour éviter une division par zéro.
    ratio = width / height if height else 0

    # On calcule la "couverture" : la proportion de l'image totale que représente
    # la zone détectée. Ex: 0.30 = la zone fait 30% de l'image originale.
    coverage = (width * height) / (original.width * original.height)

    # On valide que la zone détectée ressemble vraiment à une CNI :
    # - ratio entre 1.20 et 2.05 : une CNI a un ratio ISO ID-1 d'environ 1.586.
    #   La tolérance absorbe les légères distorsions de perspective ou d'ombre.
    # - coverage entre 0.02 et 0.65 : si la zone détectée couvre plus de 65%
    #   de l'image, on a probablement raté la détection (on "voit" encore toute la feuille A4).
    if not 1.20 <= ratio <= 2.05 or coverage > 0.65 or coverage < 0.02:
        # La zone ne ressemble pas à une CNI → on utilise l'image complète en fallback.
        return _copy_full_page(original, output_path, "crop_fallback_full_page")

    # On crée le dossier parent de l'image de sortie si nécessaire.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # On recadre l'image originale selon les coordonnées validées et on la sauvegarde.
    original.crop((left, top, right, bottom)).save(output_path, format="PNG")

    # On retourne les métadonnées du crop : chemin, statut, coordonnées et coverage.
    return {
        "image_path": str(output_path),
        "crop_status": "crop_detected",       # Le crop a réussi.
        "crop_box": [left, top, right, bottom],  # Les coordonnées exactes du crop.
        "coverage": round(coverage, 4),        # La proportion de l'image originale (4 décimales).
    }


# ─── FONCTION 3 : build_vertical_cni_composite ────────────────────────────────
# CE QU'ELLE FAIT : colle l'image recto ET l'image verso l'une sous l'autre sur
# un même "canvas" blanc, avec une ligne de séparation et des labels "RECTO"/"VERSO".
# CE QU'ON FAIT AVEC : certains modèles peuvent traiter les deux faces en une seule
# image. Cela réduit aussi le nombre d'appels API (1 appel au lieu de 2).

def build_vertical_cni_composite(recto_path: Path, verso_path: Path, output_path: Path) -> str:
    """Construit une image recto-dessus-verso pour la stratégie combinée."""

    # On ouvre l'image recto et on la prépare (correction EXIF + format RGB).
    with Image.open(recto_path) as source:
        recto = ImageOps.exif_transpose(source).convert("RGB")

    # Même chose pour l'image verso.
    with Image.open(verso_path) as source:
        verso = ImageOps.exif_transpose(source).convert("RGB")

    # On calcule la largeur cible : la plus grande des deux images.
    # Les deux images auront la même largeur finale pour un résultat propre.
    target_width = max(recto.width, verso.width)

    # On redimensionne les deux images à cette largeur commune (en gardant
    # les proportions). Si une image a déjà la bonne largeur, elle n'est pas modifiée.
    recto, verso = _resize_to_width(recto, target_width), _resize_to_width(verso, target_width)

    # On définit la hauteur de l'espace blanc entre les deux images (en pixels).
    separator = 36  # 36 pixels d'espace de séparation.

    # On crée un nouveau canvas (image vierge) de couleur blanche.
    # Sa largeur = largeur commune des deux images.
    # Sa hauteur = hauteur recto + espace séparateur + hauteur verso.
    canvas = Image.new("RGB", (target_width, recto.height + separator + verso.height), "white")

    # On colle l'image recto tout en haut du canvas (coordonnées x=0, y=0).
    canvas.paste(recto, (0, 0))
    # On colle l'image verso sous l'espace de séparation.
    canvas.paste(verso, (0, recto.height + separator))

    # On crée un objet de dessin attaché au canvas pour y ajouter des éléments.
    draw = ImageDraw.Draw(canvas)

    # On trace une ligne noire horizontale au milieu de l'espace de séparation.
    # La ligne va de x=0 à x=target_width, à y = hauteur recto + moitié du séparateur.
    draw.line(
        (0, recto.height + separator // 2, target_width, recto.height + separator // 2),
        fill="black", width=2,
    )

    # On écrit le texte "RECTO" en haut à gauche de la première image.
    draw.text((8, 8), "RECTO", fill="black")

    # On écrit le texte "VERSO" en haut à gauche de la deuxième image.
    draw.text((8, recto.height + separator + 8), "VERSO", fill="black")

    # On crée le dossier parent du fichier de sortie si nécessaire.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # On sauvegarde l'image composite finale au format PNG.
    canvas.save(output_path, format="PNG")

    # On retourne le chemin de l'image composite sous forme de texte.
    return str(output_path)


# ─── FONCTION INTERNE : _copy_full_page ───────────────────────────────────────
# CE QU'ELLE FAIT : quand le crop automatique échoue (la CNI n'a pas pu être
# détectée), on sauvegarde simplement l'image complète et on enregistre
# le statut d'échec pour que le reste du code sache ce qui s'est passé.
# CE QU'ON FAIT AVEC : utilisée comme "plan B" dans crop_cni_from_a4.

def _copy_full_page(image: Image.Image, output_path: Path, status: str) -> dict[str, Any]:
    """Enregistre le repli A4 complet et le statut expliquant le choix."""

    # On crée le dossier parent si nécessaire.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # On sauvegarde l'image complète (sans crop) au format PNG.
    image.save(output_path, format="PNG")

    # On retourne les métadonnées avec crop_box=None et coverage=None pour
    # signifier qu'aucun recadrage n'a eu lieu.
    return {"image_path": str(output_path), "crop_status": status, "crop_box": None, "coverage": None}


# ─── FONCTION INTERNE : _resize_to_width ─────────────────────────────────────
# CE QU'ELLE FAIT : redimensionne une image à une largeur donnée en gardant
# les proportions (pour éviter de déformer le document).
# CE QU'ON FAIT AVEC : utilisée dans build_vertical_cni_composite pour que
# recto et verso aient exactement la même largeur.

def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    """Redimensionne proportionnellement sans déformer le document."""

    # Si l'image a déjà exactement la bonne largeur, on la retourne sans rien faire.
    if image.width == width:
        return image

    # On calcule la nouvelle hauteur proportionnelle à la largeur cible.
    # Formule : nouvelle_hauteur = hauteur_originale × (nouvelle_largeur / largeur_originale).
    # round() arrondit au pixel entier le plus proche.
    # Image.Resampling.LANCZOS est l'algorithme de redimensionnement de meilleure
    # qualité disponible dans Pillow (préserve bien les détails des textes).
    return image.resize(
        (width, round(image.height * width / image.width)),
        Image.Resampling.LANCZOS,
    )
