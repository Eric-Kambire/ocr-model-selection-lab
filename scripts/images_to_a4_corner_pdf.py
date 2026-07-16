"""Convertit des images brutes en PDF A4 avec image positionnée dans un coin.

Le script est volontairement interactif par défaut : il liste les images du
dossier, puis accepte ``all`` ou une sélection comme ``1,3-5``. Chaque image
produit son propre PDF, avec le même nom de base, afin de pouvoir ensuite être
utilisée par le scanner de dossiers CNI.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import fitz
from PIL import Image, ImageOps


# Formats volontairement limités aux images usuelles d'un scanner ou téléphone.
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# Une page A4 mesure 210 x 297 mm. PyMuPDF travaille en points PDF (72 points/pouce).
MM_TO_POINTS = 72 / 25.4
A4_WIDTH_POINTS = 210 * MM_TO_POINTS
A4_HEIGHT_POINTS = 297 * MM_TO_POINTS


def list_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Retourne les images prises en charge, dans un ordre stable.

    Args:
        input_dir: Dossier dont le contenu doit être inspecté.
        recursive: Si vrai, inspecte aussi les sous-dossiers.

    Returns:
        Liste de chemins image triés par chemin relatif.

    Raises:
        NotADirectoryError: Si le dossier indiqué n'existe pas.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Dossier introuvable : {input_dir}")

    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path.relative_to(input_dir)).lower(),
    )


def parse_selection(selection: str, total: int) -> list[int]:
    """Convertit ``all`` ou ``1,3-5`` en positions d'images indexées à zéro.

    Args:
        selection: Choix saisi par la personne qui exécute le script.
        total: Nombre total d'images listées.

    Returns:
        Positions distinctes à traiter, dans l'ordre demandé.

    Raises:
        ValueError: Si le format est invalide ou si un index est hors liste.
    """
    normalized = selection.strip().lower()
    if normalized in {"all", "tout", "*"}:
        return list(range(total))
    if not normalized:
        raise ValueError("La sélection est vide. Utilisez all ou, par exemple, 1,3-5.")

    positions: list[int] = []
    for part in normalized.split(","):
        token = part.strip()
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", token)
        if not match:
            raise ValueError(f"Sélection invalide : {token!r}. Exemple valide : 1,3-5")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first < 1 or last < first or last > total:
            raise ValueError(f"Intervalle invalide : {token!r}. Les positions vont de 1 à {total}.")
        for number in range(first, last + 1):
            position = number - 1
            if position not in positions:
                positions.append(position)
    return positions


def image_to_jpeg_bytes(image_path: Path, quality: int = 95) -> tuple[bytes, int, int]:
    """Lit une image, applique son orientation EXIF et retourne un JPEG portable.

    La conversion JPEG évite que les formats WebP, TIFF ou PNG avec transparence
    posent problème à la génération du PDF. L'image originale n'est jamais modifiée.
    """
    with Image.open(image_path) as source:
        corrected = ImageOps.exif_transpose(source)
        if corrected.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", corrected.size, "white")
            alpha = corrected.getchannel("A")
            background.paste(corrected, mask=alpha)
            converted = background
        else:
            converted = corrected.convert("RGB")
        width, height = converted.size
        payload = io.BytesIO()
        converted.save(payload, format="JPEG", quality=quality, optimize=True)
    return payload.getvalue(), width, height


def a4_corner_rect(
    image_width: int,
    image_height: int,
    *,
    corner: str,
    margin_mm: float,
    max_width_mm: float,
    max_height_mm: float,
) -> fitz.Rect:
    """Calcule le rectangle d'une image, sans la déformer, sur une feuille A4."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Les dimensions de l'image doivent être strictement positives.")
    if min(margin_mm, max_width_mm, max_height_mm) <= 0:
        raise ValueError("La marge et les dimensions maximales doivent être strictement positives.")

    margin = margin_mm * MM_TO_POINTS
    max_width = max_width_mm * MM_TO_POINTS
    max_height = max_height_mm * MM_TO_POINTS
    if 2 * margin + max_width > A4_WIDTH_POINTS or 2 * margin + max_height > A4_HEIGHT_POINTS:
        raise ValueError("L'image maximale et les marges ne tiennent pas sur une feuille A4.")

    scale = min(max_width / image_width, max_height / image_height)
    width = image_width * scale
    height = image_height * scale
    left = margin if corner.endswith("left") else A4_WIDTH_POINTS - margin - width
    top = margin if corner.startswith("top") else A4_HEIGHT_POINTS - margin - height
    return fitz.Rect(left, top, left + width, top + height)


def create_a4_pdf(
    image_path: Path,
    output_path: Path,
    *,
    corner: str = "top-right",
    margin_mm: float = 12,
    max_width_mm: float = 120,
    max_height_mm: float = 90,
) -> None:
    """Crée un PDF d'une page A4 contenant une image dans le coin demandé.

    Args:
        image_path: Fichier image source, non modifié.
        output_path: PDF à créer. Son dossier parent est créé au besoin.
        corner: ``top-left``, ``top-right``, ``bottom-left`` ou ``bottom-right``.
        margin_mm: Distance entre l'image et les bords A4.
        max_width_mm: Largeur maximale réservée à l'image.
        max_height_mm: Hauteur maximale réservée à l'image.
    """
    if corner not in {"top-left", "top-right", "bottom-left", "bottom-right"}:
        raise ValueError("corner doit être top-left, top-right, bottom-left ou bottom-right.")

    jpeg, width, height = image_to_jpeg_bytes(image_path)
    rectangle = a4_corner_rect(
        width,
        height,
        corner=corner,
        margin_mm=margin_mm,
        max_width_mm=max_width_mm,
        max_height_mm=max_height_mm,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Chaque document contient exactement une page A4. ``insert_image`` garde
    # l'image nette et son ratio d'origine, sans rééchantillonner la source.
    document = fitz.open()
    try:
        page = document.new_page(width=A4_WIDTH_POINTS, height=A4_HEIGHT_POINTS)
        page.insert_image(rectangle, stream=jpeg, keep_proportion=True)
        document.save(output_path, deflate=True, garbage=4)
    finally:
        document.close()


def choose_images(images: list[Path]) -> list[Path]:
    """Affiche une liste numérotée et demande une sélection interactive."""
    for number, image_path in enumerate(images, start=1):
        print(f"{number:>3}. {image_path.name}")
    print("\nChoix : all (tout) ou une liste, par exemple 1,3-5.")
    selection = input("Images à convertir : ")
    return [images[position] for position in parse_selection(selection, len(images))]


def build_parser() -> argparse.ArgumentParser:
    """Construit l'interface en ligne de commande du convertisseur."""
    parser = argparse.ArgumentParser(
        description="Place des images dans un coin d'une page A4 PDF, une image par PDF."
    )
    parser.add_argument("input_dir", nargs="?", help="Dossier contenant les images brutes.")
    parser.add_argument("--output-dir", help="Dossier de sortie. Par défaut : le dossier source.")
    parser.add_argument("--select", help="all ou une sélection non interactive, ex. 1,3-5.")
    parser.add_argument("--recursive", action="store_true", help="Inclut les images des sous-dossiers.")
    parser.add_argument("--corner", choices=["top-left", "top-right", "bottom-left", "bottom-right"], default="top-right")
    parser.add_argument("--margin-mm", type=float, default=12, help="Marge A4 en mm (défaut : 12).")
    parser.add_argument("--max-width-mm", type=float, default=120, help="Largeur maximale de l'image en mm (défaut : 120).")
    parser.add_argument("--max-height-mm", type=float, default=90, help="Hauteur maximale de l'image en mm (défaut : 90).")
    parser.add_argument("--overwrite", action="store_true", help="Remplace un PDF déjà existant.")
    return parser


def main() -> None:
    """Point d'entrée : demande le dossier, sélectionne les images et les convertit."""
    args = build_parser().parse_args()
    input_text = args.input_dir or input("Chemin du dossier d'images : ").strip()
    input_dir = Path(input_text).expanduser().resolve()
    images = list_images(input_dir, recursive=args.recursive)
    if not images:
        print(f"Aucune image prise en charge dans : {input_dir}")
        return

    selected_images = (
        [images[position] for position in parse_selection(args.select, len(images))]
        if args.select is not None
        else choose_images(images)
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    print(f"\nSortie : {output_dir}\nCoin A4 : {args.corner}\n")

    created = 0
    skipped = 0
    for image_path in selected_images:
        # Avec --recursive, le sous-dossier source est conservé dans la sortie.
        relative_parent = image_path.parent.relative_to(input_dir) if args.recursive else Path()
        output_path = output_dir / relative_parent / f"{image_path.stem}.pdf"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[ignoré] existe déjà : {output_path}")
            continue
        create_a4_pdf(
            image_path,
            output_path,
            corner=args.corner,
            margin_mm=args.margin_mm,
            max_width_mm=args.max_width_mm,
            max_height_mm=args.max_height_mm,
        )
        created += 1
        print(f"[créé] {output_path}")

    print(f"\nTerminé : {created} PDF créé(s), {skipped} ignoré(s).")


if __name__ == "__main__":
    main()
