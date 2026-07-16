"""Assemble des paires d'images CNI Recto/Verso en un collage vertical.

Les paires sont reconnues avec le même préfixe avant le dernier mot Recto ou
Verso. Par exemple, ``123_CIN_Recto.jpg`` et ``123_CIN_Verso.png`` deviennent
``123_CIN_RectoVerso.png`` avec le recto en haut et le verso en bas.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SIDE_PATTERN = re.compile(r"^(?P<pair_key>.+?)[\s_.-]+(?P<side>recto|verso)$", re.IGNORECASE)


def classify_image(image_path: Path) -> tuple[str, str] | None:
    """Lit la clé commune et la face Recto/Verso depuis un nom d'image."""
    match = SIDE_PATTERN.fullmatch(image_path.stem)
    if not match:
        return None
    pair_key = match.group("pair_key").strip(" ._-")
    return (pair_key, match.group("side").lower()) if pair_key else None


def find_image_pairs(input_dir: Path, recursive: bool = False) -> tuple[list[dict[str, Path | str]], list[Path], list[str]]:
    """Détecte les paires complètes et retourne aussi les anomalies visibles."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Dossier introuvable : {input_dir}")
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    grouped: dict[str, dict[str, Path]] = {}
    unmatched: list[Path] = []
    issues: list[str] = []

    for image_path in sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path.relative_to(input_dir)).lower(),
    ):
        classified = classify_image(image_path)
        if classified is None:
            unmatched.append(image_path)
            continue
        pair_key, side = classified
        faces = grouped.setdefault(pair_key.casefold(), {})
        if side in faces:
            issues.append(f"Doublon {side} pour '{pair_key}' : {faces[side].name} et {image_path.name}")
            continue
        faces[side] = image_path

    pairs: list[dict[str, Path | str]] = []
    for key, faces in grouped.items():
        if "recto" not in faces or "verso" not in faces:
            unmatched.extend(faces.values())
            missing = "verso" if "recto" in faces else "recto"
            issues.append(f"Paire incomplète '{key}' : {missing} absent")
            continue
        pair_key = classify_image(faces["recto"])
        if pair_key is None:
            issues.append(f"Nom recto invalide : {faces['recto'].name}")
            continue
        pairs.append({"pair_key": pair_key[0], "recto": faces["recto"], "verso": faces["verso"]})
    return pairs, unmatched, issues


def _open_rgb(image_path: Path) -> Image.Image:
    """Ouvre une image avec orientation EXIF correcte et fond blanc si alpha."""
    with Image.open(image_path) as source:
        normalized = ImageOps.exif_transpose(source)
        if normalized.mode in {"RGBA", "LA"}:
            canvas = Image.new("RGB", normalized.size, "white")
            canvas.paste(normalized, mask=normalized.getchannel("A"))
            return canvas
        return normalized.convert("RGB")


def combine_vertical(recto_path: Path, verso_path: Path, output_path: Path, *, separator_px: int = 24, output_format: str = "png", jpeg_quality: int = 95, png_compress_level: int = 6) -> None:
    """Colle recto au-dessus du verso, sans crop ni redimensionnement par défaut.

    Le PNG est le format par défaut : il est sans perte. Choisir JPEG active
    une compression avec une qualité réglable de 1 à 100.
    """
    if separator_px < 0:
        raise ValueError("separator_px doit être positif ou nul.")
    if output_format not in {"png", "jpeg"}:
        raise ValueError("output_format doit être png ou jpeg.")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality doit être compris entre 1 et 100.")
    if not 0 <= png_compress_level <= 9:
        raise ValueError("png_compress_level doit être compris entre 0 et 9.")

    recto = _open_rgb(recto_path)
    verso = _open_rgb(verso_path)
    width = max(recto.width, verso.width)
    height = recto.height + separator_px + verso.height
    collage = Image.new("RGB", (width, height), "white")

    # Les deux faces sont centrées, mais leurs pixels ne sont jamais coupés,
    # agrandis ou rétrécis : la qualité source est donc conservée en PNG.
    collage.paste(recto, ((width - recto.width) // 2, 0))
    collage.paste(verso, ((width - verso.width) // 2, recto.height + separator_px))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "png":
        collage.save(output_path, format="PNG", compress_level=png_compress_level)
    else:
        collage.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)


def build_parser() -> argparse.ArgumentParser:
    """Construit les options du collage d'images CNI."""
    parser = argparse.ArgumentParser(description="Assemble les paires image Recto/Verso CNI en une image verticale.")
    parser.add_argument("input_dir", help="Dossier contenant les images Recto/Verso.")
    parser.add_argument("--output-dir", help="Sortie. Par défaut : <input_dir>/combined_images.")
    parser.add_argument("--recursive", action="store_true", help="Inclut les sous-dossiers.")
    parser.add_argument("--format", choices=["png", "jpeg"], default="png", help="png = sans perte (défaut), jpeg = compressé.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Qualité JPEG 1-100 (défaut : 95).")
    parser.add_argument("--png-compress-level", type=int, default=6, help="Compression PNG sans perte 0-9 (défaut : 6).")
    parser.add_argument("--separator-px", type=int, default=24, help="Espace blanc entre recto et verso (défaut : 24).")
    parser.add_argument("--overwrite", action="store_true", help="Remplace une image combinée existante.")
    parser.add_argument("--dry-run", action="store_true", help="Liste les paires sans créer d'image.")
    return parser


def main() -> None:
    """Liste les paires puis produit les collages demandés."""
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir / "combined_images"
    extension = ".png" if args.format == "png" else ".jpg"
    pairs, unmatched, issues = find_image_pairs(input_dir, recursive=args.recursive)

    print(f"Paires complètes détectées : {len(pairs)}")
    for issue in issues:
        print(f"[alerte] {issue}")
    for image_path in unmatched:
        print(f"[non apparié] {image_path}")
    if args.dry_run:
        for pair in pairs:
            print(f"[paire] {pair['pair_key']} → {Path(pair['recto']).name} + {Path(pair['verso']).name}")
        print("Dry-run : aucune image créée.")
        return

    created = 0
    skipped = 0
    for pair in pairs:
        recto = Path(pair["recto"])
        verso = Path(pair["verso"])
        output_path = output_dir / f"{pair['pair_key']}_RectoVerso{extension}"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[ignoré] existe déjà : {output_path}")
            continue
        combine_vertical(
            recto,
            verso,
            output_path,
            separator_px=args.separator_px,
            output_format=args.format,
            jpeg_quality=args.jpeg_quality,
            png_compress_level=args.png_compress_level,
        )
        created += 1
        print(f"[créé] {output_path}")
    print(f"Terminé : {created} collage(s) créé(s), {skipped} ignoré(s).")


if __name__ == "__main__":
    main()
