"""Extrait automatiquement une CNI placée sur une page PDF A4.

Ce script réutilise exactement les fonctions de production du benchmark CNI :
il ne contient pas une seconde logique de recadrage à maintenir.

Exemples PowerShell :

    python scripts/crop_cni_from_pdf.py "D:/data/client_CIN_Recto.pdf"
    python scripts/crop_cni_from_pdf.py "D:/data/client_CIN_Recto.pdf" --output "D:/sortie/recto.png" --dpi 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permet d'exécuter le script depuis n'importe quel dossier, tout en important
# le module du projet situé un niveau au-dessus de ``scripts``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark.cni_images import crop_cni_from_a4, render_single_page_pdf


def crop_cni_pdf(pdf_path: Path, output_path: Path, dpi: int = 300) -> dict[str, object]:
    """Rend le PDF mono-page puis recadre sa CNI.

    Entrées :
        pdf_path : PDF contenant exactement une page A4 et une CNI.
        output_path : image PNG finale de la CNI, ou page entière en fallback.
        dpi : résolution de rendu entre 72 et 600.

    Sortie :
        métadonnées avec le statut de détection, le chemin et le rectangle.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier doit être un PDF : {pdf_path.name}")

    # Image intermédiaire : elle est gardée près de la sortie pour permettre
    # de comparer visuellement la page A4 au crop lors d'un diagnostic.
    rendered_page = output_path.with_name(f"{output_path.stem}_page.png")
    render_info = render_single_page_pdf(pdf_path, rendered_page, dpi=dpi)
    crop_info = crop_cni_from_a4(rendered_page, output_path)

    return {
        "source_pdf": str(pdf_path.resolve()),
        "rendered_page": render_info,
        "crop": crop_info,
    }


def main() -> None:
    """Parse les paramètres CLI et affiche un rapport JSON lisible."""
    parser = argparse.ArgumentParser(description="Rend un PDF A4 puis extrait la zone CNI détectée.")
    parser.add_argument("pdf", type=Path, help="Chemin du PDF mono-page à traiter.")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG de sortie. Par défaut : <nom>_cni.png dans le dossier du PDF.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Résolution de rendu, entre 72 et 600 (défaut : 300).")
    args = parser.parse_args()

    pdf_path = args.pdf.expanduser().resolve()
    output_path = (args.output.expanduser().resolve() if args.output else pdf_path.with_name(f"{pdf_path.stem}_cni.png"))
    result = crop_cni_pdf(pdf_path, output_path, dpi=args.dpi)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
