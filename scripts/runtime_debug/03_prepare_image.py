"""Convertit un PDF/JPEG/PNG comme le workflow CNI, sans lancer Ollama."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from PIL import Image


## Racine du dépôt utilisée pour importer le véritable prétraitement CNI.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

## Aucune conversion n'est réécrite ici : on teste exactement la fonction applicative.
from ocr_benchmark.cni_preprocessing import prepare_cni_source


def sha256_file(path: Path) -> str:
    """Calcule l'empreinte du fichier sans le charger entièrement en mémoire."""

    ## SHA-256 permet de confirmer que deux essais utilisent exactement le même fichier.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        ## Lecture par blocs de 1 Mio pour ne pas saturer la mémoire avec un gros PDF.
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_image(source: Path, output: Path, dpi: int) -> dict:
    """Appelle exactement la fonction de normalisation du workflow CNI."""

    ## Le chronomètre couvre conversion PDF/image et écriture du PNG.
    started = time.perf_counter()
    ## prepare_cni_source accepte les formats reconnus par le workflow CNI.
    metadata = prepare_cni_source(source, output, dpi)
    ## Pillow rouvre le résultat pour vérifier qu'il s'agit d'une image valide.
    with Image.open(output) as image:
        image_info = {
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        }
    ## Le rapport compare source et résultat : taille, hash et dimensions.
    return {
        "status": "success",
        "elapsed_seconds": time.perf_counter() - started,
        "source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            **image_info,
        },
        "workflow_metadata": metadata,
    }


def main() -> None:
    ## `source` est obligatoire et peut être un PDF, JPEG ou PNG.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Document PDF/JPEG/PNG à préparer.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI utilisé pour rendre une page PDF ; sans effet destructif volontaire sur une image.",
    )
    parser.add_argument("--output", type=Path, help="Chemin du PNG préparé.")
    parser.add_argument("--report", type=Path, help="Chemin du diagnostic JSON.")
    args = parser.parse_args()

    ## Sans --output, le résultat est rangé dans un dossier de run dédié.
    output = args.output or (
        ROOT_DIR / "runs" / "runtime_debug" / "prepared" / f"{args.source.stem}.png"
    )
    ## Les erreurs de lecture et de conversion deviennent un JSON exploitable.
    try:
        if not args.source.is_file():
            raise FileNotFoundError(args.source)
        report = prepare_image(args.source, output, args.dpi)
    except Exception as exc:
        report = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
    ## Le rapport est toujours écrit, y compris en cas d'échec.
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    report_path = args.report or output.with_suffix(".diagnostic.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    print(f"\nRapport écrit dans : {report_path}")


## Point d'entrée du script autonome.
if __name__ == "__main__":
    main()
