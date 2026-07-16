"""Regroupe des paires PDF recto/verso dans des dossiers clients aléatoires.

Une paire est reconnue lorsque les deux fichiers ont le même préfixe avant le
dernier mot ``Recto`` ou ``Verso``. Exemples reconnus :

    123_CIN_Recto.pdf + 123_CIN_Verso.pdf
    scan-456-recto.pdf + scan-456-verso.pdf

Les PDF sont copiés par défaut ; l'option ``--move`` doit être choisie
explicitement pour déplacer les originaux.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Literal


# Le mot recto/verso doit être le dernier élément du nom, avant l'extension.
# Les séparateurs usuels sont volontairement tous acceptés.
SIDE_PATTERN = re.compile(r"^(?P<pair_key>.+?)[\s_.-]+(?P<side>recto|verso)$", re.IGNORECASE)


# Les paires restent de simples dictionnaires pour garder le script direct et
# facilement débogable, sans introduire de classe métier supplémentaire.
PdfPair = dict[str, str | Path]


def classify_pdf(pdf_path: Path) -> tuple[str, Literal["recto", "verso"]] | None:
    """Retourne la clé de paire et la face lue dans le nom d'un PDF.

    Args:
        pdf_path: PDF à classifier selon son nom de fichier.

    Returns:
        ``(clé, face)`` lorsque le nom se termine par recto/verso ; sinon ``None``.
    """
    match = SIDE_PATTERN.fullmatch(pdf_path.stem)
    if not match:
        return None
    pair_key = match.group("pair_key").strip(" ._-")
    if not pair_key:
        return None
    side = match.group("side").lower()
    return pair_key, "recto" if side == "recto" else "verso"


def find_pdf_pairs(input_dir: Path, recursive: bool = False) -> tuple[list[PdfPair], list[Path], list[str]]:
    """Détecte les paires recto/verso et liste les anomalies sans les masquer.

    Args:
        input_dir: Dossier qui contient les PDF source.
        recursive: Inclut les sous-dossiers quand cette option est activée.

    Returns:
        Trois listes : paires complètes, PDF non appariés, et doublons/anomalies.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Dossier introuvable : {input_dir}")

    candidates = input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf")
    grouped: dict[str, dict[str, Path]] = {}
    unmatched: list[Path] = []
    issues: list[str] = []

    for pdf_path in sorted(candidates, key=lambda path: str(path.relative_to(input_dir)).lower()):
        classified = classify_pdf(pdf_path)
        if classified is None:
            unmatched.append(pdf_path)
            continue
        pair_key, side = classified
        key = pair_key.casefold()
        faces = grouped.setdefault(key, {})
        if side in faces:
            issues.append(f"Doublon {side} pour '{pair_key}' : {faces[side].name} et {pdf_path.name}")
            continue
        faces[side] = pdf_path

    pairs: list[PdfPair] = []
    for key, faces in grouped.items():
        if "recto" not in faces or "verso" not in faces:
            unmatched.extend(faces.values())
            missing = "verso" if "recto" in faces else "recto"
            issues.append(f"Paire incomplète '{key}' : {missing} absent")
            continue
        recto_classification = classify_pdf(faces["recto"])
        if recto_classification is None:  # Protection : impossible après le regroupement ci-dessus.
            issues.append(f"Nom recto invalide : {faces['recto'].name}")
            continue
        pairs.append(
            {
                "pair_key": recto_classification[0],
                "recto": faces["recto"],
                "verso": faces["verso"],
            }
        )
    return pairs, unmatched, issues


def generate_client_id(existing_ids: set[str], prefix: str = "client-", length: int = 12) -> str:
    """Crée un identifiant aléatoire unique pour un dossier client.

    Args:
        existing_ids: Identifiants déjà réservés dans le dossier de sortie.
        prefix: Préfixe lisible du dossier client.
        length: Nombre de caractères hexadécimaux aléatoires.

    Returns:
        Identifiant unique, par exemple ``client-a1b2c3d4e5f6``.
    """
    if length < 6:
        raise ValueError("La longueur d'identifiant doit être au moins 6.")
    while True:
        candidate = f"{prefix}{uuid.uuid4().hex[:length]}"
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate


def create_client_directories(
    pairs: list[PdfPair],
    output_dir: Path,
    *,
    move_files: bool = False,
    id_prefix: str = "client-",
    id_length: int = 12,
) -> list[dict[str, str]]:
    """Copie ou déplace les paires dans leurs dossiers clients et écrit le mapping.

    Les noms de PDF restent inchangés. Ainsi, la détection ultérieure de
    ``Recto`` et ``Verso`` par le scanner CNI continue à fonctionner.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = {path.name for path in output_dir.iterdir() if path.is_dir()}
    mapping: list[dict[str, str]] = []

    for pair in pairs:
        client_id = generate_client_id(existing_ids, prefix=id_prefix, length=id_length)
        client_dir = output_dir / client_id
        client_dir.mkdir()
        copied_paths: dict[str, str] = {}
        for side in ("recto", "verso"):
            source = pair[side]
            if not isinstance(source, Path):  # Protection explicite du contrat interne.
                raise TypeError(f"Le chemin {side} doit être un Path, reçu : {type(source).__name__}")
            destination = client_dir / source.name
            if move_files:
                shutil.move(str(source), str(destination))
            else:
                shutil.copy2(source, destination)
            copied_paths[side] = str(destination.relative_to(output_dir))
        mapping.append(
            {
                "folder_client_id": client_id,
                "source_pair_key": pair["pair_key"],
                "recto_pdf": copied_paths["recto"],
                "verso_pdf": copied_paths["verso"],
                "operation": "move" if move_files else "copy",
            }
        )

    # Le manifest est indispensable : l'ID aléatoire reste traçable vers le
    # numéro d'origine même si les PDF sont anonymisés dans l'interface.
    manifest_path = output_dir / "client_mapping.json"
    manifest_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    """Construit les options du script de regroupement."""
    parser = argparse.ArgumentParser(
        description="Regroupe les paires PDF Recto/Verso dans des dossiers clients aléatoires."
    )
    parser.add_argument("input_dir", help="Dossier plat contenant les PDF Recto/Verso.")
    parser.add_argument(
        "--output-dir",
        help="Dossier racine clients créé. Par défaut : <input_dir>/clients_generated.",
    )
    parser.add_argument("--recursive", action="store_true", help="Cherche aussi les PDF dans les sous-dossiers.")
    parser.add_argument("--move", action="store_true", help="Déplace les PDF au lieu de les copier.")
    parser.add_argument("--id-prefix", default="client-", help="Préfixe des dossiers aléatoires.")
    parser.add_argument("--id-length", type=int, default=12, help="Longueur aléatoire hexadécimale (défaut : 12).")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les paires sans créer de dossier.")
    return parser


def main() -> None:
    """Point d'entrée : trouve les paires, signale les erreurs et crée les clients."""
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir / "clients_generated"

    pairs, unmatched, issues = find_pdf_pairs(input_dir, recursive=args.recursive)
    print(f"Paires complètes détectées : {len(pairs)}")
    for pair in pairs:
        print(f"  [paire] {pair['pair_key']} → {pair['recto'].name} + {pair['verso'].name}")
    for issue in issues:
        print(f"  [alerte] {issue}")
    for pdf_path in unmatched:
        print(f"  [non apparié] {pdf_path}")

    if args.dry_run:
        print("\nDry-run : aucune donnée n'a été copiée ou déplacée.")
        return
    if not pairs:
        print("\nAucune paire valide : aucun dossier client créé.")
        return

    mapping = create_client_directories(
        pairs,
        output_dir,
        move_files=args.move,
        id_prefix=args.id_prefix,
        id_length=args.id_length,
    )
    print(f"\nTerminé : {len(mapping)} dossier(s) client créés dans {output_dir}")
    print(f"Correspondance écrite dans : {output_dir / 'client_mapping.json'}")


if __name__ == "__main__":
    main()
