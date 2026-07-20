"""Découverte des entrées CNI, import des labels JSONB et ZIP sécurisé.

Le nom du dossier est l'identifiant client canonique. Le préfixe du PDF reste
une métadonnée et ne sert jamais à rechercher un label.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

LOGGER = logging.getLogger(__name__)
DEFAULT_RECTO_SUFFIX = "_CIN_Recto"
DEFAULT_VERSO_SUFFIX = "_CIN_Verso"
SUPPORTED_CNI_SOURCE_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg")


def _side_patterns(recto_suffix: str, verso_suffix: str) -> dict[str, re.Pattern[str]]:
    """Construit des patrons sûrs pour les sources PDF, JPEG et PNG."""
    suffixes = {"recto": str(recto_suffix or "").strip(), "verso": str(verso_suffix or "").strip()}
    if not all(suffixes.values()):
        raise ValueError("Les suffixes recto et verso ne peuvent pas être vides.")
    extensions = "|".join(re.escape(value) for value in SUPPORTED_CNI_SOURCE_SUFFIXES)
    return {side: re.compile(rf"^(?P<document_id>.+){re.escape(suffix)}(?P<extension>{extensions})$", re.IGNORECASE) for side, suffix in suffixes.items()}


def scan_cni_clients(
    clients_root: Path,
    labels_root: Path | None = None,
    *,
    recto_suffix: str = DEFAULT_RECTO_SUFFIX,
    verso_suffix: str = DEFAULT_VERSO_SUFFIX,
) -> list[dict[str, Any]]:
    """Construit un diagnostic d'entrée pour chaque sous-dossier client."""
    if not clients_root.is_dir():
        raise FileNotFoundError(f"Clients folder not found: {clients_root}")
    records: list[dict[str, Any]] = []
    side_filename = _side_patterns(recto_suffix, verso_suffix)
    for folder in sorted(path for path in clients_root.iterdir() if path.is_dir()):
        record: dict[str, Any] = {
            "folder_client_id": folder.name, "client_dir": str(folder),
            "recto_source": None, "verso_source": None, "recto_format": None, "verso_format": None,
            "recto_pdf": None, "verso_pdf": None,
            "recto_document_id": None, "verso_document_id": None,
            "label_source": str(labels_root / f"{folder.name}.jsonb") if labels_root else None,
            "label_path": str(folder / f"{folder.name}.json"),
            "label_status": "label_root_not_set" if labels_root is None else "label_not_found",
            "status": "ready", "issues": [],
        }
        # Seuls les deux PDF respectant le contrat comptent. Les PNG source ou
        # autres fichiers peuvent rester dans le dossier sans gêner le scan.
        for candidate in folder.iterdir():
            if not candidate.is_file():
                continue
            for side, matcher in side_filename.items():
                match = matcher.match(candidate.name)
                if not match:
                    continue
                if record[f"{side}_source"] is not None:
                    record["issues"].append(f"duplicate_{side}_source")
                else:
                    record[f"{side}_source"] = str(candidate)
                    record[f"{side}_format"] = candidate.suffix.lower().lstrip(".")
                    record[f"{side}_pdf"] = str(candidate)
                    record[f"{side}_document_id"] = match.group("document_id")
        # L'absence d'une face invalide l'entrée, sans effacer les informations
        # déjà relevées : le tableau de diagnostic reste donc exploitable.
        for side in ("recto", "verso"):
            if record[f"{side}_source"] is None:
                record["issues"].append(f"missing_{side}_source")
        if record["recto_document_id"] and record["verso_document_id"] and record["recto_document_id"] != record["verso_document_id"]:
            record["issues"].append("document_id_differs_between_sides")
        if labels_root and Path(record["label_source"]).is_file():
            record["label_status"] = "label_available"
        elif Path(record["label_path"]).is_file():
            record["label_status"] = "label_materialized"
        if record["issues"]:
            record["status"] = "invalid_input"
        records.append(record)
    LOGGER.info("CNI client scan complete | root=%s | clients=%d", clients_root, len(records))
    return records


def materialize_cni_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copie un JSONB texte UTF-8 valide dans le dossier client correspondant."""
    updated: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        source_text, target_text = record.get("label_source"), record.get("label_path")
        if not source_text or not target_text:
            updated.append(record)
            continue
        source, target = Path(source_text), Path(target_text)
        if not source.is_file():
            # Un label manquant n'empêche pas l'extraction OCR ; il rend
            # simplement l'évaluation d'accuracy indisponible pour ce client.
            record["label_status"] = "label_not_found"
            updated.append(record)
            continue
        try:
            # Parser avant l'écriture protège un JSON déjà matérialisé contre
            # un nouveau fichier JSONB illisible ou incomplet.
            value = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(value, (dict, list)):
                raise ValueError("JSON label must be an object or array")
            _atomic_write_json(target, value)
            record["label_status"] = "label_materialized"
            LOGGER.info("CNI label materialized | client=%s | target=%s", record["folder_client_id"], target)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            record["label_status"] = "label_parse_failed"
            record["issues"] = [*record.get("issues", []), f"label_parse_failed:{type(exc).__name__}"]
            LOGGER.warning("CNI label parsing failed | client=%s | source=%s | error=%s", record["folder_client_id"], source, exc)
        updated.append(record)
    return updated


def import_cni_zip(zip_path: Path, imports_root: Path) -> dict[str, Any]:
    """Extrait une archive de test dans le répertoire local d'import."""
    if not zip_path.is_file() or zip_path.suffix.lower() != ".zip":
        raise ValueError("A .zip archive is required for CNI import.")
    imports_root.mkdir(parents=True, exist_ok=True)
    destination = imports_root / f"cni-import-{time.strftime('%Y%m%d-%H%M%S')}"
    destination.mkdir(parents=True, exist_ok=False)
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                # Protection ZIP-slip : une archive ne peut pas écrire hors du
                # dossier d'import avec un chemin absolu ou contenant ``..``.
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")
                if member.is_dir():
                    continue
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
    except Exception:
        # Ne jamais laisser un import partiel qui serait ensuite pris pour un
        # jeu de test valide par le scanner.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    LOGGER.info("CNI ZIP imported | archive=%s | destination=%s | files=%d", zip_path, destination, extracted)
    return {"import_root": str(destination), "files": extracted}


def write_cni_json(path: Path, value: dict[str, Any]) -> None:
    """Écrit un artefact CNI en JSON UTF-8 lisible de façon atomique."""
    _atomic_write_json(path, value)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Écrit d'abord un temporaire voisin, puis remplace l'artefact final."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    # ``replace`` évite qu'un refresh de l'interface lise un JSON à moitié écrit.
    temporary.replace(path)
