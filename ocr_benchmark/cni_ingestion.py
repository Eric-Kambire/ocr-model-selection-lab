"""CNI input discovery, external-label materialisation and safe ZIP import.

The folder name is the only canonical client identifier. A PDF prefix is kept
as document metadata, but is deliberately never used to find a label.
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
_SIDE_FILENAME = {
    # ``document_id`` may differ from the parent folder/client ID. Keeping the
    # two concepts separate prevents a scanner-generated filename from breaking
    # the label association.
    "recto": re.compile(r"^(?P<document_id>.+)_cin_recto\.pdf$", re.IGNORECASE),
    "verso": re.compile(r"^(?P<document_id>.+)_cin_verso\.pdf$", re.IGNORECASE),
}


def scan_cni_clients(clients_root: Path, labels_root: Path | None = None) -> list[dict[str, Any]]:
    """Build one readiness record per client subfolder.

    ``status`` concerns only input validity (one recto and one verso PDF).
    ``label_status`` is separate so an OCR run can proceed even when the future
    accuracy score is unavailable. The returned records are plain dictionaries
    on purpose: they can be rendered by Gradio and persisted without a custom
    serializer.
    """
    if not clients_root.is_dir():
        raise FileNotFoundError(f"Clients folder not found: {clients_root}")
    records: list[dict[str, Any]] = []
    for folder in sorted(path for path in clients_root.iterdir() if path.is_dir()):
        record: dict[str, Any] = {
            "folder_client_id": folder.name, "client_dir": str(folder), "recto_pdf": None, "verso_pdf": None,
            "recto_document_id": None, "verso_document_id": None,
            "label_source": str(labels_root / f"{folder.name}.jsonb") if labels_root else None,
            "label_path": str(folder / f"{folder.name}.json"),
            "label_status": "label_root_not_set" if labels_root is None else "label_not_found",
            "status": "ready", "issues": [],
        }
        # Only the two strict PDF patterns belong to the benchmark contract;
        # source PNGs or arbitrary files may coexist in the client directory.
        for candidate in folder.iterdir():
            if not candidate.is_file():
                continue
            for side, matcher in _SIDE_FILENAME.items():
                match = matcher.match(candidate.name)
                if not match:
                    continue
                if record[f"{side}_pdf"] is not None:
                    record["issues"].append(f"duplicate_{side}_pdf")
                else:
                    record[f"{side}_pdf"] = str(candidate)
                    record[f"{side}_document_id"] = match.group("document_id")
        for side in ("recto", "verso"):
            if record[f"{side}_pdf"] is None:
                record["issues"].append(f"missing_{side}_pdf")
        if record["recto_document_id"] and record["verso_document_id"] and record["recto_document_id"] != record["verso_document_id"]:
            record["issues"].append("document_id_differs_between_sides")
        if labels_root and Path(record["label_source"]).is_file():
            record["label_status"] = "label_available"
        if record["issues"]:
            record["status"] = "invalid_input"
        records.append(record)
    LOGGER.info("CNI client scan complete | root=%s | clients=%d", clients_root, len(records))
    return records


def materialize_cni_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy valid external UTF-8 JSONB text into the matching client folder.

    The current integration assumes ``.jsonb`` contains text JSON. It parses
    before replacing the target so a corrupt label never destroys a previously
    materialised JSON file. Missing or malformed labels remain explicit status
    values rather than silent empty dictionaries.
    """
    updated: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        source_text, target_text = record.get("label_source"), record.get("label_path")
        if not source_text or not target_text:
            updated.append(record)
            continue
        source, target = Path(source_text), Path(target_text)
        if not source.is_file():
            record["label_status"] = "label_not_found"
            updated.append(record)
            continue
        try:
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
    """Extract a portable test archive below ``imports_root``.

    Archives are user input. Absolute paths and ``..`` components are rejected
    before writing anything, which prevents ZIP-slip writes outside the local
    import directory. A failed import removes its partial destination.
    """
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
        shutil.rmtree(destination, ignore_errors=True)
        raise
    LOGGER.info("CNI ZIP imported | archive=%s | destination=%s | files=%d", zip_path, destination, extracted)
    return {"import_root": str(destination), "files": extracted}


def write_cni_json(path: Path, value: dict[str, Any]) -> None:
    """Persist one CNI artefact atomically as readable UTF-8 JSON."""
    _atomic_write_json(path, value)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write to a sibling temporary file, then replace the final artefact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
