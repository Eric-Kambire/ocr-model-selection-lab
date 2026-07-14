"""File-oriented preparation helpers for Moroccan CNI benchmark runs.

The module intentionally uses small typed functions and dictionaries instead of
an object hierarchy.  It never uses the identifier embedded in a PDF filename
to look up a label: the directory name is the authoritative client identifier.
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

from PIL import Image, ImageDraw, ImageOps

LOGGER = logging.getLogger(__name__)

RECTO_FIELDS = (
    "cin",
    "nom",
    "prenom",
    "date_naissance",
    "ville_naissance",
    "date_validite",
)
VERSO_FIELDS = ("cin", "date_validite", "adresse")
DEFAULT_CNI_FIELD_CONFIG = {
    "recto": [
        {"key": "cin", "type": "text"},
        {"key": "nom", "type": "text"},
        {"key": "prenom", "type": "text"},
        {"key": "date_naissance", "type": "date"},
        {"key": "ville_naissance", "type": "text"},
        {"key": "date_validite", "type": "date"},
    ],
    "verso": [
        {"key": "cin", "type": "text"},
        {"key": "date_validite", "type": "date"},
        {"key": "adresse", "type": "text"},
    ],
}
_SIDE_FILENAME = {
    "recto": re.compile(r"^(?P<document_id>.+)_cin_recto\.pdf$", re.IGNORECASE),
    "verso": re.compile(r"^(?P<document_id>.+)_cin_verso\.pdf$", re.IGNORECASE),
}


def load_cni_field_config(config_path: Path | None = None) -> dict[str, list[dict[str, str]]]:
    """Return the editable CNI field configuration.

    Args:
        config_path: Optional JSON configuration.  If omitted or unavailable,
            the stable first-version field list is returned.

    Returns:
        A mapping with ``recto`` and ``verso`` lists.  Each item has a ``key``
        and a logical ``type``.  Invalid files raise ``ValueError`` rather than
        silently changing the extraction contract.
    """
    if config_path is None or not config_path.is_file():
        return json.loads(json.dumps(DEFAULT_CNI_FIELD_CONFIG))
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid CNI field configuration: {config_path}: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(value.get(side), list) for side in ("recto", "verso")):
        raise ValueError("CNI field configuration must contain recto and verso arrays.")
    for side in ("recto", "verso"):
        for item in value[side]:
            if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                raise ValueError(f"Invalid field declaration for {side}.")
    return value


def build_cni_prompt(side: str, fields: dict[str, list[dict[str, str]]] | None = None) -> str:
    """Build the strict JSON-only prompt for one CNI side.

    Values must come from Latin/French text visible on the card.  The Arabic
    text is useful to understand labels but must not be translated or invented.
    """
    if side not in {"recto", "verso"}:
        raise ValueError("side must be 'recto' or 'verso'.")
    config = fields or load_cni_field_config()
    keys = [str(item["key"]) for item in config[side]]
    schema = {key: None for key in keys}
    return (
        "You extract structured data from the " + side.upper() + " side of a Moroccan national identity card (CNI).\n"
        "Read only the Latin/French value printed on the card. Arabic text may help identify a label, "
        "but do not translate, transliterate, guess, or add fields.\n"
        "Return ONLY one valid JSON object. Do not use Markdown, code fences, comments, or prose.\n"
        "For an unreadable or absent value, use null. Preserve spelling. Format a clearly readable date as YYYY-MM-DD.\n"
        "Required JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def build_combined_cni_prompt(fields: dict[str, list[dict[str, str]]] | None = None) -> str:
    """Build the prompt for a composite image with recto above verso."""
    config = fields or load_cni_field_config()
    schema = {
        "recto": {str(item["key"]): None for item in config["recto"]},
        "verso": {str(item["key"]): None for item in config["verso"]},
    }
    return (
        "The image contains two sides of the same Moroccan national identity card: RECTO at the top and VERSO at the bottom.\n"
        "Extract only the Latin/French values visible on each side. Do not translate Arabic text, infer missing values, "
        "or add fields. Return ONLY one valid JSON object, with null for unreadable values.\n"
        "Format a clearly readable date as YYYY-MM-DD. Required schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def scan_cni_clients(clients_root: Path, labels_root: Path | None = None) -> list[dict[str, Any]]:
    """Scan client folders and report their PDF/label readiness.

    The parent directory name is the canonical ``folder_client_id``.  PDF
    prefixes are captured only as document metadata and are deliberately not
    compared with that identifier.
    """
    if not clients_root.is_dir():
        raise FileNotFoundError(f"Clients folder not found: {clients_root}")

    records: list[dict[str, Any]] = []
    for folder in sorted(path for path in clients_root.iterdir() if path.is_dir()):
        record: dict[str, Any] = {
            "folder_client_id": folder.name,
            "client_dir": str(folder),
            "recto_pdf": None,
            "verso_pdf": None,
            "recto_document_id": None,
            "verso_document_id": None,
            "label_source": str(labels_root / f"{folder.name}.jsonb") if labels_root else None,
            "label_path": str(folder / f"{folder.name}.json"),
            "label_status": "label_root_not_set" if labels_root is None else "label_not_found",
            "status": "ready",
            "issues": [],
        }
        for candidate in folder.iterdir():
            if not candidate.is_file():
                continue
            for side, matcher in _SIDE_FILENAME.items():
                match = matcher.match(candidate.name)
                if match:
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
    """Convert external text JSONB files to JSON beside matching client folders.

    Records without a source label remain processable and receive an explicit
    ``label_not_found`` status. Existing labels are atomically replaced only
    after the external JSON has been parsed successfully.
    """
    updated: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        source_text = record.get("label_source")
        target_text = record.get("label_path")
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
    """Safely extract a portable CNI test archive outside the Git dataset.

    ZIP entries containing an absolute path or ``..`` are rejected.  The caller
    receives the extraction directory and can choose its ``clients`` subfolder
    as ``CLIENTS_ROOT`` in the interface.
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


def render_single_page_pdf(pdf_path: Path, output_path: Path, dpi: int = 300) -> dict[str, Any]:
    """Render exactly one PDF page to PNG using PyMuPDF.

    Raises a clear error when the PDF is empty or contains more than one page;
    the CNI input contract is deliberately one face per one-page PDF.
    """
    if dpi < 72 or dpi > 600:
        raise ValueError("CNI render DPI must be between 72 and 600.")
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render CNI PDFs. Install requirements.txt.") from exc
    with fitz.open(pdf_path) as document:
        if document.page_count != 1:
            raise ValueError(f"Expected exactly one PDF page, found {document.page_count}: {pdf_path.name}")
        page = document.load_page(0)
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(output_path))
    with Image.open(output_path) as image:
        width, height = image.size
    return {"image_path": str(output_path), "width": width, "height": height, "dpi": dpi}


def crop_cni_from_a4(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Try to crop the non-white CNI area from an A4 scan.

    The heuristic is intentionally conservative. If the detected region does
    not resemble a card, the original page is copied to the output and the
    returned status makes this fallback visible to the UI and logs.
    """
    with Image.open(source_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    gray = ImageOps.grayscale(original)
    mask = gray.point(lambda pixel: 255 if pixel < 242 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return _copy_full_page(original, output_path, "crop_not_detected")

    left, top, right, bottom = bbox
    padding = max(12, int(max(original.size) * 0.015))
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(original.width, right + padding), min(original.height, bottom + padding)
    width, height = right - left, bottom - top
    ratio = width / height if height else 0
    coverage = (width * height) / (original.width * original.height)
    # ID-1 card ratio is ~1.586. Perspective and scanner shadows justify a
    # tolerant interval, while a full A4 page must always remain a fallback.
    if not (1.20 <= ratio <= 2.05) or coverage > 0.65 or coverage < 0.02:
        return _copy_full_page(original, output_path, "crop_fallback_full_page")

    cropped = original.crop((left, top, right, bottom))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path, format="PNG")
    return {
        "image_path": str(output_path),
        "crop_status": "crop_detected",
        "crop_box": [left, top, right, bottom],
        "coverage": round(coverage, 4),
    }


def build_vertical_cni_composite(recto_path: Path, verso_path: Path, output_path: Path) -> str:
    """Create a single image with recto above verso for the combined strategy."""
    with Image.open(recto_path) as source:
        recto = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(verso_path) as source:
        verso = ImageOps.exif_transpose(source).convert("RGB")
    target_width = max(recto.width, verso.width)
    recto = _resize_to_width(recto, target_width)
    verso = _resize_to_width(verso, target_width)
    separator_height = 36
    canvas = Image.new("RGB", (target_width, recto.height + separator_height + verso.height), "white")
    canvas.paste(recto, (0, 0))
    canvas.paste(verso, (0, recto.height + separator_height))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, recto.height + separator_height // 2, target_width, recto.height + separator_height // 2), fill="black", width=2)
    draw.text((8, 8), "RECTO", fill="black")
    draw.text((8, recto.height + separator_height + 8), "VERSO", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return str(output_path)


def parse_cni_json_response(raw_text: str, side: str, fields: dict[str, list[dict[str, str]]] | None = None) -> tuple[dict[str, str | None], str | None]:
    """Parse a model JSON reply into exactly the configured keys for one side."""
    if side not in {"recto", "verso"}:
        raise ValueError("side must be 'recto' or 'verso'.")
    config = fields or load_cni_field_config()
    keys = [str(item["key"]) for item in config[side]]
    empty = {key: None for key in keys}
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        return empty, "model_response_not_json"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return empty, "model_response_invalid_json"
    if not isinstance(parsed, dict):
        return empty, "model_response_not_object"
    return ({key: _string_or_none(parsed.get(key)) for key in keys}, None)


def parse_combined_cni_json_response(raw_text: str, fields: dict[str, list[dict[str, str]]] | None = None) -> tuple[dict[str, str | None], dict[str, str | None], str | None]:
    """Parse a composite-image response containing nested recto and verso JSON."""
    config = fields or load_cni_field_config()
    candidate = _extract_json_object(raw_text)
    empty_recto = {str(item["key"]): None for item in config["recto"]}
    empty_verso = {str(item["key"]): None for item in config["verso"]}
    if candidate is None:
        return empty_recto, empty_verso, "model_response_not_json"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return empty_recto, empty_verso, "model_response_invalid_json"
    if not isinstance(parsed, dict):
        return empty_recto, empty_verso, "model_response_not_object"
    recto = parsed.get("recto") if isinstance(parsed.get("recto"), dict) else {}
    verso = parsed.get("verso") if isinstance(parsed.get("verso"), dict) else {}
    return (
        {key: _string_or_none(recto.get(key)) for key in empty_recto},
        {key: _string_or_none(verso.get(key)) for key in empty_verso},
        None,
    )


def build_cni_global_json(client: dict[str, Any], recto: dict[str, str | None], verso: dict[str, str | None]) -> dict[str, Any]:
    """Merge side extraction without comparing it to the external label."""
    cin_value, cin_coherent = _merge_cross_side_value(recto.get("cin"), verso.get("cin"))
    validity_value, validity_coherent = _merge_cross_side_value(recto.get("date_validite"), verso.get("date_validite"))
    return {
        "folder_client_id": client["folder_client_id"],
        "recto_document_id": client.get("recto_document_id"),
        "verso_document_id": client.get("verso_document_id"),
        "label_status": client.get("label_status"),
        "recto": recto,
        "verso": verso,
        "cin_recto": recto.get("cin"),
        "cin_verso": verso.get("cin"),
        "cin_fusionne": cin_value,
        "cin_coherent": cin_coherent,
        "nom": recto.get("nom"),
        "prenom": recto.get("prenom"),
        "date_naissance": recto.get("date_naissance"),
        "ville_naissance": recto.get("ville_naissance"),
        "date_validite_recto": recto.get("date_validite"),
        "date_validite_verso": verso.get("date_validite"),
        "date_validite_fusionnee": validity_value,
        "date_validite_coherente": validity_coherent,
        "adresse": verso.get("adresse"),
    }


def write_cni_json(path: Path, value: dict[str, Any]) -> None:
    """Persist one CNI artefact atomically with UTF-8 JSON formatting."""
    _atomic_write_json(path, value)


def _copy_full_page(image: Image.Image, output_path: Path, status: str) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return {"image_path": str(output_path), "crop_status": status, "crop_box": None, "coverage": None}


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _extract_json_object(text: str) -> str | None:
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:]).strip()
    first, last = value.find("{"), value.rfind("}")
    return value[first : last + 1] if first >= 0 and last > first else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text or None
    return None


def _merge_cross_side_value(recto_value: str | None, verso_value: str | None) -> tuple[str | None, bool | None]:
    if recto_value and verso_value:
        if _normalise_for_match(recto_value) == _normalise_for_match(verso_value):
            return recto_value, True
        return None, False
    return recto_value or verso_value, None


def _normalise_for_match(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
