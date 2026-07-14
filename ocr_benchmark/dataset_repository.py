from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024
_WRITE_LOCK = threading.Lock()


class DatasetRepository:
    """Validates and atomically appends user-labeled images to the catalog."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.dataset_root = self.project_root / "dataset"
        self.catalog_path = self.dataset_root / "dataset.json"
        self.upload_root = self.dataset_root / "user_uploads"

    def add_labeled_image(
        self,
        source_path: str | Path,
        ground_truth: str,
        category: str,
        description: str = "",
    ) -> dict[str, Any]:
        # Resolve the source before validation so paths cannot escape the
        # caller's filesystem silently. The destination is always generated
        # below our controlled ``dataset/user_uploads`` directory.
        source = Path(source_path).resolve()
        label = ground_truth.strip()
        normalized_category = category.strip().lower().replace(" ", "_")

        if not source.is_file():
            raise ValueError("Le fichier image est introuvable.")
        if source.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("Format invalide. Utilisez JPG, JPEG, PNG ou WEBP.")
        if source.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("L’image dépasse la limite de 15 Mio.")
        if not label:
            raise ValueError("Le label / ground truth est obligatoire.")
        if len(label) > 100_000:
            raise ValueError("Le label dépasse 100 000 caractères.")
        if not normalized_category:
            raise ValueError("La catégorie est obligatoire.")

        try:
            with Image.open(source) as image:
                image.verify()
        except Exception as exc:
            raise ValueError("Le fichier n’est pas une image valide.") from exc

        self.upload_root.mkdir(parents=True, exist_ok=True)
        destination = self.upload_root / f"{uuid.uuid4().hex}{source.suffix.lower()}"
        relative_path = destination.relative_to(self.project_root).as_posix()
        record = {
            "image_path": relative_path,
            "ground_truth": label,
            "category": normalized_category,
            "description": description.strip() or "Donnée ajoutée manuellement.",
            "source": "user_upload",
        }

        # The lock protects concurrent Gradio callbacks. Copying the image and
        # replacing the JSON catalogue are one logical operation: on failure,
        # remove the copied file so the catalogue never points at a ghost file.
        with _WRITE_LOCK:
            catalog = self._read_catalog()
            shutil.copy2(source, destination)
            try:
                catalog.append(record)
                self._atomic_write(catalog)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        return record

    def _read_catalog(self) -> list[dict[str, Any]]:
        with self.catalog_path.open("r", encoding="utf-8") as stream:
            catalog = json.load(stream)
        if not isinstance(catalog, list):
            raise ValueError("Le catalogue dataset.json doit être une liste.")
        return catalog

    def _atomic_write(self, catalog: list[dict[str, Any]]) -> None:
        # os.replace is atomic on the same filesystem, so readers see either
        # the old complete catalogue or the new complete catalogue, never a
        # partially-written JSON document.
        temporary = self.catalog_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.catalog_path)
