"""Gestion des runs persistants et de leur rétention locale.

Ce stockage par fichiers convient à une instance interne unique. Les fonctions
ne connaissent ni Gradio ni un modèle OCR et seront remplaçables par un dépôt
PostgreSQL/MinIO lorsque cette infrastructure sera disponible.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

_RUN_DIRECTORY = re.compile(r"^(?:cni-)?\d{8}-\d{6}-[a-f0-9]{8}$")


def list_run_ids(runs_root: Path) -> list[str]:
    """Liste uniquement les dossiers de run générés par l'application."""
    if not runs_root.is_dir():
        return []
    return sorted(
        (path.name for path in runs_root.iterdir() if path.is_dir() and _RUN_DIRECTORY.fullmatch(path.name)),
        reverse=True,
    )


def load_run_results(runs_root: Path, run_id: str) -> list[dict[str, Any]]:
    """Charge un run en empêchant toute traversée de chemin depuis l'UI."""
    if not _RUN_DIRECTORY.fullmatch(str(run_id)):
        raise ValueError("Identifiant de run invalide.")
    results_path = runs_root / run_id / "results.json"
    if not results_path.is_file():
        results_path = runs_root / run_id / "cni_results.json"
    with results_path.open("r", encoding="utf-8") as stream:
        results = json.load(stream)
    if not isinstance(results, list):
        raise ValueError("Le fichier de résultats doit contenir une liste JSON.")
    return results


def purge_expired_runs(runs_root: Path, *, retention_days: int | None, now: float | None = None) -> list[str]:
    """Supprime les runs expirés, jamais un chemin arbitraire.

    ``None`` ou une valeur négative désactive la suppression automatique.
    Une valeur de zéro supprime les runs dès le prochain démarrage, ce qui est
    utile sur une machine de démonstration sans conservation d'artefacts.
    """
    if retention_days is None or retention_days < 0 or not runs_root.is_dir():
        return []
    cutoff = (now if now is not None else time.time()) - retention_days * 86_400
    deleted: list[str] = []
    for run_id in list_run_ids(runs_root):
        directory = runs_root / run_id
        if directory.stat().st_mtime < cutoff:
            shutil.rmtree(directory, ignore_errors=False)
            deleted.append(run_id)
    return deleted
